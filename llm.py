import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer

from xlstm import xLSTM

class LLM(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        # Read the YAML configuration file
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Extract the configuration values
        self.num_layers = self.config['model']['num_layers']
        self.input_dim = self.config['model']['input_dim']
        self.num_heads = self.config['model']['num_heads']
        self.head_dim = self.config['model']['head_dim']
        self.projection_factor = self.config['model']['projection_factor']
        self.kernel_size = self.config['model']['kernel_size']

        # Configure the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer']['path'])
        self.tokenizer.add_special_tokens(self.config['tokenizer']['special_tokens'])
        self.vocab_size = self.tokenizer.vocab_size + len(self.tokenizer.added_tokens_decoder)

        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.input_dim)

        # xLSTM stacks
        self.model = xLSTM(
            num_layers = self.num_layers,
            input_dim = self.input_dim,
            num_heads = self.num_heads,
            head_dim = self.head_dim,
            projection_factor = self.projection_factor,
            kernel_size = self.kernel_size
        )

        # Output Layer
        self.out_proj = nn.Linear(self.input_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        lr = self.config['optimizer']['lr'] if 'lr' in self.config['optimizer'] else 1e-4
        weight_decay = self.config['optimizer']['weight_decay'] if 'weight_decay' in self.config['optimizer'] else 0.01
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.reset_parameters()

    def to(self, device):
        self = super().to(device)
        return self

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.model.reset_parameters()
        self.out_proj.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def generate(self, text, hid=None, token_limit=300, temp=0.9, top_k=50):
        self.model.eval()
        input_ids = self.tokenizer(text, return_tensors='pt', padding=True).input_ids
        input_ids = input_ids.to(self.device)

        batch_size = input_ids.shape[0]

        hid = None
        num_tokens = 0
        pred = input_ids

        # Create a tensor with process IDs, one for each item in the batch
        process_ids = torch.arange(batch_size, device=self.device)

        # Decode the input tokens and create a dictionary mapping process IDs to decoded tokens
        decoded_tokens = {
            int(pid): self.tokenizer.decode(raw, skip_special_tokens=True)
            for pid, raw in zip(process_ids, input_ids)
        }

        yield decoded_tokens

        while num_tokens < token_limit:
            if pred.ndim < 2:
                pred = pred.unsqueeze(-1)

            pred = self.embedding(pred)
            out, hid = self.model(pred, hid, batch_first=True)
            out = self.out_proj(out)
            logits = self.softmax(out)
            logits = logits[:, -1, :]

            probs = torch.softmax(logits / temp, dim=-1)
            indices = probs.topk(k=self.vocab_size - top_k, largest=False, sorted=False).indices
            probs.scatter_(dim=-1, index=indices, src=torch.zeros_like(probs))
            probs /= probs.sum(dim=-1, keepdim=True)

            pred = torch.multinomial(probs, num_samples=1, replacement=True).squeeze()

            num_tokens += 1
            mask = pred != self.tokenizer.eos_token_id

            pred = pred[mask]
            process_ids = process_ids[mask]
            hid = [[val[mask] for val in layer] for layer in hid]

            decoded_tokens = {
                int(pid): self.tokenizer.decode(raw, skip_special_tokens=True)
                for pid, raw in zip(process_ids, pred)
            }

            yield decoded_tokens