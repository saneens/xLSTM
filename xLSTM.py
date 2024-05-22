import torch
import torch.nn as nn

from mLSTM import mLSTM

from einops import rearrange

class xLSTM(nn.Module):
    def __init__(
        self, 
        num_layers,
        vocab_size,
        input_dim, 
        num_heads, 
        head_dim, 
        projection_factor=2, 
        kernel_size=4
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_num = num_heads
        self.projection_factor = projection_factor
        self.ker_size = kernel_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, input_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(mLSTM(
                input_dim,
                num_heads,
                head_dim,
                projection_factor,
                kernel_size
            ))
        
        self.out_proj = nn.Linear(input_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()
        
        
    def forward(self, x, hid=None, batch_first=False):
        x = torch.atleast_2d(x)
        x = self.embedding(x)
        if hid is None:
            hid = self.init_hidden(x.size(0))

        if batch_first: x = rearrange(x, 'b s i -> s b i')

        out = []
        for inp in x:
            for i in range(self.num_layers):
                inp, hid[i] = self.layers[i](inp, hid[i])
            out.append(inp)

        out = torch.stack(out, dim=1 if batch_first else 0)
        out = self.out_proj(out)
        out = self.softmax(out)
        return out, hid


    def init_hidden(self, batch_size):
        hid = []
        for i in range(self.num_layers):
            hid.append(self.layers[i].init_hidden(batch_size))
        return hid


    def reset_parameters(self):
        for i in range(self.num_layers):
            self.layers[i].reset_parameters()