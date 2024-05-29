import torch
import torch.nn as nn
from math import sqrt
from torch import Tensor
from torch.nn.functional import silu, gelu

class mLSTM(nn.Module):
    def __init__(
        self, 
        input_dim, 
        num_heads, 
        head_dim, 
        projection_factor=2, 
        kernel_size=4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.projection_dim = int(projection_factor * input_dim)
        
        self.input_normalization = nn.LayerNorm(input_dim)
        self.hidden_normalization = nn.GroupNorm(num_heads, self.hidden_dim)
        
        self.left_projection = nn.Linear(input_dim, self.projection_dim)
        self.right_projection = nn.Linear(input_dim, self.hidden_dim)
        self.down_projection = nn.Linear(self.hidden_dim, input_dim)
        
        self.causal_convolution = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1))
        
        self.skip_connection = nn.Conv1d(self.projection_dim, self.hidden_dim, kernel_size=1, bias=False)
        
        self.input_gate = nn.Linear(self.projection_dim, num_heads)
        self.forget_gate = nn.Linear(self.projection_dim, num_heads)
        self.output_gate = nn.Linear(self.projection_dim, self.hidden_dim)
        
        self.query_linear = nn.Linear(self.projection_dim, self.hidden_dim)
        self.key_linear = nn.Linear(self.projection_dim, self.hidden_dim)
        self.value_linear = nn.Linear(self.projection_dim, self.hidden_dim)
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def init_hidden(self, batch_size):
        c0 = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, device=self.device)
        n0 = torch.ones(batch_size, self.num_heads, self.head_dim, device=self.device)
        m0 = torch.zeros(batch_size, self.num_heads, device=self.device)
        return c0, n0, m0
    
    def forward(self, sequence, hidden):
        c_prev, n_prev, m_prev = hidden
        batch_size = c_prev.shape[0]
        
        xn = self.input_normalization(sequence)
        lt = self.left_projection(xn)
        rt = self.right_projection(xn)

        lc = self.causal_convolution(lt.view(batch_size, 1, self.projection_dim))[..., :(self.projection_dim)]
        lc = silu(lc).squeeze()
        
        qt = self.query_linear(lc)
        kt = self.key_linear(lc) / sqrt(self.head_dim)
        vt = self.value_linear(lt)

        qt = qt.view(batch_size, self.num_heads, self.head_dim)
        kt = kt.view(batch_size, self.num_heads, self.head_dim)
        vt = vt.view(batch_size, self.num_heads, self.head_dim)
        
        it = self.input_gate(lc)
        ft = self.forget_gate(lc)
        ot = self.output_gate(lt)
        
        mt = torch.max(ft + m_prev, it)
        it = torch.exp(it - mt)
        ft = torch.exp(ft - mt + m_prev)
        ot = torch.sigmoid(ot)
        
        rem_new = vt.view(batch_size, self.num_heads, self.head_dim, 1) @ kt.view(batch_size, self.num_heads, 1, self.head_dim)
        ct = ft.view(*ft.shape, 1, 1) * c_prev + it.view(*it.shape, 1, 1) * rem_new
        nt = ft.unsqueeze(-1) * n_prev + it.unsqueeze(-1) * kt
        
        max_nqt = (
            nt.view(batch_size, self.num_heads, 1, self.head_dim) 
            @ qt.view(batch_size, self.num_heads, self.head_dim, 1)
        ).clamp(max=1).squeeze(-1)
        ht_tilda = (ct @ qt.unsqueeze(-1)).squeeze(-1) / max_nqt
        ht = ot * ht_tilda.view(batch_size, self.num_heads * self.head_dim)
        
        lc = lc.unsqueeze(-1)
        out = self.hidden_normalization(ht) + self.skip_connection(lc).squeeze()
        out = out * silu(rt)
        out = self.down_projection(out)
        
        return out + sequence, (ct, nt, mt)

    def reset_parameters(self):
        self.input_normalization.reset_parameters()
        self.left_projection.reset_parameters()
        self.right_projection.reset_parameters()
        self.down_projection.reset_parameters()
        self.causal_convolution.reset_parameters()
        self.skip_connection.reset_parameters()
        self.input_gate.reset_parameters()
        self.forget_gate.reset_parameters()
        self.output_gate.reset_parameters()
        self.query_linear.reset_parameters()
        self.key_linear.reset_parameters()
        self.value_linear.reset_parameters()