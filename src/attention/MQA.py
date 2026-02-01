
import torch
import torch.nn as nn
import torch.nn.functional as F
from eval.config import Config  
config = Config() 

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA)
    - Multiple Q heads
    - Single shared K and V
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        # Queries: per head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Keys & Values: SINGLE shared projection
        self.k_proj = nn.Linear(n_embd, self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.head_dim, bias=False)

        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        # ---- Q ----
        q = self.q_proj(x)                       # (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim)
        q = q.transpose(1, 2)                    # (B, H, T, D)

        # ---- K, V (shared) ----
        k = self.k_proj(x)                       # (B, T, D)
        v = self.v_proj(x)                       # (B, T, D)

        k = k.unsqueeze(1)                       # (B, 1, T, D)
        v = v.unsqueeze(1)                       # (B, 1, T, D)

        # ---- Attention ----
        att = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)

        att = att.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # ---- Output ----
        out = att @ v                             # (B, H, T, D)
        out = out.transpose(1, 2).contiguous()    # (B, T, H, D)
        out = out.view(B, T, C)                   # (B, T, C)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out
