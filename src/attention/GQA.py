import torch
import torch.nn as nn
import torch.nn.functional as F
from eval.config import Config  
config = Config()  

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    """

    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        assert n_embd % n_head == 0 
        assert n_head % n_kv_head == 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.groups = n_head // n_kv_head

        # Q: full heads
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)

        # K/V: fewer heads
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)

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
        q = q.transpose(1, 2)                    # (B, Hq, T, D)

        # ---- K, V ----
        k = self.k_proj(x)                       # (B, T, Hkv*D)
        v = self.v_proj(x)

        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        k = k.transpose(1, 2)                    # (B, Hkv, T, D)
        v = v.transpose(1, 2)                    # (B, Hkv, T, D)

        # ---- Expand K/V to match Q heads ----
        k = k.repeat_interleave(self.groups, dim=1)  # (B, Hq, T, D)
        v = v.repeat_interleave(self.groups, dim=1)  # (B, Hq, T, D)

        # ---- Attention ----
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, Hq, T, T)

        att = att.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # ---- Output ----
        out = att @ v                            # (B, Hq, T, D)
        out = out.transpose(1, 2).contiguous()   # (B, T, Hq, D)
        out = out.view(B, T, C)                  # (B, T, C)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out
