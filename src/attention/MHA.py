import torch
import torch.nn as nn
import torch.nn.functional as F
from eval.config import Config  
config = Config()  

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(num_heads * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Concatenate along channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    """One head of causal self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # masking here 
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)
        self.scale = head_size ** -0.5   

    def forward(self, x):
        B, T, _ = x.shape

        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # Attention scores
        wei = (q @ k.transpose(-2, -1)) * self.scale  # (B, T, T)

        # Causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted sum
        out = wei @ v  # (B, T, head_size)
        return out

