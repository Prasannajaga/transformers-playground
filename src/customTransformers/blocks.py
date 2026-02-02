import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention , MultiQueryAttention, GroupedQueryAttention
from FFN import FFN_REGISTRY

class MHA_BLOCK(nn.Module): 
    """Transformer block: attention + MLP"""

    def __init__(self, n_embd, n_head, ffn_type: str = 'relu'):
        super().__init__()
        assert n_embd % n_head == 0 
        assert ffn_type in FFN_REGISTRY, f"Unknown FFN type: {ffn_type}"

        head_size = n_embd // n_head

        self.attention = MultiHeadAttention(n_head, head_size)
        self.feedForward = FFN_REGISTRY[ffn_type](n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LN Transformer (stable)
        x = x + self.attention(self.ln1(x))
        x = x + self.feedForward(self.ln2(x))
        return x
    

class MQA_BLOCK(nn.Module):
    """
    Transformer block with MQA and pluggable FFN
    """

    def __init__(self, n_embd: int, n_head: int, ffn_type: str = 'relu'):
        super().__init__()
        assert n_embd % n_head == 0 
        assert ffn_type in FFN_REGISTRY, f"Unknown FFN type: {ffn_type}"

        self.attention = MultiQueryAttention(n_embd, n_head)
        self.feedForward = FFN_REGISTRY[ffn_type](n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feedForward(self.ln2(x))
        return x


class GQA_BLOCK(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, ffn_type: str = 'relu'):
        super().__init__()

        self.attn = GroupedQueryAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head
        ) 
        self.ffwd = FFN_REGISTRY[ffn_type](n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
