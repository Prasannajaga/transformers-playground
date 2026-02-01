"""
SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network

Math:
    SwiGLU(x) = W2 · [ SiLU(Wg · x) ⊙ (Wv · x) ]

Why:
    Combines smooth activation with multiplicative gating.
    Designed for large-scale Transformer training.

What it offers:
    • Strongest FFN expressivity per parameter
    • Better quality than GELU at similar compute
    • Used in LLaMA, Mistral, PaLM

Notes:
    • Requires two projections + gating
    • Typically uses smaller expansion ratio (~2.5–3×)
"""


import torch.nn as nn
import torch.nn.functional as F
from eval.config import Config
config = Config() 

class SWIGLU_FFN(nn.Module):
    """SwiGLU Feed-Forward Network (used in LLaMA-style models)"""

    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = 4 * n_embd

        self.w1 = nn.Linear(n_embd, hidden_dim)
        self.w2 = nn.Linear(n_embd, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(
            self.w3(F.silu(self.w1(x)) * self.w2(x))
        )
