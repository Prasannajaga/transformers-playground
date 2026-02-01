"""
SiLU (Swish) Feed-Forward Network

Math:
    SiLU(x) = x · σ(x)
    FFN(x) = W2 · SiLU(W1 · x)

Why:
    Smooth, non-monotonic activation.
    Preserves small negative values.

What it offers:
    • Strong gradient flow
    • Slightly better convergence than GELU
    • Common in modern architectures

Notes:
    • Still ungated
    • Often used as a building block for SwiGLU
"""



import torch.nn as nn
from eval.config import Config
config = Config()  

class SILU_FFN(nn.Module):
    """Position-wise feed-forward network with SiLU (Swish) activation"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
