"""
ReLU Feed-Forward Network

Math:
    FFN(x) = W2 · ReLU(W1 · x)

Why:
    Original Transformer FFN.
    Simple non-linearity to introduce capacity beyond linear attention.

What it offers:
    • Cheap to compute
    • Sparse activations
    • Baseline reference

Notes:
    • Can suffer from dead neurons
    • Largely replaced by GELU / SwiGLU in modern models
""" 


import torch.nn as nn
from eval.config import Config
config = Config() 

class RELU_FFN(nn.Module):
    """Position-wise feed-forward network with RELU activation"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),   
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)