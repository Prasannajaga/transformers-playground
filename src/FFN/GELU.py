import torch.nn as nn
from eval.config import Config
config = Config() 

class GELU_FFN(nn.Module):
    """Position-wise feed-forward network with GELU activation"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
