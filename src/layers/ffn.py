import torch.nn as nn
import torch.nn.functional as F


class RELU_FFN(nn.Module):

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GELU_FFN(nn.Module):

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SILU_FFN(nn.Module):

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SWIGLU_FFN(nn.Module):

    def __init__(self, n_embd, expansion_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(expansion_ratio * n_embd)

        self.w1 = nn.Linear(n_embd, hidden_dim)
        self.w2 = nn.Linear(n_embd, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.w3(F.silu(self.w1(x)) * self.w2(x))
        )


FFN_REGISTRY = {
    "relu": RELU_FFN,
    "gelu": GELU_FFN,
    "silu": SILU_FFN,
    "swiglu": SWIGLU_FFN,
}
