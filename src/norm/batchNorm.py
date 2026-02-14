import torch
import torch.nn as nn


class BatchNorm(nn.Module):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            dims = [i for i in range(x.dim()) if i != 1]
            batch_mean = x.mean(dim=dims)
            batch_var = x.var(dim=dims, unbiased=False)

            if self.track_running_stats:
                self.num_batches_tracked += 1
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
                )

            mean = batch_mean
            var = batch_var
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                dims = [i for i in range(x.dim()) if i != 1]
                mean = x.mean(dim=dims)
                var = x.var(dim=dims, unbiased=False)

        shape = [1] * x.dim()
        shape[1] = self.num_features
        mean = mean.view(shape)
        var = var.view(shape)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            weight = self.weight.view(shape)
            bias = self.bias.view(shape)
            x_norm = x_norm * weight + bias

        return x_norm
