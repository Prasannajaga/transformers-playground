import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"student_logits and teacher_logits must match shape, got "
                f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
            )

        temp = self.temperature
        s_log_prob = F.log_softmax(student_logits / temp, dim=-1)
        t_prob = F.softmax(teacher_logits / temp, dim=-1)
        loss = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)

        if mask is not None:
            bool_mask = mask.bool()
            loss = loss.masked_fill(~bool_mask, 0.0)
            if self.reduction == "mean":
                denom = bool_mask.sum().clamp(min=1)
                loss = loss.sum() / denom
            elif self.reduction == "sum":
                loss = loss.sum()
        else:
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()

        return loss * (temp ** 2)


KD_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "kl_div": KLDivergenceLoss,
    "kldiv": KLDivergenceLoss,
    "kl": KLDivergenceLoss,
}


def build_kd_loss(name: str = "kl_div", **kwargs) -> nn.Module:
    key = name.strip().lower()
    if key not in KD_LOSS_REGISTRY:
        valid = ", ".join(sorted(KD_LOSS_REGISTRY.keys()))
        raise ValueError(f"Unsupported KD loss '{name}'. Supported: {valid}")
    return KD_LOSS_REGISTRY[key](**kwargs)
