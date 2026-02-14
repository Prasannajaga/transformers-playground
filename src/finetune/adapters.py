import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, lora_alpha: int, lora_dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        if not isinstance(base, nn.Linear):
            raise TypeError("LoraLinear expects nn.Linear")

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def merge(self) -> nn.Linear:
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base.weight.data.add_(delta.to(self.base.weight.dtype))
        return self.base


class QloraLinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, lora_alpha: int, lora_dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        if not isinstance(base, nn.Linear):
            raise TypeError("QloraLinear expects nn.Linear")

        qweight, qscale = self._quantize_4bit(base.weight.detach())
        self.register_buffer("qweight", qweight)
        self.register_buffer("qscale", qscale)

        if base.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", base.bias.detach().clone())

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @staticmethod
    def _quantize_4bit(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 7.0
        qweight = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
        return qweight, scale

    def _dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.qweight.to(dtype=dtype) * self.qscale.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self._dequant_weight(x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        base = F.linear(x, base_weight, bias)
        return base + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.lora_A.weight.device,
            dtype=self.lora_A.weight.dtype,
        )
        base_weight = self._dequant_weight(self.lora_A.weight.dtype)
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        merged.weight.data.copy_(base_weight + delta)
        if self.bias is not None and merged.bias is not None:
            merged.bias.data.copy_(self.bias.to(dtype=merged.bias.dtype))
        return merged


def _get_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _apply_adapter(
    model: nn.Module,
    *,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    adapter_cls: type[nn.Module],
) -> nn.Module:
    model.requires_grad_(False)
    replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue

        parent, child_name = _get_parent(model, name)
        setattr(
            parent,
            child_name,
            adapter_cls(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
        )
        replaced += 1

    if replaced == 0:
        raise RuntimeError("No target modules matched for adapter")
    return model


def apply_lora(
    model: nn.Module,
    *,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> nn.Module:
    return _apply_adapter(
        model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        adapter_cls=LoraLinear,
    )


def apply_qlora(
    model: nn.Module,
    *,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> nn.Module:
    return _apply_adapter(
        model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        adapter_cls=QloraLinear,
    )


def merge_lora_weights(model: nn.Module) -> nn.Module:
    for name, module in list(model.named_modules()):
        if not isinstance(module, (LoraLinear, QloraLinear)):
            continue
        parent, child_name = _get_parent(model, name)
        setattr(parent, child_name, module.merge())
    return model


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 1:
            return batch[0], batch[0]
        return batch[0], batch[1]
    if isinstance(batch, dict):
        inp = None
        for key in ("input_ids", "inputs", "input"):
            if key in batch and batch[key] is not None:
                inp = batch[key]
                break
        tgt = None
        for key in ("labels", "targets", "targets_ids"):
            if key in batch and batch[key] is not None:
                tgt = batch[key]
                break
        if inp is None:
            raise ValueError("Dict batch missing input_ids/inputs key")
        if tgt is None:
            tgt = inp
        return inp, tgt
    return batch, batch


def _forward_adapter_batch(model: nn.Module, batch, inputs: torch.Tensor, targets: torch.Tensor):
    if isinstance(batch, dict):
        model_inputs = {"input_ids": inputs, "labels": targets}
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(inputs.device, non_blocking=True)
        return model(**model_inputs)
    try:
        return model(input_ids=inputs, labels=targets)
    except TypeError:
        try:
            return model(inputs, targets)
        except TypeError:
            return model(inputs)


def _adapter_loss(model_out, targets: torch.Tensor) -> torch.Tensor:
    if isinstance(model_out, dict):
        if model_out.get("loss") is not None:
            return model_out["loss"]
        logits = model_out.get("logits")
    elif hasattr(model_out, "loss") and getattr(model_out, "loss") is not None:
        return model_out.loss
    elif hasattr(model_out, "logits"):
        logits = model_out.logits
    elif isinstance(model_out, (tuple, list)):
        if len(model_out) > 1 and isinstance(model_out[1], torch.Tensor) and model_out[1].ndim == 0:
            return model_out[1]
        logits = model_out[0] if len(model_out) > 0 else None
    else:
        logits = model_out

    if not isinstance(logits, torch.Tensor):
        raise ValueError("Adapter training requires model loss or logits tensor")
    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape [B, T, V], got {tuple(logits.shape)}")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = targets[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
 