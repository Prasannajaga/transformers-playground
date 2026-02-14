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


def train_adapters(
    *,
    model: nn.Module,
    train_dataloader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 3e-4,
    grad_accum_steps: int = 1,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip_norm: float = 1.0,
    log_interval_steps: int = 10,
) -> None:
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable adapter parameters found")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    accum_steps = max(1, int(grad_accum_steps))

    try:
        steps_per_epoch = len(train_dataloader)
    except Exception:
        steps_per_epoch = 0
    total_steps = 0
    if steps_per_epoch > 0:
        total_steps = int(math.ceil(steps_per_epoch * float(max(1, epochs)) / accum_steps))

    def lr_factor(step: int) -> float:
        if total_steps <= 0:
            if warmup_steps > 0:
                return min(1.0, float(step) / float(max(1, warmup_steps)))
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps == warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_count = 0

    pbar = tqdm(total=total_steps if total_steps > 0 else None, desc="Adapter Training", unit="step", dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    last_batch = None
    for _ in range(max(1, int(epochs))):
        for batch in train_dataloader:
            last_batch = batch
            inputs, targets = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(enabled=amp_enabled, device_type=device.type, dtype=amp_dtype):
                model_out = _forward_adapter_batch(model, batch, inputs, targets)
                loss = _adapter_loss(model_out, targets)

            running_loss += float(loss.detach().item())
            running_count += 1

            scaled_loss = loss / accum_steps
            if scaler_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_step += 1

            if micro_step < accum_steps:
                continue

            if scaler_enabled:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
            if grad_clip_norm and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

            global_step += 1
            factor = lr_factor(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr * factor

            avg_loss = running_loss / max(1, running_count)
            if global_step % max(1, int(log_interval_steps)) == 0:
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
            pbar.update(1)

    if micro_step > 0 and last_batch is not None:
        if scaler_enabled:
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
        if grad_clip_norm and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
        if scaler_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        factor = lr_factor(global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        avg_loss = running_loss / max(1, running_count)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        pbar.update(1)

    pbar.close()
