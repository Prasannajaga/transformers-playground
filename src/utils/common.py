import json
import math
import os
import gzip
from dataclasses import dataclass, is_dataclass, asdict, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.optim import Optimizer


def _make_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (Path,)):
        return str(value)
    # torch types that commonly appear in configs
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    return str(value)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return _make_jsonable(asdict(config))
    if isinstance(config, Mapping):
        return _make_jsonable(dict(config))
    if hasattr(config, "__dict__") and config.__dict__:
        return _make_jsonable(dict(config.__dict__))

    # Fallback for "config objects" with class attrs (e.g., FineTuneConfig)
    out: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        out[name] = _make_jsonable(value)
    return out


def get_lr(it: int, config: Any) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Supports:
    - FineTuneConfig-style: max_lr, min_lr, warmup_iters, lr_decay_iters
    - TrainingConfig-style: lr, warmup_steps, total_steps (min_lr defaults to 0.0)
    """
    it = int(it)

    max_lr = getattr(config, "max_lr", None)
    if max_lr is None:
        max_lr = float(getattr(config, "lr"))
    else:
        max_lr = float(max_lr)

    min_lr = float(getattr(config, "min_lr", 0.0))
    warmup = int(getattr(config, "warmup_iters", getattr(config, "warmup_steps", 0)) or 0)
    decay = getattr(config, "lr_decay_iters", getattr(config, "total_steps", None))
    decay = None if decay is None else int(decay)

    if warmup > 0 and it < warmup:
        return max_lr * float(it) / float(max(1, warmup))

    if decay is None:
        return max_lr

    if it >= decay:
        return min_lr

    if decay <= warmup:
        return max_lr

    decay_ratio = float(it - warmup) / float(max(1, decay - warmup))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def _unpack_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, (list, tuple)):
        if len(batch) == 1:
            return batch[0], batch[0]
        return batch[0], batch[1]
    if isinstance(batch, dict):
        inp = batch.get("input_ids") or batch.get("inputs") or batch.get("input")
        tgt = batch.get("labels") or batch.get("targets") or batch.get("target_ids")
        if inp is None:
            raise ValueError("Dict batch missing input_ids/inputs key")
        if tgt is None:
            tgt = inp
        return inp, tgt
    return batch, batch


def _compute_loss(model_out: Any, targets: Any) -> torch.Tensor:
    if isinstance(model_out, dict) and "loss" in model_out:
        return model_out["loss"]
    if isinstance(model_out, (tuple, list)):
        logits = model_out[0]
    else:
        logits = model_out

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model output must be a Tensor (or a tuple/list whose first item is a Tensor).")

    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape [B, T, V], got {tuple(logits.shape)}")

    bsz, seq_len, vocab = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(bsz * seq_len, vocab),
        targets.reshape(bsz * seq_len),
    )


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataloader: Any,
    config: Optional[Any] = None,
    *,
    device: Optional[torch.device] = None,
    num_batches: Optional[int] = None,
    mixed_precision: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """Estimate average loss over a few validation batches."""
    model.eval()

    if device is None:
        if config is not None and hasattr(config, "device"):
            device = torch.device(getattr(config, "device"))
        else:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

    if num_batches is None:
        if config is not None and hasattr(config, "eval_iters"):
            num_batches = int(getattr(config, "eval_iters"))
        elif config is not None and hasattr(config, "validation_batch_count"):
            num_batches = int(getattr(config, "validation_batch_count"))
        else:
            num_batches = 10

    if mixed_precision is None:
        if config is not None and hasattr(config, "mixed_precision"):
            mixed_precision = bool(getattr(config, "mixed_precision"))
        elif config is not None and hasattr(config, "use_amp"):
            mixed_precision = bool(getattr(config, "use_amp"))
        else:
            mixed_precision = False

    if amp_dtype is None:
        amp_dtype = torch.float16

    device_type = "cuda" if device.type == "cuda" else "cpu"
    losses: list[float] = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=True) if hasattr(inputs, "to") else inputs
        targets = targets.to(device, non_blocking=True) if hasattr(targets, "to") else targets

        with torch.amp.autocast(device_type=device_type, enabled=mixed_precision, dtype=amp_dtype):
            try:
                out = model(inputs, targets)
                if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
                    loss = out[1]
                else:
                    loss = _compute_loss(out, targets)
            except TypeError:
                out = model(inputs)
                loss = _compute_loss(out, targets)

        if loss is None:
            continue
        if isinstance(loss, torch.Tensor):
            if torch.isnan(loss).any():
                continue
            losses.append(float(loss.detach().item()))
        else:
            losses.append(float(loss))

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


@dataclass(slots=True)
class ModelCheckpoint:
    training_step: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: Optional[dict[str, Any]] = None
    scaler_state_dict: Optional[dict[str, Any]] = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "training_step": int(self.training_step),
            "model_state_dict": self.model_state_dict,
        }
        if self.optimizer_state_dict is not None:
            payload["optimizer_state_dict"] = self.optimizer_state_dict
        if self.scaler_state_dict is not None:
            payload["scaler_state_dict"] = self.scaler_state_dict
        if self.config:
            payload["config"] = self.config
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def save_checkpoint(
    *,
    model: nn.Module,
    checkpoint_dir: str,
    step: int,
    prefix: str = "ckpt",
    optimizer: Optional[Optimizer] = None,
    scaler: Optional[Any] = None,
    config: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    is_final: bool = False,
    step_digits: int = 7,
    save_optimizer_state: Optional[bool] = None,
) -> str:
    """
    Save a checkpoint with a stable on-disk layout:

      checkpoint_dir/
        prefix/
          config.json
          tokenizer/           (optional)
          prefix_0000123.pt
          prefix_final_0000123.pt  (if is_final=True)
    """
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must be provided")

    step = int(step)
    prefix_dir = os.path.join(checkpoint_dir, str(prefix))
    os.makedirs(prefix_dir, exist_ok=True)

    config_dict = _config_to_dict(config)

    config_path = os.path.join(prefix_dir, "config.json")
    if config_dict and not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    if tokenizer is not None:
        tokenizer_dir = os.path.join(prefix_dir, "tokenizer")
        if not os.path.exists(tokenizer_dir):
            try:
                tokenizer.save_pretrained(tokenizer_dir)
            except Exception:
                pass

    if save_optimizer_state is None:
        save_optimizer_state = bool(getattr(config, "save_optimizer_state", True)) if config is not None else True

    optimizer_state = optimizer.state_dict() if (optimizer is not None and save_optimizer_state) else None
    scaler_state = None
    if scaler is not None:
        try:
            scaler_state = scaler.state_dict()
        except Exception:
            scaler_state = None

    ckpt = ModelCheckpoint(
        training_step=step,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer_state,
        scaler_state_dict=scaler_state,
        config=config_dict,
    )

    if step_digits and step_digits > 0:
        step_str = f"{step:0{int(step_digits)}d}"
    else:
        step_str = str(step)
    fname = f"{prefix}_final_{step_str}.pt" if is_final else f"{prefix}_{step_str}.pt"
    path = os.path.join(prefix_dir, fname)
    torch.save(ckpt.to_dict(), path)
    return path


def save_file_text(data , path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

def read_file_text(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data