"""prune.py — Llama 3.2 1B Pruning: structured → unstructured → layerwise → random."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM


MODEL_ID: str = "meta-llama/Llama-3.2-1B-Instruct"
DTYPE: torch.dtype = torch.bfloat16
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

STRUCTURED_SPARSITY: float = 0.30
UNSTRUCTURED_SPARSITY: float = 0.50
LAYERWISE_SPARSITY: float = 0.40
RANDOM_SPARSITY: float = 0.50

HISTOGRAM_BINS: int = 10_000
SEPARATOR: str = "=" * 72


@dataclass(frozen=True)
class PruneConfig:
    model_id: str
    dtype: torch.dtype
    device: str
    structured_sparsity: float
    unstructured_sparsity: float
    layerwise_sparsity: float
    random_sparsity: float


def build_config() -> PruneConfig:
    return PruneConfig(
        model_id=MODEL_ID,
        dtype=DTYPE,
        device=DEVICE,
        structured_sparsity=STRUCTURED_SPARSITY,
        unstructured_sparsity=UNSTRUCTURED_SPARSITY,
        layerwise_sparsity=LAYERWISE_SPARSITY,
        random_sparsity=RANDOM_SPARSITY,
    )


@dataclass
class ModelStats:
    label: str
    total_params: int
    nonzero_params: int
    zero_params: int
    sparsity_pct: float
    size_mb: float
    layer_sparsities: Dict[str, float] = field(default_factory=dict)


def compute_stats(model: torch.nn.Module, label: str) -> ModelStats:
    total = nonzero = 0
    element_size = 0
    layer_sparsities: Dict[str, float] = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            numel = param.numel()
            nz = int(param.data.count_nonzero().item())
            total += numel
            nonzero += nz
            if not element_size:
                element_size = param.element_size()
            layer_sparsities[name] = round(1.0 - nz / numel, 4) if numel else 0.0

    zero = total - nonzero
    sparsity = zero / total if total else 0.0

    return ModelStats(
        label=label,
        total_params=total,
        nonzero_params=nonzero,
        zero_params=zero,
        sparsity_pct=round(sparsity * 100, 2),
        size_mb=round((total * (element_size or 2)) / (1024 ** 2), 2),
        layer_sparsities=layer_sparsities,
    )


def print_stats(stats: ModelStats) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  📊  {stats.label}")
    print(SEPARATOR)
    print(f"  Total params : {stats.total_params:>15,}")
    print(f"  Non-zero     : {stats.nonzero_params:>15,}")
    print(f"  Pruned       : {stats.zero_params:>15,}")
    print(f"  Sparsity     : {stats.sparsity_pct:>14.2f} %")
    print(f"  Size         : {stats.size_mb:>14.2f} MB")
    top = sorted(stats.layer_sparsities.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, sp in top:
        print(f"    {sp*100:5.1f}% {'█' * int(sp * 20):<20} {name}")
    print(SEPARATOR)


def _linear_modules(model: torch.nn.Module) -> List[Tuple[torch.nn.Module, str]]:
    return [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Linear)]


def apply_structured_pruning(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    with torch.no_grad():
        for module, _ in _linear_modules(model):
            if module.weight.shape[0] < 2:
                continue
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return model


def _global_magnitude_threshold(model: torch.nn.Module, amount: float) -> float:
    """Two-pass histogram approach — never materializes all weights at once."""
    global_min = float("inf")
    global_max = 0.0
    total_elements = 0

    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            mag = m.weight.data.abs()
            lo, hi = mag.min().item(), mag.max().item()
            if lo < global_min:
                global_min = lo
            if hi > global_max:
                global_max = hi
            total_elements += mag.numel()
            del mag

    hist = torch.zeros(HISTOGRAM_BINS, dtype=torch.long)

    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            layer_hist = torch.histc(
                m.weight.data.abs().float(), bins=HISTOGRAM_BINS, min=global_min, max=global_max,
            )
            hist += layer_hist.long()
            del layer_hist

    target = int(amount * total_elements)
    cumsum = hist.cumsum(0)
    bin_idx = int((cumsum >= target).nonzero(as_tuple=True)[0][0].item())
    bin_width = (global_max - global_min) / HISTOGRAM_BINS

    return global_min + (bin_idx + 1) * bin_width


def apply_unstructured_global_pruning(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    """Manual global threshold + per-layer mask — replaces PyTorch's memory-heavy global_unstructured."""
    threshold = _global_magnitude_threshold(model, amount)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                mask = m.weight.data.abs() >= threshold
                m.weight.data.mul_(mask)
                del mask
    return model


def apply_layerwise_pruning(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    with torch.no_grad():
        for module, _ in _linear_modules(model):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def apply_random_pruning(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    with torch.no_grad():
        for module, _ in _linear_modules(model):
            prune.random_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def _load_fresh(cfg: PruneConfig) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.dtype, device_map=cfg.device, low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _free(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


STRATEGIES = [
    ("Structured",            "structured_sparsity",    apply_structured_pruning),
    ("Global Unstructured",   "unstructured_sparsity",  apply_unstructured_global_pruning),
    ("Layer-wise",            "layerwise_sparsity",     apply_layerwise_pruning),
    ("Random Unstructured",   "random_sparsity",        apply_random_pruning),
]


def main() -> None:
    cfg = build_config()

    print(f"\n  🦙 Loading {cfg.model_id}  (device={cfg.device}, dtype={cfg.dtype})")
    base_model = _load_fresh(cfg)
    baseline_stats = compute_stats(base_model, "Baseline (no pruning)")
    print_stats(baseline_stats)
    _free(base_model)

    all_stats: List[ModelStats] = [baseline_stats]

    for idx, (name, sparsity_attr, prune_fn) in enumerate(STRATEGIES, 1):
        amount = getattr(cfg, sparsity_attr)
        print(f"\n  ✂️  [{idx}/{len(STRATEGIES)}] {name} Pruning (sparsity={amount:.0%})")

        model = _load_fresh(cfg)
        prune_fn(model, amount)
        stats = compute_stats(model, f"{name} ({amount:.0%})")
        print_stats(stats)
        all_stats.append(stats)
        _free(model)

    print(f"\n{SEPARATOR}")
    print("  📋  PRUNING SUMMARY")
    print(SEPARATOR)
    print(f"  {'Strategy':<42} {'Sparsity':>10}  {'Non-zero':>14}  {'Size (MB)':>10}")
    print(f"  {'─'*42} {'─'*10}  {'─'*14}  {'─'*10}")
    for s in all_stats:
        print(f"  {s.label:<42} {s.sparsity_pct:>9.2f}%  {s.nonzero_params:>14,}  {s.size_mb:>9.2f}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
