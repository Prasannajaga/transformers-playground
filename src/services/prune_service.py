"""
prune_service.py — Production-grade model pruning service.

Business logic only. No UI, no CLI, no config construction.
Receives an already-loaded model and returns pruned stats / saves on demand.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.utils.prune as prune

from constants.prune_constants import (
    HISTOGRAM_BINS,
    LAYER_ALIAS_MAP,
    SEPARATOR,
    PruneType,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerStats:
    """Per-layer pruning stats with an alias for readability."""
    raw_name: str
    alias: str
    total: int
    nonzero: int
    zero: int
    sparsity_pct: float


@dataclass
class PruneStats:
    """Immutable snapshot of a model's pruning state."""
    label: str
    prune_type: Optional[PruneType]
    sparsity_target: float
    total_params: int
    nonzero_params: int
    zero_params: int
    sparsity_pct: float
    dense_size_mb: float
    effective_size_mb: float
    savings_pct: float
    layers: List[LayerStats] = field(default_factory=list)


def _resolve_alias(param_name: str) -> str:
    """First-match substring lookup against LAYER_ALIAS_MAP."""
    for fragment, alias in LAYER_ALIAS_MAP.items():
        if fragment in param_name:
            return alias
    return param_name


def compute_stats(
    model: torch.nn.Module,
    label: str,
    prune_type: Optional[PruneType] = None,
    sparsity_target: float = 0.0,
) -> PruneStats:
    total = nonzero = 0
    element_size = 0
    layers: List[LayerStats] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            numel = param.numel()
            nz = int(param.data.count_nonzero().item())
            total += numel
            nonzero += nz
            if not element_size:
                element_size = param.element_size()

            z = numel - nz
            sp = round((1.0 - nz / numel) * 100, 2) if numel else 0.0
            layers.append(LayerStats(
                raw_name=name,
                alias=_resolve_alias(name),
                total=numel,
                nonzero=nz,
                zero=z,
                sparsity_pct=sp,
            ))

    zero = total - nonzero
    sparsity = zero / total if total else 0.0
    bytes_per_param = element_size or 2
    dense_mb = (total * bytes_per_param) / (1024 ** 2)
    effective_mb = (nonzero * bytes_per_param) / (1024 ** 2)
    savings = (1.0 - effective_mb / dense_mb) * 100 if dense_mb else 0.0

    return PruneStats(
        label=label,
        prune_type=prune_type,
        sparsity_target=sparsity_target,
        total_params=total,
        nonzero_params=nonzero,
        zero_params=zero,
        sparsity_pct=round(sparsity * 100, 2),
        dense_size_mb=round(dense_mb, 2),
        effective_size_mb=round(effective_mb, 2),
        savings_pct=round(savings, 2),
        layers=layers,
    )


def print_stats(stats: PruneStats) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  📊  {stats.label}")
    if stats.prune_type:
        print(f"  Type   : {stats.prune_type.value}")
        print(f"  Target : {stats.sparsity_target:.0%}")
    print(SEPARATOR)
    print(f"  Total params : {stats.total_params:>15,}")
    print(f"  Non-zero     : {stats.nonzero_params:>15,}")
    print(f"  Pruned       : {stats.zero_params:>15,}")
    print(f"  Sparsity     : {stats.sparsity_pct:>14.2f} %")
    print(f"  Dense size   : {stats.dense_size_mb:>14.2f} MB  (on-disk, all params stored)")
    print(f"  Effective    : {stats.effective_size_mb:>14.2f} MB  (nonzero weights only)")
    print(f"  Savings      : {stats.savings_pct:>14.2f} %   (with sparse serialization)")
    print()
    print("  Top-10 sparsest layers:")
    top10 = sorted(stats.layers, key=lambda l: l.sparsity_pct, reverse=True)[:10]
    for ls in top10:
        bar = "█" * int(ls.sparsity_pct / 5)
        print(f"    {ls.sparsity_pct:5.1f}%  {bar:<20}  {ls.alias:<16}  {ls.raw_name}")
    print(SEPARATOR)


# ──────────────────────────────────────────────────────────────────────────────
# Pruning strategies (pure functions, no side effects beyond model mutation)
# ──────────────────────────────────────────────────────────────────────────────

def _linear_modules(model: torch.nn.Module) -> List[Tuple[torch.nn.Module, str]]:
    return [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Linear)]


def _structured(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    with torch.no_grad():
        for module, _ in _linear_modules(model):
            if module.weight.shape[0] < 2:
                continue
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return model


def _global_magnitude_threshold(model: torch.nn.Module, amount: float) -> float:
    """Two-pass histogram — never materialises full weight tensor."""
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
                m.weight.data.abs().float(),
                bins=HISTOGRAM_BINS,
                min=global_min,
                max=global_max,
            )
            hist += layer_hist.long()
            del layer_hist

    target = int(amount * total_elements)
    cumsum = hist.cumsum(0)
    bin_idx = int((cumsum >= target).nonzero(as_tuple=True)[0][0].item())
    bin_width = (global_max - global_min) / HISTOGRAM_BINS
    return global_min + (bin_idx + 1) * bin_width


def _global_unstructured(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    threshold = _global_magnitude_threshold(model, amount)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                mask = m.weight.data.abs() >= threshold
                m.weight.data.mul_(mask)
                del mask
    return model


def _layerwise(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    with torch.no_grad():
        for module, _ in _linear_modules(model):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def _random(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    with torch.no_grad():
        for module, _ in _linear_modules(model):
            prune.random_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


# Strategy registry
_STRATEGY_FN: Dict[PruneType, Callable[[torch.nn.Module, float], torch.nn.Module]] = {
    PruneType.STRUCTURED: _structured,
    PruneType.GLOBAL_UNSTRUCTURED: _global_unstructured,
    PruneType.LAYERWISE: _layerwise,
    PruneType.RANDOM: _random,
}


# ──────────────────────────────────────────────────────────────────────────────
# PruneWrapper — the public API
# ──────────────────────────────────────────────────────────────────────────────

class PruneWrapper:
    """
    Production-grade pruning wrapper.

    Usage:
        wrapper = PruneWrapper(model)
        stats   = wrapper.prune(PruneType.GLOBAL_UNSTRUCTURED, sparsity=0.5)
        wrapper.save("/path/to/output")
    """

    def __init__(self, model: torch.nn.Module) -> None:
        if model is None:
            raise ValueError("Model must not be None")
        self._model = model
        self._model.eval()
        self._pruned = False
        self._stats: Optional[PruneStats] = None
        logger.info("PruneWrapper initialised  (params=%s)", self._param_count())

    def _param_count(self) -> str:
        total = sum(p.numel() for p in self._model.parameters())
        return f"{total:,}"

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def is_pruned(self) -> bool:
        return self._pruned

    @property
    def stats(self) -> Optional[PruneStats]:
        return self._stats

    def baseline(self) -> PruneStats:
        """Compute and print pre-pruning stats."""
        stats = compute_stats(self._model, "Baseline (no pruning)")
        print_stats(stats)
        return stats

    def prune(self, prune_type: PruneType, sparsity: float) -> PruneStats:
        """
        Apply the requested pruning strategy in-place.

        Args:
            prune_type: One of PruneType enum members.
            sparsity:   Fraction of weights to zero (0–1).

        Returns:
            PruneStats snapshot after pruning.

        Raises:
            ValueError: Invalid sparsity range or unknown prune type.
            RuntimeError: Model already pruned (create a new wrapper).
        """
        if self._pruned:
            raise RuntimeError(
                "Model already pruned. Load a fresh model and create a new PruneWrapper."
            )
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"Sparsity must be in (0, 1), got {sparsity}")
        if prune_type not in _STRATEGY_FN:
            raise ValueError(f"Unknown prune type: {prune_type}")

        logger.info("Pruning  type=%s  sparsity=%.2f", prune_type.value, sparsity)

        fn = _STRATEGY_FN[prune_type]
        fn(self._model, sparsity)
        self._pruned = True

        self._stats = compute_stats(
            self._model,
            label=f"{prune_type.value} ({sparsity:.0%})",
            prune_type=prune_type,
            sparsity_target=sparsity,
        )
        print_stats(self._stats)
        return self._stats

    def save(self, output_path: str) -> Path:
        """
        Save the pruned model + tokenizer to disk.

        Args:
            output_path: Directory path to save the model.

        Returns:
            Resolved Path of the output directory.

        Raises:
            RuntimeError: If prune() has not been called yet.
        """
        if not self._pruned:
            raise RuntimeError("Cannot save — model has not been pruned yet.")

        dest = Path(output_path).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        logger.info("Saving pruned model to %s", dest)
        self._model.save_pretrained(dest, safe_serialization=True)
        logger.info("Model saved  (%s)", dest)

        return dest

    def free(self) -> None:
        """Release model memory explicitly."""
        del self._model
        self._model = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("Model memory released")
