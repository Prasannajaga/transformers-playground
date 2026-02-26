"""Llama 3.2 1B Instruct — structured pruning via torch_pruning."""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import torch
import torch_pruning as tp
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.config import PruneConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _param_count(model: torch.nn.Module) -> str:
    return f"{sum(p.numel() for p in model.parameters()):,}"


def main() -> None:
    cfg = PruneConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading %s  (dtype=%s, device=%s)", cfg.model_id, cfg.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=cfg.dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    logger.info("Params before pruning: %s", _param_count(model))

    with torch.no_grad():
        example_inputs = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)

    # Map attention head structure so pruner handles multi-head dims correctly
    num_heads = {}
    for layer in model.model.layers:
        num_heads[layer.self_attn.q_proj] = model.config.num_attention_heads
        num_heads[layer.self_attn.k_proj] = model.config.num_key_value_heads
        num_heads[layer.self_attn.v_proj] = model.config.num_key_value_heads

    importance = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.lm_head]

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=cfg.prune_ratio,
        ignored_layers=ignored_layers,
        output_transform=lambda out: out.logits,
        num_heads=num_heads,
        global_pruning=True,
        iterative_steps=1,
        round_to=model.config.num_attention_heads,
    )

    logger.info("Pruning at ratio=%.0f%%…", cfg.prune_ratio * 100)
    with torch.no_grad():
        pruner.step()

    logger.info("Params after pruning:  %s", _param_count(model))

    with torch.no_grad():
        out = model(example_inputs)
    logger.info("Forward pass OK — logits shape: %s", tuple(out.logits.shape))

    dest = Path(cfg.output_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Saving pruned model to %s", dest)
    model.save_pretrained(dest, safe_serialization=True)

    if cfg.save_tokenizer:
        tokenizer.save_pretrained(dest)
        logger.info("Tokenizer saved")

    del model, pruner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()