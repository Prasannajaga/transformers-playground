"""
prune_wrapper.py — CLI entry point for production model pruning.

Loads a model, applies a chosen pruning strategy via PruneWrapper,
prints stats with aliased layer names, and optionally saves to disk.

Usage:
    python -m scripts.prune_wrapper \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --prune-type global_unstructured \
        --sparsity 0.5 \
        --save-path ./pruned_output
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants.prune_constants import PruneType
from services.prune_service import PruneWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune a HuggingFace model and optionally save it.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path  (e.g. meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--prune-type",
        type=str,
        required=True,
        choices=[pt.value for pt in PruneType],
        help="Pruning strategy to apply",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        required=True,
        help="Target sparsity ratio (0–1, exclusive)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save the pruned model. Omit to skip saving.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for loading (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--save-tokenizer",
        action="store_true",
        default=False,
        help="Also save the tokenizer alongside the pruned model",
    )
    return parser.parse_args()


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main() -> None:
    args = _parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPE_MAP[args.dtype]
    prune_type = PruneType(args.prune_type)

    logger.info("Loading model  %s  (device=%s, dtype=%s)", args.model, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    wrapper = PruneWrapper(model)
    wrapper.baseline()  

    stats = wrapper.prune(prune_type, sparsity=args.sparsity)

    if args.save_path:
        saved = wrapper.save(args.save_path) 

        if args.save_tokenizer:
            logger.info("Saving tokenizer to  %s", saved)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            tokenizer.save_pretrained(saved)
            logger.info("Tokenizer saved")

    wrapper.free()
    logger.info("Done.")


if __name__ == "__main__":
    main()
