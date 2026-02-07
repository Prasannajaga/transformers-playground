#!/usr/bin/env python3
"""
Convert an old-format checkpoint and push it to Hugging Face Hub.

Old checkpoint format:
    {
      "step": int,
      "model_state": ...,
      "optimizer_state": ...,
      "scaler_state": ...
    }

This script:
1) Loads old checkpoint weights.
2) Remaps legacy keys to current DecodeTransformer architecture.
3) Validates weights against current model definition.
4) Writes HF artifacts (config.json, model.safetensors, tokenizer, README).
5) Uploads to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer

# Ensure imports work when executed as `python src/scripts/test-model.py`
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from customTransformers.decodeTransfomer import DecodeTransformer
from scripts.hf_deploy import create_hf_config, create_model_card, save_model_safetensors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("test-model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert old checkpoint format to HF artifacts and upload to Hub."
    )
    parser.add_argument("--model_model_pathpath", required=True, help="Path to .pt checkpoint or folder")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo ID: username/model-name")
    parser.add_argument("--config_path", default=None, help="Optional JSON config with n_layer/n_embd/n_head/block_size")
    parser.add_argument("--tokenizer_path", default="gpt2", help="Tokenizer path or HF tokenizer ID")
    parser.add_argument("--token", default=None, help="HF token (optional if already logged in)")
    parser.add_argument("--revision", default="main", help="Target branch/tag")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--attention", choices=["MHA", "MQA"], default=None, help="Override attention type")
    parser.add_argument("--ffn_type", default=None, help="Override FFN type (relu/swiglu/...)")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    parser.add_argument("--n_layer", type=int, default=None, help="Override n_layer")
    parser.add_argument("--n_embd", type=int, default=None, help="Override n_embd")
    parser.add_argument("--n_head", type=int, default=None, help="Override n_head")
    parser.add_argument("--block_size", type=int, default=None, help="Override block_size")
    parser.add_argument("--output_dir", default=None, help="Optional local artifact output directory")
    parser.add_argument("--no_upload", action="store_true", help="Prepare artifacts locally but skip Hub upload")
    parser.add_argument("--non_strict", action="store_true", help="Allow missing/unexpected keys when validating")
    return parser.parse_args()


def _resolve_checkpoint_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {model_path}")
    if path.is_file():
        return path
    candidates = sorted(list(path.glob("*.pt")) + list(path.glob("*.pth")))
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth files found in: {model_path}")
    return candidates[-1]


def _load_checkpoint(model_path: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Path]:
    checkpoint_file = _resolve_checkpoint_path(model_path)
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(payload)}")

    state_dict = (
        payload.get("model_state")
        or payload.get("model_state_dict")
        or payload.get("state_dict")
    )
    if state_dict is None:
        # Support raw state dict
        if all(hasattr(v, "shape") for v in payload.values()):
            state_dict = payload
        else:
            raise KeyError(
                f"Unsupported checkpoint format. Found keys: {list(payload.keys())}"
            )
    log.info("Loaded checkpoint: %s", checkpoint_file)
    log.info("Checkpoint keys: %s", list(payload.keys()))
    log.info("Model parameter entries: %d", len(state_dict))
    return payload, state_dict, checkpoint_file


def _remap_legacy_keys(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    remapped: Dict[str, torch.Tensor] = {}
    replaced = 0

    for key, value in state_dict.items():
        new_key = key.replace(".attn.", ".attention.").replace(".ffn.", ".feedForward.")
        if new_key != key:
            replaced += 1
        if new_key in remapped:
            raise KeyError(f"Key collision after remap: {new_key}")
        remapped[new_key] = value
    return remapped, replaced


def _infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    if "token_emb.weight" in state_dict:
        vocab_size, n_embd = state_dict["token_emb.weight"].shape
        cfg["vocab_size"] = int(vocab_size)
        cfg["n_embd"] = int(n_embd)
    if "position_emb.weight" in state_dict:
        block_size = state_dict["position_emb.weight"].shape[0]
        cfg["block_size"] = int(block_size)

    layer_ids = set()
    for key in state_dict:
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "transformer_blocks" and parts[1].isdigit():
            layer_ids.add(int(parts[1]))
    if layer_ids:
        cfg["n_layer"] = max(layer_ids) + 1

    head_ids = set()
    for key in state_dict:
        parts = key.split(".")
        if (
            len(parts) > 5
            and parts[0] == "transformer_blocks"
            and parts[1] == "0"
            and parts[2] == "attention"
            and parts[3] == "heads"
            and parts[4].isdigit()
        ):
            head_ids.add(int(parts[4]))
    if head_ids:
        cfg["n_head"] = max(head_ids) + 1

    return cfg


def _load_json_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return json.load(f)


def _merge_config(
    inferred: Dict[str, Any],
    json_cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(json_cfg)
    # Checkpoint-derived architecture fields are the source of truth.
    merged.update(inferred)

    overrides = {
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "block_size": args.block_size,
        "attention": args.attention,
        "ffn_type": args.ffn_type,
        "dropout": args.dropout,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    merged.setdefault("attention", "MHA")
    merged.setdefault("ffn_type", "relu")
    merged.setdefault("dropout", 0.1)
    return merged


def _validate_with_current_architecture(
    state_dict: Dict[str, torch.Tensor],
    merged_cfg: Dict[str, Any],
    strict: bool,
) -> None:
    required = ["n_layer", "n_embd", "n_head", "block_size"]
    missing = [k for k in required if k not in merged_cfg]
    if missing:
        raise ValueError(f"Could not infer required model config fields: {missing}")

    model = DecodeTransformer(
        num_layers=int(merged_cfg["n_layer"]),
        n_emb=int(merged_cfg["n_embd"]),
        n_head=int(merged_cfg["n_head"]),
        vocab_size=int(merged_cfg["vocab_size"]),
        block_size=int(merged_cfg["block_size"]),
        dropout=float(merged_cfg.get("dropout", 0.1)),
        attention=str(merged_cfg.get("attention", "MHA")),
        ffn_type=str(merged_cfg.get("ffn_type", "relu")),
    )

    if strict:
        model.load_state_dict(state_dict, strict=True)
        log.info("State dict validated with strict=True")
    else:
        incompat = model.load_state_dict(state_dict, strict=False)
        log.warning("Validated with strict=False")
        log.warning("Missing keys: %d", len(incompat.missing_keys))
        log.warning("Unexpected keys: %d", len(incompat.unexpected_keys))


def _load_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    log.info("Loaded tokenizer: %s", tokenizer_path)
    return tokenizer


def _authenticate_hf(token: str | None) -> str | None:
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
        return hf_token

    # Use cached credentials if available
    api = HfApi()
    try:
        api.whoami()
        return None
    except Exception as e:
        raise RuntimeError(
            "HuggingFace authentication required. Run `huggingface-cli login` "
            "or pass `--token`."
        ) from e


def _prepare_and_upload(
    args: argparse.Namespace,
    state_dict: Dict[str, torch.Tensor],
    merged_cfg: Dict[str, Any],
    source_step: int | None,
) -> str:
    if args.no_upload and not args.output_dir:
        raise ValueError("--no_upload requires --output_dir so artifacts are retained locally.")

    hf_token = None if args.no_upload else _authenticate_hf(args.token)
    tokenizer = _load_tokenizer(args.tokenizer_path)

    state_vocab = int(state_dict["token_emb.weight"].shape[0])
    tokenizer_vocab = int(len(tokenizer))
    if tokenizer_vocab != state_vocab:
        log.warning(
            "Tokenizer size (%d) does not match checkpoint vocab (%d). "
            "Using checkpoint vocab in config.",
            tokenizer_vocab,
            state_vocab,
        )

    hf_config = create_hf_config(merged_cfg, vocab_size=state_vocab)
    metadata = {"total_steps": source_step} if source_step is not None else {}

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_artifacts(output_dir, args.repo_id, hf_config, state_dict, tokenizer, metadata)
        if args.no_upload:
            return str(output_dir)
        _upload_folder(output_dir, args.repo_id, args.revision, args.private, hf_token, source_step)
        return f"https://huggingface.co/{args.repo_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, args.repo_id, hf_config, state_dict, tokenizer, metadata)
        if args.no_upload:
            raise RuntimeError("--no_upload requires --output_dir so artifacts are not deleted.")
        _upload_folder(output_dir, args.repo_id, args.revision, args.private, hf_token, source_step)
    return f"https://huggingface.co/{args.repo_id}"


def _write_artifacts(
    output_dir: Path,
    repo_id: str,
    hf_config: Any,
    state_dict: Dict[str, torch.Tensor],
    tokenizer: Any,
    metadata: Dict[str, Any],
) -> None:
    hf_config.save_pretrained(output_dir)
    save_model_safetensors(state_dict, output_dir)
    tokenizer.save_pretrained(output_dir)
    readme = create_model_card(repo_id=repo_id, hf_config=hf_config, training_metadata=metadata)
    (output_dir / "README.md").write_text(readme)
    log.info("Prepared HF artifacts at: %s", output_dir)


def _upload_folder(
    output_dir: Path,
    repo_id: str,
    revision: str,
    private: bool,
    hf_token: str | None,
    source_step: int | None,
) -> None:
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    msg = f"Upload converted checkpoint (step={source_step})"
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        revision=revision,
        commit_message=msg,
    )
    log.info("Uploaded model to: https://huggingface.co/%s", repo_id)


def main() -> None:
    args = parse_args()
    payload, state_dict, ckpt_file = _load_checkpoint(args.model_path)

    remapped_state_dict, remapped_count = _remap_legacy_keys(state_dict)
    if remapped_count:
        log.info("Remapped %d legacy key(s) (.attn -> .attention, .ffn -> .feedForward)", remapped_count)
    else:
        log.info("No legacy key remapping needed")

    inferred_cfg = _infer_config_from_state_dict(remapped_state_dict)
    json_cfg = _load_json_config(args.config_path)
    merged_cfg = _merge_config(inferred_cfg, json_cfg, args)
    if "vocab_size" not in merged_cfg and "token_emb.weight" in remapped_state_dict:
        merged_cfg["vocab_size"] = int(remapped_state_dict["token_emb.weight"].shape[0])

    log.info("Using model config: %s", {k: merged_cfg.get(k) for k in ["n_layer", "n_embd", "n_head", "block_size", "attention", "ffn_type", "dropout"]})
    _validate_with_current_architecture(remapped_state_dict, merged_cfg, strict=not args.non_strict)

    source_step = payload.get("step") or payload.get("training_step")
    url_or_path = _prepare_and_upload(args, remapped_state_dict, merged_cfg, source_step)

    if args.no_upload:
        log.info("Artifacts written locally: %s", url_or_path)
    else:
        log.info("Deployment complete: %s", url_or_path)
    log.info("Source checkpoint: %s", ckpt_file)


if __name__ == "__main__":
    main()
