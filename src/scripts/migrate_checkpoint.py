#!/usr/bin/env python3
"""
Checkpoint Migration Script

Migrates old checkpoint format to the new architecture format.
Old format: {step, model_state, optimizer_state, scaler_state}
New format: {step, model_state, optimizer_state, scaler_state, config, tokenizer_path}

Usage:
    python migrate_checkpoint.py --input <old_ckpt_path> --output <new_ckpt_path>
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.train import TrainingConfig
from customTransformers.decodeTransfomer import DecodeTransformer


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOKENIZER = "openai-community/gpt2"
DEFAULT_ATTENTION = "MHA"
DEFAULT_FFN_TYPE = "relu"
DEFAULT_DROPOUT = 0.1


# ============================================================================
# Migration Functions
# ============================================================================

def load_old_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load checkpoint from old format.
    
    Args:
        path: Path to the old checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        Dictionary containing checkpoint data
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If checkpoint cannot be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    print(f"[INFO] Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    
    # Validate required keys
    required_keys = {"model_state"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise RuntimeError(f"Checkpoint missing required keys: {missing_keys}")
    
    print(f"[INFO] Checkpoint keys: {list(checkpoint.keys())}")
    print(f"[INFO] Step: {checkpoint.get('step', 'N/A')}")
    
    return checkpoint


def infer_model_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """
    Infer model configuration from state dict keys and tensor shapes.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary with inferred model configuration
    """
    config = {}
    
    # Infer vocab_size and n_embd from token embedding
    if "token_emb.weight" in state_dict:
        vocab_size, n_embd = state_dict["token_emb.weight"].shape
        config["vocab_size"] = vocab_size
        config["n_embd"] = n_embd
        print(f"[INFO] Inferred vocab_size={vocab_size}, n_embd={n_embd}")
    
    # Infer block_size from position embedding
    if "position_emb.weight" in state_dict:
        block_size, _ = state_dict["position_emb.weight"].shape
        config["block_size"] = block_size
        print(f"[INFO] Inferred block_size={block_size}")
    
    # Count transformer blocks/layers
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith("transformer_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    
    if layer_indices:
        num_layers = max(layer_indices) + 1
        config["num_layers"] = num_layers
        print(f"[INFO] Inferred num_layers={num_layers}")
    
    # Infer n_head from attention layer (try to find attention weight)
    for key in state_dict.keys():
        if "attention" in key and "weight" in key:
            # For MHA, the Q/K/V projections can help infer heads
            # This is a heuristic - may need adjustment
            break
    
    return config


def create_model_from_config(
    config: TrainingConfig,
    vocab_size: int,
    attention: str = DEFAULT_ATTENTION,
    ffn_type: str = DEFAULT_FFN_TYPE,
    dropout: float = DEFAULT_DROPOUT,
) -> DecodeTransformer:
    """
    Create a DecodeTransformer model from TrainingConfig.
    
    Args:
        config: Training configuration
        vocab_size: Vocabulary size from tokenizer
        attention: Attention type ("MHA" or "MQA")
        ffn_type: Feed-forward network type
        dropout: Dropout rate
        
    Returns:
        Initialized DecodeTransformer model
    """
    print(f"[INFO] Creating DecodeTransformer model...")
    print(f"  - num_layers: {config.n_layer}")
    print(f"  - n_embd: {config.n_embd}")
    print(f"  - n_head: {config.n_head}")
    print(f"  - block_size: {config.block_size}")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - attention: {attention}")
    print(f"  - ffn_type: {ffn_type}")
    
    model = DecodeTransformer(
        num_layers=config.n_layer,
        n_emb=config.n_embd,
        n_head=config.n_head,
        vocab_size=vocab_size,
        block_size=config.block_size,
        dropout=dropout,
        attention=attention,
        ffn_type=ffn_type,
    )
    
    return model


def migrate_state_dict(
    old_state_dict: Dict[str, torch.Tensor],
    new_model: DecodeTransformer,
) -> Dict[str, torch.Tensor]:
    """
    Migrate old state dict to new model format.
    
    This function handles any key renaming or transformations needed
    between the old and new checkpoint formats.
    
    Args:
        old_state_dict: State dict from old checkpoint
        new_model: Target model to migrate to
        
    Returns:
        Migrated state dict compatible with new model
    """
    new_state_dict = new_model.state_dict()
    migrated_state_dict = {}
    
    print(f"\n[INFO] Migrating state dict...")
    print(f"  - Old state dict keys: {len(old_state_dict)}")
    print(f"  - New model keys: {len(new_state_dict)}")
    
    # Track migration status
    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    shape_mismatches = []
    
    # Direct key mapping (old_key -> new_key)
    # Add any key renames here if architecture changed
    key_mapping = {}
    
    for new_key in new_state_dict.keys():
        old_key = key_mapping.get(new_key, new_key)
        
        if old_key in old_state_dict:
            old_tensor = old_state_dict[old_key]
            new_tensor = new_state_dict[new_key]
            
            if old_tensor.shape == new_tensor.shape:
                migrated_state_dict[new_key] = old_tensor
                matched_keys.append(new_key)
            else:
                shape_mismatches.append(
                    f"{new_key}: old={old_tensor.shape} vs new={new_tensor.shape}"
                )
        else:
            missing_keys.append(new_key)
    
    # Check for unexpected keys in old checkpoint
    for old_key in old_state_dict.keys():
        mapped_key = None
        for nk, ok in key_mapping.items():
            if ok == old_key:
                mapped_key = nk
                break
        
        if mapped_key is None and old_key not in new_state_dict:
            unexpected_keys.append(old_key)
    
    # Report results
    print(f"\n[MIGRATION REPORT]")
    print(f"  ✓ Matched keys: {len(matched_keys)}")
    
    if missing_keys:
        print(f"  ! Missing keys (using random init): {len(missing_keys)}")
        for key in missing_keys[:5]:
            print(f"      - {key}")
        if len(missing_keys) > 5:
            print(f"      ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"  ! Unexpected keys (ignored): {len(unexpected_keys)}")
        for key in unexpected_keys[:5]:
            print(f"      - {key}")
        if len(unexpected_keys) > 5:
            print(f"      ... and {len(unexpected_keys) - 5} more")
    
    if shape_mismatches:
        print(f"  ✗ Shape mismatches: {len(shape_mismatches)}")
        for mismatch in shape_mismatches:
            print(f"      - {mismatch}")
        raise RuntimeError("Shape mismatches detected - cannot migrate checkpoint")
    
    return migrated_state_dict


def save_new_checkpoint(
    output_path: str,
    model_state: Dict[str, torch.Tensor],
    step: int,
    config: TrainingConfig,
    tokenizer_name: str,
    optimizer_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save checkpoint in new format.
    
    Args:
        output_path: Path to save the new checkpoint
        model_state: Migrated model state dict
        step: Training step
        config: Training configuration
        tokenizer_name: Tokenizer identifier
        optimizer_state: Optional optimizer state (if available)
        scaler_state: Optional scaler state (if available)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Build config dict from dataclass
    config_dict = {
        "n_layer": config.n_layer,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "block_size": config.block_size,
        "device": config.device,
        "mixed_precision": config.mixed_precision,
        "amp_dtype": config.amp_dtype,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "use_top_k": config.use_top_k,
        "top_k": config.top_k,
        "use_repetition_penalty": config.use_repetition_penalty,
        "repetition_penalty": config.repetition_penalty,
        "stop_on_eos": config.stop_on_eos,
    }
    
    checkpoint = {
        "step": step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scaler_state": scaler_state,
        "config": config_dict,
        "tokenizer_path": tokenizer_name,
    }
    
    print(f"\n[INFO] Saving new checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    
    # Verify saved checkpoint
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] Checkpoint saved successfully ({file_size:.2f} MB)")


def migrate_checkpoint(
    input_path: str,
    output_path: str,
    config: TrainingConfig,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    attention: str = DEFAULT_ATTENTION,
    ffn_type: str = DEFAULT_FFN_TYPE,
    device: str = "cpu",
) -> None:
    """
    Main migration function.
    
    Args:
        input_path: Path to old checkpoint
        output_path: Path to save new checkpoint
        config: Training configuration
        tokenizer_name: HuggingFace tokenizer name
        attention: Attention type for model
        ffn_type: FFN type for model
        device: Device to use for migration
    """
    device = torch.device(device)
    
    print("=" * 60)
    print("  Checkpoint Migration Tool")
    print("=" * 60)
    
    # Load tokenizer to get vocab size
    print(f"\n[INFO] Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    print(f"[INFO] Tokenizer vocab size: {vocab_size}")
    
    # Load old checkpoint
    old_checkpoint = load_old_checkpoint(input_path, device)
    old_state_dict = old_checkpoint["model_state"]
    
    # Infer config from state dict for validation
    inferred_config = infer_model_config_from_state_dict(old_state_dict)
    
    # Validate config matches
    if "vocab_size" in inferred_config and inferred_config["vocab_size"] != vocab_size:
        print(f"[WARN] Vocab size mismatch: inferred={inferred_config['vocab_size']}, tokenizer={vocab_size}")
    
    # Create new model
    new_model = create_model_from_config(
        config=config,
        vocab_size=vocab_size,
        attention=attention,
        ffn_type=ffn_type,
    )
    new_model.to(device)
    
    # Migrate state dict
    migrated_state_dict = migrate_state_dict(old_state_dict, new_model)
    
    # Load migrated state into model to verify
    print(f"\n[INFO] Loading migrated state dict into model...")
    new_model.load_state_dict(migrated_state_dict, strict=False)
    print(f"[INFO] State dict loaded successfully")
    
    # Save new checkpoint
    save_new_checkpoint(
        output_path=output_path,
        model_state=new_model.state_dict(),
        step=old_checkpoint.get("step", 0),
        config=config,
        tokenizer_name=tokenizer_name,
        optimizer_state=old_checkpoint.get("optimizer_state"),
        scaler_state=old_checkpoint.get("scaler_state"),
    )
    
    print("\n" + "=" * 60)
    print("  Migration Complete!")
    print("=" * 60)


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate old checkpoint format to new architecture format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to old checkpoint file",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save migrated checkpoint",
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HuggingFace tokenizer name",
    )
    
    parser.add_argument(
        "--attention",
        type=str,
        default=DEFAULT_ATTENTION,
        choices=["MHA", "MQA"],
        help="Attention type",
    )
    
    parser.add_argument(
        "--ffn-type",
        type=str,
        default=DEFAULT_FFN_TYPE,
        choices=["relu", "gelu", "swiglu"],
        help="Feed-forward network type",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for migration",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load default config
    config = TrainingConfig()
    
    migrate_checkpoint(
        input_path=args.input,
        output_path=args.output,
        config=config,
        tokenizer_name=args.tokenizer,
        attention=args.attention,
        ffn_type=args.ffn_type,
        device=args.device,
    )


if __name__ == "__main__":
    main()
