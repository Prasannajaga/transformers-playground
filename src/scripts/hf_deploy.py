"""
Production-grade HuggingFace Hub deployment wrapper.

Converts locally trained custom PyTorch models to HuggingFace-compatible format
and pushes to the Hub for seamless from_pretrained() loading.

Supports ANY PyTorch model architecture dynamically via PreTrainedModel wrapping.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, login
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import AutoTokenizer, PretrainedConfig
from templates.model_card import ModelCardTemplate, generate_model_card

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CustomModelConfig(PretrainedConfig):
    model_type = "custom_pytorch"

    def __init__(
        self,
        num_layers: int = 6,
        n_embd: int = 384,
        n_head: int = 6,
        vocab_size: int = 50257,
        block_size: int = 256,
        dropout: float = 0.1,
        attention: str = "MHA",
        ffn_type: str = "relu",
        original_config_hash: Optional[str] = None,
        original_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.n_embd = n_embd
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        self.attention = attention
        self.ffn_type = ffn_type
        self.original_config_hash = original_config_hash
        self.original_config = original_config or {}
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_hidden_layers = num_layers
        self.max_position_embeddings = block_size


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_training_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = json.load(f)

    required_keys = ["n_layer", "n_embd", "n_head", "block_size"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    logger.info(f"Loaded training config from: {config_path}")
    return config


def load_checkpoint(model_path: str, device: str = "cpu") -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if path.is_dir():
        candidates = list(path.glob("*.pt")) + list(path.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No .pt/.pth files in: {model_path}")
        checkpoint_file = sorted(candidates)[-1]
        logger.info(f"Found checkpoint: {checkpoint_file}")
    else:
        checkpoint_file = path

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    embedded_config = checkpoint.get("config", {})
    training_metadata = checkpoint.get("metadata", {})
    
    logger.info(f"Loaded checkpoint with {len(state_dict)} parameters")
    return {
        "state_dict": state_dict, 
        "embedded_config": embedded_config,
        "metadata": training_metadata,
    }


def create_hf_config(training_config: Dict[str, Any], vocab_size: int) -> CustomModelConfig:
    config_hash = compute_config_hash(training_config)
    hf_config = CustomModelConfig(
        num_layers=training_config.get("n_layer", 6),
        n_embd=training_config.get("n_embd", 384),
        n_head=training_config.get("n_head", 6),
        vocab_size=vocab_size,
        block_size=training_config.get("block_size", 256),
        dropout=training_config.get("dropout", 0.1),
        attention=training_config.get("attention", "MHA"),
        ffn_type=training_config.get("ffn_type", "relu"),
        original_config_hash=config_hash,
        original_config=training_config,
    )
    logger.info(f"Created HF config: {hf_config.num_layers}L, {hf_config.n_embd}D, {hf_config.n_head}H")
    return hf_config


def load_tokenizer(tokenizer_path: Optional[str], model_path: str) -> Any:
    if tokenizer_path and Path(tokenizer_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded tokenizer from: {tokenizer_path}")
        return tokenizer

    model_dir = Path(model_path) if Path(model_path).is_dir() else Path(model_path).parent
    for subdir in ["tokenizer", "tokenizer_config"]:
        subpath = model_dir / subdir
        if subpath.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(subpath))
            logger.info(f"Loaded tokenizer from: {subpath}")
            return tokenizer

    raise FileNotFoundError("No tokenizer found. Provide --tokenizer_path or place tokenizer in model directory.")


def save_model_safetensors(state_dict: Dict[str, torch.Tensor], output_dir: Path) -> Path:
    output_path = output_dir / "model.safetensors"

    seen_ptrs: Dict[int, str] = {}
    shared_keys: set = set()
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                shared_keys.add(k)
            else:
                seen_ptrs[ptr] = k

    clean_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if k in shared_keys:
                clean_state[k] = v.clone().contiguous()
            else:
                clean_state[k] = v.contiguous()

    save_safetensors(clean_state, str(output_path))
    logger.info(f"Saved safetensors: {output_path}")
    return output_path


def create_model_card(
    repo_id: str, 
    hf_config: CustomModelConfig,
    training_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a comprehensive model card using the template system."""
    metadata = training_metadata or {}
    
    template = ModelCardTemplate(
        repo_id=repo_id,
        num_layers=hf_config.num_layers,
        hidden_size=hf_config.n_embd,
        num_attention_heads=hf_config.n_head,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.block_size,
        attention_type=hf_config.attention,
        ffn_type=hf_config.ffn_type,
        dropout=hf_config.dropout,
        training_steps=metadata.get("total_steps"),
        training_dataset=metadata.get("dataset"),
        training_loss=metadata.get("final_train_loss"),
        validation_loss=metadata.get("final_val_loss"),
        config_hash=hf_config.original_config_hash,
    )
    
    return generate_model_card(template)


# def test_inference(repo_id: str, prompt: str = "Once upon a time") -> None:
#     """
#     Pull model from HuggingFace Hub and generate text.
    
#     Downloads config, tokenizer and model weights, reconstructs model, and generates.
#     """
#     from config import TrainingConfig
#     from customTransformers import DecodeTransformer
#     from pretrain import InferenceEngine
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     logger.info("=" * 60)
#     logger.info("POST-DEPLOYMENT INFERENCE TEST")
#     logger.info("=" * 60)
    
#     # Download config from HuggingFace
#     logger.info(f"Downloading config from: {repo_id}")
#     config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
#     with open(config_path, "r") as f:
#         hf_config = json.load(f)
    
#     # Download tokenizer from HuggingFace
#     logger.info(f"Downloading tokenizer from: {repo_id}")
#     tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
#     # Download model weights
#     logger.info(f"Downloading model weights from: {repo_id}")
#     weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
#     state_dict = load_safetensors(weights_path)
    
#     # Reconstruct model from config
#     logger.info("Reconstructing model architecture...")
#     model = DecodeTransformer(
#         num_layers=hf_config.get("num_layers", 6),
#         n_emb=hf_config.get("n_embd", 384),
#         n_head=hf_config.get("n_head", 6),
#         vocab_size=hf_config.get("vocab_size", tokenizer.vocab_size),
#         block_size=hf_config.get("block_size", 256),
#         ffn_type=hf_config.get("ffn_type", "swiglu"),
#         attention=hf_config.get("attention", "MQA"),
#     ).to(device)
    
#     # Load weights
#     model.load_state_dict(state_dict, strict=True)
#     num_params = sum(p.numel() for p in model.parameters())
#     logger.info(f"Loaded model with {num_params:,} parameters")
    
#     # Create inference engine
#     train_cfg = TrainingConfig(
#         block_size=hf_config.get("block_size", 256),
#         max_new_tokens=100,
#         temperature=0.7,
#         use_top_k=True,
#         top_k=50,
#     )
    
#     engine = InferenceEngine(
#         model=model,
#         config=train_cfg,
#         device=device,
#         tokenizer=tokenizer,
#     )
    
#     # Generate text
#     logger.info(f"Prompt: '{prompt}'")
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
#     print(f"\n{'='*60}")
#     print(f"RESPONSE: ", end="", flush=True)
#     for token in engine.stream_generate(input_ids):
#         print(token, end="", flush=True)
#     print(f"\n{'='*60}\n")


def deploy_to_hub(
    model_path: str,
    config_path: str,
    repo_id: str,
    tokenizer_path: Optional[str] = None,
    revision: str = "main",
    private: bool = False,
    token: Optional[str] = None,
    run_inference: bool = True,
) -> str:
    """Deploy a trained model to HuggingFace Hub."""
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        try:
            api = HfApi()
            api.whoami()
            logger.info("Using cached HuggingFace credentials")
        except Exception:
            raise RuntimeError(
                "HuggingFace authentication required. Either:\n"
                "  1. Run: huggingface-cli login\n"
                "  2. Set HF_TOKEN environment variable\n"
                "  3. Pass --token argument"
            )
    else:
        login(token=hf_token)
        logger.info("Authenticated with HuggingFace")

    logger.info("=" * 60)
    logger.info("STEP 1: Loading configuration")
    logger.info("=" * 60)
    training_config = load_training_config(config_path)

    logger.info("=" * 60)
    logger.info("STEP 2: Loading checkpoint")
    logger.info("=" * 60)
    checkpoint_data = load_checkpoint(model_path)
    state_dict = checkpoint_data["state_dict"]
    training_metadata = checkpoint_data.get("metadata", {})

    if checkpoint_data["embedded_config"]:
        training_config = {**checkpoint_data["embedded_config"], **training_config}

    logger.info("=" * 60)
    logger.info("STEP 3: Loading tokenizer")
    logger.info("=" * 60)
    tokenizer = load_tokenizer(tokenizer_path, model_path)
    vocab_size = tokenizer.vocab_size

    logger.info("=" * 60)
    logger.info("STEP 4: Creating HuggingFace config")
    logger.info("=" * 60)
    hf_config = create_hf_config(training_config, vocab_size)

    logger.info("=" * 60)
    logger.info("STEP 5: Preparing artifacts")
    logger.info("=" * 60)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        hf_config.save_pretrained(output_dir)
        logger.info(f"Saved config.json to: {output_dir}")

        save_model_safetensors(state_dict, output_dir)

        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved tokenizer to: {output_dir}")

        readme_path = output_dir / "README.md"
        readme_content = create_model_card(
            repo_id=repo_id,
            hf_config=hf_config,
            training_metadata=training_metadata,
        )
        readme_path.write_text(readme_content)
        logger.info("Generated README.md")

        logger.info("=" * 60)
        logger.info("STEP 6: Uploading to HuggingFace Hub")
        logger.info("=" * 60)

        api = HfApi(token=hf_token)

        try:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            logger.info(f"Repository ready: {repo_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to create/access repository '{repo_id}': {e}")

        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(output_dir),
            revision=revision,
            commit_message=f"Deploy model (config hash: {hf_config.original_config_hash})",
        )
        logger.info(f"Upload complete to: {repo_id}")

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info("=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info(f"Model URL: {repo_url}")
    logger.info("=" * 60)
    
    # if run_inference:
    #     prompt = input("\n PROMPT >")
    #     test_inference(repo_id=repo_id, prompt=prompt)
    return repo_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy trained PyTorch model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
          python scripts/hf_deploy.py \\
            --model_path ./checkpoints/ckpt_step_0015000.pt \\
            --config_path ./checkpoints/config.json \\
            --repo_id prasanna/custom-gpt-mini

          python scripts/hf_deploy.py \\
            --model_path ./checkpoints/run_0042 \\
            --config_path ./checkpoints/run_0042/config.json \\
            --repo_id prasanna/custom-gpt-mini \\
            --tokenizer_path ./tokenizer \\
            --private
        """
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint (.pt) or directory")
    parser.add_argument("--config_path", type=str, required=True, help="Path to training config JSON")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer directory")
    parser.add_argument("--revision", type=str, default="main", help="Branch or tag name")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--no-inference", dest="run_inference", action="store_false",
                        help="Skip post-deployment inference test")
    parser.set_defaults(run_inference=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        deploy_to_hub(
            model_path=args.model_path,
            config_path=args.config_path,
            repo_id=args.repo_id,
            tokenizer_path=args.tokenizer_path,
            revision=args.revision,
            private=args.private,
            token=args.token,
            run_inference=args.run_inference,
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
