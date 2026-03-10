import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, login
from safetensors.torch import save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def convert_to_safetensors(model_path: Path) -> None:
    """Converts .bin or .pt model files to .safetensors if they don't already exist."""
    for filepath in model_path.rglob("*"):
        if filepath.suffix in [".bin", ".pt"] and filepath.is_file():
            # Skip training arguments and optimizers as we typically only want model weights
            if "training_args" in filepath.name or "optimizer" in filepath.name:
                continue
            
            safetensors_path = filepath.with_suffix(".safetensors")
            if not safetensors_path.exists():
                logger.info(f"Converting {filepath.name} to safetensors format...")
                try:
                    state_dict = torch.load(filepath, map_location="cpu", weights_only=False)
                    
                    if isinstance(state_dict, dict):
                        # Extract the actual weights if it's a nested dictionary
                        if "model_state_dict" in state_dict:
                            state_dict = state_dict["model_state_dict"]
                        elif "state_dict" in state_dict:
                            state_dict = state_dict["state_dict"]
                        
                        # Filter to only include tensors and handle shared memory
                        tensors_only = {}
                        seen_ptrs = set()
                        for k, v in state_dict.items():
                            if isinstance(v, torch.Tensor):
                                ptr = v.untyped_storage().data_ptr()
                                if ptr in seen_ptrs:
                                    tensors_only[str(k)] = v.clone()
                                else:
                                    tensors_only[str(k)] = v
                                    seen_ptrs.add(ptr)
                                    
                        if tensors_only:
                            save_file(tensors_only, safetensors_path)
                            logger.info(f"Successfully created {safetensors_path.name}")
                        else:
                            logger.warning(f"No valid tensors found in {filepath.name}. Skipping conversion.")
                except Exception as e:
                    logger.error(f"Failed to convert {filepath.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local model to Hugging Face Hub efficiently.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local model directory")
    parser.add_argument("--repo-name", type=str, required=True, help="Hugging Face repo name (e.g., username/repo_name)")
    parser.add_argument("--model-name", type=str, default="model", help="Name of the model (for commit message)")
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token", default=os.getenv("HF_TOKEN"))
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Model path does not exist or is not a valid directory: {model_path}")
        sys.exit(1)
        
    # Handle HF Authentication
    if args.hf_token:
        login(token=args.hf_token)
    elif "HF_TOKEN" not in os.environ:
        logger.warning("No explicit HF token provided. Falling back to cached credentials if available.")
        
    # Pre-process: convert any raw PyTorch checkpoints to safetensors
    convert_to_safetensors(model_path)
    
    api = HfApi()
    
    try:
        # Create repo (no-op if it already exists, defaults to private to be safe)
        api.create_repo(repo_id=args.repo_name, repo_type="model", exist_ok=True, private=True)
        logger.info(f"Target repository: https://huggingface.co/{args.repo_name}")
        
        # Identify files to ignore 
        # (Exclude .pt/.bin files as requested, alongside any git data)
        ignore_patterns = ["*.pt", "*.bin", ".git/*"]
        
        logger.info(f"Starting efficient upload from {args.model_path}...")
        logger.info(f"Ignoring files matching: {ignore_patterns}")
        
        # upload_folder uses efficient concurrent uploading and skips already-uploaded files
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_name,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            commit_message=f"Upload model: {args.model_name}"
        )
        logger.info(f"✅ Successfully uploaded model to Hugging Face Hub!")
        
    except Exception as e:
        logger.error(f"An error occurred during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
