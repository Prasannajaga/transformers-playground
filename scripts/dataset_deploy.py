"""
Production-grade HuggingFace Dataset deployment script.

Uploads datasets to HuggingFace Hub with proper metadata and configuration.
Supports multiple formats: Arrow, Parquet, CSV, JSON, and HuggingFace Dataset objects.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi, login, repo_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetDeployConfig:
    """Configuration for dataset deployment operations."""
    
    dataset_path: str
    repo_id: str
    private: bool = False
    token: Optional[str] = None
    revision: str = "main"
    commit_message: Optional[str] = None
    dataset_format: str = "auto"  # auto, arrow, parquet, csv, json, disk
    preserve_card: bool = False  # Keep existing README.md when updating
    
    # Metadata
    description: Optional[str] = None
    license: Optional[str] = None
    tags: Optional[List[str]] = None
    language: Optional[List[str]] = None
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.dataset_path:
            raise ValueError("dataset_path cannot be empty")
        
        if not self.repo_id:
            raise ValueError("repo_id must be specified (format: username/dataset-name)")
        
        if "/" not in self.repo_id:
            raise ValueError(
                f"Invalid repo_id format: '{self.repo_id}'. "
                "Expected format: 'username/dataset-name'"
            )
        
        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.dataset_path}"
            )
        
        valid_formats = ["auto", "arrow", "parquet", "csv", "json", "disk"]
        if self.dataset_format not in valid_formats:
            raise ValueError(
                f"Invalid dataset_format: '{self.dataset_format}'. "
                f"Must be one of: {valid_formats}"
            )
        
        logger.info("✓ Configuration validated successfully")


def check_repo_exists(repo_id: str, token: Optional[str] = None) -> bool:
    """
    Check if a repository already exists on HuggingFace Hub.
    
    Args:
        repo_id: Repository ID to check
        token: Optional HF token
        
    Returns:
        True if repo exists, False otherwise
    """
    try:
        exists = repo_exists(repo_id=repo_id, repo_type="dataset", token=token)
        return exists
    except Exception as e:
        logger.warning(f"Could not check repo existence: {e}")
        return False


def authenticate(token: Optional[str] = None) -> None:
    """
    Authenticate with HuggingFace Hub.
    
    Args:
        token: Optional HF token. If None, uses cached credentials or env var.
    """
    try:
        if token:
            login(token=token, add_to_git_credential=True)
            logger.info("✓ Authenticated with provided token")
        else:
            # Try using cached credentials or HF_TOKEN env var
            login()
            logger.info("✓ Authenticated using cached/environment credentials")
    except Exception as e:
        logger.error(
            "Authentication failed. Please run 'huggingface-cli login' "
            "or provide a valid token via --token"
        )
        raise RuntimeError(f"Authentication failed: {e}")


def detect_format(dataset_path: str) -> str:
    """
    Auto-detect dataset format based on path and contents.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Detected format: 'arrow', 'parquet', 'csv', 'json', or 'disk'
    """
    path = Path(dataset_path)
    
    # Check if it's a HuggingFace dataset directory
    if path.is_dir():
        if (path / "dataset_info.json").exists():
            logger.info(f"Detected format: HuggingFace dataset directory")
            return "disk"
        
        # Check for common file patterns
        files = list(path.iterdir())
        if any(f.suffix == ".arrow" for f in files):
            logger.info(f"Detected format: Arrow")
            return "arrow"
        elif any(f.suffix == ".parquet" for f in files):
            logger.info(f"Detected format: Parquet")
            return "parquet"
        elif any(f.suffix == ".csv" for f in files):
            logger.info(f"Detected format: CSV")
            return "csv"
        elif any(f.suffix == ".json" or f.suffix == ".jsonl" for f in files):
            logger.info(f"Detected format: JSON")
            return "json"
    
    # Single file
    elif path.is_file():
        suffix = path.suffix.lower()
        format_map = {
            ".arrow": "arrow",
            ".parquet": "parquet",
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "json",
        }
        if suffix in format_map:
            detected = format_map[suffix]
            logger.info(f"Detected format: {detected}")
            return detected
    
    raise ValueError(
        f"Could not auto-detect format for: {dataset_path}. "
        "Please specify --format explicitly."
    )


def load_dataset_from_path(
    dataset_path: str,
    dataset_format: str = "auto"
) -> DatasetDict:
    """
    Load dataset from local path with format detection.
    
    Args:
        dataset_path: Path to the dataset
        dataset_format: Format of the dataset (auto for auto-detection)
        
    Returns:
        Loaded DatasetDict or Dataset
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    path = Path(dataset_path)
    
    # Auto-detect if needed
    if dataset_format == "auto":
        dataset_format = detect_format(dataset_path)
    
    # Load based on format
    try:
        if dataset_format == "disk":
            dataset = load_from_disk(str(path))
            logger.info("✓ Loaded from HuggingFace dataset directory")
            
        elif dataset_format == "arrow":
            if path.is_dir():
                dataset = load_dataset("arrow", data_dir=str(path))
            else:
                dataset = load_dataset("arrow", data_files=str(path))
            logger.info("✓ Loaded from Arrow format")
            
        elif dataset_format == "parquet":
            if path.is_dir():
                dataset = load_dataset("parquet", data_dir=str(path))
            else:
                dataset = load_dataset("parquet", data_files=str(path))
            logger.info("✓ Loaded from Parquet format")
            
        elif dataset_format == "csv":
            if path.is_dir():
                dataset = load_dataset("csv", data_dir=str(path))
            else:
                dataset = load_dataset("csv", data_files=str(path))
            logger.info("✓ Loaded from CSV format")
            
        elif dataset_format == "json":
            if path.is_dir():
                dataset = load_dataset("json", data_dir=str(path))
            else:
                dataset = load_dataset("json", data_files=str(path))
            logger.info("✓ Loaded from JSON format")
            
        else:
            raise ValueError(f"Unsupported format: {dataset_format}")
        
        # Convert single Dataset to DatasetDict if needed
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        
        # Log dataset info
        logger.info(f"Dataset splits: {list(dataset.keys())}")
        for split, ds in dataset.items():
            logger.info(f"  {split}: {len(ds)} examples")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def create_dataset_card(
    repo_id: str,
    dataset: DatasetDict,
    description: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[List[str]] = None,
    language: Optional[List[str]] = None,
) -> str:
    """
    Create a comprehensive README.md for the dataset.
    
    Args:
        repo_id: HuggingFace repository ID
        dataset: The dataset being uploaded
        description: Dataset description
        license: License type
        tags: List of tags
        language: List of languages
        
    Returns:
        Markdown content for README.md
    """
    splits_info = []
    for split, ds in dataset.items():
        num_examples = len(ds)
        features = ds.features
        splits_info.append(f"- **{split}**: {num_examples:,} examples")
    
    splits_section = "\n".join(splits_info)
    
    # Get feature info from first split
    first_split = next(iter(dataset.values()))
    features_info = []
    for feat_name, feat_type in first_split.features.items():
        features_info.append(f"- `{feat_name}`: {feat_type}")
    features_section = "\n".join(features_info)
    
    # Build YAML header
    yaml_lines = ["---"]
    if license:
        yaml_lines.append(f"license: {license}")
    if tags:
        yaml_lines.append(f"tags:")
        for tag in tags:
            yaml_lines.append(f"  - {tag}")
    if language:
        yaml_lines.append(f"language:")
        for lang in language:
            yaml_lines.append(f"  - {lang}")
    yaml_lines.append("---")
    yaml_header = "\n".join(yaml_lines)
    
    card = f"""{yaml_header}

# Dataset Card for {repo_id.split('/')[-1]}

## Dataset Description

{description or "Dataset uploaded via automated deployment script."}

## Dataset Structure

### Data Splits

{splits_section}

### Features

{features_section}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Dataset Creation

This dataset was uploaded using the HuggingFace Dataset deployment script.

## Additional Information

For more information about the dataset, please contact the dataset creator.
"""
    
    return card


def push_to_hub(
    dataset: DatasetDict,
    config: DatasetDeployConfig,
) -> None:
    """
    Push dataset to HuggingFace Hub.
    
    Args:
        dataset: The dataset to upload
        config: Deployment configuration
    """
    # Check if repo exists
    repo_already_exists = check_repo_exists(config.repo_id, config.token)
    
    logger.info("=" * 60)
    if repo_already_exists:
        logger.info("Updating Existing Dataset on HuggingFace Hub")
        logger.info(f"⚠️  Repository '{config.repo_id}' already exists - will update")
    else:
        logger.info("Creating New Dataset on HuggingFace Hub")
        logger.info(f"✓ Repository '{config.repo_id}' will be created")
    logger.info("=" * 60)
    logger.info(f"Repository: {config.repo_id}")
    logger.info(f"Private: {config.private}")
    logger.info(f"Revision: {config.revision}")
    if repo_already_exists and config.preserve_card:
        logger.info(f"Preserve README: Yes (existing README will be kept)")
    
    try:
        # Push dataset
        logger.info("Uploading dataset...")
        dataset.push_to_hub(
            repo_id=config.repo_id,
            private=config.private,
            token=config.token,
            commit_message=config.commit_message or (
                "Update dataset via deployment script" if repo_already_exists 
                else "Upload dataset via deployment script"
            ),
            revision=config.revision,
        )
        logger.info("✓ Dataset uploaded successfully")
        
        # Create and upload README.md (unless preserving existing)
        if repo_already_exists and config.preserve_card:
            logger.info("⊙ Preserving existing dataset card (--preserve-card enabled)")
        else:
            action = "Updating" if repo_already_exists else "Creating"
            logger.info(f"{action} dataset card...")
            readme_content = create_dataset_card(
                repo_id=config.repo_id,
                dataset=dataset,
                description=config.description,
                license=config.license,
                tags=config.tags,
                language=config.language,
            )
            
            api = HfApi(token=config.token)
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=config.repo_id,
                repo_type="dataset",
                commit_message="Add dataset card" if not repo_already_exists else "Update dataset card",
            )
            logger.info(f"✓ Dataset card {'created' if not repo_already_exists else 'updated'}")
        
        logger.info("=" * 60)
        if repo_already_exists:
            logger.info("✓ Update Complete!")
        else:
            logger.info("✓ Deployment Complete!")
        logger.info("=" * 60)
        logger.info(f"Dataset available at: https://huggingface.co/datasets/{config.repo_id}")
        
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")
        raise


def deploy_dataset(config: DatasetDeployConfig) -> None:
    """
    Main deployment pipeline.
    
    Args:
        config: Deployment configuration
    """
    try:
        # Validate configuration
        config.validate()
        
        # Authenticate
        authenticate(token=config.token)
        
        # Load dataset
        dataset = load_dataset_from_path(
            dataset_path=config.dataset_path,
            dataset_format=config.dataset_format,
        )
        
        # Push to hub
        push_to_hub(dataset=dataset, config=config)
        
        logger.info("✓ Dataset deployment completed successfully!")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy datasets to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy a new dataset
  python dataset-deploy.py --dataset-path ./my_dataset --repo-id username/my-dataset

  # Deploy with auto-detected format
  python dataset-deploy.py --dataset-path ./data.csv --repo-id username/csv-dataset

  # Update existing dataset and preserve its README
  python dataset-deploy.py \\
    --dataset-path ./updated_data.json \\
    --repo-id username/my-dataset \\
    --preserve-card

  # Deploy private dataset with metadata
  python dataset-deploy.py \\
    --dataset-path ./data \\
    --repo-id username/private-data \\
    --private \\
    --description "My private dataset" \\
    --license "mit" \\
    --tags nlp classification

  # Deploy with explicit format
  python dataset-deploy.py \\
    --dataset-path ./data \\
    --repo-id username/parquet-data \\
    --format parquet
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset (file or directory)",
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/dataset-name)",
    )
    
    # Optional configuration
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "arrow", "parquet", "csv", "json", "disk"],
        help="Dataset format (default: auto-detect)",
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, uses cached/env by default)",
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision to push to (default: main)",
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message",
    )
    
    parser.add_argument(
        "--preserve-card",
        action="store_true",
        help="Keep existing README.md when updating (don't overwrite dataset card)",
    )
    
    # Metadata
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Dataset description for the README",
    )
    
    parser.add_argument(
        "--license",
        type=str,
        default=None,
        help="Dataset license (e.g., mit, apache-2.0, cc-by-4.0)",
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Dataset tags (space-separated)",
    )
    
    parser.add_argument(
        "--language",
        type=str,
        nargs="+",
        default=None,
        help="Dataset languages (space-separated, e.g., en fr de)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Build configuration
    config = DatasetDeployConfig(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        revision=args.revision,
        commit_message=args.commit_message,
        dataset_format=args.format,
        preserve_card=args.preserve_card,
        description=args.description,
        license=args.license,
        tags=args.tags,
        language=args.language,
    )
    
    # Execute deployment
    deploy_dataset(config)


if __name__ == "__main__":
    main()
