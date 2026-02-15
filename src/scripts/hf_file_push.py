"""
Push a single file to a HuggingFace Hub repository.

Usage:
  python src/scripts/hf_file_push.py \
    --file_path path/to/SmolLM2-360M-Instruct.Q8_0.gguf \
    --repo_id username/my-model

  python src/scripts/hf_file_push.py \
    --file_path model.gguf \
    --repo_id username/my-model \
    --path_in_repo models/model.gguf \
    --private \
    --commit_message "Add quantized GGUF model"
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a single file to a HuggingFace Hub repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/scripts/hf_file_push.py \\
    --file_path SmolLM2-360M-Instruct.Q8_0.gguf \\
    --repo_id username/my-model

  python src/scripts/hf_file_push.py \\
    --file_path model.gguf \\
    --repo_id username/my-model \\
    --path_in_repo gguf/model.gguf \\
    --repo_type model \\
    --private
        """,
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Local path to the file to upload",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/my-model)",
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default=None,
        help="Destination path inside the repo (default: filename of --file_path)",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type (default: model)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch or tag (default: main)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it doesn't exist yet",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, uses cached/env by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import: heavy deps only after arg validation
    from utils.hf_wrapper import HFWrapper

    logger.info(f"File       : {args.file_path}")
    logger.info(f"Repo       : {args.repo_id}")
    logger.info(f"Repo type  : {args.repo_type}")
    logger.info(f"Destination: {args.path_in_repo or '(same as filename)'}")

    try:
        url = HFWrapper.push_file(
            file_path=args.file_path,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
            private=args.private,
            token=args.token,
        )
        logger.info(f"✓ Upload complete: {url}")
    except FileNotFoundError as exc:
        logger.error(f"File not found: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        logger.error(f"Authentication error: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Upload failed: {exc}")
        raise


if __name__ == "__main__":
    main()
