#!/usr/bin/env python3
"""
Script to load a checkpoint file and upload it to Google Cloud Storage.

Usage:
    python upload_checkpoint_to_gcs.py <checkpoint_path> <bucket_name> [--blob-name <name>]

Examples:
    python upload_checkpoint_to_gcs.py ./checkpoints/ckpt_step_0015000.pt gs://my-bucket
    python upload_checkpoint_to_gcs.py ./checkpoints/final.pt gs://my-bucket --blob-name models/final.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.cloud_storage import upload_to_gcs


def _check_gcp_auth() -> bool:
    """
    Check if GCP authentication is configured.
    
    Returns:
        True if credentials are available, False otherwise.
    """
    try:
        from google.cloud import storage
        from google.auth import default
        from google.auth.exceptions import DefaultCredentialsError
        
        try:
            credentials, project = default()
            return True
        except DefaultCredentialsError:
            return False
    except ImportError:
        print("Error: google-cloud-storage not installed", file=sys.stderr)
        print("Install with: pip install google-cloud-storage", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload a checkpoint file to Google Cloud Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./checkpoints/ckpt_step_0015000.pt gs://my-bucket
  %(prog)s ./checkpoints/final.pt gs://my-bucket --blob-name models/final.pt
        """
    )
    
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file to upload"
    )
    
    parser.add_argument(
        "bucket_name",
        type=str,
        help="GCS bucket name (with or without 'gs://' prefix)"
    )
    
    parser.add_argument(
        "--blob-name",
        type=str,
        default=None,
        help="Destination blob name in bucket (defaults to checkpoint filename)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the checkpoint file exists before uploading"
    )
    
    args = parser.parse_args()
    
    # Check GCP authentication first
    if not _check_gcp_auth():
        print("\n" + "="*70, file=sys.stderr)
        print("ERROR: GCP authentication not configured", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print("\nTo authenticate, choose ONE of the following methods:\n", file=sys.stderr)
        print("1. Application Default Credentials (recommended for local development):", file=sys.stderr)
        print("   gcloud auth application-default login\n", file=sys.stderr)
        print("2. Service Account Key (for production/CI):", file=sys.stderr)
        print("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json\n", file=sys.stderr)
        print("3. Vertex AI Custom Job (automatic in Vertex AI environment)", file=sys.stderr)
        print("="*70, file=sys.stderr)
        sys.exit(1)
    
    # Validate checkpoint path
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Path is not a file: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    # Get file size for logging
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Bucket: {args.bucket_name}")
    print(f"Blob name: {args.blob_name or os.path.basename(checkpoint_path)}")
    print()
    
    # Optional verification step
    if args.verify:
        try:
            import torch
            print("Verifying checkpoint integrity...")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"✓ Checkpoint valid (training step: {checkpoint.get('training_step', 'unknown')})")
            del checkpoint
        except Exception as e:
            print(f"Warning: Checkpoint verification failed: {e}", file=sys.stderr)
            response = input("Continue with upload? [y/N]: ")
            if response.lower() != 'y':
                print("Upload cancelled.")
                sys.exit(0)
    
    # Upload to GCS
    try:
        print("Uploading to GCS...")
        gcs_uri = upload_to_gcs(
            local_path=checkpoint_path,
            bucket_name=args.bucket_name,
            destination_blob=args.blob_name,
        )
        print(f"✓ Upload successful: {gcs_uri}")
        return 0
    
    except Exception as e:
        print(f"✗ Upload failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
