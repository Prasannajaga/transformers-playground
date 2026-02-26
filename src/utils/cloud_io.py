"""
Cloud storage utilities for uploading model checkpoints.
Supports Google Cloud Storage (GCS) and Google Drive.
"""

import os
import shutil
from typing import Optional


def _normalize_chunk_size(chunk_size_mb: int) -> int:
    if chunk_size_mb <= 0:
        raise ValueError("chunk_size_mb must be > 0")

    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunk_multiple = 256 * 1024

    if chunk_size_bytes % chunk_multiple == 0:
        return chunk_size_bytes

    return ((chunk_size_bytes // chunk_multiple) + 1) * chunk_multiple


def upload_to_gcs(
    local_path: str,
    bucket_name: str,
    destination_blob: Optional[str] = None,
    chunk_size_mb: Optional[int] = None,
    resumable_threshold_mb: int = 128,
    timeout: int = 600,
) -> str:
    """
    Upload a file to Google Cloud Storage.

    Args:
        local_path: Absolute path to the local file.
        bucket_name: GCS bucket name (with or without 'gs://' prefix).
        destination_blob: Blob name in bucket. Defaults to filename.
        chunk_size_mb: Optional upload chunk size in MB. If omitted, chunking is auto-enabled for large files.
        resumable_threshold_mb: Auto-enable chunked resumable uploads when file >= threshold MB.
        timeout: Upload timeout in seconds.

    Returns:
        GCS URI of the uploaded file (gs://bucket/blob).

    Raises:
        ImportError: If google-cloud-storage is not installed.
        Exception: If upload fails.
    """
    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError(
            "google-cloud-storage is required for GCS uploads. "
            "Install with: pip install google-cloud-storage"
        ) from e

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    # Normalize bucket name (remove gs:// prefix if present)
    bucket_name = bucket_name.replace("gs://", "").strip("/")

    if destination_blob is None:
        destination_blob = os.path.basename(local_path)

    file_size_bytes = os.path.getsize(local_path)
    threshold_bytes = resumable_threshold_mb * 1024 * 1024

    selected_chunk_size = chunk_size_mb
    if selected_chunk_size is None and file_size_bytes >= threshold_bytes:
        selected_chunk_size = 64

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    if selected_chunk_size is not None:
        blob.chunk_size = _normalize_chunk_size(selected_chunk_size)

    blob.upload_from_filename(local_path, timeout=timeout)

    gcs_uri = f"gs://{bucket_name}/{destination_blob}"
    return gcs_uri


def mount_google_drive(mount_path: str = "/content/drive") -> str:
    """
    Mount Google Drive (Colab environment).

    Args:
        mount_path: Path to mount Google Drive.

    Returns:
        Path to MyDrive within mounted Drive.

    Raises:
        RuntimeError: If not in Colab or mount fails.
    """
    try:
        from google.colab import drive
        drive.mount(mount_path, force_remount=False)
        return os.path.join(mount_path, "MyDrive")
    except ImportError:
        # Not in Colab - check if already mounted
        my_drive_path = os.path.join(mount_path, "MyDrive")
        if os.path.exists(my_drive_path):
            return my_drive_path
        raise RuntimeError(
            "Google Drive mount not available. "
            "Either run in Colab or ensure Drive is pre-mounted."
        )


def upload_to_gdrive(
    local_path: str,
    folder_id: Optional[str] = None,
    mount_path: str = "/content/drive"
) -> str:
    """
    Upload a file to Google Drive.

    Args:
        local_path: Absolute path to the local file.
        folder_id: Optional subfolder name within MyDrive/checkpoints.
        mount_path: Google Drive mount path.

    Returns:
        Path to the uploaded file in Google Drive.
    """
    my_drive = mount_google_drive(mount_path)

    # Default destination: MyDrive/checkpoints or MyDrive/checkpoints/{folder_id}
    if folder_id:
        dest_dir = os.path.join(my_drive, "checkpoints", folder_id)
    else:
        dest_dir = os.path.join(my_drive, "checkpoints")

    os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(local_path)
    dest_path = os.path.join(dest_dir, filename)

    # Copy as complete file (no chunking)
    shutil.copy2(local_path, dest_path)

    return dest_path
