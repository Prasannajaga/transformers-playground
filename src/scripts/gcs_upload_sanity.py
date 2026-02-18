import logging
import os
import sys
import time
from pathlib import Path
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


def split_gs_uri(gs_uri: str) -> tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    content = gs_uri[5:]
    if "/" in content:
        bucket, prefix = content.split("/", 1)
    else:
        bucket, prefix = content, ""
    return bucket, prefix.strip("/")


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _required_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def check_environment(logger: logging.Logger) -> None:
    import torch

    logger.info("--- environment check ---")

    # ── CUDA devices ──────────────────────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    logger.info("cuda.available=%s", cuda_available)
    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info("cuda.device_count=%d", device_count)
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024 ** 3)
            logger.info(
                "cuda.device[%d] name=%s compute=%d.%d vram=%.2fGB",
                idx, props.name, props.major, props.minor, total_gb,
            )
    else:
        logger.warning("cuda.available=False — no GPU detected")

    # ── Distributed GPU (NCCL) ────────────────────────────────────────────────
    try:
        import torch.distributed as dist
        nccl_available = dist.is_nccl_available()
        logger.info("distributed.nccl_available=%s", nccl_available)

        world_size_env = os.getenv("WORLD_SIZE")
        if world_size_env:
            logger.info("distributed.WORLD_SIZE=%s", world_size_env)
            logger.info("distributed.RANK=%s", os.getenv("RANK", "<not-set>"))
            logger.info("distributed.LOCAL_RANK=%s", os.getenv("LOCAL_RANK", "<not-set>"))
        else:
            logger.info("distributed.WORLD_SIZE=<not-set> (single-node run)")
    except Exception as exc:
        logger.warning("distributed check failed: %s", exc)

    # ── Flash Attention ───────────────────────────────────────────────────────
    try:
        import flash_attn
        logger.info("flash_attn.available=True version=%s", flash_attn.__version__)
    except ImportError:
        logger.info("flash_attn.available=False (package not installed)")
    except Exception as exc:
        logger.warning("flash_attn check failed: %s", exc)

    # ── Modern torch architecture features ────────────────────────────────────
    try:
        import torch.nn.functional as F
        sdpa_available = hasattr(F, "scaled_dot_product_attention")
        logger.info("torch.scaled_dot_product_attention=%s", sdpa_available)
    except Exception as exc:
        logger.warning("sdpa check failed: %s", exc)

    try:
        compile_available = hasattr(torch, "compile")
        logger.info("torch.compile=%s", compile_available)
    except Exception as exc:
        logger.warning("torch.compile check failed: %s", exc)

    try:
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
        logger.info("cuda.bf16_supported=%s", bf16_supported)
    except Exception as exc:
        logger.warning("bf16 check failed: %s", exc)

    try:
        torch_version = torch.__version__
        logger.info("torch.version=%s", torch_version)
    except Exception as exc:
        logger.warning("torch version check failed: %s", exc)

    logger.info("--- environment check complete ---")


def run() -> int:
    import torch

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    check_environment(logger)

    aip_model_dir = os.getenv("AIP_MODEL_DIR")
    logger.info("AIP_MODEL_DIR=%s", aip_model_dir or "<not-set>")

    target_dir = aip_model_dir
    if not target_dir:
        bucket_name = os.getenv("BUCKET_NAME", "").replace("gs://", "").strip("/")
        if not bucket_name:
            raise ValueError("AIP_MODEL_DIR and BUCKET_NAME are both missing")
        target_dir = f"gs://{bucket_name}/model-artifacts/{int(time.time())}"
        logger.warning("AIP_MODEL_DIR unavailable. Falling back to target_dir=%s", target_dir)

    bucket_from_target, prefix_from_target = split_gs_uri(target_dir)
    local_dir = Path(os.getenv("LOCAL_MODEL_DIR", "/tmp")).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = f"sample_model_{int(time.time())}.pt"
    local_file = local_dir / artifact_name

    model_blob = {
        "weights": torch.randn(4, 4),
        "bias": torch.randn(4),
        "meta": {
            "source": "vertex-upload-sanity",
            "timestamp": int(time.time()),
            "shape": [4, 4],
        },
    }
    torch.save(model_blob, local_file)
    logger.info("Saved sample model locally: %s", local_file)

    destination_blob = f"{prefix_from_target}/{artifact_name}" if prefix_from_target else artifact_name
    logger.info("Uploading to gs://%s/%s", bucket_from_target, destination_blob)

    chunk_size_mb = _optional_int_env("UPLOAD_CHUNK_SIZE_MB")
    threshold_mb = _required_int_env("UPLOAD_THRESHOLD_MB", 128)
    timeout = _required_int_env("UPLOAD_TIMEOUT_SEC", 600)

    gcs_uri = upload_to_gcs(
        local_path=str(local_file),
        bucket_name=bucket_from_target,
        destination_blob=destination_blob,
        chunk_size_mb=chunk_size_mb,
        resumable_threshold_mb=threshold_mb,
        timeout=timeout,
    )

    logger.info("Upload successful: %s", gcs_uri)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        logging.getLogger(__name__).exception("Upload sanity script failed: %s", exc)
        sys.exit(1)
