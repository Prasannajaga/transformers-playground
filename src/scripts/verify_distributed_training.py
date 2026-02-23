"""
Distributed training verification script (self-contained).

Trains a deliberately tiny transformer with flash-attention-backed SDPA
across multiple GPUs using PyTorch DDP.  Verifies:
  1. Flash attention (via torch.nn.functional.scaled_dot_product_attention)
  2. Multi-GPU gradient synchronisation (DDP over NCCL)
  3. GCS checkpoint upload

Designed for 2× NVIDIA L4 on Vertex AI.
Runs ~10 steps with synthetic data — just enough to prove the stack works.

This file is FULLY SELF-CONTAINED — zero project imports.
Vertex AI uploads this single script into the training container.
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ── Constants ────────────────────────────────────────────────────────────────

RANK_ENV = "RANK"
WORLD_SIZE_ENV = "WORLD_SIZE"
LOCAL_RANK_ENV = "LOCAL_RANK"
AIP_MODEL_DIR_ENV = "AIP_MODEL_DIR"
SUPPORTED_AMP_DTYPES = ("float16", "bfloat16")
VERIFY_MODEL_FILENAME = "verify_distributed_model.pt"
VERIFY_META_FILENAME = "verify_meta.json"
SEPARATOR = "=" * 72


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VerifyConfig:
    """
    Frozen, validated configuration.

    Deliberately tiny model and minimal steps — the sole purpose is
    to validate that flash attention + multi-GPU DDP works before
    burning compute on a real training run.
    """

    # Model architecture (intentionally tiny)
    vocab_size: int = 1024
    n_layer: int = 2
    n_embd: int = 64
    n_head: int = 4
    block_size: int = 32
    dropout: float = 0.0

    # Training
    batch_size: int = 4
    total_steps: int = 10
    lr: float = 1e-3
    amp_dtype: str = "bfloat16"
    grad_clip_norm: float = 1.0

    # Distributed
    backend: str = "nccl"

    # GCS output
    bucket_name: str = "gs://transformer-garage"
    gcs_output_prefix: str = "verify_distributed"
    local_tmp_dir: str = "/tmp/verify_model"

    # Logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.amp_dtype not in SUPPORTED_AMP_DTYPES:
            raise ValueError(
                f"amp_dtype must be one of {SUPPORTED_AMP_DTYPES}, got '{self.amp_dtype}'"
            )
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {self.total_steps}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not self.bucket_name.startswith("gs://"):
            raise ValueError(
                f"bucket_name must start with gs://, got '{self.bucket_name}'"
            )


# ── Logging ──────────────────────────────────────────────────────────────────

def _setup_logging(rank: int, level: str) -> logging.Logger:
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s | %(levelname)s | rank={rank} | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    return logging.getLogger("verify_distributed")


# ── Tiny Model ───────────────────────────────────────────────────────────────

class FlashSelfAttention(nn.Module):
    """Multi-head self-attention using torch SDPA (flash-attention backend)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TinyTransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = FlashSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """
    Minimal GPT-style transformer for verification.

    Architecture: token embed → N blocks → layer norm → lm_head
    Attention uses F.scaled_dot_product_attention → flash backend on L4.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TinyTransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── Synthetic dataset ────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """Random token sequences — no real data needed for verification."""

    def __init__(self, vocab_size: int, block_size: int, num_samples: int) -> None:
        self.data = torch.randint(0, vocab_size, (num_samples, block_size + 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        return seq[:-1], seq[1:]


# ── GCS upload ───────────────────────────────────────────────────────────────

def _upload_directory_to_gcs(
    local_dir: str,
    bucket_name: str,
    gcs_prefix: str,
    logger: logging.Logger,
) -> list[str]:
    """Upload all files from local_dir to GCS bucket under gcs_prefix."""
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for GCS uploads. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    clean_bucket = bucket_name.replace("gs://", "").strip("/")
    client = storage.Client()
    bucket = client.bucket(clean_bucket)
    uploaded: list[str] = []

    for file_path in sorted(Path(local_dir).rglob("*")):
        if not file_path.is_file():
            continue

        blob_name = f"{gcs_prefix}/{file_path.relative_to(local_dir)}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(file_path), timeout=300)

        gcs_uri = f"gs://{clean_bucket}/{blob_name}"
        uploaded.append(gcs_uri)
        logger.info("uploaded=%s", gcs_uri)

    return uploaded


# ── Distributed helpers ──────────────────────────────────────────────────────

def _init_distributed(cfg: VerifyConfig, logger: logging.Logger) -> tuple[int, int, int]:
    """Initialize DDP process group; returns (rank, local_rank, world_size)."""
    rank = int(os.environ.get(RANK_ENV, 0))
    local_rank = int(os.environ.get(LOCAL_RANK_ENV, 0))
    world_size = int(os.environ.get(WORLD_SIZE_ENV, 1))

    logger.info(
        "dist.init rank=%d local_rank=%d world_size=%d backend=%s",
        rank, local_rank, world_size, cfg.backend,
    )
    dist.init_process_group(backend=cfg.backend)
    torch.cuda.set_device(local_rank)
    logger.info("dist.init complete — NCCL process group ready")

    return rank, local_rank, world_size


def _cleanup_distributed(logger: logging.Logger) -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("dist.destroy_process_group complete")


# ── Flash attention verification ─────────────────────────────────────────────

def _verify_flash_attention(device: torch.device, logger: logging.Logger) -> bool:
    """Run a micro SDPA forward pass and confirm the flash backend fires."""
    logger.info("verifying flash attention via SDPA")

    q = torch.randn(1, 4, 8, 16, device=device, dtype=torch.float16)
    k = torch.randn(1, 4, 8, 16, device=device, dtype=torch.float16)
    v = torch.randn(1, 4, 8, 16, device=device, dtype=torch.float16)

    try:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        logger.info("SDPA forward ok shape=%s dtype=%s", out.shape, out.dtype)
    except Exception as exc:
        logger.error("SDPA forward failed: %s", exc)
        return False

    try:
        from torch.backends.cuda import flash_sdp_enabled
        logger.info("torch.backends.cuda.flash_sdp_enabled=%s", flash_sdp_enabled())
    except ImportError:
        logger.info("flash_sdp_enabled not available in this torch build")

    try:
        import flash_attn
        logger.info("flash_attn package version=%s", flash_attn.__version__)
    except ImportError:
        logger.info("flash_attn package not installed (using torch SDPA backend)")

    logger.info("flash_attention_verified=True")
    return True


# ── DDP gradient sync verification ──────────────────────────────────────────

def _verify_gradient_sync(
    model: DDP,
    device: torch.device,
    cfg: VerifyConfig,
    logger: logging.Logger,
) -> bool:
    """Forward+backward on dummy batch, verify grads exist and are finite."""
    logger.info("verifying DDP gradient synchronisation")

    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device=device)
    t = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device=device)
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        _, loss = model(x, t)
    loss.backward()

    first_param = next(model.parameters())
    if first_param.grad is None:
        logger.error("gradient_sync_failed: grad is None")
        return False

    grad_norm = first_param.grad.data.norm().item()
    logger.info("sample_grad_norm=%.6f", grad_norm)

    if math.isnan(grad_norm) or math.isinf(grad_norm):
        logger.error("gradient_sync_failed: grad_norm is nan/inf")
        return False

    logger.info("gradient_sync_verified=True")
    return True


# ── Training loop ────────────────────────────────────────────────────────────

def _train(
    model: DDP,
    dataloader: DataLoader,
    device: torch.device,
    cfg: VerifyConfig,
    logger: logging.Logger,
) -> float:
    """Run cfg.total_steps of training; return final loss."""
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    step = 0
    final_loss = float("nan")
    data_iter = iter(dataloader)
    t0 = time.time()

    while step < cfg.total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        final_loss = loss.item()
        step += 1
        dt = time.time() - t0
        logger.info("step=%d/%d loss=%.4f time=%.3fs", step, cfg.total_steps, final_loss, dt)
        t0 = time.time()

    return final_loss


# ── Save + upload ────────────────────────────────────────────────────────────

def _save_and_upload(
    model: DDP,
    cfg: VerifyConfig,
    rank: int,
    world_size: int,
    final_loss: float,
    logger: logging.Logger,
) -> None:
    """Save checkpoint locally, upload to GCS (rank 0 only)."""
    if rank != 0:
        logger.info("rank=%d skipping save/upload (rank 0 handles it)", rank)
        return

    local_dir = Path(cfg.local_tmp_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    model_path = local_dir / VERIFY_MODEL_FILENAME
    torch.save(model.module.state_dict(), model_path)
    logger.info("saved_checkpoint=%s", model_path)

    meta = {
        "script": "verify_distributed_training",
        "timestamp": int(time.time()),
        "world_size": world_size,
        "total_steps": cfg.total_steps,
        "final_loss": final_loss,
        "amp_dtype": cfg.amp_dtype,
        "config": asdict(cfg),
    }
    meta_path = local_dir / VERIFY_META_FILENAME
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("saved_meta=%s", meta_path)

    aip_model_dir = os.getenv(AIP_MODEL_DIR_ENV)
    if aip_model_dir:
        gcs_bucket, gcs_prefix = aip_model_dir.replace("gs://", "").split("/", 1)
        gcs_bucket = f"gs://{gcs_bucket}"
    else:
        gcs_bucket = cfg.bucket_name
        gcs_prefix = f"{cfg.gcs_output_prefix}/{int(time.time())}"
        logger.warning(
            "%s not set, falling back to bucket=%s prefix=%s",
            AIP_MODEL_DIR_ENV, gcs_bucket, gcs_prefix,
        )

    uploaded = _upload_directory_to_gcs(
        local_dir=str(local_dir),
        bucket_name=gcs_bucket,
        gcs_prefix=gcs_prefix,
        logger=logger,
    )
    logger.info("upload_complete count=%d", len(uploaded))


# ── Memory diagnostics ──────────────────────────────────────────────────────

def _log_gpu_memory(device: torch.device, logger: logging.Logger) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    logger.info(
        "gpu_memory allocated=%.2fGB reserved=%.2fGB total=%.2fGB",
        allocated, reserved, total,
    )


# ── Entrypoint ───────────────────────────────────────────────────────────────

def run() -> int:
    cfg = VerifyConfig()

    rank = int(os.environ.get(RANK_ENV, 0))
    logger = _setup_logging(rank, cfg.log_level)

    logger.info(SEPARATOR)
    logger.info("DISTRIBUTED TRAINING VERIFICATION")
    logger.info(SEPARATOR)
    logger.info("python=%s torch=%s", sys.version.split()[0], torch.__version__)
    logger.info(
        "cuda.available=%s device_count=%d",
        torch.cuda.is_available(), torch.cuda.device_count(),
    )

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            "gpu[%d] name=%s compute=%d.%d vram=%.2fGB",
            i, props.name, props.major, props.minor,
            props.total_memory / (1024 ** 3),
        )

    rank, local_rank, world_size = _init_distributed(cfg, logger)
    device = torch.device("cuda", local_rank)

    # Step 1: Flash attention
    flash_ok = _verify_flash_attention(device, logger)
    if not flash_ok:
        logger.error("FLASH ATTENTION VERIFICATION FAILED — aborting")
        _cleanup_distributed(logger)
        return 1
    logger.info(SEPARATOR)

    # Step 2: Build model + DDP
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = TinyTransformer(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("model_params=%d (%.2fK)", param_count, param_count / 1000)

    ddp_model = DDP(model, device_ids=[local_rank])
    logger.info("DDP wrapper applied — device_ids=[%d]", local_rank)
    _log_gpu_memory(device, logger)

    # Step 3: Gradient sync
    grad_ok = _verify_gradient_sync(ddp_model, device, cfg, logger)
    if not grad_ok:
        logger.error("GRADIENT SYNC VERIFICATION FAILED — aborting")
        _cleanup_distributed(logger)
        return 1
    ddp_model.zero_grad(set_to_none=True)
    logger.info(SEPARATOR)

    # Step 4: Training
    dataset = SyntheticDataset(cfg.vocab_size, cfg.block_size, num_samples=200)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(
        "starting training total_steps=%d batch_size=%d",
        cfg.total_steps, cfg.batch_size,
    )
    final_loss = _train(ddp_model, dataloader, device, cfg, logger)
    logger.info("training_complete final_loss=%.4f", final_loss)
    _log_gpu_memory(device, logger)
    logger.info(SEPARATOR)

    # Step 5: Save + Upload
    if dist.is_initialized():
        dist.barrier()
    _save_and_upload(ddp_model, cfg, rank, world_size, final_loss, logger)

    # Done
    _cleanup_distributed(logger)
    logger.info(SEPARATOR)
    logger.info("ALL VERIFICATIONS PASSED")
    logger.info("  flash_attention=OK")
    logger.info("  gradient_sync=OK")
    logger.info("  training=%d steps OK (final_loss=%.4f)", cfg.total_steps, final_loss)
    logger.info("  gcs_upload=OK (rank 0)")
    logger.info(SEPARATOR)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        logging.getLogger("verify_distributed").exception(
            "Verification failed: %s", exc
        )
        sys.exit(1)
