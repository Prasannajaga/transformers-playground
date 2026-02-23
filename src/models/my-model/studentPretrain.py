from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from customTransformers import GQATransformer


@dataclass
class ModelConfig:
    num_layers: int = 12
    n_emb: int = 192
    n_head: int = 6
    n_kv_head: int = 2
    block_size: int = 256
    dropout: float = 0.0


@dataclass
class TrainConfig:
    tokenizer_name: str = "unsloth/Llama-3.2-1B-Instruct"
    openwebtext_dataset: str = "openwebtext"
    openwebtext_split: str = "train"
    openwebtext_samples: int = 1_000_000
    openwebtext_max_updates: int = 0

    batch_size: int = 8
    grad_accum_steps: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    domain_epochs: int = 6
    log_interval: int = 20
    save_interval: int = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "outputs" / "llama32_student_pretrain")
    parser.add_argument("--prasanna-data", type=Path, default=ROOT_DIR / "datasets" / "prasanna_data.json")
    parser.add_argument("--prasanna-text", type=Path, default=ROOT_DIR / "outputs" / "llama32_student_pretrain" / "prasanna_text.jsonl")
    parser.add_argument("--tokenizer-name", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokenizer-local-path", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=ROOT_DIR / "outputs" / "hf_cache")
    parser.add_argument("--openwebtext-samples", type=int, default=1_000_000)
    parser.add_argument("--openwebtext-max-updates", type=int, default=0)
    parser.add_argument("--domain-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-init-checkpoint", type=Path, default=None)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--n-emb", type=int, default=192)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("student_pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step <= warmup_steps:
        return max_lr * float(step) / float(max(1, warmup_steps))
    if total_steps <= 0:
        return max_lr
    if step >= total_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def load_tokenizer(*, tokenizer_name: str, tokenizer_local_path: Path | None, logger: logging.Logger):
    source = tokenizer_name
    kwargs: dict[str, object] = {"use_fast": True}

    if tokenizer_local_path is not None and tokenizer_local_path.exists():
        source = str(tokenizer_local_path)
        kwargs["local_files_only"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, fix_mistral_regex=True, **kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("tokenizer_source=%s vocab_size=%d", source, len(tokenizer))
    return tokenizer


def convert_prasanna_to_text(
    *,
    data_path: Path,
    output_path: Path,
    tokenizer,
    logger: logging.Logger,
) -> int:
    with data_path.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            messages = row.get("messages", [])
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not text:
                continue
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

    logger.info("converted_prasanna_records=%d output=%s", count, output_path)
    return count


class PackedTextDataset(IterableDataset):
    def __init__(self, text_iter_factory: Callable[[], Iterator[str]], tokenizer, block_size: int):
        self.text_iter_factory = text_iter_factory
        self.tokenizer = tokenizer
        self.block_size = int(block_size)

    def __iter__(self):
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = 0

        buffer: list[int] = []
        for text in self.text_iter_factory():
            if not text:
                continue
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            buffer.extend(token_ids)
            buffer.append(eos_id)

            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                buffer = buffer[self.block_size + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def openwebtext_iter_factory(
    *,
    dataset_name: str,
    split: str,
    max_samples: int,
    cache_dir: Path | None = None,
) -> Callable[[], Iterator[str]]:
    def _iter() -> Iterator[str]:
        kwargs: dict[str, object] = {"split": split, "streaming": True}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(cache_dir)
        dataset = load_dataset(dataset_name, **kwargs)
        stream = islice(dataset, max_samples) if max_samples > 0 else dataset
        for row in stream:
            text = row.get("text")
            if text:
                yield text

    return _iter


def jsonl_text_iter_factory(path: Path) -> Callable[[], Iterator[str]]:
    def _iter() -> Iterator[str]:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = payload.get("text")
                if text:
                    yield text

    return _iter


def build_student_model(model_cfg: ModelConfig, vocab_size: int, device: torch.device) -> GQATransformer:
    model = GQATransformer(
        num_layers=model_cfg.num_layers,
        n_emb=model_cfg.n_emb,
        n_head=model_cfg.n_head,
        n_kv_head=model_cfg.n_kv_head,
        vocab_size=vocab_size,
        block_size=model_cfg.block_size,
        dropout=model_cfg.dropout,
    )
    return model.to(device)


def load_student_checkpoint(model: GQATransformer, checkpoint_path: Path, logger: logging.Logger) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        step = int(payload.get("global_update_step", 0))
    elif isinstance(payload, dict):
        state_dict = payload
        step = 0
    else:
        raise ValueError("Unsupported student checkpoint format")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "loaded_student_init=%s missing=%d unexpected=%d",
        checkpoint_path,
        len(missing),
        len(unexpected),
    )
    return step


def save_checkpoint(
    *,
    model: GQATransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    output_root: Path,
    name: str,
    global_update_step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    phase: str,
) -> Path:
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "global_update_step": int(global_update_step),
            "phase": phase,
            "student_arch": {
                "num_layers": model_cfg.num_layers,
                "n_emb": model_cfg.n_emb,
                "n_head": model_cfg.n_head,
                "n_kv_head": model_cfg.n_kv_head,
                "block_size": model_cfg.block_size,
                "vocab_size": model.token_emb.num_embeddings,
            },
            "train_config": asdict(train_cfg),
        },
        path,
    )
    return path


def train_phase(
    *,
    phase_name: str,
    model: GQATransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    output_root: Path,
    logger: logging.Logger,
    global_update_step: int,
    max_updates: int,
    total_steps_for_lr: int,
) -> int:
    model.train()
    use_amp = device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    phase_update = 0
    log_loss = 0.0
    log_tokens = 0
    t0 = time.perf_counter()

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            loss = loss / max(1, cfg.grad_accum_steps)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        micro_step += 1
        log_loss += float(loss.detach().item() * max(1, cfg.grad_accum_steps))
        log_tokens += int(y.numel())

        if micro_step % max(1, cfg.grad_accum_steps) != 0:
            continue

        phase_update += 1
        global_update_step += 1

        lr = build_lr(
            step=global_update_step,
            total_steps=total_steps_for_lr,
            warmup_steps=cfg.warmup_steps,
            max_lr=cfg.learning_rate,
            min_lr=cfg.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if phase_update % cfg.log_interval == 0:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            avg_loss = log_loss / cfg.log_interval
            tok_per_s = log_tokens / elapsed
            gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
            logger.info(
                "%s step=%d global_step=%d loss=%.4f lr=%.6e tok/s=%.1f gpu_mem=%.2fGB",
                phase_name,
                phase_update,
                global_update_step,
                avg_loss,
                lr,
                tok_per_s,
                gpu_mem,
            )
            log_loss = 0.0
            log_tokens = 0
            t0 = time.perf_counter()

        if global_update_step % cfg.save_interval == 0:
            ckpt = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                output_root=output_root,
                name=f"{phase_name}_step_{global_update_step:07d}",
                global_update_step=global_update_step,
                model_cfg=model_cfg,
                train_cfg=cfg,
                phase=phase_name,
            )
            logger.info("saved_checkpoint=%s", ckpt)

        if max_updates > 0 and phase_update >= max_updates:
            break

    logger.info("phase_complete=%s phase_updates=%d global_step=%d", phase_name, phase_update, global_update_step)
    return global_update_step


def main() -> None:
    args = parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(logs_dir / f"student_pretrain_{timestamp}.log")

    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    model_cfg = ModelConfig(
        num_layers=args.num_layers,
        n_emb=args.n_emb,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        block_size=args.block_size,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        tokenizer_name=args.tokenizer_name,
        openwebtext_samples=args.openwebtext_samples,
        openwebtext_max_updates=args.openwebtext_max_updates,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        domain_epochs=args.domain_epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    logger.info("output_root=%s", output_root)
    logger.info("hf_cache_dir=%s", args.hf_cache_dir)
    logger.info("device=%s amp_dtype=%s", device, amp_dtype)
    logger.info("model_cfg=%s", asdict(model_cfg))
    logger.info("train_cfg=%s", asdict(train_cfg))

    tokenizer = load_tokenizer(
        tokenizer_name=train_cfg.tokenizer_name,
        tokenizer_local_path=args.tokenizer_local_path,
        logger=logger,
    )

    converted_count = convert_prasanna_to_text(
        data_path=args.prasanna_data,
        output_path=args.prasanna_text,
        tokenizer=tokenizer,
        logger=logger,
    )
    if converted_count == 0:
        raise ValueError("No records found after converting prasanna dataset")

    model = build_student_model(model_cfg=model_cfg, vocab_size=len(tokenizer), device=device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("student_params=%s", f"{total_params:,}")

    if args.student_init_checkpoint is not None:
        _ = load_student_checkpoint(model=model, checkpoint_path=args.student_init_checkpoint, logger=logger)

    fused_ok = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=train_cfg.weight_decay,
        fused=fused_ok,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    global_update_step = 0

    openweb_ds = PackedTextDataset(
        text_iter_factory=openwebtext_iter_factory(
            dataset_name=train_cfg.openwebtext_dataset,
            split=train_cfg.openwebtext_split,
            max_samples=train_cfg.openwebtext_samples,
            cache_dir=args.hf_cache_dir,
        ),
        tokenizer=tokenizer,
        block_size=model_cfg.block_size,
    )
    openweb_loader = DataLoader(
        openweb_ds,
        batch_size=train_cfg.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    logger.info(
        "starting_phase=openwebtext samples=%d max_updates=%d",
        train_cfg.openwebtext_samples,
        train_cfg.openwebtext_max_updates,
    )
    global_update_step = train_phase(
        phase_name="openwebtext",
        model=model,
        dataloader=openweb_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        amp_dtype=amp_dtype,
        cfg=train_cfg,
        model_cfg=model_cfg,
        output_root=output_root,
        logger=logger,
        global_update_step=global_update_step,
        max_updates=train_cfg.openwebtext_max_updates,
        total_steps_for_lr=train_cfg.openwebtext_max_updates,
    )

    domain_ds = PackedTextDataset(
        text_iter_factory=jsonl_text_iter_factory(args.prasanna_text),
        tokenizer=tokenizer,
        block_size=model_cfg.block_size,
    )
    domain_loader = DataLoader(
        domain_ds,
        batch_size=train_cfg.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    logger.info("starting_phase=prasanna_domain epochs=%d", train_cfg.domain_epochs)
    for epoch in range(train_cfg.domain_epochs):
        logger.info("domain_epoch=%d/%d", epoch + 1, train_cfg.domain_epochs)
        global_update_step = train_phase(
            phase_name="prasanna_domain",
            model=model,
            dataloader=domain_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            cfg=train_cfg,
            model_cfg=model_cfg,
            output_root=output_root,
            logger=logger,
            global_update_step=global_update_step,
            max_updates=0,
            total_steps_for_lr=0,
        )

    final_ckpt = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        output_root=output_root,
        name="student_pretrained_final",
        global_update_step=global_update_step,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        phase="final",
    )
    tokenizer_dir = output_root / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    logger.info("final_checkpoint=%s", final_ckpt)
    logger.info("tokenizer_saved=%s", tokenizer_dir)

    summary = {
        "output_root": str(output_root),
        "device": str(device),
        "global_update_step": global_update_step,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "final_checkpoint": str(final_ckpt),
        "tokenizer_dir": str(tokenizer_dir),
        "prasanna_text": str(args.prasanna_text),
        "hf_cache_dir": str(args.hf_cache_dir),
    }
    with (output_root / "pretrain_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    cleanup_cuda()


if __name__ == "__main__":
    main()
