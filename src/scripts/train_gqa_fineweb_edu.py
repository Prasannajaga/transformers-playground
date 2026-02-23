import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from itertools import islice

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

from customTransformers import GQATransformer


@dataclass
class TrainConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = None
    split: str = "train"
    max_docs: int = 10_000_000

    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    block_size: int = 512

    n_layer: int = 10
    n_embd: int = 512
    n_head: int = 8
    n_kv_head: int = 2
    dropout: float = 0.1

    train_batch_size: int = 8
    grad_accum_steps: int = 8
    max_steps: int = 50_000

    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0

    ckpt_dir: str = "./checkpoints/gqa_fineweb_edu"
    ckpt_interval: int = 25_000

    log_interval: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PackedIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size, max_docs=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_docs = max_docs

    def __iter__(self):
        buffer = []
        it = self.dataset
        if self.max_docs is not None:
            it = islice(it, self.max_docs)
        for ex in it:
            text = ex.get("text")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(self.tokenizer.eos_token_id)

            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                buffer = buffer[self.block_size + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/train_gqa_fineweb_edu.log"),
        ],
    )
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    cfg = TrainConfig()
    setup_logging()
    log = logging.getLogger("train")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    ds = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.split,
        streaming=True,
    )
    sample_rows = list(islice(ds, 3))
    for i, row in enumerate(sample_rows):
        log.info("dataset_sample_%d=%s", i, (row.get("text") or "")[:400].replace("\n", " "))
    log.info("dataset_streaming=1 max_docs=%s", cfg.max_docs)
 
    train_ds = PackedIterableDataset(ds, tokenizer, cfg.block_size, cfg.max_docs)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        num_workers=0,
        pin_memory=True,
    )
    train_iter = cycle(train_loader)

    model = GQATransformer(
        num_layers=cfg.n_layer,
        n_emb=cfg.n_embd,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(cfg.device)

    fused_ok = cfg.device == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        fused=fused_ok,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def save_hf_checkpoint(step, final=False):
        name = "final" if final else f"step_{step:07d}"
        out_dir = os.path.join(cfg.ckpt_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
        torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
        torch.save(scaler.state_dict(), os.path.join(out_dir, "scaler.pt"))

        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        tokenizer.save_pretrained(out_dir)
        log.info("checkpoint_saved=%s", out_dir)

    model.train()
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = next(train_iter)
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps

            total_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        lr = get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tokens = cfg.train_batch_size * cfg.grad_accum_steps * cfg.block_size
            tok_per_s = tokens / max(dt, 1e-6)
            log.info(
                "step=%d loss=%.4f lr=%.6f tok/s=%.1f",
                step,
                total_loss,
                lr,
                tok_per_s,
            )
            t0 = time.time()

        if step % cfg.ckpt_interval == 0 and step != cfg.max_steps:
            save_hf_checkpoint(step, final=False)

    save_hf_checkpoint(cfg.max_steps, final=True)


if __name__ == "__main__":
    main()
