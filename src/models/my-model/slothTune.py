from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from customTransformers import GQATransformer
from utils.kd_loss import build_kd_loss


def get_unsloth_wrapper():
    from utils.unsloth_wrapper import UnslothWrapper

    return UnslothWrapper


@dataclass
class TeacherConfig:
    model_name: str = "unsloth/Llama-3.2-1B-Instruct" #"meta-llama/Llama-3.2-1B-Instruct"
    max_seq_length: int = 1024
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    train_batch_size: int = 2
    grad_accum_steps: int = 8
    warmup_steps: int = 20
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    output_dir: str = "outputs/llama32_teacher"


@dataclass
class StudentConfig:
    num_layers: int = 12
    n_emb: int = 192
    n_head: int = 6
    n_kv_head: int = 2
    block_size: int = 256
    dropout: float = 0.0


@dataclass
class DistillConfig:
    train_batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 8
    learning_rate: float = 1.5e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    temperature: float = 1.5
    alpha_kd: float = 0.6
    kd_loss_name: str = "kl_div"
    log_interval: int = 20
    save_interval: int = 200
    output_dir: str = "outputs/llama32_student_kd"


@dataclass
class EvalConfig:
    batch_size: int = 2
    num_generation_samples: int = 20
    max_new_tokens: int = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=ROOT_DIR / "datasets" / "prasanna_data.json")
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "outputs" / "llama32_unsloth_kd")
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--student-epochs", type=int, default=8)
    parser.add_argument("--kd-loss", type=str, default="kl_div")
    parser.add_argument("--student-init-checkpoint", type=Path, default=None)
    parser.add_argument("--save-merged-teacher", action="store_true")
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sloth_kd")
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


def safe_json(data: Any) -> Any:
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    if isinstance(data, dict):
        return {str(k): safe_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [safe_json(v) for v in data]
    return str(data)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step) / float(max(1, warmup_steps))
    if step >= total_steps:
        return min_lr
    if total_steps <= warmup_steps:
        return max_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def load_and_split_dataset(
    *,
    data_path: Path,
    eval_split: float,
    seed: int,
    max_samples: int,
    max_eval_samples: int,
) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if max_samples > 0:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    if eval_split <= 0.0 or len(dataset) < 2:
        eval_count = min(max_eval_samples if max_eval_samples > 0 else 64, len(dataset))
        eval_ds = dataset.select(range(eval_count))
        return dataset, eval_ds

    split = dataset.train_test_split(test_size=eval_split, seed=seed, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]

    if max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    return train_ds, eval_ds


def build_teacher(
    cfg: TeacherConfig,
    *,
    logger: logging.Logger,
    bf16: bool,
) -> tuple[torch.nn.Module, Any]:
    unsloth_wrapper = get_unsloth_wrapper()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        logger.info("HF_TOKEN is not set. Ensure you are authenticated if model access is gated.")
    logger.info("Loading teacher model: %s", cfg.model_name)
    model, tokenizer = unsloth_wrapper.load_model_and_tokenizer(
        model_name=cfg.model_name,
        model_type="language",
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=False,
        token=hf_token,
    )

    model = unsloth_wrapper.get_peft_model(
        model=model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    total, trainable = count_parameters(model)
    logger.info("Teacher params total=%s trainable=%s", f"{total:,}", f"{trainable:,}")
    logger.info("Teacher precision bf16=%s fp16=%s", bf16, not bf16)
    return model, tokenizer


def train_teacher(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Dataset,
    cfg: TeacherConfig,
    output_root: Path,
    seed: int,
    bf16: bool,
    logger: logging.Logger,
    save_merged_teacher: bool,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    unsloth_wrapper = get_unsloth_wrapper()
    formatted = unsloth_wrapper.format_chat_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        messages_field="messages",
        output_field="text",
        add_generation_prompt=False,
        num_proc=2,
    )

    teacher_output_dir = Path(cfg.output_dir)
    teacher_output_dir.mkdir(parents=True, exist_ok=True)

    train_args = {
        "per_device_train_batch_size": cfg.train_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "warmup_steps": cfg.warmup_steps,
        "num_train_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "logging_steps": cfg.logging_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": 2,
        "output_dir": str(teacher_output_dir),
        "bf16": bf16,
        "fp16": not bf16,
        "seed": seed,
        "report_to": [],
    }

    trainer = unsloth_wrapper.create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted,
        args_kwargs=train_args,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        dataset_num_proc=2,
        packing=True,
    )

    logger.info("Starting teacher LoRA fine-tuning")
    unsloth_wrapper.train(trainer=trainer)

    adapter_dir = output_root / "teacher-adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Saved teacher LoRA adapter to %s", adapter_dir)

    if save_merged_teacher:
        merged_dir = output_root / "teacher-merged"
        logger.info("Saving merged teacher model to %s", merged_dir)
        unsloth_wrapper.save_pretrained_merged(
            model=model,
            save_directory=merged_dir,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            push_to_hub=False,
            token=os.getenv("HF_TOKEN"),
        )

    history = [safe_json(item) for item in trainer.state.log_history]
    with (output_root / "teacher_train_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    model = unsloth_wrapper.for_inference(model)
    model.eval()
    return model, history


def tokenize_chat_dataset(dataset: Dataset, tokenizer: Any, block_size: int) -> Dataset:
    def _tokenize(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        input_ids_list: list[list[int]] = []
        attention_mask_list: list[list[int]] = []
        loss_mask_list: list[list[int]] = []

        for conversation in examples["messages"]:
            prompt_messages: list[dict[str, str]] = []
            for message in conversation:
                if message.get("role") == "assistant":
                    break
                prompt_messages.append(message)

            full_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )

            full_ids = full_ids[:block_size]
            attn_mask = [1] * len(full_ids)

            assistant_start = min(len(prompt_ids), len(full_ids))
            loss_mask = [0] * len(full_ids)
            for idx in range(assistant_start, len(full_ids)):
                loss_mask[idx] = 1

            if sum(loss_mask) < 2:
                loss_mask = attn_mask.copy()

            input_ids_list.append(full_ids)
            attention_mask_list.append(attn_mask)
            loss_mask_list.append(loss_mask)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "loss_mask": loss_mask_list,
        }

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=2,
    )


def build_collate_fn(pad_token_id: int, block_size: int):
    def _collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = min(max(len(item["input_ids"]) for item in batch), block_size)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        loss_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            ids = item["input_ids"][:max_len]
            mask = item["attention_mask"][:max_len]
            target_mask = item.get("loss_mask", mask)[:max_len]
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = torch.tensor(mask, dtype=torch.long)
            loss_mask[i, :length] = torch.tensor(target_mask, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}

    return _collate


def build_student_model(student_cfg: StudentConfig, tokenizer: Any, device: torch.device) -> GQATransformer:
    model = GQATransformer(
        num_layers=student_cfg.num_layers,
        n_emb=student_cfg.n_emb,
        n_head=student_cfg.n_head,
        n_kv_head=student_cfg.n_kv_head,
        vocab_size=len(tokenizer),
        block_size=student_cfg.block_size,
        dropout=student_cfg.dropout,
    )
    return model.to(device)


def load_student_init_checkpoint(
    *,
    model: GQATransformer,
    checkpoint_path: Path | None,
    expected_vocab_size: int,
    logger: logging.Logger,
) -> None:
    if checkpoint_path is None:
        return
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Student init checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        arch = payload.get("student_arch", {})
    elif isinstance(payload, dict):
        state_dict = payload
        arch = {}
    else:
        raise ValueError("Unsupported student checkpoint format")

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict must be a dict")

    ckpt_vocab = arch.get("vocab_size")
    if ckpt_vocab is not None and int(ckpt_vocab) != int(expected_vocab_size):
        raise ValueError(
            f"Checkpoint vocab_size={ckpt_vocab} does not match tokenizer vocab_size={expected_vocab_size}"
        )

    sanitized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            sanitized[key[len("module.") :]] = value
        else:
            sanitized[key] = value

    missing, unexpected = model.load_state_dict(sanitized, strict=False)
    logger.info(
        "Loaded student init checkpoint from %s (missing=%d, unexpected=%d)",
        checkpoint_path,
        len(missing),
        len(unexpected),
    )


def reduce_teacher_to_student_vocab(
    teacher_logits: torch.Tensor,
    student_vocab_size: int,
) -> torch.Tensor:
    if teacher_logits.size(-1) == student_vocab_size:
        return teacher_logits
    if teacher_logits.size(-1) > student_vocab_size:
        return teacher_logits[..., :student_vocab_size]
    pad = student_vocab_size - teacher_logits.size(-1)
    return F.pad(teacher_logits, (0, pad), value=-30.0)


def train_student_kd(
    *,
    student: GQATransformer,
    teacher: torch.nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    cfg: DistillConfig,
    output_root: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False


    kd_loss_fn = build_kd_loss(
        name=cfg.kd_loss_name,
        temperature=cfg.temperature,
        reduction="mean",
    ).to(device)

    logger.info("KD loss=%s temperature=%.3f", kd_loss_fn.__class__.__name__, cfg.temperature)

    student.train()
    fused_ok = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused_ok,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps)))
    total_updates = max(1, updates_per_epoch * cfg.num_epochs)

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    global_micro_step = 0
    update_step = 0
    log_kd = 0.0
    log_ce = 0.0
    log_loss = 0.0
    log_tokens = 0
    t_start = time.perf_counter()

    student_output_dir = Path(cfg.output_dir)
    student_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                    dtype=amp_dtype,
                ):
                    teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_out.logits[:, :-1, :]

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=amp_dtype,
            ):
                student_logits, _ = student(input_ids)
                student_logits = student_logits[:, :-1, :]
                student_logits = student_logits.contiguous()
                teacher_logits = reduce_teacher_to_student_vocab(
                    teacher_logits=teacher_logits,
                    student_vocab_size=student_logits.size(-1),
                )

                target_ids = input_ids[:, 1:].contiguous()
                target_mask = loss_mask[:, 1:].contiguous().bool()
                mask_count = target_mask.sum().clamp(min=1)

                kd_loss = kd_loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    mask=target_mask,
                )

                ce_per_token = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)),
                    target_ids.reshape(-1),
                    reduction="none",
                ).view_as(target_ids)
                ce_loss = ce_per_token.masked_fill(~target_mask, 0.0).sum() / mask_count

                loss = cfg.alpha_kd * kd_loss + (1.0 - cfg.alpha_kd) * ce_loss
                loss = loss / max(1, cfg.grad_accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_micro_step += 1
            log_kd += float(kd_loss.detach().item())
            log_ce += float(ce_loss.detach().item())
            log_loss += float((loss.detach().item() * max(1, cfg.grad_accum_steps)))
            log_tokens += int(target_mask.sum().item())

            del teacher_out
            del teacher_logits
            del student_logits
            del ce_per_token
            del target_ids
            del target_mask

            if global_micro_step % max(1, cfg.grad_accum_steps) != 0:
                continue

            update_step += 1
            lr = build_lr(
                step=update_step,
                total_steps=total_updates,
                warmup_steps=cfg.warmup_steps,
                max_lr=cfg.learning_rate,
                min_lr=cfg.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if update_step % cfg.log_interval == 0:
                elapsed = max(time.perf_counter() - t_start, 1e-6)
                tok_per_s = log_tokens / elapsed
                avg_kd = log_kd / cfg.log_interval
                avg_ce = log_ce / cfg.log_interval
                avg_loss = log_loss / cfg.log_interval
                if device.type == "cuda":
                    mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                else:
                    mem_gb = 0.0

                record = {
                    "step": update_step,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "kd_loss": avg_kd,
                    "ce_loss": avg_ce,
                    "lr": lr,
                    "tokens_per_sec": tok_per_s,
                    "gpu_mem_gb": mem_gb,
                }
                history.append(record)
                logger.info(
                    "student_step=%d/%d loss=%.4f kd=%.4f ce=%.4f lr=%.6e tok/s=%.1f gpu_mem=%.2fGB",
                    update_step,
                    total_updates,
                    avg_loss,
                    avg_kd,
                    avg_ce,
                    lr,
                    tok_per_s,
                    mem_gb,
                )
                log_kd = 0.0
                log_ce = 0.0
                log_loss = 0.0
                log_tokens = 0
                t_start = time.perf_counter()

            if update_step % cfg.save_interval == 0:
                ckpt_path = student_output_dir / f"student_step_{update_step:06d}.pt"
                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "student_config": asdict(cfg),
                        "step": update_step,
                    },
                    ckpt_path,
                )
                logger.info("Saved student checkpoint: %s", ckpt_path)

            if device.type == "cuda" and update_step % max(1, cfg.log_interval) == 0:
                cleanup_cuda()

    final_path = student_output_dir / "student_final.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "student_arch": {
                "num_layers": student.num_layers if hasattr(student, "num_layers") else None,
                "n_emb": student.token_emb.embedding_dim,
                "n_head": student.blocks[0].attn.n_head,
                "n_kv_head": student.blocks[0].attn.n_kv_head,
                "block_size": student.block_size,
                "vocab_size": student.token_emb.num_embeddings,
            },
            "distill_config": asdict(cfg),
            "history": history,
        },
        final_path,
    )
    tokenizer.save_pretrained(student_output_dir / "tokenizer")
    logger.info("Saved final student checkpoint: %s", final_path)

    with (output_root / "student_train_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    return history


def compute_loss_teacher(
    *,
    teacher: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    teacher.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            if input_ids.shape[1] < 2:
                continue

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=amp_dtype):
                logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
                logits = logits[:, :-1, :]
                labels = input_ids[:, 1:]
                target_mask = loss_mask[:, 1:].bool()
                mask_count = target_mask.sum().clamp(min=1)

                per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                ).view_as(labels)
                loss = per_token.masked_fill(~target_mask, 0.0).sum() / mask_count

            total_loss += float(loss.item())
            steps += 1

    return total_loss / max(1, steps)


def compute_loss_student(
    *,
    student: GQATransformer,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    student.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            if input_ids.shape[1] < 2:
                continue

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=amp_dtype):
                logits, _ = student(input_ids)
                logits = logits[:, :-1, :]
                labels = input_ids[:, 1:]
                target_mask = loss_mask[:, 1:].bool()
                mask_count = target_mask.sum().clamp(min=1)

                per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                ).view_as(labels)
                loss = per_token.masked_fill(~target_mask, 0.0).sum() / mask_count

            total_loss += float(loss.item())
            steps += 1

    return total_loss / max(1, steps)


def extract_prompt_reference(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    prompt: list[dict[str, str]] = []
    reference = ""
    for item in messages:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "assistant":
            reference = content
            break
        prompt.append({"role": role, "content": content})
    return prompt, reference


def generate_teacher(
    *,
    teacher: torch.nn.Module,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, float]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    attention_mask = torch.ones_like(prompt_ids, device=device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    start = time.perf_counter()
    with torch.no_grad():
        output = teacher.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    latency = time.perf_counter() - start

    response_ids = output[0, prompt_ids.shape[1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response, latency


def generate_student(
    *,
    student: GQATransformer,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, float]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    if prompt_ids.shape[1] > student.block_size:
        prompt_ids = prompt_ids[:, -student.block_size :]

    eos_token_id = tokenizer.eos_token_id

    start = time.perf_counter()
    with torch.no_grad():
        output = student.generate(
            idx=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
            repetition_penalty=1.1,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
    latency = time.perf_counter() - start

    response_ids = output[0, prompt_ids.shape[1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response, latency


def evaluate_models(
    *,
    teacher: torch.nn.Module,
    student: GQATransformer,
    tokenizer: Any,
    eval_raw: Dataset,
    eval_loader: DataLoader,
    cfg: EvalConfig,
    output_root: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    logger: logging.Logger,
) -> dict[str, Any]:
    teacher_loss = compute_loss_teacher(
        teacher=teacher,
        dataloader=eval_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    student_loss = compute_loss_student(
        student=student,
        dataloader=eval_loader,
        device=device,
        amp_dtype=amp_dtype,
    )

    teacher_ppl = math.exp(min(20.0, teacher_loss))
    student_ppl = math.exp(min(20.0, student_loss))

    sample_count = min(cfg.num_generation_samples, len(eval_raw))
    rows = eval_raw.select(range(sample_count))

    comparisons: list[dict[str, Any]] = []
    teacher_ref_f1 = 0.0
    student_ref_f1 = 0.0
    teacher_student_f1 = 0.0
    teacher_latency = 0.0
    student_latency = 0.0

    for idx, row in enumerate(rows):
        prompt, reference = extract_prompt_reference(row["messages"])
        if not prompt:
            continue

        teacher_text, t_latency = generate_teacher(
            teacher=teacher,
            tokenizer=tokenizer,
            prompt_messages=prompt,
            max_new_tokens=cfg.max_new_tokens,
            device=device,
        )
        student_text, s_latency = generate_student(
            student=student,
            tokenizer=tokenizer,
            prompt_messages=prompt,
            max_new_tokens=cfg.max_new_tokens,
            device=device,
        )

        t_ref = token_f1(teacher_text, reference)
        s_ref = token_f1(student_text, reference)
        t_s = token_f1(student_text, teacher_text)

        teacher_ref_f1 += t_ref
        student_ref_f1 += s_ref
        teacher_student_f1 += t_s
        teacher_latency += t_latency
        student_latency += s_latency

        comparisons.append(
            {
                "index": idx,
                "prompt": prompt,
                "reference": reference,
                "teacher_output": teacher_text,
                "student_output": student_text,
                "teacher_ref_f1": t_ref,
                "student_ref_f1": s_ref,
                "teacher_student_f1": t_s,
                "teacher_latency_sec": t_latency,
                "student_latency_sec": s_latency,
            }
        )

        logger.info(
            "eval_sample=%d teacher_ref_f1=%.4f student_ref_f1=%.4f teacher_student_f1=%.4f",
            idx,
            t_ref,
            s_ref,
            t_s,
        )

    denom = max(1, len(comparisons))
    report = {
        "teacher_eval_loss": teacher_loss,
        "student_eval_loss": student_loss,
        "teacher_perplexity": teacher_ppl,
        "student_perplexity": student_ppl,
        "teacher_ref_f1": teacher_ref_f1 / denom,
        "student_ref_f1": student_ref_f1 / denom,
        "teacher_student_f1": teacher_student_f1 / denom,
        "teacher_avg_latency_sec": teacher_latency / denom,
        "student_avg_latency_sec": student_latency / denom,
        "num_generation_samples": len(comparisons),
        "comparisons": comparisons,
    }

    with (output_root / "evaluation_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return report


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Unsloth + KD pipeline.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    metrics_dir = output_root / "metrics"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(logs_dir / f"sloth_kd_{timestamp}.log")

    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    teacher_cfg = TeacherConfig(
        num_epochs=args.teacher_epochs,
        output_dir=str(output_root / "teacher"),
    )
    student_cfg = StudentConfig()
    distill_cfg = DistillConfig(
        num_epochs=args.student_epochs,
        kd_loss_name=args.kd_loss,
        output_dir=str(output_root / "student"),
    )
    eval_cfg = EvalConfig()

    logger.info("Starting teacher->student distillation pipeline")
    logger.info("data_path=%s", args.data_path)
    logger.info("output_root=%s", output_root)
    logger.info("device=%s bf16=%s", device, bf16)

    train_raw, eval_raw = load_and_split_dataset(
        data_path=args.data_path,
        eval_split=args.eval_split,
        seed=args.seed,
        max_samples=args.max_samples,
        max_eval_samples=args.max_eval_samples,
    )
    logger.info("dataset train=%d eval=%d", len(train_raw), len(eval_raw))

    teacher_model, tokenizer = build_teacher(
        cfg=teacher_cfg,
        logger=logger,
        bf16=bf16,
    )

    teacher_model, teacher_train_history = train_teacher(
        model=teacher_model,
        tokenizer=tokenizer,
        train_dataset=train_raw,
        cfg=teacher_cfg,
        output_root=output_root,
        seed=args.seed,
        bf16=bf16,
        logger=logger,
        save_merged_teacher=args.save_merged_teacher,
    )

    tokenized_train = tokenize_chat_dataset(train_raw, tokenizer, student_cfg.block_size)
    tokenized_eval = tokenize_chat_dataset(eval_raw, tokenizer, student_cfg.block_size)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    collate_fn = build_collate_fn(pad_token_id=pad_token_id, block_size=student_cfg.block_size)

    train_loader = DataLoader(
        tokenized_train,
        batch_size=distill_cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        tokenized_eval,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    student_model = build_student_model(student_cfg=student_cfg, tokenizer=tokenizer, device=device)
    load_student_init_checkpoint(
        model=student_model,
        checkpoint_path=args.student_init_checkpoint,
        expected_vocab_size=len(tokenizer),
        logger=logger,
    )
    total_params, trainable_params = count_parameters(student_model)
    logger.info(
        "student_params total=%s trainable=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )
    if not (28_000_000 <= total_params <= 32_000_000):
        raise ValueError(
            f"Student parameter count must stay near 30M. Found {total_params:,} parameters."
        )

    student_history = train_student_kd(
        student=student_model,
        teacher=teacher_model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        cfg=distill_cfg,
        output_root=output_root,
        device=device,
        amp_dtype=amp_dtype,
        logger=logger,
    )

    report = evaluate_models(
        teacher=teacher_model,
        student=student_model,
        tokenizer=tokenizer,
        eval_raw=eval_raw,
        eval_loader=eval_loader,
        cfg=eval_cfg,
        output_root=output_root,
        device=device,
        amp_dtype=amp_dtype,
        logger=logger,
    )

    summary = {
        "seed": args.seed,
        "data_path": str(args.data_path),
        "output_root": str(output_root),
        "student_init_checkpoint": str(args.student_init_checkpoint) if args.student_init_checkpoint else None,
        "teacher_config": asdict(teacher_cfg),
        "student_config": asdict(student_cfg),
        "distill_config": asdict(distill_cfg),
        "eval_config": asdict(eval_cfg),
        "teacher_train_steps": len(teacher_train_history),
        "student_train_steps": len(student_history),
        "evaluation": report,
    }

    with (metrics_dir / "run_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(safe_json(summary), fp, indent=2)

    logger.info(
        "Completed. teacher_loss=%.4f student_loss=%.4f teacher_ref_f1=%.4f student_ref_f1=%.4f",
        report["teacher_eval_loss"],
        report["student_eval_loss"],
        report["teacher_ref_f1"],
        report["student_ref_f1"],
    )

    cleanup_cuda()


if __name__ == "__main__":
    main()
