from pathlib import Path
import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig
from finetune.adapters import apply_lora, merge_lora_weights
from pretrain.inference import InferenceEngine
from utils import HFWrapper
from utils.common import build_causal_lm_collate_fn, resolve_dtype


LOGGER = logging.getLogger("my_model_finetune")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


DATA_FILE = Path("src/models/my-model/data/prasanna_data.json")
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = Path("custom-lora")
MERGED_DIR = Path("Prasanna-SmolLM-360M-merged")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

TRAINING_CONFIG = dict(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=5,
    optim="adamw",
    val_split_ratio=0.1,
)

INFERENCE_MESSAGES = [
    {
        "role": "system",
        "content": "You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills.",
    },
    {"role": "user", "content": "tell me more about prasanna?"},
]


def build_dataset(
    *,
    data_file: Path,
    tokenizer: object,
    max_seq_length: int,
    num_proc: int = 2,
) -> object:
    return HFWrapper.get_dataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split="train",
        num_proc=num_proc,
        messages_field="messages",
        add_generation_prompt=False,
    )


def split_train_val_dataset(dataset: object, val_split_ratio: float) -> tuple[object, object | None]:
    if val_split_ratio <= 0:
        return dataset, None
    if not hasattr(dataset, "train_test_split"):
        return dataset, None
    if len(dataset) < 2:
        return dataset, None
    split = dataset.train_test_split(test_size=val_split_ratio, seed=42, shuffle=True)
    return split["train"], split["test"]


def prepare_model(
    model_name: str,
    *,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[nn.Module, object]:
    LOGGER.info("Loading %s...", model_name)
    model, tokenizer = HFWrapper.get_model_and_tokenizer(
        model_name=model_name,
        task="causal-lm",
        prefer_local=True,
        tokenizer_kwargs={"use_fast": True},
        model_kwargs={"device_map": None, "dtype": resolve_dtype()},
    )

    model = apply_lora(
        model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.config.use_cache = False
    return model, tokenizer


def create_dataloader(dataset: object, tokenizer: object, batch_size: int, *, shuffle: bool) -> DataLoader:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_causal_lm_collate_fn(pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )


def build_training_config(output_dir: Path) -> TrainingConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float16"
    return TrainingConfig(
        device=device,
        train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        grad_accum_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        optimizer=TRAINING_CONFIG["optim"],
        ckpt_dir=str(output_dir),
        ckpt_interval_steps=0,
        save_optimizer_state=False,
        use_amp=device == "cuda",
        amp_dtype=amp_dtype,
        total_steps=None,
        log_interval_steps=TRAINING_CONFIG["logging_steps"],
        max_new_tokens=100,
        temperature=0.7,
    )


def _loss_from_output(model_out: object, labels: torch.Tensor) -> torch.Tensor:
    if hasattr(model_out, "loss") and model_out.loss is not None:
        return model_out.loss
    if isinstance(model_out, dict) and model_out.get("loss") is not None:
        return model_out["loss"]
    if isinstance(model_out, (tuple, list)) and len(model_out) > 1:
        maybe_loss = model_out[1]
        if isinstance(maybe_loss, torch.Tensor) and maybe_loss.ndim == 0:
            return maybe_loss

    if hasattr(model_out, "logits"):
        logits = model_out.logits
    elif isinstance(model_out, dict):
        logits = model_out.get("logits")
    elif isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        logits = model_out[0]
    else:
        logits = model_out

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("Could not derive logits/loss from model output")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate_validation_loss(
    *,
    model: nn.Module,
    val_dataloader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float | None:
    if val_dataloader is None:
        return None
    if len(val_dataloader) == 0:
        return None

    model.eval()
    total = 0.0
    count = 0
    for input_ids, labels in val_dataloader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(enabled=use_amp and device.type == "cuda", device_type=device.type, dtype=amp_dtype):
            model_out = model(input_ids=input_ids, labels=labels)
            loss = _loss_from_output(model_out, labels)
        total += float(loss.detach().item())
        count += 1

    model.train()
    if count == 0:
        return None
    return total / count


def train_model(
    *,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    config: TrainingConfig,
    epochs: int,
) -> None:
    device = torch.device(config.device)
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for LoRA training")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    use_amp = bool(config.use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    scaler_enabled = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    grad_accum_steps = max(1, int(config.grad_accum_steps))
    warmup_steps = max(0, int(config.warmup_steps))

    total_steps = 0
    if len(train_dataloader) > 0:
        total_steps = int(math.ceil((len(train_dataloader) * max(1, epochs)) / grad_accum_steps))

    def lr_factor(step: int) -> float:
        if total_steps <= 0:
            if warmup_steps > 0:
                return min(1.0, float(step) / float(max(1, warmup_steps)))
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps == warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(1, max(1, epochs) + 1):
        epoch_loss = 0.0
        epoch_count = 0

        for input_ids, labels in train_dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type, dtype=amp_dtype):
                model_out = model(input_ids=input_ids, labels=labels)
                loss = _loss_from_output(model_out, labels)

            scaled_loss = loss / grad_accum_steps
            if scaler_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro_step += 1
            loss_value = float(loss.detach().item())
            epoch_loss += loss_value
            epoch_count += 1
            running_loss += loss_value
            running_count += 1

            if micro_step < grad_accum_steps:
                continue

            if scaler_enabled:
                scaler.unscale_(optimizer)
            if config.grad_clip_norm and config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)

            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

            global_step += 1
            factor = lr_factor(global_step)
            for group in optimizer.param_groups:
                group["lr"] = config.lr * factor

            if global_step % max(1, int(config.log_interval_steps)) == 0:
                avg_loss = running_loss / max(1, running_count)
                LOGGER.info(
                    "step=%d train_loss=%.4f lr=%.2e",
                    global_step,
                    avg_loss,
                    optimizer.param_groups[0]["lr"],
                )

        if micro_step > 0:
            if scaler_enabled:
                scaler.unscale_(optimizer)
            if config.grad_clip_norm and config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

            global_step += 1
            factor = lr_factor(global_step)
            for group in optimizer.param_groups:
                group["lr"] = config.lr * factor

        val_loss = evaluate_validation_loss(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        train_epoch_loss = epoch_loss / max(1, epoch_count)
        if val_loss is None:
            LOGGER.info("epoch=%d train_loss=%.4f val_loss=n/a", epoch, train_epoch_loss)
        else:
            LOGGER.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch, train_epoch_loss, val_loss)


def run_inference(
    model: nn.Module,
    tokenizer: object,
    config: TrainingConfig,
    messages: list[dict],
) -> None:
    LOGGER.info("Running inference...")
    device = torch.device(config.device)
    model.to(device)
    engine = InferenceEngine(
        model=model,
        config=config,
        device=device,
        tokenizer=tokenizer,
    )
    output = engine.infer_chat(messages, streamText=False)
    LOGGER.info("Inference: %s", output)


def save_trained_model(model: nn.Module, tokenizer: object, output_dir: Path) -> None:
    HFWrapper.save_pretrained(
        model=model,
        tokenizer=tokenizer,
        save_dir=output_dir,
    )


def train_and_merge(
    *,
    model_name: str,
    data_file: Path,
    output_dir: Path,
    merged_dir: Path,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    max_seq_length: int,
    run_final_inference: bool = True,
) -> nn.Module:
    if not data_file.exists():
        raise SystemExit(f"Dataset not found at {data_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = prepare_model(
        model_name,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    dataset = build_dataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_proc=2,
    )
    
    train_dataset, val_dataset = split_train_val_dataset(dataset, TRAINING_CONFIG["val_split_ratio"])

    train_dataloader = create_dataloader(
        train_dataset,
        tokenizer,
        batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        shuffle=True,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = create_dataloader(
            val_dataset,
            tokenizer,
            batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            shuffle=False,
        )

    train_cfg = build_training_config(output_dir)
    LOGGER.info("Starting training...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=train_cfg,
        epochs=TRAINING_CONFIG["num_train_epochs"],
    )

    model = merge_lora_weights(model)
    save_trained_model(model, tokenizer, merged_dir)

    if run_final_inference:
        run_inference(model, tokenizer, train_cfg, INFERENCE_MESSAGES)
    return model


def main() -> None:
    train_and_merge(
        model_name=MODEL_NAME,
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        merged_dir=MERGED_DIR,
        target_modules=TARGET_MODULES,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        max_seq_length=MAX_SEQ_LENGTH,
        run_final_inference=True,
    )


if __name__ == "__main__":
    main()
