import time
import logging 
import random
import json
import gzip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader  
from transformers import GPT2Tokenizer
from customTransformers import DecodeTransformer
from config import FineTuneConfig
from customtokenizer import StoryQADataset , collate_storyqa , SFTDataset
from utils.common import estimate_loss, get_lr, save_checkpoint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="app.log",
    filemode="a",
)
log = logging.getLogger("SFT model training")  

log.info("\n\n\n Fine-Tuning SFT Initialization")
log.info("=" * 50)  
config = FineTuneConfig(max_iters=500)
log.info(f"Config loaded: device={config.device}, mixed_precision={config.mixed_precision}")
log.info(f"Hyperparams: iters={config.max_iters}, lr={config.max_lr}, batch={config.batch_size}")


log.info("Loading tokenizer and model architecture...")
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = DecodeTransformer(
    num_layers=config.n_layer,
    n_emb=config.n_embd,
    n_head=config.n_head,
    block_size=config.block_size,
    vocab_size=len(tokenizer),
)
log.info(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}") 

DATASET_PATH = "/home/prasanna/coding/transformers-playground/src/CustomDatasets/sft.json" 
with gzip.open(DATASET_PATH, "rt", encoding="utf-8") as f:
    storyqa_data = json.load(f)

# SPECIAL TOKENS
SPECIAL_TOKENS = ["<|story|>", "<|user|>", "<|assistant|>"]
tokenizer.add_tokens(SPECIAL_TOKENS) 
# Dataset should already be loaded in storyqa_data
log.info(f"Dataset ready with {len(storyqa_data)} samples.")

random.seed(42)
random.shuffle(storyqa_data)

split_idx = int(0.80 * len(storyqa_data))
train_sft = storyqa_data[:split_idx]
val_sft   = storyqa_data[split_idx:]
eval_batch_size = max(1, config.batch_size // 4)


train_data = SFTDataset(
    train_sft,
    tokenizer,
    config.block_size
)

val_data = SFTDataset(
    val_sft,
    tokenizer,
    config.block_size
)

train_loader = DataLoader(
    train_data,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
    collate_fn=collate_storyqa
)

val_loader = DataLoader(
    val_data,
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
    collate_fn=collate_storyqa
) 

xb, yb = next(iter(val_loader))
print((yb != -100).sum())  # should be > 0
print((yb == -100).sum())  # should be much larger

log.info(f"Loading pre-trained checkpoint from: {config.pretrained_checkpoint}")
ckpt = torch.load(config.pretrained_checkpoint, map_location=config.device) 
state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt.get("state_dict")
if state_dict is None:
    raise KeyError(f"Unrecognized checkpoint format (keys: {list(ckpt.keys())})")
model.resize_token_embeddings(len(tokenizer)) 
model.load_state_dict(state_dict, strict=False)
model.to(config.device)
log.info("Weights loaded and model moved to device.")

 
# Optimizer & Scaler 
decay = []
no_decay = []
for name, param in model.named_parameters():
    if param.dim() < 2 or "bias" in name or "ln" in name.lower():
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    betas=(0.9, 0.95),
    lr=config.max_lr,
)

scaler = torch.amp.grad_scaler.GradScaler(
    enabled=config.mixed_precision
) 


# Training Loop 
log.info("Starting fine-tuning...")
train_iter = iter(train_loader)
model.train()
train_start_time = time.perf_counter()
best_val = float("inf")
patience = 5
bad = 0  


for step in range(config.max_iters):
    # Learning rate schedule
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation
    if step % config.eval_interval == 0:
        val_loss = estimate_loss(model, val_loader, config)

        if val_loss < best_val:
            best_val = val_loss
            bad = 0 
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping SFT")
                break

        torch.cuda.empty_cache()
        log.info(
            f"step {step} | val_loss {val_loss:.4f} | lr {lr:.6e}"
        )
        model.train()

    # Get batch
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)
        log.info(f"Epoch completed at step {step}")

    xb = xb.to(config.device, non_blocking=True)
    yb = yb.to(config.device, non_blocking=True) 

    # Forward & backward pass
    with torch.amp.autocast_mode.autocast(
        enabled=config.mixed_precision,
        device_type="cuda" if config.device == "cuda" else "cpu",
        dtype=torch.float16,
    ):
        logits, loss = model(xb, yb)
        loss = loss / config.grad_accum_steps
        
    if step % 1000 == 0:
        log.info(f"step {step}: train_loss {loss.item() * config.grad_accum_steps:.4f}")

    scaler.scale(loss).backward()

    # Gradient update
    if (step + 1) % config.grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Checkpoint save
    if (step + 1) % config.checkpoint_interval == 0 and (step + 1) % config.max_iters != 0:
        ckpt_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step + 1,
            checkpoint_dir=config.checkpoint_dir,
            prefix=config.checkpoint_prefix,
            config=config,
            tokenizer=tokenizer,
        )
        log.info(f"Checkpoint saved: {ckpt_path}")
 
# Final Checkpoint 
final_ckpt_path = save_checkpoint(
    model=model,
    optimizer=optimizer,
    scaler=scaler,
    step=config.max_iters,
    checkpoint_dir=config.checkpoint_dir,
    prefix=config.checkpoint_prefix,
    config=config,
    tokenizer=tokenizer,
    is_final=True,
)
log.info(f"Final checkpoint saved: {final_ckpt_path}") 

# Training Summary 
total_train_time = time.perf_counter() - train_start_time
log.info(
    f"Fine-tuning complete! "
    f"Time: {total_train_time / 60:.2f} min ({total_train_time / 3600:.2f} hr)"
) 
