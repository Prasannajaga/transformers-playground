# =============================================================================
# finetune.py
# =============================================================================
import os
import sys
import math
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Imports from your existing codebase
from config import TrainingConfig
from customTransformers import DecodeTransformer 

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python finetune.py <path_to_pretrained_checkpoint.pt>")
    sys.exit(1)

checkpoint_path = sys.argv[1]

# =============================================================================
# 1. Configuration (Fine-Tuning Overrides)
# =============================================================================
# We instantiate your config but override the "Pre-training" defaults 
# with "Fine-tuning" defaults (Lower LR, fewer steps, etc.)
config = TrainingConfig(
    # System
    device="cuda" if torch.cuda.is_available() else "cpu",
    
    # Fine-tuning Hyperparameters
    lr=2e-5,                  # Crucial: Much lower than pre-training (5e-4)
    warmup_steps=100,         # Short warmup
    total_steps=2000,         # Short training duration (prevent forgetting)
    train_batch_size=16,      # Adjust based on VRAM
    grad_accum_steps=2,
    
    # Logging
    enable_logging=True,
    log_interval_steps=50,
    ckpt_interval_steps=1000, # Save halfway and at end
    ckpt_dir="./checkpoints_qa",
    
    # Model Specs (Must match your pretrained model!)
    n_layer=6,
    n_embd=384,
    n_head=6,
    block_size=256
)

os.makedirs(config.ckpt_dir, exist_ok=True)
torch.manual_seed(config.seed)

# =============================================================================
# 2. Tokenizer & Dataset (Context + Question -> Answer)
# =============================================================================
logger.info("Loading Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

class SQuADDataset(Dataset):
    def __init__(self, split="train", limit=None):
        # Load SQuAD (Context-based QA)
        ds = load_dataset("squad", split=split)
        if limit:
            ds = ds.select(range(limit))
        self.data = ds
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format: "Context: ... \n Question: ... \n Answer:"
        # We need Context because a 40M TinyStories model doesn't know real facts.
        context_text = f"Context: {item['context']}"
        question_text = f"\nQuestion: {item['question']}"
        answer_text = f"\nAnswer: {item['answers']['text'][0]}" # Take first answer
        
        # Full text for training
        full_text = f"{context_text}{question_text}{answer_text}{tokenizer.eos_token}"
        
        # Prompt only (for masking)
        prompt_text = f"{context_text}{question_text}{answer_text}" 
        # Note: We mask everything up to the start of the actual answer words usually, 
        # but simpler is to mask up to "\nAnswer:".
        mask_until = f"{context_text}{question_text}\nAnswer: "
        
        # Tokenize
        encoded = tokenizer(
            full_text, 
            truncation=True, 
            max_length=config.block_size, 
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        # Calculate Masking Index
        # We find the length of the prompt part effectively
        prompt_ids = tokenizer.encode(mask_until, add_special_tokens=False)
        mask_len = min(len(prompt_ids), config.block_size)
        
        # Apply Mask: -100 is ignored by CrossEntropyLoss
        labels[:mask_len] = -100
        
        # Also mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

logger.info("Preparing Datasets...")
train_dataset = SQuADDataset(split="train", limit=10000) # Use subset for speed
val_dataset = SQuADDataset(split="validation", limit=500)

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.train_batch_size, 
    shuffle=True, 
    num_workers=config.num_workers,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=config.eval_batch_size, 
    shuffle=False
)

# =============================================================================
# 3. Model & Checkpoint Loading
# =============================================================================
logger.info(f"Initializing Model on {config.device}...")
model = DecodeTransformer(
    num_layers=config.n_layer,
    n_emb=config.n_embd,
    n_head=config.n_head,
    block_size=config.block_size,
    vocab_size=tokenizer.vocab_size, 
).to(config.device)

logger.info(f"Loading Checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=config.device)

# Handle keys (remove _orig_mod if compiled)
state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

if config.enable_gradient_checkpointing:
    model.gradient_checkpointing_enable()

# =============================================================================
# 4. Optimization Setup
# =============================================================================
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.lr, 
    betas=config.betas, 
    eps=config.eps, 
    weight_decay=config.weight_decay
)

scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

def get_lr(step):
    # Simple linear warmup then cosine decay
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    if step > config.total_steps:
        return config.lr / 10
    decay_ratio = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.lr / 10) + coeff * (config.lr - config.lr/10)

# =============================================================================
# 5. Training Loop
# =============================================================================
model.train()
step = 0
train_iter = iter(train_loader)

logger.info("Starting Fine-Tuning...")

while step < config.total_steps:
    t0 = time.time()
    
    # --- Optimization Step ---
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    
    for _ in range(config.grad_accum_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_ids = batch['input_ids'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        # Autocast Forward
        dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.cuda.amp.autocast(enabled=config.use_amp, dtype=dtype):
            logits, loss = model(input_ids, labels)
            loss = loss / config.grad_accum_steps
        
        scaler.scale(loss).backward()
        loss_accum += loss.item()
    
    # Clip Gradient
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    
    # Update Weights
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    scaler.step(optimizer)
    scaler.update()
    
    step += 1
    t1 = time.time()
    dt = t1 - t0
    
    # --- Logging ---
    if step % config.log_interval_steps == 0:
        logger.info(f"Step {step}/{config.total_steps} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms")

    # --- Validation ---
    if step % 500 == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= config.validation_batch_count: break
                xb = batch['input_ids'].to(config.device)
                yb = batch['labels'].to(config.device)
                _, val_loss = model(xb, yb)
                val_losses.append(val_loss.item())
        avg_val = sum(val_losses) / len(val_losses) if val_losses else 0
        logger.info(f"Validation Loss: {avg_val:.4f}")
        model.train()

    # --- Checkpointing ---
    if step % config.ckpt_interval_steps == 0 or step == config.total_steps:
        ckpt_path = os.path.join(config.ckpt_dir, f"qa_finetuned_step{step}.pt")
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'config': config,
            'step': step
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

# =============================================================================
# 6. Inference Test (Quick Sanity Check)
# =============================================================================
logger.info("Running Inference Test...")
model.eval()
test_ctx = "Tiny the bear lived in a blue cave. He loved honey."
test_q = "What did Tiny love?"
prompt = f"Context: {test_ctx}\nQuestion: {test_q}\nAnswer:"

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=20, 
        temperature=0.7, 
        top_k=50,
        eos_token_id=tokenizer.eos_token_id
    ) # Assuming your DecodeTransformer has a .generate() method

decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n" + "="*40)
print(f"Prompt:\n{prompt}")
print("-" * 20)
print(f"Generated:\n{decoded}")
print("="*40)