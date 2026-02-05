import os
import sys

# Add src to sys.path to allow imports from project root
# current_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
# if src_dir not in sys.path:
#     sys.path.insert(0, src_dir)

import logging
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import GPT2Tokenizer   
from config import TrainingConfig
from customTransformers.decodeTransfomer import DecodeTransformer
from pretrain.trainer import Trainer


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
 
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Suppress verbose HTTP logs from HuggingFace libraries
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
 
# Config (ALL hyperparams here)
# =============================================================================
cfg = TrainingConfig()

device = torch.device(cfg.device if torch.cuda.is_available() else "cpu") 

# Tokenizer
# =============================================================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

 
# Dataset
# =============================================================================
dataset = load_dataset("roneneldan/TinyStories")
dataset = dataset["train"].train_test_split(test_size=0.01, seed=cfg.seed)

class PackedIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        for text in self.dataset["text"]:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(self.tokenizer.eos_token_id)

            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                buffer = buffer[self.block_size + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )

train_ds = PackedIterableDataset(dataset["train"], tokenizer, cfg.block_size)
val_ds   = PackedIterableDataset(dataset["test"], tokenizer, cfg.block_size)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.train_batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=cfg.eval_batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
)


RESUME_FROM_CHECKPOINT = False
RESUME_CKPT_PATH = os.path.join(
    cfg.ckpt_dir,
    "ckpt_step_0015000.pt"
) 

# Model
# =============================================================================
model = DecodeTransformer(
    num_layers=cfg.n_layer,
    n_emb=cfg.n_embd,
    n_head=cfg.n_head,
    block_size=cfg.block_size,
    vocab_size=vocab_size,
).to(device)

 
# Trainer
# =============================================================================
trainer = Trainer(
    model=model,
    config=cfg,
    device=device,
    tokenizer=tokenizer,
    ckpt_dir=cfg.ckpt_dir,
) 

# =============================================================================
# Resume training (if enabled)
# =============================================================================
if RESUME_FROM_CHECKPOINT:
    if not os.path.isfile(RESUME_CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {RESUME_CKPT_PATH}"
        )

    log.info(f"Resuming training from checkpoint: {RESUME_CKPT_PATH}")
    trainer.load_checkpoint(RESUME_CKPT_PATH)
    log.info(f"Resumed at global_step = {trainer.global_step}")


log.info(f"Model: {trainer.parameter_counts()}")

 
# Train
# =============================================================================
# trainer.train(
#     train_dataloader=train_loader,
#     val_dataloader=val_loader,
#     epochs=1,  # ignored if total_steps reached
# )
