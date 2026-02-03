from datasets.formatting.torch_formatter import torch
import torch
from transformers import GPT2Tokenizer
from config import TrainingConfig 
from customTransformers import DecodeTransformer
from infer import InferenceEngine
import os
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(torch.version.cuda)
# print(torch.cuda)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Model (YOU own construction)
train_cfg = TrainingConfig()
model = DecodeTransformer(
    num_layers=train_cfg.n_layer,
    n_emb=train_cfg.n_embd,
    n_head=train_cfg.n_head,
    vocab_size=tokenizer.vocab_size,
    block_size=train_cfg.block_size,
    ffn_type="swiglu",
    attention="MHA"
).to(device)

fileName = f"ckpt_step_{sys.argv[1]}.pt"
print("fileName" , fileName)

# Load checkpoint
RESUME_CKPT_PATH = os.path.join(
    train_cfg.ckpt_dir,
    fileName
) 

# Inference 
infer = InferenceEngine(
    model=model,
    config=train_cfg,
    device=device,
    tokenizer=tokenizer,
)

if not os.path.isfile(RESUME_CKPT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {RESUME_CKPT_PATH}"
    ) 

infer.load_checkpoint(RESUME_CKPT_PATH) 

# CLI
while True:
    prompt = input("\nPROMPT > ")
    if prompt in ("exit", "quit"):
        break

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print("\nRESPONSE: ", end="", flush=True)
    for tok in infer.stream_generate(input_ids):
        print(tok, end="", flush=True)

    print("\n" + "-" * 40)
