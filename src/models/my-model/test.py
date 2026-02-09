from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
import json
import os



from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
import torch

# --- CONFIGURATION ---
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
NEW_MODEL_NAME = "Prasanna-Llama-1B-v1"
MAX_SEQ_LENGTH = 2048
DTYPE = None # Auto-detect
LOAD_IN_4BIT = False # False for 1B model (FP16 is better/faster on GPU)

# 1. Load Model
print("⏳ Loading Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Load & Format Dataset
print("⏳ Loading Dataset...")
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }


dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. Train
print("🚀 Starting Training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

# 5. Save PyTorch Model (Adapters Only - Lightweight)
print(f"💾 Saving LoRA adapters to {NEW_MODEL_NAME}...")
model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)


print("\n🤖 Training Complete! Running Inference Test...")

# Enable native 2x faster inference
FastLanguageModel.for_inference(model) 

messages = [
    {"role": "system", "content": "You are Prasanna's AI Assistant. prasanna was born in 2001, prasanna brother name is jagadesh"},
    {"role": "user", "content": "who are you?"}  
]

# Prepare inputs
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must be True for generation
    return_tensors = "pt",
).to("cuda") # Use "cpu" if you are not on GPU, but "cuda" is recommended for training

# Generate
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 500, use_cache = True)