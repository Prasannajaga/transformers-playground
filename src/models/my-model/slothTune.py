from datasets import load_dataset 
import torch 
from utils import UnslothWrapper 
import os

OUTPUT_PATH = "./datasets/prasanna_data.json"
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
NEW_MODEL_NAME = "Prasanna-SmolLM-360M"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = False

print(f"⏳ Loading {MODEL_NAME}...")
model, tokenizer = UnslothWrapper.load_model_and_tokenizer(
    model_name=MODEL_NAME,
    model_type="language",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
) 

model = UnslothWrapper.get_peft_model(
    model=model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
) 

# dataset loading 
dataset = load_dataset("json", data_files=OUTPUT_PATH, split="train")
dataset = UnslothWrapper.format_chat_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    messages_field="messages",
    output_field="text",
    add_generation_prompt=False,
)

bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
TRAIN_ARGS = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 10,
    "num_train_epochs": 3,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "logging_steps": 5,
    "optim": "adamw_torch",
    "output_dir": "outputs/unsloth-sft",
    "save_strategy": "no",
    "bf16": bf16,
    "fp16": not bf16,
}

trainer = UnslothWrapper.create_sft_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args_kwargs=TRAIN_ARGS,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,
)

print("🚀 Starting Training...")
UnslothWrapper.train(trainer=trainer)  

print("💾 Merging LoRA adapters and saving model...")
UnslothWrapper.save_pretrained_merged(
    model=model,
    save_directory=NEW_MODEL_NAME,
    tokenizer=tokenizer,
    save_method="merged_16bit",
    push_to_hub=True,
    token=os.getenv("HF_TOKEN"),
)
print(f"✅ Merged model saved to {NEW_MODEL_NAME}")
