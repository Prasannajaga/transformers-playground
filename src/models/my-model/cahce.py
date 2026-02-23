from datasets import load_dataset 
import torch 
from utils.unsloth_wrapper import UnslothWrapper 
import os
import json 

OUTPUT_PATH = "./datasets/prasanna_data.json"
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
NEW_MODEL_NAME = "Prasanna-SmolLM-360M-3.2"
MAX_SEQ_LENGTH = 2048 

print(f"Loading {MODEL_NAME}...")
model, tokenizer = UnslothWrapper.load_model_and_tokenizer(
    model_name=MODEL_NAME,
    model_type="language",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_16bit=True,
) 

bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# standard config 
model = UnslothWrapper.get_peft_model(
    model=model,
    r=32,                
    lora_alpha=64,      
    lora_dropout=0.02,  
    bias="none",
    use_gradient_checkpointing="unsloth",
)

TRAIN_ARGS = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2, 
    "warmup_steps": 20,       
    "num_train_epochs": 3,    
    "learning_rate": 3e-4,    
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",    
    "output_dir": "outputs/variant1",
    "bf16": bf16,
    "fp16": not bf16,
}

# Mid config 
# model = UnslothWrapper.get_peft_model(
#     model=model,
#     r=64,                
#     lora_alpha=128,      
#     lora_dropout=0.0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
# )
# # training arguments for finetuning LORA
# TRAIN_ARGS = {
#     "per_device_train_batch_size": 4,
#     "gradient_accumulation_steps": 4, 
#     "warmup_steps": 10,
#     "num_train_epochs": 5,    
#     "learning_rate": 5e-5,    
#     "weight_decay": 0.01,
#     "lr_scheduler_type": "linear", 
#     "output_dir": "outputs/variant2",
#     "bf16": bf16,
#     "fp16": not bf16,
# }

# short run 
# 1. Update PEFT settings
# model = UnslothWrapper.get_peft_model(
#     model=model,
#     r=8,                # Very low rank (only need to learn surface style)
#     lora_alpha=16,
#     lora_dropout=0.0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
# )

# # 2. Update Training Arguments
# TRAIN_ARGS = {
#     "per_device_train_batch_size": 16, # Faster throughput
#     "gradient_accumulation_steps": 1,
#     "warmup_steps": 5,
#     "num_train_epochs": 2,    # Short run
#     "learning_rate": 5e-4,    # Aggressive LR
#     "weight_decay": 0.00,     # No decay needed for short run
#     "lr_scheduler_type": "cosine",
#     "output_dir": "outputs/variant3",
#     "bf16": bf16,
#     "fp16": not bf16,
# }

# dataset loading 
dataset = load_dataset("json", data_files=OUTPUT_PATH, split="train")
dataset = UnslothWrapper.format_chat_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    messages_field="messages",
    output_field="text",
    add_generation_prompt=False,
)

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

# trainer.state.log_history
print("Starting Training...")
UnslothWrapper.train(trainer=trainer)  

print("Merging LoRA adapters and saving model...")
UnslothWrapper.save_pretrained_merged(
    model=model,
    save_directory=NEW_MODEL_NAME,
    tokenizer=tokenizer,
    save_method="merged_16bit",
    push_to_hub=False,
    token=os.getenv("HF_TOKEN"),
)
print(f"Merged model saved to {NEW_MODEL_NAME}")


print("Converting to GGUF format...")
UnslothWrapper.save_pretrained_gguf(
    model=model,
    tokenizer=tokenizer,
    model_name=NEW_MODEL_NAME,  
    quantization_method=["q8_0" , "q5_k_m", "q4_k_s"]
)
print(f"GGUF models saved to gguf-models/{NEW_MODEL_NAME}/")

# 1. Extract the logs
logs = trainer.state.log_history

# 2. Save to a file
os.makedirs("metrics", exist_ok=True)   
with open(f"metrics/{NEW_MODEL_NAME}.json", "w") as f:
    json.dump(logs, f, indent=2)

print(f"Metrics saved to metrics/{NEW_MODEL_NAME}.json")
