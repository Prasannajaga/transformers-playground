from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
import json
import os 
from trl import SFTTrainer
from transformers import TrainingArguments , TextStreamer

# --- CONFIGURATION ---
# We switch to SmolLM2 (135M version)
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
NEW_MODEL_NAME = "Prasanna-SmolLM-135M"
MAX_SEQ_LENGTH = 1024 # 135M models don't need massive context
DTYPE = None 
LOAD_IN_4BIT = False # 135M is so small, we don't even need 4bit loading!

# 1. Load Model
print(f"⏳ Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# 2. Add LoRA Adapters
# For such a small model, we target all modules to squeeze out maximum intelligence
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Higher rank because the model is small (needs more capacity to learn)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 3. Load & Format Dataset
dataset = load_dataset("json", data_files="src/models/my-model/data/prasanna_data.json", split="train")

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Train
print("🚀 Starting Training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = True, # Critical for speed
    # Updated arguments for better generalization
    args = TrainingArguments(
        per_device_train_batch_size = 8,    
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        num_train_epochs = 3,       # Use epochs instead of max_steps to control passes
        learning_rate = 3e-4,       # Lower LR for stability
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = "outputs",
    )
)

trainer.train()

# 5. Export to GGUF (Q8_0 for Quality)
# Since the model is only 135M, we can use Q8 (High Quality) and it will STILL be small (~150MB)
# print("📦 Converting to GGUF (Q8_0)...")
# model.save_pretrained_gguf(NEW_MODEL_NAME, tokenizer, quantization_method = "q8_0")

print("\n🤖 Training Complete! Running Inference Test...")

# Enable native 2x faster inference
FastLanguageModel.for_inference(model) 

# Initialize messages with system prompt
messages = [
    {"role": "system", "content": "You are Prasanna's AI Assistant. prasanna was born in 2001, prasanna brother name is jagadesh"},
]

# Continuous conversation loop
while True:
    # Get user input dynamically
    user_input = input("\n💬 You: ")
    
    # Exit condition
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("👋 Goodbye!")
        break
    
    # Append user message to the messages list
    messages.append({"role": "user", "content": user_input})
    
    # Prepare inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must be True for generation
        return_tensors = "pt",
    ).to("cuda") # Use "cpu" if you are not on GPU, but "cuda" is recommended for training
    
    # Generate
    print("🤖 Assistant: ", end="")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 100, use_cache = True)
    
    # Decode the response and add to conversation history
    response = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response})