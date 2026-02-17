import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading models on {DEVICE}...")

# 1. Load Tokenizer (Shared between both models)
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

# 2. Load Target Model (The "Teacher")
# efficient usage: load in float16 (half precision)
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    # attn_implementation="flash_attention_2" # Optional: Faster attention if hardware supports it
)

# 3. Load Draft Model (The "Student")
# Must match the target model's dtype for compatibility
draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    # attn_implementation="flash_attention_2"
)

print("✅ Models loaded. Starting Inference...")

# --- Inference Function ---
def generate_code_speculative(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Measure time
    start_time = time.time()
    
    # The 'assistant_model' parameter triggers speculative decoding automatically
    outputs = target_model.generate(
        **inputs,
        assistant_model=draft_model,  # <--- This enables Speculative Decoding
        max_new_tokens=max_new_tokens,
        do_sample=False,              # Greedy decoding is faster and standard for code
        temperature=0.0,              # Deterministic output
        pad_token_id=tokenizer.eos_token_id
    )
    
    end_time = time.time()
    
    # Calculate stats
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
    total_time = end_time - start_time
    tps = num_tokens / total_time
    
    return generated_text, tps

# --- Run Test ---
prompt_text = """def merge_sort(arr):
    \"\"\"
    Sorts an array using merge sort algorithm.
    \"\"\""""

print(f"\n📝 Prompt: {prompt_text}")
print("-" * 40)

result, tps = generate_code_speculative(prompt_text)

print(result)
print("-" * 40)
print(f"⚡ Speed: {tps:.2f} tokens/second")