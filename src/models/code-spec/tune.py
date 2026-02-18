import torch
import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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
    dtype=torch.float16,
    device_map="auto",
    # attn_implementation="flash_attention_2" # Optional: Faster attention if hardware supports it
)

# 3. Load Draft Model (The "Student")
# Must match the target model's dtype for compatibility
draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    # attn_implementation="flash_attention_2"
)

print("✅ Models loaded. Starting Inference...")

# --- Inference Function ---
def generate_code_speculative(prompt, max_new_tokens=1024, sepculative=True):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Measure time
    start_time = time.time()
    
    generation_kwargs = dict(
        inputs,
        assistant_model=draft_model if sepculative else None,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread_results = {}
    def generate_thread():
        thread_results['outputs'] = target_model.generate(**generation_kwargs)

    t = Thread(target=generate_thread)
    t.start()
    
    # Print stream
    for text in streamer:
        print(text, end="", flush=True)

    t.join()
    outputs = thread_results['outputs']
    
    end_time = time.time()
    
    # Calculate stats
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
    total_time = end_time - start_time
    tps = num_tokens / total_time
    
    return generated_text, tps

# --- Run Test --- 
while True:
    prompt_text = input("\n📝 Enter prompt (or 'exit' to quit): ")
    if prompt_text.lower() in ["exit", "quit"]:
        break
    
    if not prompt_text.strip():
        continue

    print("-" * 40)
    result, tps = generate_code_speculative(prompt_text, sepculative=True)

    print(f"\n{result}")
    print("-" * 40)
    print(f"⚡ Speed: {tps:.2f} tokens/second") 