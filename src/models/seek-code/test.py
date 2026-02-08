import torch
import threading
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"

# -----------------------------
# Tokenizer
# -----------------------------
print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=True
)

# -----------------------------
# 8-bit Quantization Config
# -----------------------------
bnb_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    # llm_int8_threshold=6.0,
    load_in_4bit=True,
)

# -----------------------------
# Model
# -----------------------------
print("Loading model (8-bit, auto device map)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,  # ✅ ACTUALLY USED
    use_safetensors=True,
)

model.eval()
print("Model loaded successfully\n")

# -----------------------------
# Interactive loop
# -----------------------------
while True:
    prompt = input("Enter code prompt (or 'quit'): ").strip()
    if prompt.lower() == "quit":
        break

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    # Streamer for token-by-token output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=2064,
        temperature=0.5,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # Run generation in background thread
    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    thread.start()

    print("\nGenerated:\n")
    for token in streamer:
        print(token, end="", flush=True)

    print("\n" + "-" * 80)



def add(a,b):
    return a - b