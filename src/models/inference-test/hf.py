import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load Model with FlashAttention-2
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2"
)

prompt = "Write a Python function to calculate the Fibonacci sequence."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=256)
end_time = time.time()

num_generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
time_taken = end_time - start_time
tokens_per_second = num_generated_tokens / time_taken

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("\n--- Inference Stats ---")
print(f"Generated {num_generated_tokens} tokens in {time_taken:.2f} seconds")
print(f"Speed: {tokens_per_second:.2f} tokens/sec")