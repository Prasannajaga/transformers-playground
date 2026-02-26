import time

import torch
from transformers import TextStreamer

from src.utils.unsloth_wrapper import UnslothWrapper

MODEL_NAME = "Prasanna-SmolLM-360M-3.3"
MAX_SEQ_LENGTH = 1024
SYSTEM_PROMPT = "You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills."


print(f"⏳ Loading merged model from {MODEL_NAME}...")
model, tokenizer = UnslothWrapper.load_model_and_tokenizer(
    model_name=MODEL_NAME,
    model_type="language",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
)

model = UnslothWrapper.for_inference(model)
print("✅ Model loaded and ready for inference.\n")

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

messages: list[dict[str, str]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"quit", "exit", "q"}:
        print("👋 Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    attention_mask = torch.ones_like(inputs, device=model.device)

    print("Assistant: ", end="", flush=True)

    start_time = time.perf_counter()

    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        use_cache=True,
        streamer=streamer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    end_time = time.perf_counter()

    num_generated_tokens = outputs.shape[-1] - inputs.shape[-1]
    elapsed_seconds = end_time - start_time
    tokens_per_second = num_generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

    print(f"\n⚡ {num_generated_tokens} tokens in {elapsed_seconds:.2f}s ({tokens_per_second:.2f} tok/s)\n")

    generated_tokens = outputs[0][inputs.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": response})
