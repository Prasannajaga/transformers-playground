from utils import UnslothWrapper

MODEL_NAME = "Prasanna-SmolLM-360M"
MAX_SEQ_LENGTH = 1024
SYSTEM_PROMPT = (
    "You are Prasanna's AI Assistant. "
    "You answer questions about his professional background, projects, and skills."
)


print(f"⏳ Loading merged model from {MODEL_NAME}...")
model, tokenizer = UnslothWrapper.load_model_and_tokenizer(
    model_name=MODEL_NAME,
    model_type="language",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
)

model = UnslothWrapper.for_inference(model)
print("✅ Model loaded and ready for inference.\n")

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

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=True,
    )

    # Decode only the newly generated tokens (skip the input prompt)
    generated_tokens = outputs[0][inputs.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print(f"\nAssistant: {response}\n")
    messages.append({"role": "assistant", "content": response})
