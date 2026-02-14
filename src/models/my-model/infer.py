from pathlib import Path

import torch

from config import TrainingConfig
from pretrain.inference import InferenceEngine
from utils import HFWrapper


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
LOCAL_MODEL_DIR = Path("Prasanna-SmolLM-360M-merged")
SYSTEM_PROMPT = "You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills."


def _resolve_model_dtype() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if torch.cuda.is_available():
        return "float16"
    return "float32"


def load_inference_engine() -> InferenceEngine:
    local_dir = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else None
    model, tokenizer = HFWrapper.get_model_and_tokenizer(
        model_name=MODEL_NAME,
        task="causal-lm",
        model_local_dir=local_dir,
        tokenizer_local_dir=local_dir,
        prefer_local=True,
        tokenizer_kwargs={"use_fast": True},
        model_kwargs={"device_map": None, "dtype": _resolve_model_dtype()},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    amp_dtype = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float16"
    config = TrainingConfig(
        device=device.type,
        use_amp=device.type == "cuda",
        amp_dtype=amp_dtype,
        max_new_tokens=128,
        temperature=0.7,
    )

    return InferenceEngine(
        model=model,
        config=config,
        device=device,
        tokenizer=tokenizer,
    )


def main() -> None:
    engine = load_inference_engine()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        prompt = input("\nYou: ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break

        messages.append({"role": "user", "content": prompt})

        print("\nAssistant: ", end="", flush=True)
        chunks: list[str] = []
        for token in engine.infer_chat(messages, streamText=True):
            print(token, end="", flush=True)
            chunks.append(token)
        print()

        response = "".join(chunks).strip()
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
