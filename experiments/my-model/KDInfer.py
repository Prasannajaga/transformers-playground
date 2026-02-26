from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.architectures.gqa_transformer import GQATransformer


DEFAULT_SYSTEM_PROMPT = (
    "You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-path",
        type=Path,
        default=Path("outputs/llama32_unsloth_kd/teacher-merged"),
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=Path("outputs/llama32_unsloth_kd/student/student_final.pt"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("outputs/llama32_unsloth_kd/student/tokenizer"),
    )
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--student-temperature", type=float, default=0.8)
    parser.add_argument("--student-top-k", type=int, default=40)
    parser.add_argument("--student-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def infer_num_layers(state_dict: dict[str, torch.Tensor]) -> int:
    indices: set[int] = set()
    for key in state_dict.keys():
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 2:
            continue
        if parts[1].isdigit():
            indices.add(int(parts[1]))
    if not indices:
        raise ValueError("Unable to infer student num_layers from checkpoint state_dict")
    return max(indices) + 1


def load_tokenizer(tokenizer_path: Path, teacher_path: Path):
    if tokenizer_path.exists():
        return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
    return AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True, use_fast=True)


def load_teacher(teacher_path: Path, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        str(teacher_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model


def load_student(
    *,
    student_checkpoint: Path,
    tokenizer: AutoTokenizer,
    device: torch.device,
    n_head: int,
    n_kv_head: int,
    block_size: int,
) -> GQATransformer:
    payload = torch.load(student_checkpoint, map_location="cpu")
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError("student checkpoint missing model_state_dict")

    arch = payload.get("student_arch", {})

    vocab_size = int(arch.get("vocab_size") or state_dict["token_emb.weight"].shape[0])
    n_emb = int(arch.get("n_emb") or state_dict["token_emb.weight"].shape[1])
    num_layers = int(arch.get("num_layers") or infer_num_layers(state_dict))
    resolved_n_head = int(arch.get("n_head") or n_head)
    resolved_n_kv_head = int(arch.get("n_kv_head") or n_kv_head)
    resolved_block_size = int(arch.get("block_size") or block_size)

    model = GQATransformer(
        num_layers=num_layers,
        n_emb=n_emb,
        n_head=resolved_n_head,
        n_kv_head=resolved_n_kv_head,
        vocab_size=vocab_size,
        block_size=resolved_block_size,
        dropout=0.0,
    )

    model.load_state_dict(state_dict, strict=True)

    if model.token_emb.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()
    return model


def generate_teacher(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[str, float, int, float]:
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    attention_mask = torch.ones_like(prompt_ids, device=device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    do_sample = temperature > 0.0
    gen_kwargs = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**gen_kwargs)
    elapsed = max(time.perf_counter() - start, 1e-6)

    generated = output[0, prompt_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    token_count = int(generated.shape[0])
    tok_s = token_count / elapsed
    return text, elapsed, token_count, tok_s


def generate_student(
    *,
    model: GQATransformer,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    device: torch.device,
) -> tuple[str, float, int, float]:
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    if prompt_ids.shape[1] > model.block_size:
        prompt_ids = prompt_ids[:, -model.block_size :]

    eos_token_id = tokenizer.eos_token_id

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            idx=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
    elapsed = max(time.perf_counter() - start, 1e-6)

    generated = output[0, prompt_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    token_count = int(generated.shape[0])
    tok_s = token_count / elapsed
    return text, elapsed, token_count, tok_s


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = load_tokenizer(args.tokenizer_path, args.teacher_path)

    print("tokenizer" , len(tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = load_teacher(args.teacher_path, device=device, dtype=dtype)
    student = load_student(
        student_checkpoint=args.student_checkpoint,
        tokenizer=tokenizer,
        device=device,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        block_size=args.block_size,
    )

    print("student tokenizer" , student.token_emb.num_embeddings)


    print("Teacher and student loaded. Type 'exit' to quit.")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"q", "quit", "exit"}:
            break

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_text},
        ]

        teacher_text, teacher_lat, teacher_tokens, teacher_tps = generate_teacher(
            model=teacher,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.teacher_temperature,
            device=device,
        )

        student_text, student_lat, student_tokens, student_tps = generate_student(
            model=student,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.student_temperature,
            top_k=args.student_top_k,
            repetition_penalty=args.student_repetition_penalty,
            device=device,
        )

        similarity = token_f1(student_text, teacher_text)

        print("\nTeacher:")
        print(teacher_text or "<empty>")
        print(f"[latency={teacher_lat:.3f}s, tokens={teacher_tokens}, tok/s={teacher_tps:.1f}]")

        print("\nStudent:")
        print(student_text or "<empty>")
        print(f"[latency={student_lat:.3f}s, tokens={student_tokens}, tok/s={student_tps:.1f}]")

        print(f"\nToken-F1(student vs teacher): {similarity:.4f}")


if __name__ == "__main__":
    main()
