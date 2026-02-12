from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import torch

from src.utils import UnslothWrapper


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qlora_4bit": {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "load_in_16bit": False,
        "full_finetuning": False,
    },
    "lora_16bit": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,
    },
    "lora_8bit": {
        "load_in_4bit": False,
        "load_in_8bit": True,
        "load_in_16bit": False,
        "full_finetuning": False,
    },
    "full_finetune": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_in_16bit": False,
        "full_finetuning": True,
    },
}

LORA_PROFILES: dict[str, dict[str, Any]] = {
    "none": {},
    "base": {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
    },
    "high_rank": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
    },
}

SFT_PRESET: dict[str, Any] = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 10,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "logging_steps": 5,
    "optim": "adamw_torch",
    "output_dir": "outputs/unsloth-sft",
}

PRETRAIN_PRESET: dict[str, Any] = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 20,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "optim": "adamw_torch",
    "output_dir": "outputs/unsloth-pretrain",
}

GRPO_PRESET: dict[str, Any] = {
    "output_dir": "outputs/unsloth-rl-grpo",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,
    "logging_steps": 5,
    "num_generations": 2,
    "max_prompt_length": 512,
    "max_completion_length": 256,
}


def _bf16_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _with_precision(args_kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(args_kwargs)
    bf16 = _bf16_available()
    out["bf16"] = bf16
    out["fp16"] = not bf16
    return out


def _coerce_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith(("[", "{")) and value.endswith(("]", "}")):
        try:
            return json.loads(value)
        except Exception:
            pass
    return value


def _parse_overrides(values: list[str] | None) -> dict[str, Any]:
    if not values:
        return {}
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --train-arg '{item}'. Expected key=value.")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --train-arg '{item}'. Expected key=value.")
        overrides[key] = _coerce_value(raw.strip())
    return overrides


def _resolve_dataset_files(paths: list[str], dataset_format: str) -> list[str]:
    resolved: list[str] = []
    format_exts = {
        "json": {".json", ".jsonl"},
        "text": {".txt"},
        "parquet": {".parquet"},
        "csv": {".csv"},
    }.get(dataset_format, None)
    for item in paths:
        if any(ch in item for ch in ["*", "?", "["]):
            for match in sorted(Path().glob(item)):
                if match.is_file():
                    resolved.append(str(match))
            continue
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if not child.is_file():
                    continue
                if format_exts is None or child.suffix in format_exts:
                    resolved.append(str(child))
            continue
        resolved.append(str(path))
    return resolved


def _load_dataset_from_args(args: argparse.Namespace) -> Any:
    if args.dataset_source == "hf":
        if len(args.dataset_path) != 1:
            raise ValueError("For --dataset-source=hf provide a single dataset name in --dataset-path.")
        kwargs: dict[str, Any] = {}
        if args.dataset_config:
            kwargs["name"] = args.dataset_config
        return load_dataset(args.dataset_path[0], split=args.dataset_split, **kwargs)

    data_files = _resolve_dataset_files(args.dataset_path, args.dataset_format)
    return load_dataset(args.dataset_format, data_files=data_files, split=args.dataset_split)


def _build_train_args(
    preset: dict[str, Any],
    args: argparse.Namespace,
    *,
    use_precision: bool = True,
) -> dict[str, Any]:
    merged = dict(preset)
    if args.output_dir:
        merged["output_dir"] = args.output_dir
    merged.update(_parse_overrides(args.train_arg))
    return _with_precision(merged) if use_precision else merged


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _adapter_base_model(adapter_path: Path) -> str | None:
    cfg = _read_json(adapter_path / "adapter_config.json")
    if not cfg:
        return None
    base = cfg.get("base_model_name_or_path")
    return base if isinstance(base, str) else None


def _load_peft_adapter(model: Any, adapter_path: Path) -> Any:
    try:
        from peft import PeftModel
    except Exception as exc:
        raise RuntimeError(f"peft is required to load adapters: {exc}") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _load_tokenizer_from_path(path: Path):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(str(path))
    except Exception:
        return None


def _load_infer_model(args: argparse.Namespace):
    model_cfg = MODEL_PROFILES[args.model_profile]
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    base_model = args.model_name
    if adapter_path:
        if args.adapter_base and args.adapter_base != "auto":
            base_model = args.adapter_base
        else:
            inferred = _adapter_base_model(adapter_path)
            if inferred:
                base_model = inferred

    model, tokenizer = UnslothWrapper.load_model_and_tokenizer(
        model_name=base_model,
        model_type=args.model_type,
        max_seq_length=args.max_seq_length,
        trust_remote_code=args.trust_remote_code,
        **model_cfg,
    )

    if adapter_path:
        model = _load_peft_adapter(model, adapter_path)
        adapter_tokenizer = _load_tokenizer_from_path(adapter_path)
        if adapter_tokenizer is not None:
            tokenizer = adapter_tokenizer

    return model, tokenizer


def _load_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.messages_json:
        data = json.loads(Path(args.messages_json).read_text())
    elif args.messages:
        data = json.loads(args.messages)
    else:
        if not args.prompt:
            raise ValueError("Provide --prompt or --messages/--messages-json.")
        data = []
        if args.system_prompt:
            data.append({"role": "system", "content": args.system_prompt})
        data.append({"role": "user", "content": args.prompt})
    if not isinstance(data, list):
        raise ValueError("Messages must be a JSON list of {role, content} items.")
    return data


def _available_rl_algorithms() -> list[str]:
    try:
        import trl
    except Exception:
        return []

    available: list[str] = []
    for algo, (trainer_name, config_name) in UnslothWrapper._RL_ALGORITHMS.items():
        if hasattr(trl, trainer_name) and hasattr(trl, config_name):
            available.append(algo)
    return sorted(available)


def list_options() -> None:
    print("Model types:", ", ".join(sorted(UnslothWrapper._MODEL_LOADERS.keys())))
    print("Model profiles:", ", ".join(sorted(MODEL_PROFILES.keys())))
    print("LoRA profiles:", ", ".join(sorted(LORA_PROFILES.keys())))
    print("RL algorithms (wrapper map):", ", ".join(sorted(UnslothWrapper._RL_ALGORITHMS.keys())))
    print("RL algorithms (installed trl):", ", ".join(_available_rl_algorithms()))
    print("SFT preset:", SFT_PRESET)
    print("Pretrain preset:", PRETRAIN_PRESET)
    print("GRPO preset:", GRPO_PRESET)


def _load_model(args: argparse.Namespace):
    model_cfg = MODEL_PROFILES[args.model_profile]
    model, tokenizer = UnslothWrapper.load_model_and_tokenizer(
        model_name=args.model_name,
        model_type=args.model_type,
        max_seq_length=args.max_seq_length,
        trust_remote_code=args.trust_remote_code,
        **model_cfg,
    )

    if args.lora_profile != "none" and not model_cfg["full_finetuning"]:
        model = UnslothWrapper.get_peft_model(model=model, **LORA_PROFILES[args.lora_profile])

    return model, tokenizer


def run_sft(args: argparse.Namespace) -> None:
    model, tokenizer = _load_model(args)
    dataset = _load_dataset_from_args(args)
    dataset_text_field = args.dataset_text_field
    if dataset_text_field is None:
        dataset = UnslothWrapper.format_chat_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            messages_field=args.messages_field,
            output_field="text",
            add_generation_prompt=False,
            num_proc=args.dataset_num_proc,
        )
        dataset_text_field = "text"

    trainer = UnslothWrapper.create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args_kwargs=_build_train_args(SFT_PRESET, args, use_precision=True),
        use_unsloth_trainer=args.use_unsloth_trainer,
        dataset_text_field=dataset_text_field,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
    )

    print(f"SFT trainer: {trainer.__class__.__name__}")
    if args.train:
        UnslothWrapper.train(trainer=trainer)


def run_pretrain(args: argparse.Namespace) -> None:
    model, tokenizer = _load_model(args)

    raw_paths = args.raw_text_path
    if len(raw_paths) == 1:
        dataset = UnslothWrapper.create_pretraining_dataset(
            tokenizer=tokenizer,
            file_path=raw_paths[0],
            chunk_size=args.chunk_size,
            stride=args.stride,
            return_tokenized=False,
        )
    else:
        dataset = UnslothWrapper.create_pretraining_dataset(
            tokenizer=tokenizer,
            file_paths=raw_paths,
            chunk_size=args.chunk_size,
            stride=args.stride,
            return_tokenized=False,
        )

    trainer = UnslothWrapper.create_pretraining_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args_kwargs=_build_train_args(PRETRAIN_PRESET, args, use_precision=True),
        use_unsloth_trainer=True,
        dataset_text_field="auto",
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    print(f"Pretraining trainer: {trainer.__class__.__name__}")
    if args.train:
        UnslothWrapper.train(trainer=trainer)


def _build_grpo_prompt_dataset(dataset_path: str, tokenizer: Any, messages_field: str) -> Dataset:
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def to_prompt(example: dict[str, Any]) -> dict[str, str]:
        messages = example[messages_field]
        prompt_messages = messages[:-1] if len(messages) > 1 else messages
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt}

    dataset = dataset.map(to_prompt)
    drop_cols = [col for col in dataset.column_names if col != "prompt"]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)
    return dataset


def run_rl(args: argparse.Namespace) -> None:
    if args.rl_algorithm != "grpo":
        raise ValueError("This demo script implements GRPO runnable flow. Other RL algorithms require task-specific dataset schemas.")

    model, tokenizer = _load_model(args)
    if args.dataset_source != "file":
        raise ValueError("GRPO demo expects a local JSON dataset. Use --dataset-source=file.")
    if len(args.dataset_path) != 1:
        raise ValueError("GRPO demo expects a single dataset file path.")
    train_dataset = _build_grpo_prompt_dataset(args.dataset_path[0], tokenizer, args.messages_field)

    def reward_len(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = ""
            if isinstance(completion, list) and completion:
                part = completion[0]
                if isinstance(part, dict):
                    text = str(part.get("content", ""))
                else:
                    text = str(part)
            else:
                text = str(completion)
            rewards.append(min(len(text), 200) / 200.0)
        return rewards

    trainer = UnslothWrapper.create_rl_trainer(
        algorithm="grpo",
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args_kwargs=_build_train_args(GRPO_PRESET, args, use_precision=False),
        trainer_kwargs={"reward_funcs": [reward_len]},
        patch_fast_rl=True,
    )

    print(f"RL trainer: {trainer.__class__.__name__}")
    if args.train:
        UnslothWrapper.train(trainer=trainer)


def run_infer(args: argparse.Namespace) -> None:
    model, tokenizer = _load_infer_model(args)
    UnslothWrapper.for_inference(model)
    model.eval()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    messages = _load_messages(args)
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(input_ids, dict):
            input_ids = input_ids.get("input_ids")
    else:
        encoded = tokenizer(messages[-1]["content"], return_tensors="pt")
        input_ids = encoded["input_ids"]

    model_device = getattr(model, "device", None)
    if model_device is None:
        input_ids = input_ids.to(device)
    else:
        input_ids = input_ids.to(model_device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty

    if args.stream:
        from transformers import TextStreamer

        streamer = TextStreamer(tokenizer)
        model.generate(input_ids=input_ids, streamer=streamer, **gen_kwargs)
    else:
        output = model.generate(input_ids=input_ids, **gen_kwargs)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["list", "sft", "pretrain", "rl", "infer"], default="list")

    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--model-type", choices=sorted(UnslothWrapper._MODEL_LOADERS.keys()), default="language")
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES.keys()), default="qlora_4bit")
    parser.add_argument("--lora-profile", choices=sorted(LORA_PROFILES.keys()), default="base")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--dataset-source", choices=["file", "hf"], default="file")
    parser.add_argument("--dataset-format", default="json")
    parser.add_argument("--dataset-path", nargs="+", default=["src/models/my-model/data/train.jsonl"])
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-num-proc", type=int)
    parser.add_argument("--dataset-text-field")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--use-unsloth-trainer", action="store_true")

    parser.add_argument("--raw-text-path", nargs="+", default=["README.md"])
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)

    parser.add_argument("--rl-algorithm", choices=sorted(UnslothWrapper._RL_ALGORITHMS.keys()), default="grpo")

    parser.add_argument("--output-dir")
    parser.add_argument("--train-arg", action="append")

    parser.add_argument("--adapter-path")
    parser.add_argument("--adapter-base", default="auto")
    parser.add_argument("--prompt")
    parser.add_argument("--system-prompt")
    parser.add_argument("--messages")
    parser.add_argument("--messages-json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--stream", action="store_true")

    parser.add_argument("--train", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "list":
        list_options()
    elif args.mode == "sft":
        run_sft(args)
    elif args.mode == "pretrain":
        run_pretrain(args)
    elif args.mode == "rl":
        run_rl(args)
    elif args.mode == "infer":
        run_infer(args)


if __name__ == "__main__":
    main()
