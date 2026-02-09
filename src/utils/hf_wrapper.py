from __future__ import annotations
from pathlib import Path
from typing import Any
import threading

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)


class HFWrapper:
    _model_cache: dict[str, Any] = {}
    _tokenizer_cache: dict[str, Any] = {}
    _task_to_model_class = {
        "auto": AutoModel,
        "causal-lm": AutoModelForCausalLM,
        "seq2seq-lm": AutoModelForSeq2SeqLM,
        "sequence-classification": AutoModelForSequenceClassification,
        "token-classification": AutoModelForTokenClassification,
        "question-answering": AutoModelForQuestionAnswering,
    }

    @classmethod
    def _cache_key(cls, *parts: Any) -> str:
        return "::".join(str(part) for part in parts)

    @classmethod
    def _normalized_kwargs_repr(cls, kwargs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((key, repr(value)) for key, value in kwargs.items()))

    @classmethod
    def _normalize_task(cls, task: str) -> str:
        return task.strip().lower().replace("_", "-")

    @classmethod
    def _resolve_model_class(cls, task: str):
        normalized_task = cls._normalize_task(task)
        if normalized_task not in cls._task_to_model_class:
            valid_tasks = ", ".join(sorted(cls._task_to_model_class.keys()))
            raise ValueError(f"Unsupported task '{task}'. Supported: {valid_tasks}")
        return cls._task_to_model_class[normalized_task]

    @classmethod
    def _normalize_dtype(cls, torch_dtype: str | torch.dtype | None) -> str | torch.dtype | None:
        if torch_dtype is None:
            return None
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        normalized = torch_dtype.strip().lower()
        if normalized == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch_dtype '{torch_dtype}'")
        return mapping[normalized]

    @classmethod
    def build_quantization_config(
        quantization: str | None = None,
        **kwargs: Any,
    ) -> BitsAndBytesConfig | None:
        if quantization is None:
            return None
        normalized = quantization.strip().lower().replace("-", "")
        if normalized in {"int8", "8bit"}:
            config_kwargs = {"load_in_8bit": True}
            config_kwargs.update(kwargs)
            return BitsAndBytesConfig(**config_kwargs)
        if normalized in {"int4", "4bit"}:
            config_kwargs = {"load_in_4bit": True}
            config_kwargs.update(kwargs)
            return BitsAndBytesConfig(**config_kwargs)
        raise ValueError("quantization must be one of: int8, int4, 8bit, 4bit")

    @classmethod
    def _has_local_quant_config(cls, source: str) -> bool:
        local_path = Path(source)
        return local_path.is_dir() and (local_path / "quantization_config.json").exists()

    @classmethod
    def pull_model(
        model_name: str,
        local_dir: str | Path | None = None,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        local_files_only: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "repo_id": model_name,
            "revision": revision,
            "token": token,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
            "local_files_only": local_files_only,
        }
        if local_dir is not None:
            kwargs["local_dir"] = str(Path(local_dir))
            kwargs["local_dir_use_symlinks"] = False
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return snapshot_download(**kwargs)

    @classmethod
    def resolve_source(
        name: str,
        local_dir: str | Path | None = None,
        *,
        prefer_local: bool = True,
    ) -> str:
        if local_dir is None:
            return name
        local_path = Path(local_dir)
        if prefer_local and local_path.exists():
            return str(local_path)
        return name

    @classmethod
    def load_tokenizer(
        cls,
        *,
        model_name: str,
        tokenizer_name: str | None = None,
        local_dir: str | Path | None = None,
        prefer_local: bool = True,
        use_fast: bool | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        force_reload: bool = False,
        **kwargs: Any,
    ):
        source_name = tokenizer_name or model_name
        source = cls.resolve_source(source_name, local_dir, prefer_local=prefer_local)
        cache_key = cls._cache_key(
            "tokenizer",
            source,
            use_fast,
            trust_remote_code,
            local_files_only,
            cls._normalized_kwargs_repr(kwargs),
        )
        if not force_reload and cache_key in cls._tokenizer_cache:
            return cls._tokenizer_cache[cache_key]

        tokenizer_kwargs = dict(kwargs)
        if use_fast is not None:
            tokenizer_kwargs["use_fast"] = use_fast
        tokenizer_kwargs["trust_remote_code"] = trust_remote_code
        tokenizer_kwargs["local_files_only"] = local_files_only
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        cls._tokenizer_cache[cache_key] = tokenizer
        return tokenizer

    @classmethod
    def load_model(
        cls,
        *,
        model_name: str,
        task: str = "causal-lm",
        local_dir: str | Path | None = None,
        prefer_local: bool = True,
        quantization: str | None = None,
        quantization_kwargs: dict[str, Any] | None = None,
        device_map: str | dict[str, int] | None = "auto",
        torch_dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        force_reload: bool = False,
        **kwargs: Any,
    ):
        source = cls.resolve_source(model_name, local_dir, prefer_local=prefer_local)
        model_class = cls._resolve_model_class(task)
        quant_config = None
        if quantization is not None and not cls._has_local_quant_config(source):
            quant_config = cls.build_quantization_config(quantization, **(quantization_kwargs or {}))
        dtype = cls._normalize_dtype(torch_dtype)

        model_kwargs = dict(kwargs)
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model_kwargs["trust_remote_code"] = trust_remote_code
        model_kwargs["local_files_only"] = local_files_only
        if use_safetensors is not None:
            model_kwargs["use_safetensors"] = use_safetensors
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        cache_key = cls._cache_key(
            "model",
            source,
            cls._normalize_task(task),
            quantization,
            repr(quantization_kwargs),
            repr(device_map),
            repr(dtype),
            trust_remote_code,
            local_files_only,
            use_safetensors,
            cls._normalized_kwargs_repr(kwargs),
        )
        if not force_reload and cache_key in cls._model_cache:
            return cls._model_cache[cache_key]

        model = model_class.from_pretrained(source, **model_kwargs)
        model.eval()
        cls._model_cache[cache_key] = model
        return model

    @classmethod
    def save_pretrained(
        *,
        model: Any,
        tokenizer: Any | None,
        save_dir: str | Path,
        max_shard_size: str = "10GB",
    ) -> Path:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, max_shard_size=max_shard_size)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        return output_dir

    @classmethod
    def load_model_and_tokenizer(
        cls,
        *,
        model_name: str,
        tokenizer_name: str | None = None,
        task: str = "causal-lm",
        model_local_dir: str | Path | None = None,
        tokenizer_local_dir: str | Path | None = None,
        prefer_local: bool = True,
        quantization: str | None = None,
        quantization_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        save_if_downloaded_to: str | Path | None = None,
        force_reload: bool = False,
    ):
        tokenizer = cls.load_tokenizer(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            local_dir=tokenizer_local_dir if tokenizer_local_dir is not None else model_local_dir,
            prefer_local=prefer_local,
            force_reload=force_reload,
            **(tokenizer_kwargs or {}),
        )
        model = cls.load_model(
            model_name=model_name,
            task=task,
            local_dir=model_local_dir,
            prefer_local=prefer_local,
            quantization=quantization,
            quantization_kwargs=quantization_kwargs,
            force_reload=force_reload,
            **(model_kwargs or {}),
        )
        if save_if_downloaded_to is not None:
            save_path = Path(save_if_downloaded_to)
            if not save_path.exists():
                cls.save_pretrained(model=model, tokenizer=tokenizer, save_dir=save_path)
        return model, tokenizer

    @classmethod
    def generate(
        *,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 128,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        encoded = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        kwargs = {"max_new_tokens": max_new_tokens}
        if generation_kwargs:
            kwargs.update(generation_kwargs)

        with torch.no_grad():
            output = model.generate(**encoded, **kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    @classmethod
    def stream_generate(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 128,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        encoded = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "streamer": streamer}
        if generation_kwargs:
            kwargs.update(generation_kwargs)

        thread = threading.Thread(target=model.generate, kwargs={**encoded, **kwargs}, daemon=True)
        thread.start()
        try:
            for token in streamer:
                yield token
        finally:
            thread.join()

    @classmethod
    def load_pipeline(
        cls,
        *,
        pipeline_task: str,
        model_name: str,
        tokenizer_name: str | None = None,
        task: str = "causal-lm",
        model_local_dir: str | Path | None = None,
        tokenizer_local_dir: str | Path | None = None,
        prefer_local: bool = True,
        quantization: str | None = None,
        quantization_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
        force_reload: bool = False,
    ):
        model, tokenizer = cls.load_model_and_tokenizer(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            task=task,
            model_local_dir=model_local_dir,
            tokenizer_local_dir=tokenizer_local_dir,
            prefer_local=prefer_local,
            quantization=quantization,
            quantization_kwargs=quantization_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            force_reload=force_reload,
        )
        return pipeline(
            task=pipeline_task,
            model=model,
            tokenizer=tokenizer,
            **(pipeline_kwargs or {}),
        )
