from peft import PeftModelForSequenceClassification 
from pathlib import Path
from typing import Any, Iterable
import inspect

import torch
import unsloth
import trl
from datasets import Dataset
from peft import PeftModelForCausalLM
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import TrainOutput


class UnslothWrapper:
    _MODEL_LOADERS = {
        "language": "FastLanguageModel",
        "auto": "FastModel",
        "text": "FastTextModel",
        "vision": "FastVisionModel",
    }

    _RL_ALGORITHMS = {
        "grpo": ("GRPOTrainer", "GRPOConfig"),
        "gspo": ("GSPOTrainer", "GSPOConfig"),
        "dpo": ("DPOTrainer", "DPOConfig"),
        "orpo": ("ORPOTrainer", "ORPOConfig"),
        "kto": ("KTOTrainer", "KTOConfig"),
        "ppo": ("PPOTrainer", "PPOConfig"),
        "cpo": ("CPOTrainer", "CPOConfig"),
        "online_dpo": ("OnlineDPOTrainer", "OnlineDPOConfig"),
        "rloo": ("RLOOTrainer", "RLOOConfig"),
        "bco": ("BCOTrainer", "BCOConfig"),
        "gkd": ("GKDTrainer", "GKDConfig"),
        "xpo": ("XPOTrainer", "XPOConfig"),
        "nashmd": ("NashMDTrainer", "NashMDConfig"),
        "reward": ("RewardTrainer", "RewardConfig"),
        "prm": ("PRMTrainer", "PRMConfig"),
    }



    @staticmethod
    def _set_tokenizer_arg(
        trainer_class: type[Trainer],
        trainer_kwargs: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase | None,
    ) -> None:
        if tokenizer is None:
            return
        params = inspect.signature(trainer_class.__init__).parameters
        if "processing_class" in params:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

    @classmethod
    def load_model_and_tokenizer(
        cls,
        *,
        model_name: str,
        model_type: str = "language",
        max_seq_length: int = 2048,
        dtype: torch.dtype | None = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        trust_remote_code: bool = False,
        token: str | None = None,
        **kwargs: Any,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        normalized = model_type.strip().lower().replace("-", "_")
        if normalized not in cls._MODEL_LOADERS:
            valid = ", ".join(sorted(cls._MODEL_LOADERS.keys()))
            raise ValueError(f"Unsupported model_type '{model_type}'. Supported: {valid}")

        loader_name = cls._MODEL_LOADERS[normalized]
        loader = getattr(unsloth, loader_name, None)
        if loader is None:
            fallback = getattr(unsloth, "FastModel")
            loader = fallback

        return loader.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            load_in_16bit=load_in_16bit,
            full_finetuning=full_finetuning,
            trust_remote_code=trust_remote_code,
            token=token,
            **kwargs,
        )

    @classmethod
    def get_peft_model(
        cls,
        *,
        model: PreTrainedModel,
        r: int = 16,
        target_modules: list[str] | None = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: bool | str = "unsloth",
        random_state: int = 3407,
        **kwargs: Any,
    ) -> PeftModelForCausalLM :
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        model = unsloth.FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            **kwargs,
        ) 
        return model

    @staticmethod
    def format_chat_dataset(
        *,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        messages_field: str = "messages",
        output_field: str = "text",
        add_generation_prompt: bool = False,
        num_proc: int | None = None,
    ) -> Dataset:
        def _format(examples: dict[str, Any]) -> dict[str, list[str]]:
            conversations = examples[messages_field]
            texts = [
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
                for convo in conversations
            ]
            return {output_field: texts}

        map_kwargs: dict[str, Any] = {"batched": True}
        if num_proc is not None:
            map_kwargs["num_proc"] = num_proc
        return dataset.map(_format, **map_kwargs)

    @classmethod
    def create_sft_trainer(
        cls,
        *,
        model: PreTrainedModel | PeftModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        args: TrainingArguments | None = None,
        args_kwargs: dict[str, Any] | None = None,
        use_unsloth_trainer: bool = False,
        dataset_text_field: str | None = "text",
        max_seq_length: int | None = None,
        dataset_num_proc: int | None = None,
        packing: bool | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> trl.SFTTrainer:
        if use_unsloth_trainer:
            trainer_class = unsloth.UnslothTrainer
            if args is None:
                args = unsloth.UnslothTrainingArguments(**(args_kwargs or {}))
        else:
            trainer_class = trl.SFTTrainer
            if args is None:
                config_class = getattr(trl, "SFTConfig", None)
                if config_class is None:
                    config_class = TrainingArguments
                args = config_class(**(args_kwargs or {}))

        kwargs = dict(trainer_kwargs or {})
        kwargs.setdefault("model", model)
        kwargs.setdefault("train_dataset", train_dataset)
        kwargs.setdefault("args", args)
        cls._set_tokenizer_arg(trainer_class, kwargs, tokenizer)

        if dataset_text_field is not None:
            kwargs.setdefault("dataset_text_field", dataset_text_field)
        if max_seq_length is not None:
            kwargs.setdefault("max_seq_length", max_seq_length)
        if dataset_num_proc is not None:
            kwargs.setdefault("dataset_num_proc", dataset_num_proc)
        if packing is not None:
            kwargs.setdefault("packing", packing)

        return trainer_class(**kwargs)

    @classmethod
    def create_pretraining_dataset(
        cls,
        *,
        tokenizer: PreTrainedTokenizerBase,
        file_path: str | Path | None = None,
        file_paths: Iterable[str | Path] | None = None,
        text: str | None = None,
        chunk_size: int = 2048,
        stride: int = 512,
        return_tokenized: bool = False,
    ) -> Dataset:
        loader = unsloth.RawTextDataLoader(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            stride=stride,
            return_tokenized=return_tokenized,
        )

        if text is not None:
            chunks = loader.chunk_text(text, return_tokenized=return_tokenized)
            return loader.create_causal_dataset(chunks)

        if file_paths is not None:
            paths = [str(Path(path)) for path in file_paths]
            return loader.load_from_files(paths, return_tokenized=return_tokenized)

        if file_path is None:
            raise ValueError("Provide file_path, file_paths, or text")

        return loader.load_from_file(str(Path(file_path)), return_tokenized=return_tokenized)

    @classmethod
    def create_pretraining_trainer(
        cls,
        *,
        model: PreTrainedModel | PeftModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        args: TrainingArguments | None = None,
        args_kwargs: dict[str, Any] | None = None,
        use_unsloth_trainer: bool = True,
        dataset_text_field: str | None = "auto",
        max_seq_length: int | None = None,
        dataset_num_proc: int | None = None,
        packing: bool | None = True,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> trl.SFTTrainer:
        resolved_text_field = dataset_text_field
        if dataset_text_field == "auto":
            columns = getattr(train_dataset, "column_names", [])
            resolved_text_field = "text" if "text" in columns else None

        return cls.create_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=args,
            args_kwargs=args_kwargs,
            use_unsloth_trainer=use_unsloth_trainer,
            dataset_text_field=resolved_text_field,
            max_seq_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            trainer_kwargs=trainer_kwargs,
        )

    @classmethod
    def create_rl_trainer(
        cls,
        *,
        algorithm: str,
        model: PreTrainedModel | PeftModelForCausalLM,
        train_dataset: Dataset | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        args: TrainingArguments | None = None,
        args_kwargs: dict[str, Any] | None = None,
        patch_fast_rl: bool = True,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> Trainer:
        normalized = algorithm.strip().lower().replace("-", "_")

        if normalized not in cls._RL_ALGORITHMS:
            valid = ", ".join(sorted(cls._RL_ALGORITHMS.keys()))
            raise ValueError(f"Unsupported RL algorithm '{algorithm}'. Supported: {valid}")

        trainer_name, config_name = cls._RL_ALGORITHMS[normalized]
        trainer_class = getattr(trl, trainer_name, None)
        if trainer_class is None:
            raise ValueError(f"{trainer_name} is not available in your installed trl version")

        if patch_fast_rl:
            patch_algorithm = normalized.replace("_", "")
            unsloth.PatchFastRL(
                algorithm=patch_algorithm,
                FastLanguageModel=unsloth.FastLanguageModel,
            )

        if args is None:
            config_class = getattr(trl, config_name, None)
            if config_class is None:
                raise ValueError(f"{config_name} is not available in your installed trl version")
            args = config_class(**(args_kwargs or {}))

        kwargs = dict(trainer_kwargs or {})
        kwargs.setdefault("model", model)
        kwargs.setdefault("args", args)
        if train_dataset is not None:
            kwargs.setdefault("train_dataset", train_dataset)
        cls._set_tokenizer_arg(trainer_class, kwargs, tokenizer)

        return trainer_class(**kwargs)

    @staticmethod
    def train(*, trainer: Trainer, **kwargs: Any) -> TrainOutput:
        return trainer.train(**kwargs)

    @classmethod
    def for_inference(cls, model: PreTrainedModel) -> PreTrainedModel:
        return unsloth.FastLanguageModel.for_inference(model)

    @classmethod
    def for_training(cls, model: PreTrainedModel, use_gradient_checkpointing: bool = True) -> PreTrainedModel:
        return unsloth.FastLanguageModel.for_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    @staticmethod
    def save_pretrained_gguf(
        *,
        model: PreTrainedModel,
        save_directory: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        quantization_method: str = "q4_k_m",
    ) -> None:
        model.save_pretrained_gguf(
            str(Path(save_directory)),
            tokenizer,
            quantization_method=quantization_method,
        )

    @staticmethod
    def save_pretrained_merged(
        *,
        model: PreTrainedModel,
        save_directory: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        save_method: str = "merged_16bit",
        push_to_hub: bool = False,
        token: str | None = None,
    ) -> None:
        """Merge LoRA adapters into the base model and save.

        Args:
            model: The PEFT model with LoRA adapters.
            save_directory: Path to save the merged model.
            tokenizer: Tokenizer to save alongside the model.
            save_method: One of ``merged_16bit``, ``merged_4bit``, or ``lora``.
            push_to_hub: Whether to push to Hugging Face Hub after saving.
            token: Hugging Face token for pushing to hub.
        """
        save_kwargs: dict[str, Any] = {
            "save_method": save_method,
        }
        if push_to_hub:
            save_kwargs["push_to_hub"] = True
            if token is not None:
                save_kwargs["token"] = token

        model.save_pretrained_merged(
            str(Path(save_directory)),
            tokenizer,
            **save_kwargs,
        )
