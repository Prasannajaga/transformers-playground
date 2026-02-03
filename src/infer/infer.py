from config import TrainingConfig
import torch
import torch.nn as nn
from typing import Optional, Any 

class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.cfg = config
        self.device = device
        self.tokenizer = tokenizer

        self.model.eval()

        self.autocast_dtype = (
            torch.bfloat16
            if self.cfg.amp_dtype == "bfloat16"
            else torch.float16
        )
 
    # Internal sampling logic (shared) 
    def _sample_next_token(self, logits, idx):
        logits = logits / self.cfg.temperature

        # Repetition penalty (vectorized, cheap)
        if self.cfg.use_repetition_penalty:
            uniq = torch.unique(idx)
            logits[:, uniq] /= self.cfg.repetition_penalty

        # Top-K
        if self.cfg.use_top_k:
            v, _ = torch.topk(logits, self.cfg.top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # Streaming generation (token-by-token)
    @torch.no_grad()
    def stream_generate(self, input_ids):
        """
        input_ids: (1, T) already tokenized
        """
        idx = input_ids.to(self.device)

        for _ in range(self.cfg.max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]

            with torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype,
                enabled=self.cfg.use_amp,
            ):
                logits, _ = self.model(idx_cond)

            next_token_logits = logits[:, -1, :]
            next_token = self._sample_next_token(next_token_logits, idx)

            if (
                self.cfg.stop_on_eos
                and next_token.item() == self.tokenizer.eos_token_id
            ):
                break

            idx = torch.cat([idx, next_token], dim=1)

            yield self.tokenizer.decode(
                next_token[0], skip_special_tokens=True
            )

    # Non-streaming generation
    @torch.no_grad()
    def generate(self, input_ids):
        idx = input_ids.to(self.device)

        for _ in range(self.cfg.max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]

            with torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype,
                enabled=self.cfg.use_amp,
            ):
                logits, _ = self.model(idx_cond)

            next_token_logits = logits[:, -1, :]
            next_token = self._sample_next_token(next_token_logits, idx)

            if (
                self.cfg.stop_on_eos
                and next_token.item() == self.tokenizer.eos_token_id
            ):
                break

            idx = torch.cat([idx, next_token], dim=1)

        return self.tokenizer.decode(idx[0], skip_special_tokens=True)

    @torch.no_grad()
    def load_checkpoint(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model_state_dict"], strict=True) 
        return data