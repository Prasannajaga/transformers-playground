import torch
import torch.nn as nn


class KVCache(nn.Module):

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.current_len = 0

        cache_shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = k_new.shape[0]
        new_len = k_new.shape[2]
        end_pos = self.current_len + new_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: tried to write up to position {end_pos}, "
                f"but max_seq_len is {self.max_seq_len}"
            )

        self.k_cache[:batch_size, :, self.current_len:end_pos, :] = k_new
        self.v_cache[:batch_size, :, self.current_len:end_pos, :] = v_new
        self.current_len = end_pos

        return (
            self.k_cache[:batch_size, :, :end_pos, :],
            self.v_cache[:batch_size, :, :end_pos, :],
        )

    @property
    def seq_len(self) -> int:
        return self.current_len

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0

    def resize(self, new_max_seq_len: int):
        if new_max_seq_len == self.max_seq_len:
            return

        old_len = min(self.current_len, new_max_seq_len)
        batch_size, n_kv_heads, _, head_dim = self.k_cache.shape

        new_k = torch.zeros(
            batch_size, n_kv_heads, new_max_seq_len, head_dim,
            dtype=self.k_cache.dtype, device=self.k_cache.device,
        )
        new_v = torch.zeros(
            batch_size, n_kv_heads, new_max_seq_len, head_dim,
            dtype=self.v_cache.dtype, device=self.v_cache.device,
        )

        if old_len > 0:
            new_k[:, :, :old_len, :] = self.k_cache[:, :, :old_len, :]
            new_v[:, :, :old_len, :] = self.v_cache[:, :, :old_len, :]

        self.k_cache = new_k
        self.v_cache = new_v
        self.max_seq_len = new_max_seq_len
        self.current_len = old_len
