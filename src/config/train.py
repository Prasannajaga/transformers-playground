from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TrainingConfig:
    
    # System 
    device: str = "cuda"
    seed: Optional[int] = 42
    mixed_precision: bool = True
    num_workers: int = 4

    
    # Model 
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    block_size: int = 256

    
    # Data
    train_batch_size: int = 16
    eval_batch_size: int = 8
    grad_accum_steps: int = 4
    validation_batch_count: int = 20

    
    # Optimization 
    lr: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    
    # VRAM Optimization
    enable_gradient_checkpointing: bool = True  # Trade compute for memory
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16" (bfloat16 more stable, requires Ampere+)
    clear_cache_interval: int = 100  # Clear CUDA cache every N steps (0 to disable)
    log_memory_usage: bool = True  # Log GPU memory stats

    
    # LR Schedule 
    total_steps: int = 30_000
    warmup_steps: int = 2_000

    
    # Checkpointing 
    ckpt_dir: str = "./checkpoints"
    ckpt_interval_steps: int = 15000
    save_optimizer_state: bool = True

    
    # Logging 
    enable_logging: bool = True
    log_interval_steps: int = 500
    log_per_iteration_time: bool = False
