from dataclasses import dataclass
from typing import Optional, Tuple
import torch 


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
    optimizer: str = "adamw"  # Supported: adamw, adam, sgd, adafactor
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

    # Inference config here 
    max_new_tokens: int = 512
    temperature: float = 0.7

    # Sampling toggles
    use_top_k: bool = True
    top_k: int = 50
    use_repetition_penalty: bool = True
    repetition_penalty: float = 1.2
 
    stop_on_eos: bool = True



@dataclass
class DeployConfig:
    
    BUCKET_NAME = 'gs://transformer-garage' 
    LOCATION     = 'us-central1'
    COTNAINER_URI = 'us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest'
    REQUIREMEMTS = ["datasets", "transformers", "tqdm"]
    NAME = "model-training-job"
    SCRIPT_PATH = "pre_training.py" 
    REPLICA_COUNT = 1
    ACCELERATOR_COUNT = 1 
    MACHINE_TYPE = "g2-standard-4" 


@dataclass
class FineTuneConfig: 

    # # SYSTEM
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # mixed_precision = (device == "cuda")
    # num_workers = 4
    # seed = 1337

    # # MODEL (must match pre-trained checkpoint)
    # block_size = 256
    # n_layer = 12
    # n_embd = 384
    # n_head = 6
    # dropout = 0.05  # Lower dropout for fine-tuning

    # # TRAINING
    # batch_size = 4
    # grad_accum_steps = 8
    # effective_batch_size = batch_size * grad_accum_steps  # 32
    # squad_ratio = 0.9  # prioritize QA; keep a small fluency mix

    # # Learning rate (lower for fine-tuning)
    # max_lr = 6e-6
    # min_lr = 1e-5
    # warmup_iters = 500
    # lr_decay_iters = 10000
    # max_iters = 10000

    # # Evaluation
    # eval_iters = 100
    # eval_interval = 1000

    # # Checkpointing
    # checkpoint_interval = 5000
    # checkpoint_dir = "./finetuned"
    # checkpoint_prefix = "mid-train"

    # # Pre-trained checkpoint path
    # pretrained_checkpoint = "./checkpoints/model-4P_step50000.pt" 


    # MID-TRAIN 

    # SYSTEM
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # mixed_precision = (device == "cuda")
    # num_workers = 4
    # seed = 1337

    # # MODEL (must match pretrained)
    # block_size = 256
    # n_layer = 12
    # n_embd = 384
    # n_head = 6
    # dropout = 0.1   # slightly higher to avoid overfitting

    # # TRAINING
    # batch_size = 4
    # grad_accum_steps = 8
    # effective_batch_size = 32

    # # Learning rate (still learning representations)
    # max_lr = 2e-4
    # min_lr = 5e-5
    # warmup_iters = 500
    # lr_decay_iters = 2000
    # max_iters = 2000    

    # # Evaluation
    # eval_iters = 50
    # eval_interval = 500

    # # Checkpointing
    # checkpoint_interval = 1000
    # checkpoint_dir = "./finetuned"
    # checkpoint_prefix = "midtrain"

    # # Base checkpoint
    # pretrained_checkpoint = "./checkpoints/model-4P_step50000.pt"



    # SFT TRAIN 
       # SYSTEM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = (device == "cuda")
    num_workers = 4
    seed = 1337

    # MODEL (same as before)
    block_size = 256
    n_layer = 12
    n_embd = 384
    n_head = 6
    dropout = 0.05   # lower: don't inject noise into policy

    # TRAINING
    batch_size = 4
    grad_accum_steps = 8
    effective_batch_size = 32

    # Learning rate (VERY LOW)
    max_lr = 1e-5
    min_lr = 5e-6
    warmup_iters = 200
    lr_decay_iters = 2000
    max_iters = 2000   # 🔥 STOP EARLY

    # Evaluation
    eval_iters = 50
    eval_interval = 100

    # Checkpointing
    checkpoint_interval = 1000
    checkpoint_dir = "./finetuned"
    checkpoint_prefix = "sft"

    # Base checkpoint (MID-TRAINED!)
    pretrained_checkpoint = "./finetuned/midtrain/midtrain_final_0002000.pt"