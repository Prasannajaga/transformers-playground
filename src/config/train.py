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

    # Runtime
    use_amp: bool = True
    amp_dtype: str = "float16"   # or "float16"

    stop_on_eos: bool = True

    # Cloud Storage (Vertex AI / Google Drive)
    upload_to_drive: bool = False  # True: Google Drive, False: skip Drive upload
    upload_to_gcs: bool = False  # True: Google Cloud Storage, False: skip GCS upload
    gcs_bucket_name: Optional[str] = None  # GCS bucket name (optional) 
    gcs_destination_blob: Optional[str] = None  # GCS blob name (optional)
    gdrive_folder_id: Optional[str] = None  # Google Drive folder ID (optional)


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