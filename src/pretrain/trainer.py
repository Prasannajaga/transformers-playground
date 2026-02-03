"""
Production-quality training wrapper for a Transformer model. 
* Responsible for optimizer, amp, accumulation, clipping, checkpointing, logging, LR schedule
* Optimized for low VRAM (6GB) with gradient checkpointing, configurable AMP dtype, and memory management
""" 

import os
import gc
import time
import math
import sys
import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Any, Dict 
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from config.train import TrainingConfig

# Supported optimizers for dynamic selection
SUPPORTED_OPTIMIZERS = frozenset({"adamw", "adam", "sgd", "adafactor"})


class Trainer:
    """Production-grade trainer with tqdm progress bars and clean logging."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        tokenizer: Optional[Any] = None,
        ckpt_dir: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.tokenizer = tokenizer
        # checkpoint directory precedence: explicit arg > config.ckpt_dir
        self.ckpt_dir = ckpt_dir or config.ckpt_dir
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # device placement
        self.model.to(self.device)

        # AMP setup: disable on CPU automatically
        self.use_amp = bool(self.config.use_amp and self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # Gradient checkpointing for memory efficiency
        # self._setup_gradient_checkpointing()

        # Optimizer with parameter groups (no weight decay on bias & LayerNorm)
        self.optimizer = self._create_optimizer()

        # lr schedule state (manual cosine + warmup)
        if self.config.total_steps is None:
            # total_steps unknown: leave as None; trainer.train will attempt to infer
            self.total_steps = None
        else:
            self.total_steps = int(self.config.total_steps)
        self.warmup_steps = int(self.config.warmup_steps or 0)
        # store base lrs to scale
        self._base_lrs = [g.get("lr", self.config.lr) for g in self.optimizer.param_groups]
        # ensure initial lr matches config.lr if not set in param groups
        for g in self.optimizer.param_groups:
            if "lr" not in g or g["lr"] is None:
                g["lr"] = self.config.lr

        # bookkeeping
        self.global_step = 0  # increments after each optimizer.step()
        self._accum_counter = 0  

        # timer
        self._train_start_time = None
        self._last_step_time = None  

        # Loss tracking for metadata
        self._best_loss = float("inf")
        self._worst_loss = float("-inf")
        self._final_loss = None
        self._best_loss_step = 0
        self._total_tokens_trained = 0
        self._total_training_time = 0.0

        # CUDA optimizations
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._clear_cuda_cache()
        
        # Log initialization info
        if self.config.enable_logging:
            self._log_init_info()

    # -------------------------------------------------------------------------
    # Logging Helpers (Clean, non-repetitive)
    # -------------------------------------------------------------------------
    def _log(self, message: str):
        """Simple log output to stderr to not interfere with tqdm."""
        if self.config.enable_logging:
            tqdm.write(message, file=sys.stderr)
    
    def _log_init_info(self):
        """Log model and memory info at initialization."""
        params = self.parameter_counts()
        self._log(f"[Trainer] Model: {params['trainable_params_m']:.2f}M trainable / {params['total_params_m']:.2f}M total ({params['trainable_percent']:.1f}%)")
        self._log(f"[Trainer] Optimizer: {self.config.optimizer.upper()} | LR: {self.config.lr:.2e} | Weight Decay: {self.config.weight_decay}")
        if self.device.type == "cuda":
            mem = self.get_memory_summary()
            self._log(f"[Trainer] GPU Memory: {mem['allocated_gb']:.2f}GB allocated | Device: {self.device}")

    # -------------------------------------------------------------------------
    # VRAM Optimization Helpers
    # -------------------------------------------------------------------------
    def _resolve_amp_dtype(self) -> torch.dtype:
        """Resolve AMP dtype from config string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype_str = self.config.amp_dtype.lower()
        if dtype_str not in dtype_map:
            self._log(f"[Trainer] Warning: Unknown amp_dtype '{self.config.amp_dtype}', defaulting to float16")
            return torch.float16
        
        requested_dtype = dtype_map[dtype_str]
        
        # Validate bfloat16 support
        if requested_dtype == torch.bfloat16 and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                self._log("[Trainer] Warning: bfloat16 not supported, falling back to float16")
                return torch.float16
        
        return requested_dtype
    
    def _setup_gradient_checkpointing(self):
        """Enable gradient checkpointing on the model if supported."""
        if not self.config.enable_gradient_checkpointing:
            return
        
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            self._log("[Trainer] Enabled gradient checkpointing")
            return
        
        checkpointed_count = 0
        for module in self.model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
                checkpointed_count += 1
        
        if checkpointed_count > 0:
            self._log(f"[Trainer] Enabled gradient checkpointing on {checkpointed_count} modules")
        else:
            self._log("[Trainer] Warning: Gradient checkpointing not supported by model")
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache to reduce memory fragmentation."""
        if self.device.type != "cuda":
            return
        gc.collect()
        torch.cuda.empty_cache()
    
    def reset_peak_memory_stats(self):
        """Reset peak memory tracking for fresh measurement."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_memory_summary(self) -> Dict[str, float]:
        """Get current memory usage summary in GB."""
        if self.device.type != "cuda":
            return {"device": "cpu", "allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
        
        return {
            "device": str(self.device),
            "allocated_gb": round(torch.cuda.memory_allocated(self.device) / (1024 ** 3), 3),
            "reserved_gb": round(torch.cuda.memory_reserved(self.device) / (1024 ** 3), 3),
            "peak_gb": round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 3), 3),
        }

    # -------------------------------------------------------------------------
    # Initialization helpers 
    # -------------------------------------------------------------------------
    def _create_optimizer(self):
        """Create optimizer based on config. Supports: adamw, adam, sgd, adafactor."""
        optimizer_name = getattr(self.config, 'optimizer', 'adamw').lower()
        
        if optimizer_name not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: '{self.config.optimizer}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_OPTIMIZERS))}"
            )
        
        # Exclude bias and LayerNorm/Norm weights from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "layernorm" in name.lower() or "layer_norm" in name.lower() or "ln_" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Create optimizer based on config
        if optimizer_name == "adamw":
            return AdamW(param_groups, lr=self.config.lr, betas=self.config.betas, eps=self.config.eps)
        elif optimizer_name == "adam":
            return torch.optim.Adam(param_groups, lr=self.config.lr, betas=self.config.betas, eps=self.config.eps)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(param_groups, lr=self.config.lr, momentum=0.9)
        elif optimizer_name == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(param_groups, lr=self.config.lr, relative_step=False, warmup_init=False)
            except ImportError:
                raise ImportError(
                    "Adafactor requires the 'transformers' library. "
                    "Install with: pip install transformers"
                )
 
    # -------------------------------------------------------------------------
    # Learning rate schedule 
    # -------------------------------------------------------------------------
    def _get_lr_factor(self, step: int) -> float: 
        if self.total_steps is None or self.total_steps <= 0:
            if self.warmup_steps > 0:
                return min(1.0, float(step) / float(max(1, self.warmup_steps)))
            return 1.0

        step = min(step, self.total_steps)
        if step < self.warmup_steps and self.warmup_steps > 0:
            return float(step) / float(max(1, self.warmup_steps))
        
        if self.total_steps == self.warmup_steps:
            return 1.0
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _update_lr(self): 
        factor = self._get_lr_factor(self.global_step)
        for base, group in zip(self._base_lrs, self.optimizer.param_groups):
            group["lr"] = base * factor
 
    # -------------------------------------------------------------------------
    # Utilities 
    # -------------------------------------------------------------------------
    def parameter_counts(self) -> Dict[str, Any]:
        """Return parameter counts with human-readable formatting in millions."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = (trainable / total * 100.0) if total > 0 else 0.0
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_percent": pct,
            "total_params_m": total / 1e6,
            "trainable_params_m": trainable / 1e6,
        }

    # -------------------------------------------------------------------------
    # Metadata Generation (Production Debug)
    # -------------------------------------------------------------------------
    def _generate_training_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """Generate comprehensive training metadata for debugging."""
        params = self.parameter_counts()
        mem = self.get_memory_summary()
        
        # Calculate avg tokens per second
        avg_tokens_per_sec = 0.0
        if self._total_training_time > 0 and self._total_tokens_trained > 0:
            avg_tokens_per_sec = self._total_tokens_trained / self._total_training_time
        
        metadata = {
            "model_info": {
                "model_name": self.model.__class__.__name__,
                "checkpoint_path": checkpoint_path,
                "save_timestamp": datetime.now().isoformat(),
            },
            "training_statistics": {
                "total_parameters": params["total_params"],
                "trainable_parameters": params["trainable_params"],
                "total_trained_tokens": self._total_tokens_trained,
                "total_training_time_seconds": round(self._total_training_time, 2),
                "avg_tokens_per_second": round(avg_tokens_per_sec, 2),
            },
            "loss_statistics": {
                "best_loss": self._best_loss if self._best_loss != float("inf") else None,
                "worst_loss": self._worst_loss if self._worst_loss != float("-inf") else None,
                "final_loss": self._final_loss,
                "step_at_best_loss": self._best_loss_step,
            },
            "system_diagnostics": {
                "peak_gpu_memory_allocated_gb": mem.get("peak_gb", 0),
                "peak_gpu_memory_reserved_gb": mem.get("reserved_gb", 0),
                "device_type": str(self.device),
                "num_gpus": torch.cuda.device_count() if self.device.type == "cuda" else 0,
            },
            "optimizer_config": {
                "optimizer_name": getattr(self.config, 'optimizer', 'adamw'),
                "learning_rate": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "gradient_accumulation_steps": self.config.grad_accum_steps,
                "batch_size": self.config.train_batch_size,
                "effective_batch_size": self.config.train_batch_size * self.config.grad_accum_steps,
            },
        }
        return metadata

    def _save_metadata_atomically(self, metadata: Dict[str, Any], path: str):
        """Write metadata atomically (write to temp, then rename)."""
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Write to temp file first, then atomically rename
        fd, temp_path = tempfile.mkstemp(suffix='.json', dir=dir_path)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(metadata, f, indent=2)
            os.replace(temp_path, path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _save_checkpoint_metadata(self, checkpoint_path: str, checkpoint_name: str):
        """Save metadata JSON file alongside checkpoint to src/logs/."""
        # Determine metadata path in src/logs/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(os.path.dirname(script_dir), "logs")
        
        # Extract checkpoint name without extension and add .json
        base_name = os.path.splitext(checkpoint_name)[0]
        metadata_path = os.path.join(logs_dir, f"{base_name}.json")
        
        try:
            metadata = self._generate_training_metadata(checkpoint_path)
            self._save_metadata_atomically(metadata, metadata_path)
            self._log(f"[Trainer] Saved metadata: {metadata_path}")
        except Exception as e:
            self._log(f"[Trainer] Warning: Failed to save metadata: {e}")
 
    # -------------------------------------------------------------------------
    # Checkpointing 
    # -------------------------------------------------------------------------
    def save_checkpoint(self, step: Optional[int] = None, prefix: str = "ckpt") -> str:
        step = self.global_step if step is None else int(step)
        if not self.ckpt_dir:
            raise ValueError("Checkpoint directory not configured.")
        fname = f"{prefix}_step_{step:07d}.pt"
        path = os.path.join(self.ckpt_dir, fname)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "training_step": step,
            "config": asdict(self.config),
        }
        if self.config.save_optimizer_state:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scaler is not None:
            try:
                payload["scaler_state_dict"] = self.scaler.state_dict()
            except Exception:
                payload["scaler_state_dict"] = None

        torch.save(payload, path)
        self._log(f"[Trainer] Saved checkpoint: {path}")
        
        # Upload to Google Drive if configured (non-blocking on failure)
        if self.config.upload_to_drive:
            self._upload_checkpoint_to_drive(path)

        # Upload to GCS if configured (non-blocking on failure)
        if self.config.upload_to_gcs:
            self._upload_checkpoint_to_gcs(path)
        
        return path
    
    def _upload_checkpoint_to_drive(self, local_path: str) -> None: 
        try:
            from services.cloud_storage import upload_to_gdrive
            
            gdrive_path = upload_to_gdrive(
                local_path=local_path,
                folder_id=self.config.gdrive_folder_id,
            )
            self._log(f"[Trainer] Uploaded to Google Drive: {gdrive_path}")
        except Exception as e:
            self._log(f"[Trainer] Warning: Google Drive upload failed: {e}")
    
    def _upload_checkpoint_to_gcs(self, local_path: str) -> Optional[str]: 
        try:
            from services.cloud_storage import upload_to_gcs
            
            gcs_uri = upload_to_gcs(
                local_path=local_path,
                bucket_name=self.config.gcs_bucket_name,
                destination_blob=self.config.gcs_destination_blob,
            )
            self._log(f"[Trainer] Uploaded to GCS: {gcs_uri}")
            return gcs_uri
        except Exception as e:
            self._log(f"[Trainer] Warning: GCS upload failed: {e}")
            return None

    def load_checkpoint(self, path: str, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
        if map_location is None:
            map_location = self.device
        data = torch.load(path, map_location=map_location)
        self.model.load_state_dict(data["model_state_dict"], strict=True)
        if "optimizer_state_dict" in data and data["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])
            except Exception as e:
                self._log(f"[Trainer] Warning: Failed to load optimizer state: {e}")
        if "scaler_state_dict" in data and data["scaler_state_dict"] is not None:
            try:
                self.scaler.load_state_dict(data["scaler_state_dict"])
            except Exception:
                pass
        self.global_step = int(data.get("training_step", 0))
        saved_cfg = data.get("config", None)
        if saved_cfg and self.total_steps is None:
            self.total_steps = int(saved_cfg.get("total_steps")) if saved_cfg.get("total_steps") else None
            self.warmup_steps = int(saved_cfg.get("warmup_steps", self.warmup_steps))
        self._update_lr()
        self.model.to(self.device)
        self._log(f"[Trainer] Loaded checkpoint from {path} (step {self.global_step})")
        return data
 
    # -------------------------------------------------------------------------
    # Validation loss estimation  
    # -------------------------------------------------------------------------
    def estimate_validation_loss(self, val_dataloader, num_batches: Optional[int] = None) -> float: 
        """Memory-efficient validation loss estimation."""
        self.model.eval()
        num_batches = int(num_batches or self.config.validation_batch_count)
        total_loss = 0.0
        seen = 0 
        
        self._clear_cuda_cache()
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if seen >= num_batches:
                    break
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.amp.autocast(enabled=self.use_amp, device_type=self.config.device, dtype=self.amp_dtype):
                    model_out = self.model(inputs)
                    loss = self._compute_loss(model_out, targets)
                total_loss += loss.detach()
                del inputs, targets, model_out, loss
                seen += 1
        
        self._clear_cuda_cache()
        self.model.train()
        return (total_loss / max(1, seen)).item()

    # -------------------------------------------------------------------------
    # Core training primitives 
    # -------------------------------------------------------------------------
    def _unpack_batch(self, batch):
        """Standardize dataloader batch -> (inputs, targets)."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 1:
                return batch[0], batch[0]
            return batch[0], batch[1]
        if isinstance(batch, dict):
            inp = batch.get("input_ids") or batch.get("inputs") or batch.get("input")
            tgt = batch.get("labels") or batch.get("targets") or batch.get("targets_ids")
            if inp is None:
                raise ValueError("Dict batch missing input_ids/inputs key")
            if tgt is None:
                tgt = inp
            return inp, tgt
        return batch, batch

    def _compute_loss(self, logits, targets):  

        if isinstance(logits, dict) and "loss" in logits:
            return logits["loss"]
        if isinstance(logits, (tuple, list)):
            logits_tensor = logits[0]
        else:
            logits_tensor = logits

        B, T, V = logits_tensor.shape
        loss = nn.functional.cross_entropy(
            logits_tensor.reshape(-1, V),
            targets.reshape(-1),
            # ignore_index=-100,
            # reduction="mean",
        )
        return loss

    def train_step(self, batch) -> Dict[str, Any]:
        """Perform forward + backward for a single batch."""
        self.model.train()
        inputs, targets = self._unpack_batch(batch)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        accum_steps = max(1, int(self.config.grad_accum_steps)) 

        with torch.amp.autocast(enabled=self.use_amp, device_type=self.config.device, dtype=self.amp_dtype):
            model_out = self.model(inputs)
            loss = self._compute_loss(model_out, targets)

        scaled_loss = loss / accum_steps
        self.scaler.scale(scaled_loss).backward() 
        self._accum_counter += 1

        did_step = False
        grad_norm = None
        current_lr = None

        if self._accum_counter >= accum_steps:
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception:
                pass

            if self.config.grad_clip_norm and self.config.grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                grad_norm = float(grad_norm) if hasattr(grad_norm, "item") else float(grad_norm)
            else: 
                grad_norm = None

            try:
                self.scaler.step(self.optimizer)
            except Exception as e:
                self.scaler.update()
                raise e
            else:
                self.scaler.update()
                did_step = True
                self.global_step += 1
                self._update_lr()
               
            finally:
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_counter = 0 
                current_lr = float(self.optimizer.param_groups[0]["lr"])

        return {
            "loss": loss.item(),
            "did_step": did_step,
            "grad_norm": grad_norm,
            "lr": current_lr,
        }
 
    # -------------------------------------------------------------------------
    # Training loop with tqdm
    # -------------------------------------------------------------------------
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: Optional[int] = 1,
        max_steps: Optional[int] = None,
    ):
        """
        High-level training loop with tqdm progress bar.
        - train_dataloader: iterable of batches
        - val_dataloader: optional validation dataloader for periodic eval
        - epochs: number of epochs to run if total_steps not specified
        - max_steps: optional override for total optimizer steps
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        # Infer total_steps
        if max_steps is not None:
            self.total_steps = int(max_steps)
        elif self.total_steps is None:
            try:
                steps_per_epoch = len(train_dataloader)
            except Exception:
                steps_per_epoch = None
            if steps_per_epoch:
                approx_total = int(math.ceil(steps_per_epoch * float(epochs) / max(1, self.config.grad_accum_steps)))
                self.total_steps = approx_total

        self._train_start_time = time.perf_counter()
        self._last_step_time = self._train_start_time
        self.model.train()

        global_step_target = None if self.total_steps is None else int(self.total_steps)
        
        # Create progress bar
        pbar = tqdm(
            total=global_step_target,
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
        
        # Tracking for smoother progress bar updates
        running_loss = 0.0
        loss_count = 0
        last_val_loss = None

        stop_requested = False
        epoch = 0

        try:
            while epoch < epochs and not stop_requested:
                epoch += 1
                for batch in train_dataloader:
                    step_info = self.train_step(batch)
                    
                    # Accumulate loss for averaging
                    running_loss += step_info["loss"]
                    loss_count += 1
                    
                    # Track loss statistics for metadata
                    current_loss = step_info["loss"]
                    self._final_loss = current_loss
                    if current_loss < self._best_loss:
                        self._best_loss = current_loss
                        self._best_loss_step = self.global_step
                    if current_loss > self._worst_loss:
                        self._worst_loss = current_loss
                    
                    # Track tokens trained (batch_size * sequence_length)
                    # Extract from batch since inputs is scoped inside train_step
                    if isinstance(batch, (list, tuple)):
                        batch_tensor = batch[0]
                    elif isinstance(batch, dict):
                        batch_tensor = batch.get("input_ids") or batch.get("inputs") or batch.get("input")
                    else:
                        batch_tensor = batch
                    if batch_tensor is not None and hasattr(batch_tensor, 'size'):
                        batch_size = batch_tensor.size(0)
                        seq_length = batch_tensor.size(1) if batch_tensor.dim() > 1 else self.config.block_size
                        self._total_tokens_trained += batch_size * seq_length

                    if step_info["did_step"]:
                        # Update progress bar
                        avg_loss = running_loss / loss_count
                        
                        # Build postfix dict
                        postfix = {
                            "train_loss": f"{avg_loss:.4f}",
                            "lr": f"{step_info['lr']:.2e}",
                        }
                        if step_info["grad_norm"] is not None:
                            postfix["gnorm"] = f"{step_info['grad_norm']:.2f}"
                        if last_val_loss is not None:
                            postfix["val"] = f"{last_val_loss:.4f}"
                        if self.config.log_memory_usage and self.device.type == "cuda":
                            mem = self.get_memory_summary()
                            postfix["mem"] = f"{mem['allocated_gb']:.1f}GB"
                        
                        pbar.set_postfix(postfix)
                        pbar.update(1)
                        
                        # Reset running loss periodically
                        if self.global_step % 100 == 0:
                            running_loss = 0.0
                            loss_count = 0

                        # Checkpointing
                        if self.ckpt_dir and self.config.ckpt_interval_steps:
                            if self.global_step % int(self.config.ckpt_interval_steps) == 0:
                                try:
                                    self.save_checkpoint(step=self.global_step)
                                except Exception as e:
                                    self._log(f"[Trainer] Warning: Checkpoint save failed: {e}")

                        # Validation
                        if val_dataloader is not None and self.config.validation_batch_count:
                            if self.global_step % max(1, self.config.log_interval_steps) == 0:
                                try:
                                    last_val_loss = self.estimate_validation_loss(
                                        val_dataloader, 
                                        num_batches=self.config.validation_batch_count
                                    )
                                    self._log(f"[Trainer] Step {self.global_step}: val_loss={last_val_loss:.4f}")
                                except Exception as e:
                                    self._log(f"[Trainer] Warning: Validation failed: {e}")

                    # Stopping condition
                    if global_step_target is not None and self.global_step >= global_step_target:
                        stop_requested = True
                        break
                
                # Periodic memory cleanup 
                self._clear_cuda_cache()

                if stop_requested:
                    break

        finally:
            pbar.close()

        total_wall = time.perf_counter() - self._train_start_time
        self._total_training_time = total_wall
        
        # Final summary
        self._log(f"[Trainer] Training complete: {self.global_step} steps in {total_wall:.1f}s ({total_wall/60:.1f}min)")
        if self.device.type == "cuda":
            mem = self.get_memory_summary()
            self._log(f"[Trainer] Peak GPU memory: {mem['peak_gb']:.2f}GB")
        
        # Final checkpoint with metadata
        if self.ckpt_dir:
            try:
                ckpt_name = f"final_ckpt_step_{self.global_step:07d}.pt"
                ckpt_path = self.save_checkpoint(step=self.global_step, prefix="final_ckpt")
                # Generate and save metadata for final checkpoint
                self._save_checkpoint_metadata(ckpt_path, ckpt_name)
            except Exception as e:
                self._log(f"[Trainer] Warning: Final checkpoint save failed: {e}")
 
    # -------------------------------------------------------------------------
    # State dict for manual save/load
    # -------------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]: 
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": getattr(self.scaler, "state_dict", lambda: None)(),
            "global_step": self.global_step,
            "config": asdict(self.config),
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]):  
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state and state["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as e:
                self._log(f"[Trainer] Warning: optimizer state load failed: {e}")
        if "scaler_state_dict" in state and state["scaler_state_dict"] is not None:
            try:
                self.scaler.load_state_dict(state["scaler_state_dict"])
            except Exception:
                pass
        self.global_step = int(state.get("global_step", 0))
        self._update_lr()
