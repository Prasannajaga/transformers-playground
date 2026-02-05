# transformers-playground

Custom transformer model training and deployment toolkit for building, training, and deploying language models to HuggingFace Hub.

## Installation

```bash
# Clone the repository
git clone https://github.com/Prasannajaga/transformers-playground.git
cd transformers-playground

# Install dependencies
uv venv .venv
uv sync
```

## Quick Start

### Configuration

```python
from config import TrainingConfig, Presets, ModelArchConfig

# Use default config
config = TrainingConfig()

# Use a preset for your hardware
config = Presets.small_6gb_vram()

# Custom configuration with structured overrides
config = TrainingConfig.create(
    model=ModelArchConfig(n_layer=12, n_embd=768, n_head=12),
)

# Dynamic overrides
config = config.with_overrides(lr=1e-4, total_steps=50_000)
```

### Training

```python
from config import TrainingConfig
from pretrain import Trainer

config = TrainingConfig(
    n_layer=6,
    n_embd=384,
    n_head=6,
    total_steps=30_000,
)

# Initialize and train your model
# trainer = Trainer(model, config, train_loader, val_loader)
# trainer.train()
```

### Deployment to HuggingFace

```bash
python src/scripts/hf_deploy.py \
  --model_path ./checkpoints/ckpt_step_30000.pt \
  --config_path ./checkpoints/config.json \
  --repo_id username/my-model \
  --interactive  # Optional: test inference after deployment
```

### Inference

```python
from pretrain import InferenceEngine
from config import TrainingConfig

# Load model and run inference
engine = InferenceEngine(model, config, device, tokenizer)
engine.load_checkpoint("./checkpoints/ckpt_step_30000.pt")

# Stream generation
for token in engine.stream_generate(input_ids):
    print(token, end="", flush=True)
```

## Project Structure

```
src/
├── config/          # Centralized configuration (TrainingConfig, DeployConfig)
├── templates/       # Model card templates for HuggingFace
├── scripts/         # CLI tools (hf_deploy.py, deploy.py)
├── pretrain/        # Training and inference engines
├── customTransformers/  # Custom transformer architectures
├── attention/       # Attention mechanisms (MHA, GQA, MQA)
├── FFN/             # Feed-forward network variants
├── models/          # Model-specific training scripts
└── services/        # Cloud storage services
```

## Configuration Reference

### Training Presets

| Preset | VRAM | Layers | Hidden | Heads | Block Size |
|--------|------|--------|--------|-------|------------|
| `tiny_stories()` | Any | 6 | 384 | 6 | 256 |
| `small_6gb_vram()` | 6GB | 4 | 256 | 4 | 128 |
| `medium_12gb_vram()` | 12GB | 8 | 512 | 8 | 512 |
| `large_24gb_vram()` | 24GB | 12 | 768 | 12 | 1024 |

### Sub-Configurations

- `SystemConfig` - Device, seed, mixed precision
- `ModelArchConfig` - Layers, dimensions, attention type
- `OptimizerConfig` - Optimizer, learning rate, weight decay
- `VRAMConfig` - Gradient checkpointing, AMP settings
- `InferenceConfig` - Temperature, top-k, repetition penalty

## License

MIT License
