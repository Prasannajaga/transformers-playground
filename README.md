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

## Configuration

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

## CLI Scripts

### Model Deployment

Deploy trained models to HuggingFace Hub:

```bash
# Basic deployment
python src/scripts/hf_deploy.py \
  --model_path ./checkpoints/ckpt_step_30000.pt \
  --config_path ./checkpoints/config.json \
  --repo_id username/my-model

# Private deployment with custom tokenizer
python src/scripts/hf_deploy.py \
  --model_path ./checkpoints/ckpt_step_30000.pt \
  --config_path ./checkpoints/config.json \
  --repo_id username/my-model \
  --tokenizer_path ./tokenizer \
  --private
```

### Dataset Deployment

Deploy datasets to HuggingFace Hub:

```bash
# Auto-detect format
python src/scripts/dataset-deploy.py \
  --dataset-path ./my_dataset \
  --repo-id username/my-dataset

# With metadata
python src/scripts/dataset-deploy.py \
  --dataset-path ./data.csv \
  --repo-id username/csv-dataset \
  --description "My dataset description" \
  --license "mit" \
  --tags nlp classification

# Update existing dataset (preserve README)
python src/scripts/dataset-deploy.py \
  --dataset-path ./updated_data \
  --repo-id username/my-dataset \
  --preserve-card
```

### Legacy Checkpoint Conversion

Convert old checkpoint formats to HuggingFace:

```bash
# Basic conversion
python src/scripts/hf-old-conversion.py \
  --model_path ./old_checkpoints/model.pt \
  --repo_id username/converted-model \
  --config_path ./config.json

# With architecture overrides
python src/scripts/hf-old-conversion.py \
  --model_path ./old_checkpoints/model.pt \
  --repo_id username/converted-model \
  --n_layer 12 \
  --n_embd 768 \
  --attention MQA
```

## Project Structure

```text
src/
├── config/          # Centralized configuration (TrainingConfig, DeployConfig)
├── templates/       # Model card templates for HuggingFace
├── scripts/         # CLI tools (hf_deploy.py, dataset-deploy.py)
├── pretrain/        # Training and inference engines
├── customTransformers/  # Custom transformer architectures
├── attention/       # Attention mechanisms (MHA, GQA, MQA)
├── FFN/             # Feed-forward network variants
├── models/          # Model-specific training scripts
└── services/        # Cloud storage services
```
