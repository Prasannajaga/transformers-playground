# transformers-playground

A custom transformer model training and deployment toolkit built for rapid experimentation, efficient fine-tuning, and seamless deployment to Hugging Face Hub.

This repository is centered around one principle:

> Experiment → Fail → Optimize → Repeat

It provides a streamlined wrapper layer on top of modern LLM tooling to accelerate training speed, simplify inference, and reduce engineering overhead during iteration.

## Trained Models

you can find a model I trained from this playground below and their HuggingFace links:

| Model                | README                                     | HuggingFace |
|----------------------|--------------------------------------------|-------------|
| my-model             | [readme](src/models/my-model/readme.md)    |      [link](https://huggingface.co/prasannaJagadesh/my-model)       |
| mini-stories         | [readme](src/models/tiny-stories/readme.md)|      [link](https://huggingface.co/prasannaJagadesh/my-mini-stories)       |

This project was built to:

- Maximize LLM training throughput
- Reduce friction in experimentation cycles
- Standardize model training and evaluation workflows
- Simplify inference and deployment pipelines
- Improve reproducibility across experiments

## CLI Scripts

### Model Deployment

Deploy trained models to HuggingFace Hub:

```bash
python src/scripts/hf_deploy.py \
  --model_path <model-path> \
  --config_path <config-path> \
  --repo_id username/my-model
 
# .pt convert to .safetensors then upload 
python src/scripts/hf_deploy.py \
  --model_path <model-path> \
  --config_path <config-path> \
  --repo_id username/my-model \
  --tokenizer_path <tokenizer-path> \
  --
  
# direct .safetensors upload 
huggingface-cli upload prasannaJagadesh/my-model \
  ./Prasanna-SmolLM-360M-3.1 \
  --repo-type model 
```

### Dataset Deployment

Deploy datasets to HuggingFace Hub:

```bash
python src/scripts/dataset-deploy.py \
  --dataset-path <dataset-path> \
  --repo-id username/my-dataset
 
python src/scripts/dataset-deploy.py \
  --dataset-path <dataset-path> \
  --repo-id username/csv-dataset \
  --description "My dataset description" \
  --license "mit" \
  --tags nlp classification
 
python src/scripts/dataset-deploy.py \
  --dataset-path <dataset-path> \
  --repo-id username/my-dataset \
  --preserve-card
```

### Legacy Checkpoint Conversion

Convert old checkpoint formats to HuggingFace:

```bash
python src/scripts/hf-old-conversion.py \
  --model_path <model-path> \
  --repo_id username/converted-model \
  --config_path <config-path>
 
python src/scripts/hf-old-conversion.py \
  --model_path <model-path> \
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
