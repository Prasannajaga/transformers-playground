from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelCardTemplate:
    """Template configuration for model card generation."""
    
    # Repository Information
    repo_id: str
    github_repo: str = "https://github.com/Prasannajaga/transformers-playground"
    
    # Model Metadata
    model_name: str = ""
    language: str = "en"
    license: str = "mit"
    tags: List[str] = field(default_factory=lambda: [
        "pytorch",
        "causal-lm", 
        "custom-architecture",
        "transformers-playground"
    ])
    
    # Architecture Details
    num_layers: int = 6
    hidden_size: int = 384
    num_attention_heads: int = 6
    vocab_size: int = 50257
    max_sequence_length: int = 256
    attention_type: str = "MHA"
    ffn_type: str = "relu"
    dropout: float = 0.1
    
    # Training Details
    training_steps: Optional[int] = None
    training_dataset: Optional[str] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    
    # Config Hash
    config_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.model_name:
            self.model_name = self.repo_id.split('/')[-1] if '/' in self.repo_id else self.repo_id


def generate_model_card(
    template: ModelCardTemplate,
    custom_sections: Optional[Dict[str, str]] = None
) -> str: 


    tags_yaml = "\n".join(f"  - {tag}" for tag in template.tags)
    
    # Build training metrics section
    training_metrics = ""
    if any([template.training_steps, template.training_dataset, 
            template.training_loss, template.validation_loss]):
        training_metrics = """
        ## Training Details

        | Metric | Value |
        |--------|-------|"""

        if template.training_dataset:
            training_metrics += f"\n| Dataset | {template.training_dataset} |"
        if template.training_steps:
            training_metrics += f"\n| Training Steps | {template.training_steps:,} |"
        if template.training_loss is not None:
            training_metrics += f"\n| Final Training Loss | {template.training_loss:.4f} |"
        if template.validation_loss is not None:
            training_metrics += f"\n| Final Validation Loss | {template.validation_loss:.4f} |"
        training_metrics += "\n"
    
    # Build custom sections
    custom_content = ""
    if custom_sections:
        for section_name, section_text in custom_sections.items():
            custom_content += f"\n## {section_name}\n\n{section_text}\n"
    
    card = f"""---
    language: {template.language}
    license: {template.license}
    tags:
    {tags_yaml}
    library_name: transformers
    ---

    # {template.model_name}

    Custom PyTorch language model deployed via [`transformers-playground`]({template.github_repo}).

    ## Model Details

    | Parameter | Value |
    |-----------|-------|
    | Layers | {template.num_layers} |
    | Hidden Size | {template.hidden_size} |
    | Attention Heads | {template.num_attention_heads} |
    | Attention Type | {template.attention_type} |
    | FFN Type | {template.ffn_type} |
    | Vocab Size | {template.vocab_size:,} |
    | Max Sequence Length | {template.max_sequence_length} |
    | Dropout | {template.dropout} |
    {training_metrics}
    ## Usage

    ```python
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    import torch

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("{template.repo_id}")

    # Load model weights
    weights = load_file(hf_hub_download("{template.repo_id}", "model.safetensors"))

    # Initialize your model architecture and load weights
    # model.load_state_dict(weights)
    ```

    ## Quick Inference Test

    ```python
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Run inference with your model
    # output = model.generate(input_ids)
    # print(tokenizer.decode(output[0]))
    ```

    ## Repository

    - **GitHub**: [{template.github_repo}]({template.github_repo})
    - **Config Hash**: `{template.config_hash or 'N/A'}`
    {custom_content} 
    """
    return card


def generate_model_card_simple(
    repo_id: str,
    config: Any,
    github_repo: str = "https://github.com/Prasannajaga/transformers-playground"
) -> str: 
    template = ModelCardTemplate(
        repo_id=repo_id,
        github_repo=github_repo,
        num_layers=getattr(config, 'num_layers', 6),
        hidden_size=getattr(config, 'n_embd', 384),
        num_attention_heads=getattr(config, 'n_head', 6),
        vocab_size=getattr(config, 'vocab_size', 50257),
        max_sequence_length=getattr(config, 'block_size', 256),
        attention_type=getattr(config, 'attention', 'MHA'),
        ffn_type=getattr(config, 'ffn_type', 'relu'),
        dropout=getattr(config, 'dropout', 0.1),
        config_hash=getattr(config, 'original_config_hash', None),
    )
    return generate_model_card(template)
