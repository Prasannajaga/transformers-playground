# Prasanna-SmolLM-360M-3.1

A fine-tuned version of [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) trained to be a personal AI assistant that answers questions about my professional background, projects, and skills.

This model is designed to serve as a personal AI assistant on a portfolio website and it's trained for only training purpose of the finetuning & Reward model and It answers questions specifically about myself and refuses off-topic or inappropriate requests.

## Model Details

| Parameter | Value |
|---|---|
| **Base Model** | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| **Parameters** | 360M |
| **Max Sequence Length** | 1024 |
| **Fine-Tuning Method** | LoRA (via Unsloth) |
| **Merge Method** | `merged_16bit` |
| **GGUF Quantizations** | `q8_0` |

## LoRA Configuration

| Parameter | Value |
|---|---|
| **Rank (r)** | 16 |
| **Alpha** | 32 |
| **Dropout** | 0.05 |
| **Bias** | none |
| **Gradient Checkpointing** | unsloth |

## Training Arguments

| Parameter | Value |
|---|---|
| **Batch Size (per device)** | 8 |
| **Gradient Accumulation Steps** | 2 |
| **Effective Batch Size** | 16 |
| **Epochs** | 3 |
| **Learning Rate** | 2e-4 |
| **Weight Decay** | 0.01 |
| **LR Scheduler** | cosine |
| **Optimizer** | adamw_8bit |
| **Precision** | bf16 (if supported, else fp16) |
| **Packing** | enabled |
| **Dataset Workers** | 2 |

## Dataset

~2K samples curated and reviewed manually, covering:

- Biography & identity
- career & workExp
- technical skills
- tech journey
- contacts & social media
- some Refusal for refuse questions if asked not about me
- NFSW to prevent safety measure

### Format

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills."
        },
        {
            "role": "user",
            "content": "Who is Prasanna?"
        },
        {
            "role": "assistant",
            "content": "Prasanna is a driven Software Engineer based in Chennai, India..."
        }
    ]
}
```

## Usage

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("prasannaJagadesh/Prasanna-SmolLM-360M-3.1")
tokenizer = AutoTokenizer.from_pretrained("prasannaJagadesh/Prasanna-SmolLM-360M-3.1")

messages = [
    {"role": "system", "content": "You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills."},
    {"role": "user", "content": "Tell me about Prasanna."},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
  
## Warning Limitations

- Only knows about myself, not a general-purpose assistant
- Small model (360M params) very limited reasoning depth compared to larger models
- Best suited for CPU inference on constrained environments (4-8 GB RAM)
