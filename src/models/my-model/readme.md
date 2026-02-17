# Building a Personal LLM: My Journey, Trade-offs, and Final Choice

I’ve started thinking: okay, it’s about time I pre-train or fine-tune a model from scratch about *me*—my journey, my story, and my life.  
Yeah, I think it’s about time.

Initially, I was strongly inclined to pre-train a model from scratch. Then I remembered a paper I read a couple of weeks back from the Hugging Face **SmolLM** blog, which boldly says: *don’t waste compute training from scratch*. That hit hard.

So I thought, okay, let me fine tune a model instead.

Choosing the right model to fine tune turned out to be harder than I expected.

you can find the full progress [here](progress.md)

## dataset

this dataset curated and reviewed by me
I have generate around 2K samples for prevent hallucinating
from categories

- Biography & identity
- career & workExp
- technical skills
- tech journey
- contacts & social media
- some Refusal for refuse questions if asked not about me
- NFSW to prevent safety measure

example format

```json
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
        "content": "Prasanna is a driven Software Engineer based in Chennai, India. He is a college dropout who realized early on that traditional academics weren't for him. Instead, he took the self-taught route, mastering computer science concepts on his own. After building a successful career in Full Stack development, he has now pivoted his focus entirely toward Deep Learning, AI, and exploring the frontiers of AGI."
    }
]
```

## Tokenizer

You have to be extremely careful with the tokenizer.  
This is what defines how your model understands and adapts to prompts.

For example:

- A **LLaMA tokenizer will not work with Qwen**

Here’s a sample of how a prompt looks with a LLaMA-style tokenizer:

```python
"""
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 09 Feb 2026

You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Who is Prasanna?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Prasanna is a software engineer with over 3 years of experience specializing in Machine Learning, Python, and Linux systems.
<|eot_id|>
"""
```

## Optimization

## What is LORA adapter?

Low-rank adaptation helps us avoid retraining all the weights in a pretrained model. In simple terms,
we fine-tune 30% linear weights on data and then merge with original model

Say `d = 4096`, we define `rank` and `alpha`.

First, we freeze the model, then we initialize all Linear weights based on `d × rank`:

- `A = matrix(d × rank)`, `B = matrix(rank × d)`
- Then we scale by `alpha / rank`

```python
self.lora_A = nn.Linear(base.in_features, r, bias=False, device=device, dtype=dtype)
self.lora_B = nn.Linear(r, base.out_features, bias=False, device=device, dtype=dtype)
```

## What is QLORA adapter?

QLoRA is basically a quantized model + LoRA to reduce VRAM usage. During the forward pass alone, we dequantize the model to FP16 or FP32 that's the only difference.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_weight = self._dequant_weight(x.dtype)
    bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
    base = F.linear(x, base_weight, bias)
    return base + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
```
