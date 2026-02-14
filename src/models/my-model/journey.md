# Building a Personal LLM: My Journey, Trade-offs, and Final Choice

I’ve started thinking: okay, it’s about time I pre-train or fine-tune a model from scratch about *me*—my journey, my story, and my life.  
Yeah, I think it’s about time.

Initially, I was strongly inclined to pre-train a model from scratch. Then I remembered a paper I read a couple of weeks back from the Hugging Face **SmolLM** blog, which boldly says: *don’t waste compute training from scratch*. That hit hard.

So I thought, okay—let me fine-tune a model instead.

Choosing the right model to fine-tune turned out to be harder than I expected.

---

## My Goal Clear

I want to create a chatbot **about me** and deploy it on my portfolio.  
That comes with some hard constraints:

1. The model should infer smoothly on **CPU**
2. Model size should be **less than 300 MB**
3. Small enough to fit into **4 GB / 8 GB RAM**
4. Quantized and able to infer using **llama.cpp**
5. Deployed on **Cloud Run**, cost-effective (I’ve got GCP credits)
6. Freeze the base model and use **Unsloth** for LoRA fine-tuning

---

## Choosing the Model

I was very confused about which model to choose.  
I decided to go with **instruction-tuned models**, since they’re not only pre-trained on massive datasets but also fine-tuned for instruction following.

I narrowed it down to three options:

1. `llama-3.1-8b-instruct`
2. `qwen2.5-0.5b-instruct`
3. `smoll-vllm-135M / 360M-instruct`

---

## llama-3.1-8b-instruct

I always wanted to work with LLaMA, so this was my first fine-tuning experiment.

No doubt—it’s a SOTA model. Even with just a few samples, I could reach very high accuracy.

But reality hit hard.

- Model size: **2.47 GB → ~800 MB (4-bit quantized)**
- Accuracy: **~95% with just 10 samples**
- Problem: **Infra cost**

Deploying this on Cloud Functions or Cloud Run would be expensive, especially considering cold starts where the model needs to be loaded on every request.

It was a hard trade-off, but I had to drop LLaMA because of infrastructure constraints.

**Summary**

- 2.47 GB → 800 MB (4-bit)
- ~100% better accuracy
- Too expensive for my use case

---

## qwen2.5-0.5b-instruct

This is a 500 M parameter model.

I thought, okay—this could be the sweet spot.

- Quantized size: **~1 GB → 390 MB (4-bit)**
- Accuracy: **~75% with just 10 samples**

The model is trained on fewer tokens compared to LLaMA, which shows up in performance.

**Summary**

- 1 GB → 390 MB (4-bit)
- ~60% lower accuracy compared to LLaMA
- Still slightly too large for my comfort

---

## smoll-vllm-135M / 360M-instruct

Finally—models that actually fit my requirements.

### 🔹 135M Model

- Size: **270 MB → 144 MB (8-bit)**
- Accuracy: **~40%**

Not great accuracy, but the trade-off was acceptable given the size.

### 🔹 360M Model

- Size: **725 MB → 350 MB (8-bit)**
- Accuracy: **~60%**

This felt like the best balance between size, performance, and cost.

I used **8-bit quantization** for these because they’re small enough to handle it without massive degradation.

---

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

waht is LORA adapter ?

Low rank adaption which help us avoid overtraining the whole weights 
in pretrained model

say d = 4096, 
we define rank & alpha 

first we freeze the model then we initialize all Linear weights based on d x rank 

* A = matrix^d * rank , B = matrix^rank * d
* then we scale by alpha / rank 


self.lora_A = nn.Linear(base.in_features, r, bias=False, device=device, dtype=dtype)
self.lora_B = nn.Linear(r, base.out_features, bias=False, device=device, dtype=dtype)