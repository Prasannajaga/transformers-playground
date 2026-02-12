<!-- I have started think okay it's about time I pre-train or fine tune model from scratch about me, my journey, my story and my life
well I think it's about time

I was strongly to pretrain from scratch then I remember and read paper couple weeks back from hugging face smoll-Llm blog <https://huggingface.co/blog/smollm> which boldy says don't waste compute for training from scratch

I was like okay let me try to fine-tune a model from scratch
choosing model to fine tune was pretty hard to me

## my goal is clear

I have to create chatbout about me and deploy it in my portfolio
so I have some constraints

1. model should infer smoothly on CPU
2. modeul should be less than 300MB
3. model should be small enough to fit in 4GB/ 8GB RAM
4. model should be quantanized and able infer using llama.cpp
5. deploy cotainer cloud RUN cost effective (and I got GCP credits)
6. freeze the base model use unsloth for fine-tuning LORA adapters

## choosing model

I was very confused about which model to choose
i choosed isntruct model because it not only pre-trained on massive dataset but also fine-tuned on instruction following dataset

I have 3 options

1. llama-3.1-8b-instruct
2. qwen2.5-0.5b-instruct
3. smoll-vllm-135M/360M-instruct

## llama-3.1-8b-instruct

I always wanted to work with llama so I choosed this as my first fine-tuning project

I have to say it was SOTA model with 1B instruct model
I can able to reach better accuracy with just few samples but the model was sized 800MB even after 4-bit quantization,
deploying this on CLOUD function would be expensize considering I
have to load model on every request or each cold start

it was hard trade off droping llama because of the infra, it was 95% accuracy with just 10 samples

2.47 GB to 800MB compression
100% better accuracy

## qwen2.5-0.5b-instruct

this is 500M model, it was 1GB even after 4-bit quantization
I was like okay let me try to fine-tune this
I can able to reach 75% accuracy with just 10 samples since it is trained on les token comapred to llama

compression
1GB to 390MB  4bit
60% lower accuracy compared to llama

## smoll-vllm-135M / 360M

Finally the small sized and my requirement model

135M model
compression 270MB to 144MB 8bit  
I can able to reach 40% accuracy but the tradeoff was worth it

360M model
compression 725MB  to 350MB 8bit  
I can able to reach 60% accuracy but the tradeoff was worth it

I did 8 bit on these because these models are small enough to handle 8 bit quantization

## tokenizer

so you have to be careful with tokenizer, this is what defines the how your model will adapt with your prompts

for example llama tokenizer will not work with qwen model
here is some sample how it looks like

```python
 """ 
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 09 Feb 2026

You are Prasanna's AI Assistant. You answer questions about his professional background, projects, and skills.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>Who is Prasanna?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Prasanna is a software engineer with over 3 years of experience specializing in Machine Learning, Python, and Linux systems.
<|eot_id|>
"""

```

## final decision

I choosed smoll-vllm-360M-instruct model because it was 60% accuracy and 350MB size

why llama.cpp & unsloth

llama.cpp is a C++ implementation of LLaMA inference engine
it is very fast and efficient and can run on CPU

can simply run on any device with just sigle file
example:

```bash
llama-server -m ./model.gguf \
--port 8080 \
--ctx-size 4096
```

just simple as that and unsloth supports converting the LORA or QLora adapters to GGUF format which is way easier than I thought

```python
model.save_pretrained_gguf(
    model_path,
    tokenizer,
    quantization_method = "q4_k_m"
)   
```

both llama.cpp & unsloth supports hugginfgface transformers which is cherry on top -->

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
