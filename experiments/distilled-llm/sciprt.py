
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from typing import List
import os
import time
import gc


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_PATH = "llama_pruned-4/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRUNE_RATIO = 0.25   # remove 25% least important layers
OUTPUT_DIR = "./llama_pruned-4"


def load_model(model_source=MODEL_NAME):
    print("logging model", model_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def compute_layer_importance(model, tokenizer, sample_texts: List[str]):
    model.eval()
    importance_scores = torch.zeros(len(model.model.layers), device=DEVICE)

    for text in sample_texts:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        outputs = model(
            **inputs,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states  # tuple: [emb, layer1, layer2...]

        for i in range(1, len(hidden_states)):  # skip embedding layer
            importance_scores[i - 1] += hidden_states[i].abs().mean()

    importance_scores /= len(sample_texts)
    return importance_scores , outputs


def sync_model_config(model):
    layers = model.model.layers
    if len(layers) == 0:
        return

    model.config.num_hidden_layers = len(layers)

    first_layer = layers[0]
    attn = first_layer.self_attn
    head_dim = attn.head_dim
    num_q_heads = attn.q_proj.weight.shape[0] // head_dim
    num_kv_heads = attn.k_proj.weight.shape[0] // head_dim

    model.config.head_dim = int(head_dim)
    model.config.num_attention_heads = int(num_q_heads)
    model.config.num_key_value_heads = int(num_kv_heads)
    model.config.intermediate_size = int(first_layer.mlp.gate_proj.weight.shape[0])


def save_model(model, tokenizer, path):
    sync_model_config(model)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

 
# ## Depth Pruning 

# @title Depth pruning 
def prune_layers(model, importance_scores, prune_ratio):
    total_layers = len(model.model.layers)
    num_prune = int(total_layers * prune_ratio)

    print(f"Total layers: {total_layers}")
    print(f"Pruning {num_prune} layers")

    # Get least important layers
    prune_indices = torch.argsort(importance_scores)[:num_prune]
    keep_indices = sorted(
        list(set(range(total_layers)) - set(prune_indices.tolist()))
    )

    print(f"Keeping layers: {keep_indices}")

    # Replace ModuleList
    model.model.layers = nn.ModuleList(
        [model.model.layers[i] for i in keep_indices]
    )

    # Update config
    model.config.num_hidden_layers = len(keep_indices)

    return model


model, tokenizer = load_model()

# Use small representative sample
sample_texts = [
    "Explain quantum mechanics simply.",
    "Write a Python function to compute Fibonacci.",
    "Describe the benefits of pruning neural networks.",
    "How does transformer attention work?"
]

importance_scores , outputs = compute_layer_importance(
    model, tokenizer, sample_texts
)

print("important scores", importance_scores) 

model = prune_layers(model, importance_scores, PRUNE_RATIO) 
save_model(model, tokenizer, OUTPUT_DIR)

print("Pruning complete. Starting fine-tuning...")

 


def prune_attention_heads_gqa(model, prune_ratio=0.25):

    for layer in model.model.layers:

        attn = layer.self_attn

        num_q_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim

        group_size = num_q_heads // num_kv_heads

        # how many KV groups to prune
        num_groups = num_kv_heads
        prune_groups = int(num_groups * prune_ratio)

        if prune_groups == 0:
            continue

        keep_groups = list(range(prune_groups, num_groups))

        # build head indices
        keep_q_heads = []
        keep_kv_heads = keep_groups

        for g in keep_groups:
            start = g * group_size
            keep_q_heads.extend(range(start, start + group_size))

        # convert to tensor indices
        q_keep_idx = torch.cat([
            torch.arange(h * head_dim, (h + 1) * head_dim)
            for h in keep_q_heads
        ])

        kv_keep_idx = torch.cat([
            torch.arange(h * head_dim, (h + 1) * head_dim)
            for h in keep_kv_heads
        ])

        with torch.no_grad():

            # Q
            attn.q_proj.weight.data = attn.q_proj.weight.data[q_keep_idx]

            # K,V
            attn.k_proj.weight.data = attn.k_proj.weight.data[kv_keep_idx]
            attn.v_proj.weight.data = attn.v_proj.weight.data[kv_keep_idx]

            # O projection column prune
            attn.o_proj.weight.data = attn.o_proj.weight.data[:, q_keep_idx]

            # Update metadata
            new_q_heads = len(keep_q_heads)
            new_kv_heads = len(keep_kv_heads)

            attn.config.num_attention_heads = new_q_heads
            attn.config.num_key_value_heads = new_kv_heads

    torch.cuda.empty_cache()
    return model

def prune_mlp_width(model, prune_ratio=0.3):

    for layer in model.model.layers:

        mlp = layer.mlp
        hidden_dim = mlp.gate_proj.out_features

        num_prune = int(hidden_dim * prune_ratio)
        keep_dim = hidden_dim - num_prune

        if num_prune == 0:
            continue

        with torch.no_grad():

            # Importance proxy (cheap + no forward pass)
            scores = mlp.gate_proj.weight.abs().mean(dim=1)
            keep_idx = torch.topk(scores, keep_dim).indices.sort().values

            # Row prune gate + up
            mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[keep_idx]
            mlp.up_proj.weight.data = mlp.up_proj.weight.data[keep_idx]

            # Column prune down_proj
            mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, keep_idx]

            mlp.gate_proj.out_features = keep_dim
            mlp.up_proj.out_features = keep_dim
            mlp.down_proj.in_features = keep_dim

    torch.cuda.empty_cache()
    return model


def compare_models(orig_model, tokenizer, pruned_model, prompt):

    device = DEVICE
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ---------- ORIGINAL ----------
    # orig_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )

    start = time.time()
    with torch.no_grad():
        out = orig_model.generate(**inputs, max_new_tokens=80)
    orig_time = time.time() - start
    orig_text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Free original model
    del orig_model
    torch.cuda.empty_cache()
    gc.collect()

    # ---------- PRUNED ----------
    start = time.time()
    with torch.no_grad():
        out = pruned_model.generate(**inputs, max_new_tokens=80)
    pruned_time = time.time() - start
    pruned_text = tokenizer.decode(out[0], skip_special_tokens=True)

    print("\n========== RESULTS ==========")
    print(f"Original latency: {orig_time:.3f}s")
    print(f"Pruned latency:   {pruned_time:.3f}s")
    print("\n--- Original Output ---\n")
    print(orig_text)
    print("\n--- Pruned Output ---\n")
    print(pruned_text)


originalModel , _ = load_model()

model, tokenizer = load_model()
 
# Apply pruning in-place
model = prune_attention_heads_gqa(model, prune_ratio=0.60)
model = prune_mlp_width(model, prune_ratio=0.60)
 

OUTPUT_DIR = "./llama_pruned-4"
save_model(model, tokenizer, OUTPUT_DIR)
 
prompt = "Explain how transformers work."
compare_models(
    orig_model=originalModel,
    tokenizer=tokenizer,
    pruned_model=model,
    prompt=prompt
)
