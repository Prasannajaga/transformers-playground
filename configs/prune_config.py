"""
Fixed identifiers and protocol constants for model pruning.

Layer aliases follow the naming convention of transformer architectures
(LLaMA / Mistral / Qwen style) — maps raw parameter paths to human-readable names.
"""

from enum import Enum, unique


@unique
class PruneType(Enum):
    """Supported pruning strategies, ordered best → worst."""
    STRUCTURED = "structured"
    GLOBAL_UNSTRUCTURED = "global_unstructured"
    LAYERWISE = "layerwise"
    RANDOM = "random"


# ─── Layer-name fragment → human-readable alias ──────────────────────────────
# Keys are substrings matched against `param_name`, first match wins.
# Order matters: more specific fragments come first.

LAYER_ALIAS_MAP: dict[str, str] = {
    "self_attn.q_proj.weight":     "Qmatrix",
    "self_attn.k_proj.weight":     "Kmatrix",
    "self_attn.v_proj.weight":     "Vmatrix",
    "self_attn.o_proj.weight":     "Omatrix",
    "self_attn.q_proj.bias":       "Qbias",
    "self_attn.k_proj.bias":       "Kbias",
    "self_attn.v_proj.bias":       "Vbias",
    "self_attn.o_proj.bias":       "Obias",
    "mlp.gate_proj.weight":        "GateWeight",
    "mlp.up_proj.weight":          "UpWeight",
    "mlp.down_proj.weight":        "DownWeight",
    "mlp.gate_proj.bias":          "GateBias",
    "mlp.up_proj.bias":            "UpBias",
    "mlp.down_proj.bias":          "DownBias",
    "input_layernorm.weight":      "InputNorm",
    "post_attention_layernorm.weight": "PostAttnNorm",
    "model.norm.weight":           "FinalNorm",
    "model.embed_tokens.weight":   "EmbedTokens",
    "lm_head.weight":              "LMHead",
}

# Histogram resolution for memory-efficient global magnitude threshold
HISTOGRAM_BINS: int = 10_000

# Display
SEPARATOR: str = "=" * 72
