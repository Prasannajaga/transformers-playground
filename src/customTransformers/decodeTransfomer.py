import torch
import torch.nn as nn
import torch.nn.functional as F 
from customTransformers.blocks import MQA_BLOCK

class DecodeTransformer(nn.Module):

    def __init__(self, num_layers, n_emb, n_head, vocab_size, block_size , dropout=0.1):
        super().__init__() 

        self.token_emb = nn.Embedding(vocab_size, n_emb)
        self.position_emb = nn.Embedding(block_size, n_emb)
        self.drop = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            MQA_BLOCK(n_embd=n_emb, n_head=n_head)
            for _ in range(num_layers)
        ])

        # Init weights
        self.apply(self._init_weights)

        self.final_norm = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self.lm_head.weight = self.token_emb.weight 

        # Special init for residual scaling
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.position_emb.num_embeddings, (
            f"Sequence length {T} exceeds block_size "
            f"{self.position_emb.num_embeddings}"
        )
        assert idx.max() < self.token_emb.num_embeddings

         # 1. Standard Embedding
        token_emb = self.token_emb(idx) 
        
        # 2. Positional Embedding (Fixed indexing)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        position_emb = self.position_emb(pos)

        x = self.drop(token_emb + position_emb)

        for block in self.transformer_blocks: 
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape

            logits = logits.view(B * T, V)
            targets = targets.view(B * T)

            assert logits.dim() == 2
            assert targets.dim() == 1
            assert logits.size(0) == targets.size(0)

            loss = F.cross_entropy(
                logits,
                targets,
                # ignore_index=-100,
                label_smoothing=0.0
            ) 

        return logits, loss