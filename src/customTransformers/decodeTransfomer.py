import torch
import torch.nn as nn
import torch.nn.functional as F 
from customTransformers.blocks import MQA_BLOCK, MHA_BLOCK

class DecodeTransformer(nn.Module):

    def __init__(self, num_layers, n_emb, n_head, vocab_size, block_size , dropout=0.1, attention="MHA",ffn_type='relu'):
        super().__init__() 

        self.token_emb = nn.Embedding(vocab_size, n_emb)
        self.position_emb = nn.Embedding(block_size, n_emb)
        self.drop = nn.Dropout(dropout)
        self.attention = attention 
        self.transformer_blocks = nn.ModuleList([
            MHA_BLOCK(n_embd=n_emb, n_head=n_head, ffn_type=ffn_type)
            if self.attention == "MHA"
            else MQA_BLOCK(n_embd=n_emb, n_head=n_head, ffn_type=ffn_type) 
            for _ in range(num_layers)
        ])

        # Init weights
        self.apply(self._init_weights)

        self.final_norm = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self.lm_head.weight = self.token_emb.weight 
 

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
    
    # resize the embedding table after adding the special tokens 
    def resize_token_embeddings(self, new_vocab_size):
        old_vocab_size, n_emb = self.token_emb.weight.shape

        if new_vocab_size == old_vocab_size:
            return

        #   Resize token embedding 
        new_token_emb = nn.Embedding(new_vocab_size, n_emb)

        # init new weights
        torch.nn.init.normal_(new_token_emb.weight, mean=0.0, std=0.02)

        # copy old weights
        num_to_copy = min(old_vocab_size, new_vocab_size)
        new_token_emb.weight.data[:num_to_copy] = self.token_emb.weight.data[:num_to_copy]

        self.token_emb = new_token_emb

        # Resize LM head (tied weights) 
        new_lm_head = nn.Linear(n_emb, new_vocab_size, bias=False)

        # copy old weights
        new_lm_head.weight.data[:num_to_copy] = self.lm_head.weight.data[:num_to_copy]

        self.lm_head = new_lm_head

        # re-tie weights
        self.lm_head.weight = self.token_emb.weight

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=None, top_p=None, repetition_penalty=1.2, eos_token_id=None): 

        self.eval()
        block_size = self.position_emb.num_embeddings
        
        
        for _ in range(max_new_tokens): 
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:] 
            logits, _ = self.forward(idx_cond) 
            logits = logits[:, -1, :]  # (B, V)
             
            if temperature == 0.0:
                # Greedy sampling
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            else: 
                logits = logits / temperature

                # Apply repetition penalty (optimized vectorized version)
                if repetition_penalty != 1.0:
                    # Get unique tokens and their positions
                    unique_tokens = torch.unique(idx) 
                    logits[:, unique_tokens] = logits[:, unique_tokens] / repetition_penalty
                
                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # # Optional top-p (nucleus) filtering
                # if top_p is not None and top_p < 1.0:
                #     sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                #     # Remove tokens with cumulative probability above the threshold
                #     sorted_indices_to_remove = cumulative_probs > top_p
                #     # Shift the indices to the right to keep the first token above threshold
                #     sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                #     sorted_indices_to_remove[:, 0] = False
                    
                #     # Scatter back to original indexing
                #     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                #     logits[indices_to_remove] = -float('Inf')
                 
                probs = F.softmax(logits, dim=-1)  # (B, V) 
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

                if eos_token_id is not None:
                    if idx_next == eos_token_id:
                        break
             
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        return idx