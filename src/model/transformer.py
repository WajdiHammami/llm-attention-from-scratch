import torch
from src.modules.transformerBlock import TransformerBlock
from src.config.model_config import TransformerModelConfig
import torch.nn as nn




class TransformerModel(torch.nn.Module):
    def __init__(self, config: TransformerModelConfig):
        super(TransformerModel, self).__init__()
        self.config = config
        self.head_dim = config.d_model // config.num_heads
        # Embedding layer
        self.embeddings = torch.nn.Embedding(
            num_embeddings= config.vocab_size,
            embedding_dim= config.d_model
        )

        # Transformer Block
        self.transformers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)  # You can make number of layers configurable if needed
        ])

        # Final Norm
        self.norm = torch.nn.LayerNorm(config.d_model)

        # Final linear layer tied with embeddings
        if config.tie_embeddings:
            self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False) # (B,T,vocab_size)
            self.lm_head.weight = self.embeddings.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True) # (B,T,vocab_size)



    def forward(self, input_ids, cache=None, return_logits=True):
        x = self.embeddings(input_ids)  # (B, T, d_model)

        # -----------------------
        # TRAINING MODE
        # -----------------------
        if cache is None:
            for block in self.transformers:
                x, _ = block(x, cache=None)

            x = self.norm(x)

            if return_logits:
                return self.lm_head(x)
            else:
                return x

        # -----------------------
        # INFERENCE MODE
        # -----------------------
        new_cache = []

        for layer, block in enumerate(self.transformers):
            layer_cache = cache[layer]
            x, updated_cache = block(x, layer_cache)
            new_cache.append(updated_cache)

        x = self.norm(x)

        if return_logits:
            return self.lm_head(x), new_cache
        else:
            return x, new_cache
    
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0):
        """Apply top-k and/or top-p (nucleus) filtering to logits."""
        filtered = logits.clone()

        # Top-K
        if top_k > 0:
            top_k = min(top_k, filtered.size(-1))
            values, _ = torch.topk(filtered, top_k, dim=-1)
            cutoff = values[..., -1:]  # shape (B,1)
            filtered = torch.where(filtered < cutoff, torch.full_like(filtered, -float("inf")), filtered)

        # Top-P
        if top_p > 0.0:
            sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)

            # Mask tokens where cumulative prob exceeds top_p
            sorted_mask = cumsum > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            # Scatter mask back to original indices
            keep_mask = torch.zeros_like(filtered, dtype=torch.bool)
            keep_mask.scatter_(-1, sorted_idx, ~sorted_mask)

            filtered = filtered.masked_fill(~keep_mask, -float("inf"))

        return filtered

        

    def sample_next_token(self, inputs_ids, cache, temperature=1.0, top_k=40, top_p=0.85, repeat_penalty=1.0):
        self.eval()


        logits, cache = self.forward(inputs_ids[:, -1:], cache=cache)  # (B,T,vocab_size)
        # Apply temperature
        logits = logits / temperature
        if repeat_penalty != 1.0:
            for i in range(inputs_ids.size(0)):  # For each batch
                for token_id in set(inputs_ids[i].tolist()):
                    logits[i, :, token_id] /= repeat_penalty
        
        # Get logits for the last token
        logits = logits[:, -1, :]  # (B,vocab_size)
        # Apply top-k and top-p filtering
        filtered_logits = logits.clone()
        if top_k > 0 or top_p > 0.0:
            filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        # Sample from the filtered distribution
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)  # (B,vocab_size)
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        return next_token, cache

    def generate(self, input_ids, max_length, temperature=1.0, top_k=0, top_p=0.0, repeat_penalty=1.2):
        self.eval()

        cache = [ {"k": None, "v": None} for _ in range(self.config.num_layers) ]
        generated = input_ids

        for _ in range(max_length):
            next_token, cache = self.sample_next_token(
                generated,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty
                )
            
            generated = torch.cat([generated, next_token], dim=1)

        return generated