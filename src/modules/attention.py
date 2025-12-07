import torch
import torch.nn as nn
from src.modules.rope import RotaryEmbedding


class AttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None,
                 max_position_embeddings=2048, rope_base=10000):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must divide num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.heads_per_kv = num_heads // self.num_kv_heads

        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)

        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_base)


    # -------- Shaping utils --------

    def _shape_q(self, q, B, T):
        return q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x, B, T):
        return x.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)


    # -------- FORWARD PASS --------

    def forward(self, x, cache=None):
        B, T, _ = x.shape

        # Training mode / no cache
        if cache is None:
            return self._forward_train(x)

        # Inference mode / KV cache
        return self._forward_inference(x, cache)


    # ======================
    # TRAINING MODE FORWARD
    # ======================
    def _forward_train(self, x):
        B, T, _ = x.size()

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = self._shape_q(Q, B, T)  # (B, H, T, D)
        K = self._shape_kv(K, B, T) # (B, G, T, D)
        V = self._shape_kv(V, B, T) # (B, G, T, D)

        # Apply RoPE
        cos, sin = self.rope.get_cos_sin(T, x.device, x.dtype)
        Q, K = self.rope.apply_rotary(Q, K, cos, sin)

        # ===============================
        # Efficient GQA/MQA
        # ===============================

        # Map each head to its KV group
        group_map = torch.arange(self.num_heads, device=Q.device) // self.heads_per_kv

        # Select appropriate KV for each head
        K_sel = K[:, group_map]  # (B, H, T, D)
        V_sel = V[:, group_map]  # (B, H, T, D)

        # Standard attention with grouped KV
        scores = torch.matmul(Q, K_sel.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add causal mask
        causal_mask = torch.full((T, T), float("-inf"), device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        scores = scores + causal_mask

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_sel)  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out), None


    # ======================
    # INFERENCE MODE FORWARD
    # ======================
    def _forward_inference(self, x, cache):
        B, T, _ = x.shape
        assert T == 1

        # Compute projections for the new token
        Q = self.W_Q(x)
        K_new = self.W_K(x)
        V_new = self.W_V(x)

        Q = self._shape_q(Q, B, 1)        # (B,H,1,D)
        K_new = self._shape_kv(K_new, B, 1)  # (B,G,1,D)
        V_new = self._shape_kv(V_new, B, 1)  # (B,G,1,D)

        # Retrieve previous KV cache
        K_cache = cache["k"]  # (B,G,T_cache,D)
        V_cache = cache["v"]

        if K_cache is None:
            K_cat = K_new
            V_cat = V_new
        else:
            K_cat = torch.cat([K_cache, K_new], dim=2)
            V_cat = torch.cat([V_cache, V_new], dim=2)

        T_total = K_cat.size(2)

        # Apply RoPE with offset
        cos, sin = self.rope.get_cos_sin(T_total, x.device, x.dtype)

        # Apply RoPE to *last* Q only
        Q, _ = self.rope.apply_rotary(Q, K_new, cos[:, -1:], sin[:, -1:])

        # Apply RoPE to full K_cat
        _, K_cat = self.rope.apply_rotary(torch.zeros_like(K_cat), K_cat, cos, sin)

        # ===============================
        # GQA/MQA
        # ===============================
        group_map = torch.arange(self.num_heads, device=x.device) // self.heads_per_kv
        K_sel = K_cat[:, group_map]   # (B,H,T_total,D)
        V_sel = V_cat[:, group_map]   # (B,H,T_total,D)

        # Attention
        scores = torch.matmul(Q, K_sel.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_sel)  # (B,H,1,D)

        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        out = self.W_O(out)

        new_cache = {"k": K_cat, "v": V_cat}
        return out, new_cache
