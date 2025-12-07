import math
import torch
import torch.nn as nn



def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., D)
    d = x.size(-1)
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super(RotaryEmbedding, self).__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        half_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # build cos/sin cache per device/dtype
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = None


    def _build_cache(self, len_seq, device, dtype):
        if self._seq_len_cached is not None and self._seq_len_cached >= len_seq and self._cos_cached.device == device and self._cos_cached.dtype == dtype:
            return  
        
        pos = torch.arange(len_seq, device=device, dtype=dtype)
        freq = torch.outer(pos, self.inv_freq)  # (len_seq, head_dim/2)
        emb = torch.cat([freq, freq], dim=-1)  # (len_seq, head_dim)

        cos = emb.cos().to(device)
        sin = emb.sin().to(device)

        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = len_seq


    def get_cos_sin(self, seq_len, device, dtype, offset: int = 0):
        needed_len = seq_len + offset

        self._build_cache(needed_len, device, dtype)

        cos = self._cos_cached[offset:offset + seq_len, :] # (seq_len, head_dim)
        sin = self._sin_cached[offset:offset + seq_len, :] # (seq_len, head_dim)
        return cos, sin
    
    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (batch_size, num_heads, seq_len, head_dim)
        # cos, sin: (seq_len, head_dim)

        cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim)
        sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim)
        
        q_rotated = (q * cos) + (rotate_half(q) * sin)
        k_rotated = (k * cos) + (rotate_half(k) * sin)

        return q_rotated, k_rotated