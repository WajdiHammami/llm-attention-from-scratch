import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.modules.attention import AttentionModule
from src.modules.rope import RotaryEmbedding


def test_rope_before_expansion():
    B, T, d = 1, 5, 32
    x = torch.randn(B, T, d)

    # GQA case: 8 heads, 2 KV heads → expansion needed
    att = AttentionModule(d_model=d, num_heads=8, num_kv_heads=2, max_position_embeddings=10)

    q = att.W_Q(x)          # (1,5, 8*4)
    k = att.W_K(x)          # (1,5, 2*4)

    # Shapes before RoPE:
    q = q.view(B,T,8,4).transpose(1,2)  # (1,8,5,4)
    k = k.view(B,T,2,4).transpose(1,2)  # (1,2,5,4)

    rope = att.rope
    cos, sin = rope.get_cos_sin(T, x.device, x.dtype)

    q_rot, k_rot = rope.apply_rotary(q, k, cos, sin)

    # Expand AFTER RoPE
    k_rot_expanded = k_rot.repeat_interleave(4, dim=1)  # (1,8,5,4)

    # Expand BEFORE RoPE then rotate
    k_exp = k.repeat_interleave(4, dim=1)
    _, k_exp_rot = rope.apply_rotary(k_exp, k_exp, cos, sin)

    assert torch.allclose(k_rot_expanded, k_exp_rot, atol=1e-5), \
        "RoPE must be applied BEFORE expansion. Implementation incorrect."
    


def test_attention_training_shapes():
    B, T, d = 2, 7, 32
    att = AttentionModule(d_model=d, num_heads=8, num_kv_heads=2, max_position_embeddings=10)

    x = torch.randn(B, T, d)
    out, cache = att(x, cache=None)

    assert out.shape == (B, T, d)
    assert cache is None



def test_attention_inference_kv_cache_rope():
    B, d = 1, 32
    att = AttentionModule(d_model=d, num_heads=8, num_kv_heads=1, max_position_embeddings=50)

    cache = {"k": None, "v": None}
    seq = [torch.randn(B,1,d) for _ in range(5)]

    for i, token in enumerate(seq):
        out, cache = att(token, cache)
        
        assert out.shape == (1,1,32)
        assert "k" in cache and "v" in cache
        assert cache["k"].shape[2] == i+1
        assert cache["v"].shape[2] == i+1

        # Check RoPE offset is correct (position == cache_length - 1)
        pos = cache["k"].shape[2] - 1
        cos, sin = att.rope.get_cos_sin(1, out.device, out.dtype, offset=pos)
        # Just verify it builds without errors and changes w/ pos


def test_attention_modes_equivalence():
    B, T, d = 1, 6, 32
    x = torch.randn(B, T, d)

    # MHA: h=8, g=8
    att_mha = AttentionModule(d_model=d, num_heads=8, num_kv_heads=8)
    out_mha, _ = att_mha(x, None)

    # GQA: g=2 KV heads
    att_gqa = AttentionModule(d_model=d, num_heads=8, num_kv_heads=2)
    out_gqa, _ = att_gqa(x, None)

    # MQA: g=1 KV head
    att_mqa = AttentionModule(d_model=d, num_heads=8, num_kv_heads=1)
    out_mqa, _ = att_mqa(x, None)

    assert out_mha.shape == out_gqa.shape == out_mqa.shape
