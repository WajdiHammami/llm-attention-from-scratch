import torch
import pytest
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.modules.attention import AttentionModule


def test_forward_training_shapes():
    B, T, d_model = 2, 16, 32
    num_heads = 4

    attn = AttentionModule(d_model=d_model, num_heads=num_heads)
    x = torch.randn(B, T, d_model)

    out, cache = attn(x)
    assert out.shape == (B, T, d_model)
    assert cache is None


def test_cache_first_step():
    B, d_model = 2, 32
    num_heads = 4
    num_kv_heads = 1  # MQA

    attn = AttentionModule(d_model, num_heads, num_kv_heads)

    # First token
    x = torch.randn(B, 1, d_model)
    out, cache = attn(x, cache={"k": None, "v": None})

    assert out.shape == (B, 1, d_model)
    assert cache["k"].shape == (B, num_kv_heads, 1, d_model // num_heads)
    assert cache["v"].shape == (B, num_kv_heads, 1, d_model // num_heads)


def test_cache_append():
    B, d_model = 2, 32
    num_heads = 4
    num_kv_heads = 2  # GQA example

    attn = AttentionModule(d_model, num_heads, num_kv_heads)

    x1 = torch.randn(B, 1, d_model)
    out, cache = attn(x1, cache={"k": None, "v": None})

    x2 = torch.randn(B, 1, d_model)
    out, cache = attn(x2, cache)

    T = 2
    head_dim = d_model // num_heads

    assert cache["k"].shape == (B, num_kv_heads, T, head_dim)
    assert cache["v"].shape == (B, num_kv_heads, T, head_dim)


def test_kv_expansion():
    B, T, d_model = 2, 8, 32
    num_heads = 4
    num_kv_heads = 2  # GQA → 2 groups

    attn = AttentionModule(d_model, num_heads, num_kv_heads)
    x = torch.randn(B, T, d_model)

    q = attn.W_Q(x)
    k = attn.W_K(x)

    q = attn._shape_q(q, B, T)          # (B, 4, T, d)
    k = attn._shape_kv(k, B, T)         # (B, 2, T, d)

    k_expanded = attn._expand_kv(k)

    assert k_expanded.shape[1] == num_heads  # expanded to 4 heads


def test_training_vs_cache_equivalence():
    B, d_model = 2, 32
    num_heads = 4
    num_kv_heads = 1  # MQA test

    attn = AttentionModule(d_model, num_heads, num_kv_heads)

    x = torch.randn(B, 1, d_model)

    # Training mode path (full sequence)
    out_train, _ = attn(x)

    # Cache mode path (incremental)
    out_inf, _ = attn(x, cache={"k": None, "v": None})

    torch.testing.assert_close(out_train, out_inf, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_kv_heads", [4, 1, 2])  # MHA, MQA, GQA
def test_attention_variants(num_kv_heads):
    B, T, d_model = 2, 8, 32
    num_heads = 4

    attn = AttentionModule(d_model, num_heads, num_kv_heads)
    x = torch.randn(B, T, d_model)

    out, _ = attn(x)

    assert out.shape == (B, T, d_model)
