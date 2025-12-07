# tests/test_block.py
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.modules.transformerBlock import TransformerBlock


def test_block_training_shapes():
    """
    Basic smoke test:
    - cache=None → training path
    - shapes are preserved
    - no cache returned
    """
    torch.manual_seed(0)
    B, T, d_model = 2, 7, 32
    num_heads = 8

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=None,      # MHA mode
        dim_feedforward=64,
        dropout=0.1,
    )

    x = torch.randn(B, T, d_model)
    block.eval()  # turn off dropout so test is deterministic-ish

    y, cache = block(x, cache=None)

    assert y.shape == (B, T, d_model)
    assert cache is None


def test_block_inference_cache_grows_mqa():
    """
    Inference path:
    - start with cache={"k": None, "v": None}
    - feed 1 token at a time
    - cache["k"]/cache["v"] grow along time dimension
    - outputs keep correct shape (B, 1, d_model)
    """
    torch.manual_seed(0)
    B, d_model = 1, 32
    num_heads = 8
    num_kv_heads = 1  # MQA mode

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dim_feedforward=64,
        dropout=0.0,
    )
    block.eval()

    seq_len = 5
    tokens = [torch.randn(B, 1, d_model) for _ in range(seq_len)]
    cache = {"k": None, "v": None}

    for t, token in enumerate(tokens):
        y, cache = block(token, cache)

        # output shape
        assert y.shape == (B, 1, d_model)

        # cache structure
        assert isinstance(cache, dict)
        assert "k" in cache and "v" in cache
        assert cache["k"] is not None and cache["v"] is not None

        # cache should have length t+1 along time dim (dim=2)
        assert cache["k"].shape[2] == t + 1
        assert cache["v"].shape[2] == t + 1


def test_block_autoregressive_equivalence():
    """
    Check that:
    - Full pass over a sequence with cache=None (training mode)
    - Step-by-step autoregressive pass with KV cache
    give (approximately) the same hidden state for the last token.
    """
    torch.manual_seed(0)
    B, T, d_model = 1, 6, 32
    num_heads = 8
    num_kv_heads = 1  # use MQA to stress KV logic

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dim_feedforward=64,
        dropout=0.0,
    )
    block.eval()  # no dropout

    x = torch.randn(B, T, d_model)

    # Single full forward (training-style, causal mask)
    full_out, _ = block(x, cache=None)  # (B, T, d_model)
    last_full = full_out[:, -1:, :]     # (B, 1, d_model)

    # Autoregressive forward with KV cache
    cache = {"k": None, "v": None}
    last_step = None
    for t in range(T):
        xt = x[:, t:t+1, :]         # (B,1,d_model)
        last_step, cache = block(xt, cache)

    # Compare last token representation
    assert last_step.shape == last_full.shape
    diff = (last_step - last_full).abs().max().item()
    assert diff < 6e-2, f"Difference too large: {diff}"



def test_block_gqa_shapes():
    """
    Just a shape sanity check for GQA:
    - num_heads > num_kv_heads
    - block still runs, shapes okay, returns cache
    """
    torch.manual_seed(0)
    B, T, d_model = 2, 4, 64
    num_heads = 8
    num_kv_heads = 2   # GQA: 4 Q heads per KV head

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dim_feedforward=128,
        dropout=0.0,
    )
    block.eval()

    # training path
    x = torch.randn(B, T, d_model)
    y, cache = block(x, cache=None)
    assert y.shape == (B, T, d_model)
    assert cache is None

    # inference path
    cache = {"k": None, "v": None}
    token = torch.randn(B, 1, d_model)
    y_step, cache = block(token, cache)

    assert y_step.shape == (B, 1, d_model)
    assert isinstance(cache, dict)
    assert "k" in cache and "v" in cache
    assert cache["k"] is not None and cache["v"] is not None
