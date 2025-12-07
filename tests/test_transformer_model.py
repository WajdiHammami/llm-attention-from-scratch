import torch
import pytest
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model.transformer import TransformerModel, TransformerModelConfig



def test_transformer_training_shapes():
    config = TransformerModelConfig(
        d_model=32,
        num_heads=8,
        num_kv_heads=1,
        num_layers=3,
        vocab_size=100,
    )
    model = TransformerModel(config)
    model.eval()

    B, T = 2, 5
    x = torch.randint(0, config.vocab_size, (B, T))

    logits = model(x, cache=None, return_logits=True)
    assert logits.shape == (B, T, config.vocab_size)


def test_transformer_inference_shapes():
    config = TransformerModelConfig(
        d_model=32,
        num_heads=8,
        num_kv_heads=1,   # MQA
        num_layers=3,
        vocab_size=100,
    )
    model = TransformerModel(config)
    model.eval()

    B = 1
    x0 = torch.randint(0, config.vocab_size, (B, 1))

    # Initialize empty cache for each layer
    cache = [ {"k": None, "v": None} for _ in range(config.num_layers) ]

    logits, new_cache = model(x0, cache=cache, return_logits=True)

    assert logits.shape == (B, 1, config.vocab_size)
    assert len(new_cache) == config.num_layers

    for layer_cache in new_cache:
        assert "k" in layer_cache and "v" in layer_cache
        assert layer_cache["k"] is not None
        assert layer_cache["v"] is not None
        assert layer_cache["k"].shape[2] == 1
        assert layer_cache["v"].shape[2] == 1



def test_transformer_autoregressive_equivalence():
    config = TransformerModelConfig(
        d_model=32,
        num_heads=8,
        num_kv_heads=1,
        num_layers=4,
        vocab_size=200,
    )
    model = TransformerModel(config)
    model.eval()

    B, T = 1, 6
    x = torch.randint(0, config.vocab_size, (B, T))

    # Full forward
    full_logits = model(x, cache=None)     # (B, T, vocab)
    last_full = full_logits[:, -1, :]      # (B, vocab)

    # Autoregressive
    cache = [ {"k": None, "v": None} for _ in range(config.num_layers) ]

    last_logit = None
    for t in range(T):
        xt = x[:, t:t+1]                 # (B,1)
        logits, cache = model(xt, cache=cache)
        last_logit = logits[:, -1, :]    # (B,vocab)

    diff = (last_logit - last_full).abs().max().item()

    assert diff < 0.15, f"AR vs FULL too different: {diff}"



def test_transformer_tied_embeddings():
    config = TransformerModelConfig(
        d_model=32,
        num_heads=8,
        num_layers=2,
        vocab_size=50,
        tie_embeddings=True,
    )
    model = TransformerModel(config)

    assert model.lm_head.weight.data_ptr() == model.embeddings.weight.data_ptr(), \
        "Embedding and LM head weights are not tied"



def test_transformer_gqa_works():
    config = TransformerModelConfig(
        d_model=32,
        num_heads=8,
        num_kv_heads=2,     # GQA (4 Q heads per KV head)
        num_layers=3,
        vocab_size=200,
    )
    model = TransformerModel(config)
    model.eval()

    B, T = 2, 4
    x = torch.randint(0, config.vocab_size, (B, T))

    logits = model(x, cache=None)
    assert logits.shape == (B, T, config.vocab_size)



def test_transformer_kv_cache_growth():
    config = TransformerModelConfig(
        d_model=32,
        num_heads=8,
        num_kv_heads=1,
        num_layers=3,
        vocab_size=256,
    )
    model = TransformerModel(config)
    model.eval()

    B = 1
    seq_len = 5
    cache = [ {"k": None, "v": None} for _ in range(config.num_layers) ]

    # Feed sequence token-by-token
    for t in range(seq_len):
        x = torch.randint(0, config.vocab_size, (B, 1))
        logits, cache = model(x, cache)

        # Every layer should grow cache by 1 step each iteration
        for layer in range(config.num_layers):
            assert cache[layer]["k"].shape[2] == t + 1
            assert cache[layer]["v"].shape[2] == t + 1



def test_transformer_generate():
    config = TransformerModelConfig(
        d_model=32,
        num_heads=8,
        num_kv_heads=1,
        num_layers=2,
        vocab_size=100,
    )
    model = TransformerModel(config)
    model.eval()

    x0 = torch.randint(0, config.vocab_size, (1, 1))
    out = model.generate(x0, max_length=5)

    # (1, 1+5)
    assert out.shape == (1, 6)

    # All tokens must be valid IDs
    assert (out >= 0).all() and (out < config.vocab_size).all()
