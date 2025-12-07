import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.model.transformer import TransformerModel, TransformerModelConfig


def model_size(model):
    total = sum(p.numel() for p in model.parameters())
    return total * 4 / (1024**2)   # fp32 → MB


def kv_cache_memory(model, seq_len):
    
    kv_per_layer = model.config.num_kv_heads * seq_len * model.head_dim
    # K + V
    per_layer = 2 * kv_per_layer
    total = per_layer * model.config.num_layers
    return total * 2 / (1024**2)  # bf16 → 2 bytes per float


def benchmark_memory(model, seq_len=512, device="cuda"):
    model = model.to(device)
    
    print(f"Model params: {model_size(model):.2f} MB")
    print(f"KV Cache memory (@{seq_len} tokens): {kv_cache_memory(model, seq_len):.2f} MB")

    x = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x, cache=None)

    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"Activation memory (forward pass): {peak:.2f} MB")


def make_model(attention_type="mha"):
    if attention_type == "mha":
        num_kv_heads = 8
    elif attention_type == "gqa":
        num_kv_heads = 4
    elif attention_type == "mqa":
        num_kv_heads = 1

    return TransformerModel(TransformerModelConfig(
        vocab_size=32000,
        d_model=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=num_kv_heads,
        dim_feedforward=2048,
        dropout=0.1,
        tie_embeddings=True,
    ))


if __name__ == "__main__":
    for attn_type in ["mha", "gqa", "mqa"]:
        print(f"\n=== Memory Benchmark ({attn_type.upper()}) ===")
        model = make_model(attn_type)
        benchmark_memory(model, seq_len=512)
