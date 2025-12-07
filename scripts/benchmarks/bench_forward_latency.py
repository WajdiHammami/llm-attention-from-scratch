import time
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.model.transformer import TransformerModel, TransformerModelConfig


def forward_latency(model, seq_len=512, device="cuda"):
    model.eval()
    model.to(device)
    x = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    # Warm-up
    with torch.no_grad():
        _ = model(x, cache=None)

    # Measure
    start = time.time()
    with torch.no_grad():
        _ = model(x, cache=None)
    torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"Seq len {seq_len:4}: {elapsed*1000:.2f} ms")


def make_model(attention_type="mha"):
    if attention_type == "mha":
        num_kv_heads = 8
    elif attention_type == "gqa":
        num_kv_heads = 4
    elif attention_type == "mqa":
        num_kv_heads = 1

    config = TransformerModelConfig(
        vocab_size=32000,
        d_model=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=num_kv_heads,
        dim_feedforward=2048,
        dropout=0.1,
        tie_embeddings=True,
    )
    return TransformerModel(config)


if __name__ == "__main__":
    seq_lengths = [128, 256, 512, 1024]

    for attn_type in ["mha", "gqa", "mqa"]:
        print(f"\n=== Forward Latency ({attn_type.upper()}) ===")
        model = make_model(attn_type)
        for L in seq_lengths:
            forward_latency(model, seq_len=L)
