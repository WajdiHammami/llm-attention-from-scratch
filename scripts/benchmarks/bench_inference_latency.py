import time
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.model.transformer import TransformerModel, TransformerModelConfig


def benchmark_inference_latency(model, seq_len=512, device="cuda"):
    model.eval()
    model.to(device)

    # dummy prompt
    x = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    # warm-up (just to ignore CUDA lazy init)
    with torch.no_grad():
        _, cache = model(x[:, :1], cache=[{"k": None, "v": None} for _ in range(model.config.num_layers)])

    # measure autoregressive token generation
    tokens_to_generate = 50
    times = []
    input_ids = x[:, :1]

    cache = [{"k": None, "v": None} for _ in range(model.config.num_layers)]

    for _ in range(tokens_to_generate):
        start = time.time()
        with torch.no_grad():
            logits = model(input_ids[:, -1:], cache=cache)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print(f"--- Inference Latency ---")
    print(f"Model: {model.config.num_layers} layers, d={model.config.d_model}, heads={model.config.num_heads}")
    print(f"{avg*1000:.3f} ms/token  |  {1/avg:.1f} tokens/sec")


def make_model(attention_type="mha"):
    if attention_type == "mha":
        num_kv_heads = 8
    elif attention_type == "gqa":
        num_kv_heads = 4
    elif attention_type == "mqa":
        num_kv_heads = 1
    else:
        raise ValueError("Unknown attention type")

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
    print("\n=== Benchmark: MHA ===")
    benchmark_inference_latency(make_model("mha"))

    print("\n=== Benchmark: GQA ===")
    benchmark_inference_latency(make_model("gqa"))

    print("\n=== Benchmark: MQA ===")
    benchmark_inference_latency(make_model("mqa"))
