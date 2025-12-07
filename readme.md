# Modern LLM Attention From Scratch (PyTorch)

### **MHA • GQA • MQA • RoPE • KV-Cache • Training Pipeline • Benchmark Suite**

This project implements **modern LLM attention mechanisms from first principles**, including:

* Multi-Head Attention (MHA)
* Grouped-Query Attention (GQA) — *LLaMA-2 style*
* Multi-Query Attention (MQA) — *PaLM-style*
* Rotary Positional Embedding (RoPE)
* Efficient KV-cache for autoregressive decoding
* A complete PyTorch training pipeline
* Dataset preprocessing (tokenizer, chunking, collator)
* Benchmark suite for latency, memory, and throughput

The goal is an **educational** repository showing exactly how modern transformers work internally.

---

# Features

### Attention Mechanisms

* ✔ Scaled Dot-Product Attention
* ✔ Multi-Head Attention (MHA)
* ✔ Grouped-Query Attention (GQA)
* ✔ Multi-Query Attention (MQA)
* ✔ Efficient KV selection (`group_map`) — **NO tensor expansion**
* ✔ Rotary Positional Embeddings (RoPE)

### Inference Optimizations

* ✔ KV-cache
* ✔ Autoregressive decoding
* ✔ Top-k, top-p, temperature sampling
* ✔ Efficient attention computation

### Training Pipeline

* ✔ AdamW + Cosine LR with Warmup
* ✔ Mixed Precision (fp16/bf16)
* ✔ Gradient Accumulation
* ✔ Checkpointing + Resume
* ✔ Logging JSONL

### Benchmark Suite

* ✔ Inference latency (ms/token)
* ✔ Forward pass throughput
* ✔ Memory usage (model + KV cache + activations)

---

# Model Architecture

```
Embedding → [ Transformer Block × N ] → LayerNorm → LM Head
```

Each Transformer Block:

1. LayerNorm
2. Attention (MHA/GQA/MQA)
3. Residual
4. LayerNorm
5. SwiGLU Feedforward
6. Residual

---

# Attention Mechanisms (Overview)

### 🔹 Multi-Head Attention (MHA)

Each head has its own Query, Key, and Value.

### 🔹 Grouped-Query Attention (GQA)

Queries have many heads (`H`), but Keys/Values have fewer (`G`):

```
H heads → G KV groups
group = head_index // (H / G)
```

Used in **LLaMA-2** to reduce memory and improve inference speed.

### 🔹 Multi-Query Attention (MQA)

Extreme case of GQA:

```
G = 1 → all heads share the same KV
```

Used in **PaLM**, **T5**, and many production LLMs for:

* Faster KV-cache lookup
* Lower memory footprint
* Stable scaling to long context lengths

---

# Benchmarks (RTX 4070 Laptop GPU)

## 1. Inference Latency (ms per token, KV-cache enabled)

| Attention | ms/token | tokens/sec  |
| --------- | -------- | ----------- |
| **MHA**   | 5.955 ms | 167.9 tok/s |
| **GQA**   | 5.709 ms | 175.2 tok/s |
| **MQA**   | 5.711 ms | 175.1 tok/s |

### Interpretation

PyTorch lacks fused grouped-query kernels (FlashAttention-2 style), so MHA/GQA/MQA show similar latency.
This repo focuses on *correctness and educational clarity*.

In real LLM runtimes, MQA yields **2–4× speedups**.

---

## 2. Memory Benchmark

| Attention | Model Params | KV Cache @ 512 tokens | Activation Memory |
| --------- | ------------ | --------------------- | ----------------- |
| **MHA**   | 190.64 MB    | 8.00 MB               | 265.28 MB         |
| **GQA**   | 182.64 MB    | 4.00 MB               | 257.28 MB         |
| **MQA**   | 176.64 MB    | 1.00 MB               | 251.28 MB         |

### Interpretation

The KV-cache savings are **massive**:

* GQA uses **50% less memory**
* MQA uses **87.5% less memory**

This is critical for long-context inference.

---

## 3. Forward Pass Latency (Training Mode)

| Seq Length | MHA      | GQA      | MQA      |
| ---------- | -------- | -------- | -------- |
| 128        | 9.26 ms  | 6.36 ms  | 5.90 ms  |
| 256        | 9.77 ms  | 8.84 ms  | 6.60 ms  |
| 512        | 14.41 ms | 13.59 ms | 11.81 ms |
| 1024       | 46.00 ms | 44.32 ms | 42.95 ms |

### Interpretation

Forward pass shows **clear improvements** with GQA/MQA due to:

* Smaller KV projection
* Reduced memory bandwidth
* Fewer KV tensors per layer

---

# Diagrams

### GQA / MQA KV Sharing

```
Queries: H heads
Keys/Values: G groups   (G < H)

Head mapping:
head → head // (H / G)
```

```
Q0 ──┐
Q1 ──┤── uses KV group 0
Q2 ──┘

Q3 ──┐
Q4 ──┤── uses KV group 1
Q5 ──┘
```

### KV-Cache Growth per Layer

```
token 1 → store 1 key/value
token 2 → store 2 key/value
...
token T → store T key/value
```

MQA reduces memory by factor `H`.

---

# Repository Structure

```
src/
  model/
    transformer.py
  modules/
    attention.py
    rope.py
    transformerBlock.py
  data/
    tokenizer.py
    tokenized_dataset.py
    chunked_dataset.py
    collator.py

scripts/
  train.py
  generate.py
  benchmarks/
    bench_inference_latency.py
    bench_memory.py
    bench_forward_latency.py

configs/
  model_config.py
  train_config.py

README.md
```

---

# Training

```
python -m scripts.train

Supports:

* Mixed precision
* Gradient accumulation
* Resume from checkpoint
* Logging

Configs should be modified in the training script

---

# Text Generation

```
python -m scripts.generate \
  --checkpoint checkpoints/model.pt \
  --prompt "The history of machine learning" \
  --max_new_tokens 150
```

---

# Notes on GQA/MQA Performance

> PyTorch does not currently include fused grouped-query attention kernels
> (unlike FlashAttention-2, PaLM kernels, or Triton implementations).
>
> Therefore, MHA/GQA/MQA achieve **similar inference latency** in this project.
>
> Memory usage decreases significantly in GQA/MQA, matching behavior of real LLMs.
>
> For real-world speedups, FlashAttention-2 or Triton fused kernels are required.

---

# License

MIT.

---

# Acknowledgements

Inspired by architectures of **GPT**, **PaLM** and **LLaMA** research.
