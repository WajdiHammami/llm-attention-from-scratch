import torch
import argparse
from src.data.tokenizer import get_tokenizer, vocab_size
from src.model.transformer import TransformerModel, TransformerModelConfig


# -------------------------
# Generation Function
# -------------------------

def generate_text(
    model,
    tokenizer,
    prompt,
    device="cuda",
    max_new_tokens=100,
    temperature=1.1,
    top_k=40,
    top_p=0.85,
    repeat_penalty=1.2,
):
    """Generate text using KV cache."""

    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Debug prints
    print(f"Model vocab_size: {model.config.vocab_size}")
    print(f"Input token IDs: {input_ids}")
    print(f"Max token ID: {input_ids.max().item()}")
    print(f"Min token ID: {input_ids.min().item()}")
    assert input_ids.max() < model.config.vocab_size, f"Token ID {input_ids.max()} >= vocab_size {model.config.vocab_size}"

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=max_new_tokens + input_ids.size(1),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )

    return tokenizer.decode(generated[0].tolist())


# -------------------------
# Main Script
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer_path)

    # Load model config (saved inside checkpoint)
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model_config= TransformerModelConfig(
            vocab_size=vocab_size(tokenizer),
            d_model=512,
            num_layers=10,
            num_heads=8,
            num_kv_heads=2,
            dim_feedforward=2048,
            dropout=0.1,
            tie_embeddings=True,
        )
    
    # Build model
    model = TransformerModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    # Add this after loading the model:
    print(f"Config vocab_size: {model.config.vocab_size}")
    print(f"Embedding layer actual size: {model.embeddings.weight.shape[0]}")
    print(f"LM head actual size: {model.lm_head.weight.shape[0]}")
    # Format the number of parameters with commas
    print(f"Model's Number of Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify they match
    assert model.embeddings.weight.shape[0] == model.config.vocab_size, \
        f"Mismatch: embeddings has {model.embeddings.weight.shape[0]} but config says {model.config.vocab_size}"

    # Generate text
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print("\n================ GENERATED TEXT ================\n")
    print(output)
    print("\n================================================\n")


if __name__ == "__main__":
    main()
