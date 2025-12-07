import sentencepiece as spm
import argparse
import os



if __name__ == "__main__":
    print("Training SentencePiece model on training data...")
    argparser = argparse.ArgumentParser(description="Train SentencePiece Model")
    argparser.add_argument("--input", type=str, required=True, help="Path to input TXT file")
    argparser.add_argument("--model_prefix", type=str, required=True, help="Prefix for the output model files")
    argparser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size for the SentencePiece model")
    argparser.add_argument("--output", type=str, required=False, help="Path to output model file (not used)")
    args = argparser.parse_args()

    prefix_path = os.path.join(args.output, args.model_prefix) if args.output else args.model_prefix

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=prefix_path,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        # Special tokens
        unk_id=0,          # default UNK id
        bos_id=1,          # enable BOS
        eos_id=2,          # enable EOS
        pad_id=3,          # add PAD token
        # Whitespace/byte fallback for consistency
        byte_fallback=True,
        add_dummy_prefix=True,   # stabilizes leading whitespace handling
        split_by_whitespace=True # default; keep explicit for clarity
    )
    print(f"SentencePiece model trained and saved with prefix: {prefix_path}")