import os, sys

import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.training.train import Trainer
from src.data.tokenized_dataset import TokenizedDataset
from src.data.chunked_dataset import ChunkedDataset
from src.data.collator import DataCollator
from src.config.training_config import TrainingConfig
from src.model.transformer import TransformerModel
from src.config.model_config import TransformerModelConfig
from src.data.tokenizer import vocab_size


if __name__ == "__main__":

    
    # -------------------------
    # Data Preparation
    # -------------------------

    dataset = TokenizedDataset("src/data/raw/wikitext.txt")
    chunked_dataset = ChunkedDataset(dataset, chunk_size=128)
    train_dataset, val_dataset = torch.utils.data.random_split(
        chunked_dataset,
        [int(0.9 * len(chunked_dataset)), len(chunked_dataset) - int(0.9 * len(chunked_dataset))],
        generator=torch.Generator().manual_seed(42)
    )
    tokenizer = dataset.tokenizer
    collator = DataCollator(pad_token_id=tokenizer.pad_id, block_size=128)


    
    # -------------------------
    # Model and Training Configuration
    # -------------------------

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
    

    train_config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=32,
        micro_batch_size=4,
        weight_decay=0.01,
        warmup_steps=2000,
        output_dir="checkpoints/model_checkpoint_huge",
        #resume_from="checkpoints/model_checkpoint_huge/checkpoint_step_2000.pt",
        block_size=512,
        max_steps=100000,
        precision="bf16",
        log_dir="logs/model_wiki_logs",
        save_interval=2000,
        log_interval=50,
        eval_interval=2000,
    )

    print("Building model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerModel(model_config).to(device)

    print("Starting training on device:", device)
    print(f"Model's Number of Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Dataset size:", len(train_dataset), "training samples,", len(val_dataset), "validation samples")

    
    # -------------------------
    # Training
    # -------------------------

    trainer = Trainer(
        model,
        dataset=train_dataset,
        training_config=train_config,
        collator=collator,
        device=device
        )

    trainer.train(eval_dataset=val_dataset)
    trainer._save_checkpoint(train_config.output_dir, step=train_config.max_steps)
    print("Training completed and checkpoint saved.")