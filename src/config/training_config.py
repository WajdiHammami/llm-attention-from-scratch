from dataclasses import dataclass
from typing import Literal, Optional


from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)

    # Batching
    batch_size: int = 8          # per-dataloader batch
    micro_batch_size: int = 2    # gradient accumulation

    # Sequence length
    block_size: int = 512

    # Training schedule
    max_steps: int = 2000
    warmup_steps: int = 200
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 1000

    # Precision
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"

    # Checkpointing
    output_dir: str = "checkpoints"
    resume_from: str | None = None

    # Randomness
    seed: int = 42

    # Gradient clipping
    grad_clip: float = 1.0

    # Logging
    log_dir: str = "logs/training_logs"


    # Padding token id
    pad_token_id: int = 3