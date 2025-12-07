from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TransformerModelConfig:
    d_model: int
    num_heads: int
    num_kv_heads: int = None
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_position_embeddings: int = 2048
    rope_base: int = 10000
    vocab_size: int = 32000
    num_layers: int = 8
    
    # training / generation helpers
    tie_embeddings: Optional[bool] = True
    use_flash_attn: Optional[bool] = False          # optional flag if you later want it
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

