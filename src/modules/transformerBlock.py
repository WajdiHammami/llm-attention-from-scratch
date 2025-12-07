import torch
from src.modules.attention import AttentionModule


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, dim_feedforward, bias=True)
        self.w2 = torch.nn.Linear(dim_feedforward, d_model, bias=True)
        self.w3 = torch.nn.Linear(d_model, dim_feedforward, bias=False)
    
    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.normlayer1 = torch.nn.LayerNorm(d_model)
        self.attention = AttentionModule(d_model, num_heads, num_kv_heads)
        self.normlayer2 = torch.nn.LayerNorm(d_model)
        self.feedforward = SwiGLU(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x, cache=None):
        # x shape: (batch_size, seq_len, d_model)
        # cache: dict with 'k' and 'v' tensors for caching key and value projections

        # LayerNorm + Attention
        x_norm = self.normlayer1(x)
        attn_output, new_cache = self.attention(x_norm, cache)
        x = x + self.dropout(attn_output)

        # LayerNorm + Feedforward
        x_norm = self.normlayer2(x)
        ff_output = self.feedforward(x_norm)
        x = x + self.dropout(ff_output)

        return x, new_cache