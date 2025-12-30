# SPDX-FileCopyrightText: 2025 Dennis H. Wuitz <dennis.wuitz@wavelens.io>
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from ..attention.mhsa import MultiHeadSelfAttention
from ..mlp.positionwise import PositionwiseFeedForward

class TransformerDecoderLayer(nn.Module):
    """One decoder block built from scratch (no torch.nn.MultiheadAttention)."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor | None = None, tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.self_attn(x, tgt_mask)
        x = self.ln1(x + self.dropout(attn_out))
        cross_attn_out = self.cross_attn(x, src_mask, enc_output)
        x = self.ln2(x + self.dropout(cross_attn_out))
        ff_out = self.ff(x)
        x = self.ln3(x + self.dropout(ff_out))
        return x
