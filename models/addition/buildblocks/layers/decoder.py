# SPDX-FileCopyrightText: 2025 Dennis H. Wuitz <dennis.wuitz@wavelens.io>
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from ..attention.mhsa import MultiHeadSelfAttention
from ..mlp.positionwise import PositionwiseFeedForward

class TransformerDecoderLayer(nn.Module):
    """One decoder block built from scratch."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.cross_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor | None = None, tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.self_attn(x, tgt_mask)
        x = self.ln1(x + attn_out)
        cross_attn_out = self.cross_attn(x, src_mask, enc_output)
        x = self.ln2(x + cross_attn_out)
        ff_out = self.ff(x)
        x = self.ln3(x + ff_out)
        return x
