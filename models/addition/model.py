# SPDX-FileCopyrightText: 2025 Dennis H. Wuitz <dennis.wuitz@wavelens.io>
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from buildblocks import TransformerEncoderLayer

class Transformer(nn.Module):
    """A simple Encoder Only Transformer model."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, layers: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=4*d_model) for _ in range(layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embedding(x)

        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        for layer in self.encoder_layers:
            x = layer(x, mask=mask)

        out = self.fc_out(x)
        return out

