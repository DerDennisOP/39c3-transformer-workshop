# SPDX-FileCopyrightText: 2025 Dennis H. Wuitz <dennis.wuitz@wavelens.io>
#
# SPDX-License-Identifier: MIT

import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional embeddings from Vaswani et al. (2017). Adds to token embeddings.

    Args:
        d_model (int): Embedding size.
        max_len (int): Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, seq, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

