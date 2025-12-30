# SPDX-FileCopyrightText: 2025 Dennis H. Wuitz <dennis.wuitz@wavelens.io>
#
# SPDX-License-Identifier: MIT

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention using PyTorch operations only.

    Args:
        d_model (int): embedding size.
        n_heads (int): number of attention heads.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Learnable linear projections for Q, K, V and final output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute self-attention.

        Args:
            x: (batch, seq, d_model)
            mask: (batch, seq) where 0 = pad / 1 = keep, OR (batch, 1, seq, seq) attention mask of -inf/0
        Returns:
            Tensor of shape (batch, seq, d_model)
        """
        bsz, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads
        def reshape_heads(t):
            return t.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(reshape_heads, (q, k, v))  # each: (batch, heads, seq, head_dim)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, seq, seq)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # (batch, heads, seq, head_dim)
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)

