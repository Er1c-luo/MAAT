"""
Lightweight pure context branch encoder.

Consumes only explicit context features C and returns z_c for downstream modulation.
Input shape:  [B, L, C_dim]
Output shape: [B, L, d_ctx]
"""

import torch
import torch.nn as nn


class LightweightContextBranchEncoder(nn.Module):
    """Linear(C_dim->d_ctx) + 1-layer TransformerEncoder for temporal context mixing."""

    def __init__(
        self,
        c_dim: int,
        d_ctx: int = 32,
        nhead: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_ctx % nhead == 0, "d_ctx must be divisible by nhead"

        # Lift context features into encoder width.
        self.in_proj = nn.Linear(c_dim, d_ctx)

        # Single lightweight Transformer layer over time axis.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_ctx,
            nhead=nhead,
            dim_feedforward=d_ctx * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Keep output scale stable for later fusion/modulation.
        self.out_norm = nn.LayerNorm(d_ctx)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: [B, L, C_dim] context features only.
        Returns:
            z_c: [B, L, d_ctx] encoded context representation.
        """
        h = self.in_proj(c)
        h = self.encoder(h)
        z_c = self.out_norm(h)
        return z_c
