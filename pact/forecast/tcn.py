"""Dilated causal TCN (Module 2).

Receptive field (Eq. 9): ``R = 1 + 2(K − 1)(2^D − 1)``. Each residual block
has two sequential gated causal convolutions so the factor of 2 is real.
Causality is left-pad then trim the right. The head emits all H steps in one
pass, shape ``(B, H, 2)`` for CPU and memory — not autoregressive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pact.config import PactConfig
from pact.telemetry.features import FEATURE_DIM


def receptive_field(kernel_size: int, depth: int) -> int:
    """``R = 1 + 2(K − 1)(2^D − 1)`` (Eq. 9)."""

    if kernel_size < 1 or depth < 1:
        raise ValueError(f"K and D must be >= 1, got K={kernel_size}, D={depth}")
    return int(1 + 2 * (kernel_size - 1) * (2**depth - 1))


class CausalConv1d(nn.Module):
    """Dilated conv that cannot see the future: pad left, trim right."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self._trim = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self._trim,
        )

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.conv(x)
        if self._trim > 0:
            y = y[..., : -self._trim]
        return y


class GatedCausalConv(nn.Module):
    """Eq. 7: ``tanh(W_f * h) ⊙ σ(W_g * h)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.filter = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.gate = CausalConv1d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, h: Tensor) -> Tensor:
        gated: Tensor = torch.tanh(self.filter(h)) * torch.sigmoid(self.gate(h))
        return gated


class ResidualBlock(nn.Module):
    """Two sequential gated causal convs plus a 1×1 skip (Eq. 8)."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gated_1 = GatedCausalConv(in_channels, channels, kernel_size, dilation)
        self.gated_2 = GatedCausalConv(channels, channels, kernel_size, dilation)
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.drop_1(self.gated_1(x))
        h = self.drop_2(self.gated_2(h))
        residual: Tensor = h + self.skip(x)
        return residual


class DilatedTCN(nn.Module):
    """Full forecaster. ``forward`` is a single shot ``(B, C_in, L) → (B, H, 2)``."""

    def __init__(
        self,
        *,
        in_channels: int,
        window: int,
        horizon: int,
        kernel_size: int,
        depth: int,
        channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        rf = receptive_field(kernel_size, depth)
        if rf < window:
            raise ValueError(
                f"Horizon receptive field R={rf} does not cover window L={window} "
                f"(Eq. 9: R = 1 + 2(K-1)(2^D-1) >= L required)"
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.kernel_size = kernel_size
        self.depth = depth
        self.channels = channels
        blocks: list[ResidualBlock] = []
        block_in = in_channels
        for layer in range(depth):
            dilation = 2**layer
            blocks.append(
                ResidualBlock(
                    block_in, channels, kernel_size, dilation, dropout
                )
            )
            block_in = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(channels, horizon * 2)

    @classmethod
    def from_config(
        cls,
        config: PactConfig,
        *,
        in_channels: int = FEATURE_DIM,
    ) -> DilatedTCN:
        f = config.forecast
        return cls(
            in_channels=in_channels,
            window=config.telemetry.window,
            horizon=f.horizon,
            kernel_size=f.kernel_size,
            depth=f.depth,
            channels=f.channels,
            dropout=f.dropout,
        )

    def features(self, x: Tensor) -> Tensor:
        """Per-timestep hidden states ``(B, C, L)`` — used by the causality test."""

        h = x
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, L), got shape {tuple(x.shape)}")
        hidden = self.features(x)
        last = hidden[:, :, -1]
        forecast: Tensor = self.head(last).view(x.size(0), self.horizon, 2)
        return forecast

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
