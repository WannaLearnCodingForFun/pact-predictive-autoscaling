"""Sequence forecasters for Table 5: LSTM, Bi-LSTM, GRU, small Transformer.

Same input/output contract as the TCN: ``(B, C, L) → (B, H, 2)``, one shot.
LSTM hidden size can be matched to a TCN parameter count.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from pact.telemetry.features import FEATURE_DIM

SeqKind = Literal["lstm", "bilstm", "gru", "transformer"]


class RecurrentForecaster(nn.Module):
    def __init__(
        self,
        *,
        kind: Literal["lstm", "bilstm", "gru"],
        in_channels: int,
        window: int,
        horizon: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be >= 1")
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.hidden_size = hidden_size
        bidirectional = kind == "bilstm"
        drop = dropout if num_layers > 1 else 0.0
        if kind == "gru":
            self.rnn: nn.GRU | nn.LSTM = nn.GRU(
                in_channels,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=drop,
                bidirectional=bidirectional,
            )
        else:
            self.rnn = nn.LSTM(
                in_channels,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=drop,
                bidirectional=bidirectional,
            )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, horizon * 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, L), got {tuple(x.shape)}")
        seq = x.transpose(1, 2)
        out, _hidden = self.rnn(seq)
        last = out[:, -1, :]
        forecast: Tensor = self.head(last).view(x.size(0), self.horizon, 2)
        return forecast

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TinyTransformer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        window: int,
        horizon: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, horizon * 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, L), got {tuple(x.shape)}")
        seq = x.transpose(1, 2)
        length = seq.size(1)
        h = self.proj(seq) + self.pos[:, :length, :]
        causal = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(h, mask=causal)
        forecast: Tensor = self.head(encoded[:, -1, :]).view(
            x.size(0), self.horizon, 2
        )
        return forecast

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_seq_forecaster(
    kind: SeqKind,
    *,
    in_channels: int = FEATURE_DIM,
    window: int,
    horizon: int,
    hidden_size: int = 32,
    dropout: float = 0.0,
) -> nn.Module:
    if kind == "transformer":
        d_model = max(8, hidden_size - hidden_size % 4)
        nhead = 4 if d_model % 4 == 0 else 1
        return TinyTransformer(
            in_channels=in_channels,
            window=window,
            horizon=horizon,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
    return RecurrentForecaster(
        kind=kind,
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        hidden_size=hidden_size,
        dropout=dropout,
    )


def hidden_size_matching_params(
    kind: SeqKind,
    *,
    target_params: int,
    in_channels: int,
    window: int,
    horizon: int,
    lo: int = 8,
    hi: int = 128,
) -> int:
    """Return the hidden/d_model size whose param count is closest to ``target``."""

    best = lo
    best_err = abs(
        _count(kind, lo, in_channels, window, horizon) - target_params
    )
    for hidden in range(lo, hi + 1, 4):
        err = abs(
            _count(kind, hidden, in_channels, window, horizon) - target_params
        )
        if err < best_err:
            best, best_err = hidden, err
    return best


def _count(
    kind: SeqKind, hidden: int, in_channels: int, window: int, horizon: int
) -> int:
    model = build_seq_forecaster(
        kind,
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        hidden_size=hidden,
    )
    return int(sum(p.numel() for p in model.parameters()))


class ModuleForecaster:
    """Adapt a ``(B, C, L) → (B, H, 2)`` module to the loop Forecaster protocol."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._model.eval()

    def predict(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        arr = np.asarray(window, dtype=np.float32)
        x = torch.from_numpy(arr.T.copy()).unsqueeze(0)
        with torch.no_grad():
            out = self._model(x)
        return np.asarray(out[0].detach().cpu().numpy(), dtype=np.float32)
