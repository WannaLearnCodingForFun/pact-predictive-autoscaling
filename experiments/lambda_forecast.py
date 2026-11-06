"""Arrival-rate (lambda) forecasting shared by training and the harness.

The model forecasts the exogenous arrival rate, never observed utilisation:
utilisation is moved by the controller itself, so a model trained on it would
learn the controller's own policy and the evaluation would be circular.

Input channels per tick, all computable from the past and the clock alone:
    0  z-scored lambda, using train-split mean and std only
    1  sin(2 pi * local time of day)
    2  cos(2 pi * local time of day)
Targets are (T, 2) because the repository's forecasters and training helpers
emit (B, H, 2); both columns hold the z-scored lambda and channel 0 of the
output is the forecast. Channel 1 is trained on the same target and unused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pact.forecast.seq_models import build_seq_forecaster
from pact.forecast.tcn import DilatedTCN
from torch import nn

PARIS_OFFSET_S = 7200
IN_CHANNELS = 3


def lambda_features(
    lam: NDArray[np.floating], t0_s: int, dt_s: float, mean: float, std: float
) -> NDArray[np.float32]:
    """``(T, 3)`` features for a lambda series whose first bin starts at ``t0_s``."""

    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}")
    lam = np.asarray(lam, dtype=np.float64)
    t = t0_s + np.arange(lam.size, dtype=np.float64) * dt_s + PARIS_OFFSET_S
    phase = 2.0 * np.pi * (np.mod(t, 86400.0) / 86400.0)
    out = np.empty((lam.size, IN_CHANNELS), dtype=np.float32)
    out[:, 0] = (lam - mean) / std
    out[:, 1] = np.sin(phase)
    out[:, 2] = np.cos(phase)
    return out


def build_model(meta: dict[str, Any]) -> nn.Module:
    kind = meta["kind"]
    if kind == "tcn":
        return DilatedTCN(
            in_channels=IN_CHANNELS,
            window=meta["window"],
            horizon=meta["horizon"],
            kernel_size=meta["kernel_size"],
            depth=meta["depth"],
            channels=meta["channels"],
            dropout=meta["dropout"],
        )
    return build_seq_forecaster(
        kind,
        in_channels=IN_CHANNELS,
        window=meta["window"],
        horizon=meta["horizon"],
        hidden_size=meta["hidden_size"],
    )


def save_checkpoint(path: Path, model: nn.Module, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    meta = json.loads(path.with_suffix(".json").read_text())
    model = build_model(meta)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, meta


def predict_lambda(
    model: nn.Module,
    meta: dict[str, Any],
    features: NDArray[np.float32],
    origins: NDArray[np.integer],
    *,
    batch: int = 4096,
) -> NDArray[np.float64]:
    """lambda-hat ``(len(origins), H)`` in req/s for windows ending at each origin.

    The window for origin ``t`` is ``features[t-L+1 : t+1]``: nothing after
    ``t`` is read.
    """

    window = int(meta["window"])
    origins = np.asarray(origins, dtype=np.int64)
    if origins.size and origins.min() < window - 1:
        raise ValueError("origin has fewer than L ticks of history")
    if origins.size and origins.max() >= features.shape[0]:
        raise ValueError("origin beyond the end of the feature series")
    offsets = np.arange(-window + 1, 1)
    out = np.empty((origins.size, int(meta["horizon"])), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for lo in range(0, origins.size, batch):
            idx = origins[lo : lo + batch, None] + offsets[None, :]
            x = torch.from_numpy(np.transpose(features[idx], (0, 2, 1)).copy())
            z = model(x)[:, :, 0].numpy().astype(np.float64)
            out[lo : lo + batch] = z * float(meta["lam_std"]) + float(meta["lam_mean"])
    return out
