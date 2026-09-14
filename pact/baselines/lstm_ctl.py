"""LSTM forecast + Module 3 mapping with a fixed margin (Phase 8).

The LSTM is sized to match the TCN parameter count and trained with plain
(symmetric) Huber — ``κ = 1``, no extra under-prediction weight.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from pact.baselines.common import Scaler, apply_pool_bounds
from pact.capacity.mapper import map_to_demand
from pact.config import PactConfig
from pact.forecast.losses import AsymmetricHorizonHuber
from pact.forecast.seq_models import RecurrentForecaster, hidden_size_matching_params
from pact.forecast.tcn import DilatedTCN
from pact.forecast.train import chronological_window_splits, seed_everything
from pact.telemetry.collector import Observation
from pact.telemetry.features import FEATURE_DIM, SlidingWindow, feature_vector


class LstmController(Scaler):
    name = "lstm"

    def __init__(
        self,
        config: PactConfig,
        model: nn.Module,
        *,
        gamma: float = 0.0,
    ) -> None:
        self._config = config
        self._model = model
        self._model.eval()
        self._gamma = gamma
        self._window = SlidingWindow(config.telemetry.window, FEATURE_DIM)

    def desired_replicas(
        self, obs: Observation, n_current: int, now_s: float
    ) -> int:
        del now_s
        self._window.push(feature_vector(obs))
        if self._window.full:
            tensor = np.asarray(self._window.tensor(), dtype=np.float32)
            forecast = _predict(self._model, tensor)
        else:
            horizon = self._config.forecast.horizon
            forecast = [(float(obs.u), float(obs.r))] * horizon
        demand = map_to_demand(forecast, n_current, self._gamma, self._config)
        return apply_pool_bounds(n_current, demand[0], self._config)


def lstm_matching_tcn(config: PactConfig, tcn: DilatedTCN) -> RecurrentForecaster:
    hidden = hidden_size_matching_params(
        "lstm",
        target_params=tcn.parameter_count(),
        in_channels=tcn.in_channels,
        window=config.telemetry.window,
        horizon=config.forecast.horizon,
    )
    return RecurrentForecaster(
        kind="lstm",
        in_channels=tcn.in_channels,
        window=config.telemetry.window,
        horizon=config.forecast.horizon,
        hidden_size=hidden,
    )


def train_lstm(
    model: nn.Module,
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    timestamps: NDArray[np.floating],
    config: PactConfig,
    *,
    max_epochs: int = 20,
    patience: int = 5,
    seed: int = 0,
) -> nn.Module:
    """Train with symmetric Huber on the chronological train/val splits only."""

    seed_everything(seed)
    train, val, _test = chronological_window_splits(
        features,
        targets,
        timestamps,
        window=config.telemetry.window,
        horizon=config.forecast.horizon,
    )
    loss_fn = AsymmetricHorizonHuber(
        kappa=1.0, beta=0.0, huber_delta=config.forecast.huber_delta
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.forecast.lr)
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0
    for _epoch in range(max_epochs):
        _run_epoch(model, train.X, train.Y, loss_fn, optimizer)
        val_loss = _run_epoch(model, val.X, val.Y, loss_fn, None)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


def _run_epoch(
    model: nn.Module,
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    loss_fn: AsymmetricHorizonHuber,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    xb = torch.from_numpy(np.transpose(np.asarray(x, dtype=np.float32), (0, 2, 1)))
    yb = torch.from_numpy(np.asarray(y, dtype=np.float32))
    if training:
        assert optimizer is not None
        optimizer.zero_grad()
    with torch.set_grad_enabled(training):
        pred = model(xb)
        loss = loss_fn(pred, yb)
        if training:
            assert optimizer is not None
            loss.backward()
            optimizer.step()
    return float(loss.item())


def _predict(
    model: nn.Module, window: NDArray[np.float32]
) -> list[tuple[float, float]]:
    arr = np.asarray(window, dtype=np.float32)
    x = torch.from_numpy(arr.T.copy()).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(x)
    rows = np.asarray(out[0].detach().cpu().numpy(), dtype=np.float64)
    return [(float(row[0]), float(row[1])) for row in rows]


def windows_from_observations(
    observations: Sequence[Observation],
) -> NDArray[np.float32]:
    return np.asarray([feature_vector(o) for o in observations], dtype=np.float32)
