"""ARIMA forecast + Module 3 mapping with a fixed margin (Phase 8)."""

from __future__ import annotations

from collections.abc import Sequence

from pact.baselines.common import Scaler, apply_pool_bounds
from pact.capacity.mapper import map_to_demand
from pact.config import PactConfig
from pact.telemetry.collector import Observation


class ArimaController(Scaler):
    name = "arima"

    def __init__(
        self,
        config: PactConfig,
        *,
        order: tuple[int, int, int] = (1, 0, 0),
        gamma: float = 0.0,
        min_history: int = 8,
    ) -> None:
        self._config = config
        self._order = order
        self._gamma = gamma
        self._min_history = min_history
        self._u: list[float] = []
        self._r: list[float] = []

    def desired_replicas(
        self, obs: Observation, n_current: int, now_s: float
    ) -> int:
        del now_s
        self._u.append(float(obs.u))
        self._r.append(float(obs.r))
        horizon = self._config.forecast.horizon
        u_hat = _forecast_series(self._u, horizon, self._order, self._min_history)
        r_hat = _forecast_series(self._r, horizon, self._order, self._min_history)
        forecast = list(zip(u_hat, r_hat, strict=True))
        demand = map_to_demand(forecast, n_current, self._gamma, self._config)
        return apply_pool_bounds(n_current, demand[0], self._config)


def select_arima_order(
    train: Sequence[float],
    val: Sequence[float],
    *,
    budget: int,
) -> tuple[int, int, int]:
    """Pick ``(p, d, q)`` on the validation split. At most ``budget`` fits."""

    if budget < 1:
        raise ValueError("budget must be >= 1")
    candidates = (
        (1, 0, 0),
        (2, 0, 0),
        (1, 0, 1),
        (0, 1, 1),
        (2, 1, 0),
        (1, 1, 1),
    )[:budget]
    best = candidates[0]
    best_mae = float("inf")
    train_list = [float(x) for x in train]
    val_list = [float(x) for x in val]
    for order in candidates:
        history = list(train_list)
        abs_err = 0.0
        for actual in val_list:
            pred = _forecast_series(history, 1, order, min_history=3)[0]
            abs_err += abs(pred - actual)
            history.append(actual)
        mae = abs_err / max(len(val_list), 1)
        if mae < best_mae:
            best_mae = mae
            best = order
    return best


def _forecast_series(
    history: Sequence[float],
    steps: int,
    order: tuple[int, int, int],
    min_history: int,
) -> list[float]:
    series = [float(x) for x in history]
    last = series[-1] if series else 0.0
    if len(series) < min_history or steps < 1:
        return [last] * max(steps, 0)
    try:
        from statsmodels.tsa.arima.model import ARIMA

        fitted = ARIMA(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        fc = fitted.forecast(steps=steps)
        values = [max(float(x), 0.0) for x in fc]
        if len(values) < steps:
            pad = values[-1] if values else last
            values.extend([pad] * (steps - len(values)))
        return values[:steps]
    except Exception:
        return [last] * steps
