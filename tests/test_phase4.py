"""Phase 4 acceptance tests (CURSOR_SPEC.md §4)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from pact.capacity.mapper import (
    HorizonTooShortError,
    assert_horizon_actionable,
    map_to_demand,
)
from pact.config import ForecastConfig

from tests.helpers import make_config


def test_zero_forecast_returns_n_min() -> None:
    cfg = make_config(n_min=1, n_max=16, mu=40.0)
    horizon = cfg.forecast.horizon
    forecast = [[0.0, 0.0] for _ in range(horizon)]
    demand = map_to_demand(forecast, n_current=4, gamma=0.0, cfg=cfg)
    assert demand == [cfg.control.n_min] * horizon


def test_overload_forecast_returns_n_max_without_hanging() -> None:
    cfg = make_config(n_min=1, n_max=16, mu=40.0)
    # û ≫ 1 ⇒ λ̂ far above what n_max can serve; Kingman latency is inf.
    forecast = [[50.0, 1.0] for _ in range(cfg.forecast.horizon)]
    demand = map_to_demand(forecast, n_current=16, gamma=0.0, cfg=cfg)
    assert demand == [cfg.control.n_max] * cfg.forecast.horizon
    assert len(demand) == cfg.forecast.horizon


def test_raising_gamma_never_decreases_demand() -> None:
    cfg = make_config(n_min=1, n_max=16, mu=40.0)
    forecast = [[0.80, 0.40] for _ in range(cfg.forecast.horizon)]
    n_current = 4
    d0 = map_to_demand(forecast, n_current, 0.0, cfg)
    d1 = map_to_demand(forecast, n_current, 0.20, cfg)
    d2 = map_to_demand(forecast, n_current, 0.50, cfg)
    for a, b, c in zip(d0, d1, d2, strict=True):
        assert a <= b <= c


def test_horizon_assertion_fires_when_h_too_small() -> None:
    # H·Δt = 2·5 = 10s; τc + Δt_ctrl = 20 + 5 = 25s.
    short = replace(
        make_config(dt=5.0, tau_c_s=20.0),
        forecast=ForecastConfig(horizon=2),
    )
    with pytest.raises(HorizonTooShortError, match="not actionable"):
        assert_horizon_actionable(short)

    # Default H=12, Δt=5, τc=10 → 60s >= 15s, actionable.
    assert_horizon_actionable(make_config(dt=5.0, tau_c_s=10.0))
