"""Fixed replica count sized to the trace peak (Phase 8)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pact.baselines.common import Scaler, apply_pool_bounds
from pact.config import PactConfig
from pact.telemetry.collector import Observation


def peak_replicas(arrival_rates: Sequence[float], config: PactConfig) -> int:
    """``ceil(λ_peak / (μ · u*))``, clipped to pool bounds."""

    if not arrival_rates:
        raise ValueError("arrival_rates must be non-empty")
    peak = max(float(x) for x in arrival_rates)
    mu = config.capacity.mu
    u_target = config.capacity.u_target
    if mu <= 0.0 or u_target <= 0.0:
        raise ValueError("mu and u_target must be positive")
    raw = math.ceil(peak / (mu * u_target)) if peak > 0.0 else config.control.n_min
    return max(config.control.n_min, min(config.control.n_max, raw))


class StaticScaler(Scaler):
    name = "static"

    def __init__(self, config: PactConfig, n_fixed: int) -> None:
        self._config = config
        self._n = max(config.control.n_min, min(config.control.n_max, n_fixed))

    @classmethod
    def sized_to_peak(
        cls, config: PactConfig, arrival_rates: Sequence[float]
    ) -> StaticScaler:
        return cls(config, peak_replicas(arrival_rates, config))

    def desired_replicas(
        self, obs: Observation, n_current: int, now_s: float
    ) -> int:
        del obs, now_s
        return apply_pool_bounds(n_current, self._n, self._config)
