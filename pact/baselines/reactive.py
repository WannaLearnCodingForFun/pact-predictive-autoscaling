"""Kubernetes HPA control law (Phase 8).

``desired = ceil(N · u_observed / u_target)``, then the 10% tolerance band and
a scale-down stabilisation window. Reference: Kubernetes HPA algorithm.
Pool bounds and ``Δ_max`` are applied last via ``apply_pool_bounds``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pact.baselines.common import Scaler, apply_pool_bounds
from pact.config import PactConfig
from pact.telemetry.collector import Observation


@dataclass(frozen=True, kw_only=True)
class HpaConfig:
    tolerance: float = 0.1
    scale_down_window_s: float = 300.0
    scale_up_window_s: float = 0.0


class ReactiveHPA(Scaler):
    name = "reactive"

    def __init__(self, config: PactConfig, *, hpa: HpaConfig | None = None) -> None:
        self._config = config
        self._hpa = hpa or HpaConfig()
        self._history: list[tuple[float, int]] = []

    def recommend(self, obs: Observation, n_current: int) -> int:
        """Raw HPA recommendation before stabilisation and rate limits."""

        target = self._config.capacity.u_target
        if target <= 0.0:
            raise ValueError("u_target must be positive")
        n = max(n_current, 1)
        ratio = obs.u / target
        if abs(ratio - 1.0) <= self._hpa.tolerance:
            return n_current
        desired = math.ceil(n * ratio)
        return max(self._config.control.n_min, min(self._config.control.n_max, desired))

    def desired_replicas(
        self, obs: Observation, n_current: int, now_s: float
    ) -> int:
        recommended = self.recommend(obs, n_current)
        stabilized = self._stabilize(recommended, n_current, now_s)
        return apply_pool_bounds(n_current, stabilized, self._config)

    def _stabilize(self, recommended: int, n_current: int, now_s: float) -> int:
        self._history.append((now_s, recommended))
        if recommended < n_current:
            window = self._hpa.scale_down_window_s
        else:
            window = self._hpa.scale_up_window_s
        cutoff = now_s - window
        recent = [rec for ts, rec in self._history if ts >= cutoff]
        self._history = [(ts, rec) for ts, rec in self._history if ts >= cutoff]
        if not recent:
            return recommended
        # Scale-down: largest rec in the window so we do not drop too fast.
        if recommended < n_current:
            return max(recent)
        if window <= 0.0:
            return recommended
        return max(recent)
