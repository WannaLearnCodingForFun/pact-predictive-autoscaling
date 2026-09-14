"""Shared pool bounds for every baseline. Decision logic must not bypass these."""

from __future__ import annotations

from pact.config import PactConfig
from pact.telemetry.collector import Observation


def apply_pool_bounds(n_current: int, desired: int, config: PactConfig) -> int:
    """Clip to ``[n_min, n_max]`` then to ``±Δ_max`` of the current count."""

    ctrl = config.control
    bounded = max(ctrl.n_min, min(ctrl.n_max, desired))
    delta = bounded - n_current
    if delta > ctrl.delta_max:
        delta = ctrl.delta_max
    elif delta < -ctrl.delta_max:
        delta = -ctrl.delta_max
    return n_current + delta


class Scaler:
    """Decision mechanism. Plant, Δt, telemetry, and bounds are the caller's."""

    name: str = "scaler"

    def desired_replicas(
        self, obs: Observation, n_current: int, now_s: float
    ) -> int:
        raise NotImplementedError
