"""Capacity mapping (Module 3): forecast → replica demand, Algorithm 1 lines 7–16.

Pure functions, no state. Latency uses the Kingman approximation (Eq. 18).
The Eq. 19 horizon check belongs at startup so a too-short forecast cannot run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pact.config import PactConfig
from pact.sim.queue_model import kingman_p95_ms


class HorizonTooShortError(ValueError):
    """Raised when H·Δt does not cover cold start plus one control interval."""


def assert_horizon_actionable(
    config: PactConfig, *, dt_ctrl: float | None = None
) -> None:
    """Eq. 19: ``H · Δt ≥ τc + Δt_ctrl``. Call at process start, not per tick."""

    dt = config.telemetry.dt
    control_dt = dt if dt_ctrl is None else dt_ctrl
    horizon_s = config.forecast.horizon * dt
    tau_c = config.control.tau_c_s
    if horizon_s < tau_c + control_dt:
        raise HorizonTooShortError(
            f"Horizon {horizon_s}s does not cover cold start {tau_c}s — "
            "forecast is not actionable"
        )


def latency_est(n: int, arrival_rate: float, config: PactConfig) -> float:
    """p95 latency in ms (Eq. 18)."""

    cap = config.capacity
    return kingman_p95_ms(
        n, arrival_rate, mu=cap.mu, ca2=cap.ca2, cs2=cap.cs2
    )


def map_to_demand(
    forecast: Sequence[Sequence[float]],
    n_current: int,
    gamma: float,
    cfg: PactConfig,
) -> list[int]:
    """Map a CPU/memory forecast ``(H, 2)`` to per-horizon replica demand.

    ``N(t)`` is ``n_current``. ``γ`` is the adaptive margin. No internal state.
    """

    if n_current < 0:
        raise ValueError(f"n_current must be non-negative, got {n_current}")
    if not forecast:
        raise ValueError("forecast must be non-empty")

    cap = cfg.capacity
    n_min = cfg.control.n_min
    n_max = cfg.control.n_max
    if cap.u_target <= 0.0 or cap.r_target <= 0.0 or cap.mu <= 0.0:
        raise ValueError("u_target, r_target, and mu must be positive")

    demand: list[int] = []
    for step in forecast:
        if len(step) < 2:
            raise ValueError("each forecast step must be [u_hat, r_hat]")
        u_hat = max(float(step[0]), 0.0)
        r_hat = max(float(step[1]), 0.0)
        # Eq. 14
        lam_hat = u_hat * n_current * cap.mu
        # Eq. 15–16
        n_cpu = math.ceil((1.0 + gamma) * lam_hat / (cap.mu * cap.u_target))
        n_mem = math.ceil(r_hat * n_current / cap.r_target)
        # Eq. 17
        n = max(n_cpu, n_mem)
        # Eq. 18: n strictly increases and is bounded by n_max, so this halts.
        while (
            latency_est(n, lam_hat, cfg) > cap.slo_ms and n < n_max
        ):
            n += 1
        demand.append(_clip(n, n_min, n_max))
    return demand


def _clip(n: int, n_min: int, n_max: int) -> int:
    return max(n_min, min(n_max, n))
