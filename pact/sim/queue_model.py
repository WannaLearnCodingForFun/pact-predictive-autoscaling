"""Analytical M/M/c-style pool simulator with cold-start delay.

A replica requested at tick t contributes nothing until tick
``t + ceil(τc / Δt)``. Latency uses the Kingman / Allen–Cunneen approximation
(Eq. 18).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pact.config import PactConfig, cold_start_ticks

# p95 of an exponential with mean T is -ln(0.05) · T.
_P95_EXPONENTIAL = -math.log(0.05)


@dataclass(frozen=True)
class PoolState:
    utilisation: float
    memory_fraction: float
    p95_latency_ms: float
    requests_served: float
    n_ready: int
    n_pending: int


class PoolSimulator:
    """Pool of N servers. ``n_replicas`` is the desired count, not the ready count."""

    def __init__(
        self,
        config: PactConfig,
        *,
        n_initial: int | None = None,
    ) -> None:
        self._capacity = config.capacity
        self._telemetry = config.telemetry
        self._control = config.control
        self._sim = config.simulator
        self._delay = cold_start_ticks(config.control.tau_c_s, config.telemetry.dt)
        initial = (
            self._control.n_min if n_initial is None else n_initial
        )
        self._n_ready = _clip(
            initial, self._control.n_min, self._control.n_max
        )
        self._pending: list[int] = []
        self._tick = 0
        self._last_state: PoolState | None = None
        self._last_arrival_rate: float | None = None

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def delay_ticks(self) -> int:
        return self._delay

    @property
    def n_ready(self) -> int:
        return self._n_ready

    @property
    def last_state(self) -> PoolState | None:
        return self._last_state

    @property
    def last_arrival_rate(self) -> float | None:
        return self._last_arrival_rate

    def step(self, arrival_rate: float, n_replicas: int) -> PoolState:
        """Advance one control interval.

        Replicas requested this tick become ready at ``tick + delay_ticks``.
        """

        if arrival_rate < 0.0:
            raise ValueError(f"arrival_rate must be non-negative, got {arrival_rate}")
        self._reconcile(n_replicas)
        self._promote(self._tick)
        state = self._observe(arrival_rate)
        self._last_state = state
        self._last_arrival_rate = arrival_rate
        self._tick += 1
        return state

    def _reconcile(self, n_replicas: int) -> None:
        desired = _clip(n_replicas, self._control.n_min, self._control.n_max)
        current = self._n_ready + len(self._pending)
        if desired > current:
            ready_at = self._tick + self._delay
            self._pending.extend([ready_at] * (desired - current))
        elif desired < current:
            to_remove = current - desired
            cancelled = min(to_remove, len(self._pending))
            if cancelled:
                del self._pending[-cancelled:]
            remaining = to_remove - cancelled
            self._n_ready -= remaining

    def _promote(self, tick: int) -> None:
        still_pending: list[int] = []
        for ready_at in self._pending:
            if ready_at <= tick:
                self._n_ready += 1
            else:
                still_pending.append(ready_at)
        self._pending = still_pending

    def _observe(self, arrival_rate: float) -> PoolState:
        n = self._n_ready
        mu = self._capacity.mu
        capacity_rate = n * mu
        served_rate = min(arrival_rate, capacity_rate) if n > 0 else 0.0
        utilisation = 0.0 if capacity_rate == 0.0 else served_rate / capacity_rate
        memory = self._sim.mem_baseline + self._sim.mem_load_coeff * utilisation
        latency = kingman_p95_ms(
            n,
            arrival_rate,
            mu=mu,
            ca2=self._capacity.ca2,
            cs2=self._capacity.cs2,
        )
        requests = served_rate * self._telemetry.dt
        return PoolState(
            utilisation=utilisation,
            memory_fraction=memory,
            p95_latency_ms=latency,
            requests_served=requests,
            n_ready=n,
            n_pending=len(self._pending),
        )


def kingman_p95_ms(
    n: int,
    arrival_rate: float,
    *,
    mu: float,
    ca2: float,
    cs2: float,
) -> float:
    """p95 sojourn time (ms) from the Kingman approximation (Eq. 18).

    Waiting time:

    ``W_q = ((c_a² + c_s²) / 2) · (ρ / (1 − ρ)) · (1 / (n μ))``

    for ``ρ = λ / (n μ) < 1``. Sojourn mean is ``1/μ + W_q``. p95 assumes an
    exponential tail, which is exact for M/M/1 (c_a² = c_s² = 1, n = 1).
    """

    if n <= 0 or mu <= 0.0:
        return math.inf
    rho = arrival_rate / (n * mu)
    if rho >= 1.0:
        return math.inf
    if rho < 0.0:
        raise ValueError(f"arrival_rate must be non-negative, got {arrival_rate}")
    wait_s = ((ca2 + cs2) / 2.0) * (rho / (1.0 - rho)) * (1.0 / (n * mu))
    sojourn_s = (1.0 / mu) + wait_s
    return _P95_EXPONENTIAL * sojourn_s * 1000.0


def _clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))
