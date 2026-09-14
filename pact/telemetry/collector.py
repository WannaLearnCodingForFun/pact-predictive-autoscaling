"""Observation collection: scrape, cumulative-to-rate conversion, Eq. 1 vector.

Docker/cAdvisor counters are cumulative. CPU is consumed nanoseconds (or
seconds, depending on the exporter); network and block I/O are byte totals.
A decrease is a counter reset (container restart) and must not yield a
negative rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from pact.config import PactConfig, PrometheusConfig
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.prometheus import PrometheusClient

NS_PER_SECOND = 1_000_000_000.0


class ScalarSource(Protocol):
    def scalar(self, query: str) -> float: ...


@dataclass(frozen=True)
class Observation:
    """Observation vector of Eq. 1: ``[u, r, λ, ℓ_p95, N]`` plus a timestamp."""

    u: float
    r: float
    lam: float
    ell_p95: float
    n: float
    t_s: float = 0.0

    def as_vector(self) -> tuple[float, ...]:
        return (self.u, self.r, self.lam, self.ell_p95, self.n)


class Collector(ABC):
    @abstractmethod
    def collect(self, t_s: float) -> Observation:
        """Return the Eq. 1 observation at timestamp ``t_s`` (seconds)."""


def counter_to_rate(
    current: float,
    previous: float | None,
    dt: float,
) -> tuple[float, float]:
    """Convert a cumulative counter to a per-second rate.

    Returns ``(rate, current)`` so the caller stores ``current`` as the next
    previous. The first sample (``previous is None``) yields rate 0.

    If ``current < previous``, the counter reset: the increase is ``current``
    (counts since restart), never ``current - previous``.
    """

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if current < 0.0:
        raise ValueError(f"counter must be non-negative, got {current}")
    if previous is None:
        return 0.0, current
    delta = current if current < previous else current - previous
    return delta / dt, current


def cpu_ns_counter_to_cores(
    current_ns: float,
    previous_ns: float | None,
    dt: float,
) -> tuple[float, float]:
    """CPU consumed-nanoseconds counter → cores used (rate / 1e9)."""

    rate_ns, new_prev = counter_to_rate(current_ns, previous_ns, dt)
    return rate_ns / NS_PER_SECOND, new_prev


class SimCollector(Collector):
    """Reads the latest ``PoolSimulator`` step. No counters to difference."""

    def __init__(self, simulator: PoolSimulator) -> None:
        self._simulator = simulator

    def collect(self, t_s: float) -> Observation:
        state = self._simulator.last_state
        lam = self._simulator.last_arrival_rate
        if state is None or lam is None:
            raise RuntimeError("PoolSimulator has not produced an observation yet")
        return Observation(
            u=state.utilisation,
            r=state.memory_fraction,
            lam=lam,
            ell_p95=state.p95_latency_ms,
            n=float(state.n_ready),
            t_s=t_s,
        )


class PrometheusCollector(Collector):
    """Scrapes cAdvisor via Prometheus and converts counters to rates over Δt."""

    def __init__(
        self,
        config: PactConfig,
        *,
        client: ScalarSource | None = None,
    ) -> None:
        self._dt = config.telemetry.dt
        self._prom: PrometheusConfig = config.prometheus
        self._client = client or PrometheusClient(config.prometheus)
        self._prev: dict[str, float] = {}

    def collect(self, t_s: float) -> Observation:
        cpu_raw = self._client.scalar(self._prom.cpu_counter_query)
        mem_ws = self._client.scalar(self._prom.memory_working_set_query)
        mem_lim = self._client.scalar(self._prom.memory_limit_query)
        req_raw = self._client.scalar(self._prom.requests_total_query)
        ell_p95 = self._client.scalar(self._prom.latency_p95_query)
        n = self._client.scalar(self._prom.replica_count_query)
        # Network and block I/O are cumulative byte totals — convert so a
        # reset cannot leak a negative rate into later channels.
        self._rate("net_rx", self._client.scalar(self._prom.network_rx_bytes_query))
        self._rate("net_tx", self._client.scalar(self._prom.network_tx_bytes_query))
        self._rate("fs_read", self._client.scalar(self._prom.fs_read_bytes_query))
        self._rate("fs_write", self._client.scalar(self._prom.fs_write_bytes_query))

        cpu_rate, self._prev["cpu"] = counter_to_rate(
            cpu_raw, self._prev.get("cpu"), self._dt
        )
        if self._prom.cpu_counter_in_nanoseconds:
            cpu_rate = cpu_rate / NS_PER_SECOND
        lam, self._prev["requests"] = counter_to_rate(
            req_raw, self._prev.get("requests"), self._dt
        )
        n_replicas = max(n, 0.0)
        u = cpu_rate / max(n_replicas, 1.0)
        r = 0.0 if mem_lim <= 0.0 else mem_ws / mem_lim
        return Observation(
            u=u,
            r=r,
            lam=lam,
            ell_p95=ell_p95,
            n=n_replicas,
            t_s=t_s,
        )

    def _rate(self, key: str, current: float) -> float:
        rate, self._prev[key] = counter_to_rate(
            current, self._prev.get(key), self._dt
        )
        return rate
