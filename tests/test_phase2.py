"""Phase 2 acceptance tests (CURSOR_SPEC.md §2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pact.config import PrometheusConfig
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import (
    Observation,
    PrometheusCollector,
    SimCollector,
    counter_to_rate,
    cpu_ns_counter_to_cores,
)
from pact.telemetry.features import (
    FEATURE_DIM,
    EWMASmoother,
    FrozenNormaliserError,
    MinMaxNormaliser,
    SlidingWindow,
    feature_vector,
    rho,
    time_of_day_features,
)
from pact.telemetry.prometheus import PrometheusClient

from tests.helpers import make_config


def test_normaliser_refuses_fit_after_freeze() -> None:
    norm = MinMaxNormaliser()
    # min=[0, 10], max=[10, 20] — (5, 15) → (0.5, 0.5)
    norm.fit([[0.0, 10.0], [10.0, 20.0]])
    assert norm.transform([5.0, 15.0]) == pytest.approx([0.5, 0.5])
    norm.freeze()
    with pytest.raises(FrozenNormaliserError):
        norm.fit([[1.0, 2.0]])
    # Frozen stats still transform.
    assert norm.transform([10.0, 20.0]) == pytest.approx([1.0, 1.0])


def test_incremental_window_matches_naive() -> None:
    length = 3
    window = SlidingWindow(length=length, n_channels=2)
    history: list[list[float]] = []
    samples = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]
    for sample in samples:
        window.push(sample)
        history.append(sample)
        naive = history[-length:]
        assert window.tensor() == naive


def test_counter_to_rate_handles_reset() -> None:
    dt = 5.0
    # First sample: no previous → rate 0.
    rate0, prev = counter_to_rate(100.0, None, dt)
    assert rate0 == 0.0
    # 350 − 100 over 5 s → 50 / s
    rate1, prev = counter_to_rate(350.0, prev, dt)
    assert rate1 == pytest.approx(50.0)
    # Container restart: counter drops to 10. Must not be (10 − 350) / 5 = −68.
    rate_reset, prev = counter_to_rate(10.0, prev, dt)
    assert rate_reset >= 0.0
    assert rate_reset == pytest.approx(10.0 / dt)
    # Resume after restart: 60 − 10 over 5 s → 10 / s
    rate2, _ = counter_to_rate(60.0, prev, dt)
    assert rate2 == pytest.approx(10.0)
    assert rate_reset >= 0.0 and rate1 >= 0.0 and rate2 >= 0.0


def test_cpu_ns_counter_to_cores_handles_reset() -> None:
    dt = 5.0
    ns = 1_000_000_000.0
    # 2 cores for 5 s → 10e9 ns; rate = 2 cores
    cores1, prev = cpu_ns_counter_to_cores(10.0 * ns, 0.0, dt)
    assert cores1 == pytest.approx(2.0)
    cores_reset, _ = cpu_ns_counter_to_cores(0.0, prev, dt)
    assert cores_reset >= 0.0
    assert cores_reset == pytest.approx(0.0)


def test_ewma_hand_computed() -> None:
    # α = 0.5: s0 = 1; s1 = 0.5·3 + 0.5·1 = 2
    smoother = EWMASmoother(alpha=0.5)
    assert smoother.update([1.0]) == pytest.approx([1.0])
    assert smoother.update([3.0]) == pytest.approx([2.0])


def test_rho_and_time_of_day_in_feature_vector() -> None:
    assert rho(40.0, 2.0) == pytest.approx(20.0)
    assert rho(40.0, 0.0) == pytest.approx(40.0)
    sin_t, cos_t = time_of_day_features(0.0)
    assert sin_t == pytest.approx(0.0)
    assert cos_t == pytest.approx(1.0)
    sin_6h, cos_6h = time_of_day_features(21600.0)
    assert sin_6h == pytest.approx(1.0)
    assert cos_6h == pytest.approx(0.0)
    obs = Observation(u=0.4, r=0.5, lam=40.0, ell_p95=100.0, n=2.0, t_s=0.0)
    vec = feature_vector(obs)
    assert len(vec) == FEATURE_DIM
    assert vec[5] == pytest.approx(20.0)
    assert vec[6] == pytest.approx(0.0)
    assert vec[7] == pytest.approx(1.0)


def test_sim_collector_reads_pool_state() -> None:
    cfg = make_config(dt=5.0, mu=40.0, tau_c_s=0.0)
    sim = PoolSimulator(cfg, n_initial=2)
    collector = SimCollector(sim)
    with pytest.raises(RuntimeError):
        collector.collect(0.0)
    state = sim.step(arrival_rate=20.0, n_replicas=2)
    obs = collector.collect(t_s=12.0)
    assert obs.u == pytest.approx(state.utilisation)
    assert obs.r == pytest.approx(state.memory_fraction)
    assert obs.lam == pytest.approx(20.0)
    assert obs.ell_p95 == pytest.approx(state.p95_latency_ms)
    assert obs.n == pytest.approx(2.0)
    assert obs.t_s == pytest.approx(12.0)
    assert obs.as_vector() == (obs.u, obs.r, obs.lam, obs.ell_p95, obs.n)


def test_prometheus_collector_rates_and_reset() -> None:
    cfg = make_config(dt=5.0)
    prom = cfg.prometheus
    values = {
        prom.cpu_counter_query: 10.0,
        prom.memory_working_set_query: 50.0,
        prom.memory_limit_query: 100.0,
        prom.requests_total_query: 100.0,
        prom.latency_p95_query: 80.0,
        prom.replica_count_query: 2.0,
        prom.network_rx_bytes_query: 1000.0,
        prom.network_tx_bytes_query: 2000.0,
        prom.fs_read_bytes_query: 3000.0,
        prom.fs_write_bytes_query: 4000.0,
    }
    collector = PrometheusCollector(cfg, client=_StubProm(values))
    first = collector.collect(0.0)
    assert first.lam == pytest.approx(0.0)
    assert first.r == pytest.approx(0.5)
    assert first.n == pytest.approx(2.0)

    values[prom.cpu_counter_query] = 20.0
    values[prom.requests_total_query] = 200.0
    second = collector.collect(5.0)
    # Δrequests = 100 over 5 s → λ = 20; Δcpu = 10 over 5 s → 2 cores; u = 2/2
    assert second.lam == pytest.approx(20.0)
    assert second.u == pytest.approx(1.0)

    values[prom.cpu_counter_query] = 1.0
    values[prom.requests_total_query] = 5.0
    reset = collector.collect(10.0)
    assert reset.lam >= 0.0
    assert reset.u >= 0.0
    assert reset.lam == pytest.approx(5.0 / 5.0)


def test_normaliser_persist(tmp_path: Path) -> None:
    path = tmp_path / "norm.json"
    norm = MinMaxNormaliser()
    norm.fit([[0.0], [4.0]])
    norm.freeze()
    norm.dump(path)
    loaded = MinMaxNormaliser.load(path)
    assert loaded.frozen
    assert loaded.transform([2.0]) == pytest.approx([0.5])
    with pytest.raises(FrozenNormaliserError):
        loaded.fit([[0.0]])


def test_prometheus_client_parses_vector() -> None:

    payload = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"name": "a"}, "value": [1.0, "2.5"]},
                {"metric": {"name": "b"}, "value": [1.0, "1.5"]},
            ],
        },
    }
    client = PrometheusClient(
        PrometheusConfig(),
        fetch=lambda _url: payload,
    )
    assert client.scalar("up") == pytest.approx(4.0)


class _StubProm:
    def __init__(self, values: dict[str, float]) -> None:
        self._values = values

    def scalar(self, query: str) -> float:
        return self._values[query]
