"""Phase 1 acceptance tests (CURSOR_SPEC.md §1.3)."""

from __future__ import annotations

import math

import pytest
from pact.sim.queue_model import PoolSimulator, kingman_p95_ms
from pact.sim.replay import TraceReplay

from tests.helpers import make_config


def test_step_increase_raises_utilisation_and_latency() -> None:
    """Fixed replica count: a step-up in arrival rate raises u and p95."""

    cfg = make_config(dt=5.0, mu=40.0, tau_c_s=10.0)
    sim = PoolSimulator(cfg, n_initial=2)
    low = sim.step(arrival_rate=20.0, n_replicas=2)
    high = sim.step(arrival_rate=60.0, n_replicas=2)

    assert low.n_ready == 2
    assert high.n_ready == 2
    assert high.utilisation > low.utilisation
    assert high.p95_latency_ms > low.p95_latency_ms
    # ρ_low = 20/(2*40) = 0.25; ρ_high = 60/80 = 0.75
    assert low.utilisation == pytest.approx(0.25)
    assert high.utilisation == pytest.approx(0.75)


def test_cold_start_replica_has_no_effect_until_ready() -> None:
    """Replica requested at t is inert at t+τ/Δt−1 and live at t+τ/Δt."""

    dt = 5.0
    tau_c_s = 10.0
    delay = math.ceil(tau_c_s / dt)  # 2
    cfg = make_config(dt=dt, mu=40.0, tau_c_s=tau_c_s)
    sim = PoolSimulator(cfg, n_initial=1)
    assert sim.delay_ticks == delay

    arrival = 20.0
    # Request the extra replica at tick t = 0.
    at_request = sim.step(arrival_rate=arrival, n_replicas=2)
    before_ready: list[float] = []
    before_n: list[int] = []
    for _ in range(delay - 1):
        state = sim.step(arrival_rate=arrival, n_replicas=2)
        before_ready.append(state.utilisation)
        before_n.append(state.n_ready)

    at_ready = sim.step(arrival_rate=arrival, n_replicas=2)

    assert at_request.n_ready == 1
    assert at_request.n_pending == 1
    assert before_n == [1]
    assert before_ready[0] == pytest.approx(at_request.utilisation)
    assert at_ready.n_ready == 2
    assert at_ready.n_pending == 0
    assert at_ready.utilisation < at_request.utilisation
    assert at_ready.p95_latency_ms < at_request.p95_latency_ms
    assert at_request.utilisation == pytest.approx(0.5)
    assert at_ready.utilisation == pytest.approx(0.25)


def test_replay_drives_simulator_from_trace() -> None:
    """Replay folds a trace at the requested compression and steps the pool."""

    cfg = make_config(dt=5.0, mu=40.0, tau_c_s=0.0)
    sim = PoolSimulator(cfg, n_initial=2)
    replay = TraceReplay(
        arrival_rates=[10.0, 30.0, 20.0, 40.0],
        n_replicas=2,
        compression=2,
    )
    result = replay.run(sim)

    assert result.arrival_rates == pytest.approx((20.0, 30.0))
    assert result.n_replicas == (2, 2)
    assert len(result.states) == 2
    assert result.states[0].n_ready == 2
    assert result.states[0].utilisation == pytest.approx(20.0 / 80.0)
    assert result.states[1].utilisation == pytest.approx(30.0 / 80.0)


def test_kingman_mm1_matches_closed_form() -> None:
    # M/M/1: E[T] = 1/(μ−λ); p95 = −ln(0.05) · E[T] · 1000 ms
    mu = 40.0
    lam = 20.0
    expected = -math.log(0.05) / (mu - lam) * 1000.0
    got = kingman_p95_ms(1, lam, mu=mu, ca2=1.0, cs2=1.0)
    assert got == pytest.approx(expected)
