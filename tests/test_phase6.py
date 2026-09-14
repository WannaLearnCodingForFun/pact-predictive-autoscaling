"""Phase 6 acceptance tests: drift monitor, τ-aligned ring buffer."""

from __future__ import annotations

from dataclasses import replace

from pact.config import PactConfig
from pact.drift.monitor import DriftMonitor

from tests.helpers import make_config


class _RecordingScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def schedule(self) -> None:
        self.calls += 1


def test_ring_buffer_aligns_prediction_at_t_minus_tau() -> None:
    """N_predicted(t | t−τ) is the horizon element issued at t−τ targeting t.

    At issue tick 0, demand[k] = Ñ(1+k), so the prediction of tick 2 is
    demand[1] = 20. Observing 25 at tick 2 yields e = 5.
    """

    cfg = make_config(dt=5.0, tau_c_s=10.0)
    monitor = DriftMonitor(cfg, tau=2)
    monitor.step(n_required_observed=0, predicted_demand=[10, 20, 30, 40])
    monitor.step(n_required_observed=0, predicted_demand=[11, 21, 31, 41])
    snap = monitor.step(n_required_observed=25, predicted_demand=[12, 22, 32, 42])
    assert monitor.predicted_at(target_tick=2, issued_at=0) == 20
    assert snap.error == 5.0
    assert monitor.last_error == 5.0


def test_sustained_under_prediction_drives_gamma_to_max() -> None:
    cfg = _with_drift(make_config(dt=5.0, tau_c_s=5.0), gamma_max=0.50)
    monitor = DriftMonitor(cfg, tau=1, gamma=0.0)
    pred = [3] * cfg.forecast.horizon
    for _ in range(80):
        monitor.step(n_required_observed=8, predicted_demand=pred)
    assert monitor.gamma == cfg.drift.gamma_max


def test_sustained_over_prediction_drives_gamma_to_min() -> None:
    cfg = _with_drift(
        make_config(dt=5.0, tau_c_s=5.0), gamma_min=0.0, gamma_max=0.50
    )
    monitor = DriftMonitor(cfg, tau=1, gamma=0.50)
    pred = [8] * cfg.forecast.horizon
    for _ in range(80):
        monitor.step(n_required_observed=3, predicted_demand=pred)
    assert monitor.gamma == cfg.drift.gamma_min


def test_single_spike_does_not_move_gamma_more_than_kappa_times_spike() -> None:
    kappa = 0.02
    spike = 10.0
    cfg = _with_drift(
        make_config(dt=5.0, tau_c_s=5.0),
        eta=0.2,
        kappa_gamma=kappa,
        gamma_min=0.0,
        gamma_max=0.50,
        xi=100.0,
    )
    monitor = DriftMonitor(cfg, tau=1, gamma=0.0)
    pred = [0] * cfg.forecast.horizon
    monitor.step(n_required_observed=0, predicted_demand=pred)
    monitor.step(n_required_observed=0, predicted_demand=pred)
    before = monitor.gamma
    snap = monitor.step(n_required_observed=int(spike), predicted_demand=pred)
    assert snap.error == spike
    # ē = η·spike = 2, Δγ = κ_γ·2 = 0.04 < κ_γ·spike = 0.20
    delta = snap.gamma - before
    assert delta == kappa * cfg.drift.eta * spike
    assert delta <= kappa * spike
    for _ in range(40):
        monitor.step(n_required_observed=0, predicted_demand=pred)
    assert monitor.gamma <= kappa * spike + 1e-12


def test_drift_trigger_schedules_refit_after_w_consecutive() -> None:
    scheduler = _RecordingScheduler()
    cfg = _with_drift(
        make_config(dt=5.0, tau_c_s=5.0),
        eta=1.0,
        xi=1.0,
        drift_window=5,
        kappa_gamma=0.0,
    )
    monitor = DriftMonitor(cfg, tau=1, scheduler=scheduler)
    pred = [0] * cfg.forecast.horizon
    monitor.step(0, pred)
    for _ in range(4):
        monitor.step(n_required_observed=3, predicted_demand=pred)
        assert scheduler.calls == 0
    monitor.step(n_required_observed=3, predicted_demand=pred)
    assert scheduler.calls == 1
    monitor.step(n_required_observed=3, predicted_demand=pred)
    assert scheduler.calls == 1


def _with_drift(
    cfg: PactConfig,
    *,
    eta: float | None = None,
    kappa_gamma: float | None = None,
    gamma_min: float | None = None,
    gamma_max: float | None = None,
    xi: float | None = None,
    drift_window: int | None = None,
) -> PactConfig:
    d = cfg.drift
    return replace(
        cfg,
        drift=replace(
            d,
            eta=d.eta if eta is None else eta,
            kappa_gamma=d.kappa_gamma if kappa_gamma is None else kappa_gamma,
            gamma_min=d.gamma_min if gamma_min is None else gamma_min,
            gamma_max=d.gamma_max if gamma_max is None else gamma_max,
            xi=d.xi if xi is None else xi,
            drift_window=d.drift_window if drift_window is None else drift_window,
        ),
    )
