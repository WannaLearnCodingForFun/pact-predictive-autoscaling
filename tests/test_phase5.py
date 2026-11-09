"""Remaining Phase 5 acceptance tests (CURSOR_SPEC.md §5)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

import pytest
from pact.config import PactConfig, cold_start_ticks
from pact.control.gates import apply_gates
from pact.control.mpc import apply_plant, solve_mpc
from pact.control.safety import evaluate_safety, immediate_scale_up
from pact.eval.metrics import scale_action_count

from tests.helpers import make_config


def test_alternating_demand_inside_deadband_is_zero_actions() -> None:
    # |u| < θ zeros ±1, so θ=2 is the integer deadband around a 1-replica wobble.
    cfg = _control(
        make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16),
        deadband=2,
        cooldown_s=0.0,
    )
    series = [4 + (t % 2) for t in range(80)]
    actions, _n = _closed_loop(series, cfg, utilisation=[0.0] * 80)
    assert all(u == 0 for u in actions)


def test_raising_s_churn_monotonically_decreases_action_count() -> None:
    series = [5 + ((t * 7 + 3) % 5) - 2 for t in range(90)]
    counts: list[int] = []
    for s_churn in (0.0, 0.2, 0.45):
        cfg = _control(
            make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16),
            s_churn=s_churn,
            deadband=0,
            cooldown_s=0.0,
        )
        _actions, n_hist = _closed_loop(series, cfg, utilisation=[0.0] * 90)
        counts.append(scale_action_count(n_hist))
    assert counts[0] > counts[1] > counts[2]


def test_safety_ceiling_fires_regardless_of_cooldown() -> None:
    cfg = _control(
        make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16),
        u_emergency=0.92,
        cooldown_s=60.0,
        deadband=2,
        delta_max=1,
    )
    n = 4
    # Just scaled up; cooldown would block a scale-down and deadband would
    # swallow a +1, and Δ_max=1 would cap a gated scale-up.
    decision = evaluate_safety(0.99, n, cfg)
    assert decision.active is True
    assert decision.u > cfg.control.delta_max
    gated = apply_gates(
        -3.0, n, cfg.control, now_s=5.0, last_scale_up_s=0.0
    )
    assert gated.u == 0
    assert immediate_scale_up(n, 0.99, cfg) == decision.u


def test_safety_override_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    with caplog.at_level(logging.WARNING, logger="pact.control.safety"):
        decision = evaluate_safety(0.95, 4, cfg)
    assert decision.active is True
    assert decision.u > 0
    assert any("safety override" in rec.message for rec in caplog.records)
    quiet = evaluate_safety(0.92, 4, cfg)
    assert quiet.active is False


def test_safety_at_n_max_logs_and_returns_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    with caplog.at_level(logging.WARNING, logger="pact.control.safety"):
        decision = evaluate_safety(0.99, 16, cfg)
    assert decision.active is True
    assert decision.u == 0
    assert any("safety override" in rec.message for rec in caplog.records)


def test_identical_traces_produce_identical_actions() -> None:
    cfg = _control(
        make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16),
        deadband=1,
        cooldown_s=60.0,
        u_emergency=0.92,
    )
    series = [4 + ((t * 5 + 1) % 6) for t in range(70)]
    utilisation = [0.50] * 70
    utilisation[30] = 0.99
    utilisation[31] = 0.97
    first_u, first_n = _closed_loop(series, cfg, utilisation=utilisation)
    second_u, second_n = _closed_loop(series, cfg, utilisation=utilisation)
    assert first_u == second_u
    assert first_n == second_n


def _control(
    cfg: PactConfig,
    *,
    s_churn: float | None = None,
    deadband: int | None = None,
    cooldown_s: float | None = None,
    u_emergency: float | None = None,
    delta_max: int | None = None,
) -> PactConfig:
    ctrl = cfg.control
    return replace(
        cfg,
        control=replace(
            ctrl,
            s_churn=ctrl.s_churn if s_churn is None else s_churn,
            deadband=ctrl.deadband if deadband is None else deadband,
            cooldown_s=ctrl.cooldown_s if cooldown_s is None else cooldown_s,
            u_emergency=(
                ctrl.u_emergency if u_emergency is None else u_emergency
            ),
            delta_max=ctrl.delta_max if delta_max is None else delta_max,
        ),
    )


def _closed_loop(
    series: Sequence[int],
    cfg: PactConfig,
    *,
    utilisation: Sequence[float],
) -> tuple[list[int], list[int]]:
    tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
    horizon = cfg.forecast.horizon
    n = int(series[0])
    pending = [0] * tau
    last_u = 0
    last_scale_up_s: float | None = None
    actions: list[int] = []
    replicas: list[int] = [n]
    dt = cfg.telemetry.dt
    ticks = len(series) - horizon - 1
    for t in range(ticks):
        now_s = t * dt
        util = float(utilisation[t])
        safety = evaluate_safety(util, n, cfg)
        if safety.active:
            u = safety.u
            if u > 0:
                last_scale_up_s = now_s
        else:
            forecast = [int(series[t + 1 + k]) for k in range(horizon)]
            sol = solve_mpc(n, forecast, pending, last_u, cfg)
            gated = apply_gates(
                sol.u,
                n,
                cfg.control,
                now_s=now_s,
                last_scale_up_s=last_scale_up_s,
            )
            u = gated.u
            last_scale_up_s = gated.last_scale_up_s
        actions.append(u)
        n, pending = apply_plant(
            n, pending, u, tau, cfg.control.n_min, cfg.control.n_max
        )
        last_u = u
        replicas.append(n)
    return actions, replicas
