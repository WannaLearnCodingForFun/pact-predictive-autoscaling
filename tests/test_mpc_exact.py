"""Tests for the globally optimal integer MPC solver.

The substantive test is ``test_matches_brute_force``: it enumerates the whole
(2*Delta_max+1)^H decision space on configurations small enough to do so and
checks that ``solve_mpc_exact`` attains that optimum. Everything the solver
relies on -- dead-time truncation, tail separability, the saturating plant --
is exercised through that comparison rather than asserted separately.
"""

from __future__ import annotations

import itertools
import random

from pact.config import (
    CapacityConfig,
    ControlConfig,
    DriftConfig,
    ForecastConfig,
    PactConfig,
    SimulatorConfig,
    TelemetryConfig,
    cold_start_ticks,
)
from pact.control.mpc import _pending_of_length, _pipe_len, solve_mpc
from pact.control.mpc_exact import (
    landing_lag,
    plan_cost,
    solve_mpc_exact,
    tracking_relevant_actions,
)

TOL = 1e-6


def _cfg(*, dt=5.0, horizon=4, tau_c_s=10.0, delta_max=2, n_min=1, n_max=8,
         r=0.15, s=0.60, q_up=8.0, q_down=1.0) -> PactConfig:
    return PactConfig(
        telemetry=TelemetryConfig(dt=dt),
        forecast=ForecastConfig(horizon=horizon),
        capacity=CapacityConfig(),
        control=ControlConfig(
            tau_c_s=tau_c_s, delta_max=delta_max, n_min=n_min, n_max=n_max,
            r_act=r, s_churn=s, q_up=q_up, q_down=q_down,
        ),
        drift=DriftConfig(),
        simulator=SimulatorConfig(),
    )


def _brute_force(n, demand, pending, last_u, cfg):
    ctrl = cfg.control
    tau = cold_start_ticks(ctrl.tau_c_s, cfg.telemetry.dt)
    horizon = min(len(demand), cfg.forecast.horizon)
    pipe = _pending_of_length(pending, _pipe_len(tau))
    dem = [int(x) for x in demand[:horizon]]
    us = list(range(-ctrl.delta_max, ctrl.delta_max + 1))
    best, best_us = float("inf"), set()
    for plan in itertools.product(us, repeat=horizon):
        c = plan_cost(n, pipe, plan, dem, last_u, tau, ctrl)
        if c is None:
            continue
        if c < best - TOL:
            best, best_us = c, {plan[0]}
        elif abs(c - best) <= TOL:
            best_us.add(plan[0])
    return best, best_us


CASES = [
    dict(horizon=4, tau_c_s=10.0, delta_max=2, n_min=1, n_max=8),
    dict(horizon=5, tau_c_s=10.0, delta_max=2, n_min=1, n_max=8),
    dict(horizon=4, tau_c_s=20.0, delta_max=2, n_min=1, n_max=8),
    dict(horizon=4, tau_c_s=0.0, delta_max=2, n_min=1, n_max=8),
    dict(horizon=5, tau_c_s=15.0, delta_max=2, n_min=2, n_max=6),
    dict(horizon=6, tau_c_s=5.0, delta_max=2, n_min=1, n_max=8),
    dict(horizon=6, tau_c_s=30.0, delta_max=1, n_min=1, n_max=10),
]


def test_matches_brute_force():
    """Exhaustive check of global optimality across both dead-time regimes."""
    rng = random.Random(20261030)
    checked = 0
    for case in CASES:
        cfg = _cfg(**case)
        tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
        pipe_len = _pipe_len(tau)
        lo, hi = cfg.control.n_min, cfg.control.n_max
        dmax = cfg.control.delta_max
        for _ in range(12):
            n = rng.randint(lo, hi)
            pipe = [rng.randint(-1, 1) for _ in range(pipe_len)]
            if not (lo <= n + sum(pipe) <= hi):
                pipe = [0] * pipe_len
            demand = [rng.randint(lo, hi) for _ in range(cfg.forecast.horizon)]
            last_u = rng.randint(-dmax, dmax)

            bf_cost, bf_us = _brute_force(n, demand, pipe, last_u, cfg)
            if bf_cost == float("inf"):
                continue
            got = solve_mpc_exact(n, demand, pipe, last_u, cfg)
            assert abs(got.cost - bf_cost) <= TOL, (
                f"cost {got.cost} != brute force {bf_cost} for {case}"
            )
            assert got.u in bf_us, (
                f"first action {got.u} not optimal (optimal: {sorted(bf_us)})"
            )
            checked += 1
    assert checked >= 60, f"only {checked} cases were feasible"


def test_truncation_count():
    """m = max(1, H - lag + 1), with lag = 1 when tau <= 1."""
    assert landing_lag(0) == 1
    assert landing_lag(1) == 1
    assert landing_lag(9) == 9
    assert tracking_relevant_actions(12, 9) == 4
    assert tracking_relevant_actions(12, 1) == 12
    assert tracking_relevant_actions(6, 6) == 1
    assert tracking_relevant_actions(4, 20) == 1  # never below one


def test_tail_is_not_all_zero():
    """With s > 0 the cost-optimal tail ramps down rather than snapping to 0.

    This is the property the greedy completion violates, so it is worth
    pinning: it is the reason greedy completion is not exact.
    """
    cfg = _cfg(horizon=12, tau_c_s=45.0, delta_max=4, n_min=1, n_max=16)
    ctrl = cfg.control
    r, s = ctrl.r_act, ctrl.s_churn
    zero_tail = s * (0 - 4) ** 2
    ramp = (r * 2**2 + s * (2 - 4) ** 2) + (r * 1**2 + s * (1 - 2) ** 2) \
        + s * (0 - 1) ** 2
    assert ramp < zero_tail, f"ramp {ramp} should beat zero tail {zero_tail}"


def test_never_worse_than_greedy_completion():
    """The exact objective is never above the greedy-completion objective."""
    cfg = _cfg(horizon=12, tau_c_s=45.0, delta_max=4, n_min=1, n_max=16)
    ctrl = cfg.control
    tau = cold_start_ticks(ctrl.tau_c_s, cfg.telemetry.dt)
    pipe_len = _pipe_len(tau)
    rng = random.Random(11)
    compared = 0
    for _ in range(40):
        n = rng.randint(ctrl.n_min, ctrl.n_max)
        pipe = [0] * pipe_len
        base = rng.randint(ctrl.n_min, ctrl.n_max)
        slope = rng.choice([-2, -1, 0, 1, 2])
        demand = [
            max(ctrl.n_min, min(ctrl.n_max, base + slope * k + rng.randint(-1, 1)))
            for k in range(cfg.forecast.horizon)
        ]
        last_u = rng.randint(-ctrl.delta_max, ctrl.delta_max)
        ex = solve_mpc_exact(n, demand, pipe, last_u, cfg)
        gr = solve_mpc(n, demand, pipe, last_u, cfg)
        gr_true = plan_cost(n, pipe, gr.plan, demand, last_u, tau, ctrl)
        if gr_true is None:
            continue
        assert ex.cost <= gr_true + TOL, (
            f"exact {ex.cost} worse than greedy {gr_true}"
        )
        compared += 1
    assert compared >= 30


def test_respects_rate_limit_and_pool_bounds():
    cfg = _cfg(horizon=12, tau_c_s=45.0, delta_max=4, n_min=1, n_max=16)
    tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
    pipe = [0] * _pipe_len(tau)
    got = solve_mpc_exact(16, [16] * 12, pipe, 0, cfg)
    assert -4 <= got.u <= 4
    assert 1 <= 16 + got.u <= 16
    got = solve_mpc_exact(1, [1] * 12, pipe, 0, cfg)
    assert 1 <= 1 + got.u <= 16


def test_plan_length_matches_horizon():
    cfg = _cfg(horizon=12, tau_c_s=45.0, delta_max=4, n_min=1, n_max=16)
    tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
    pipe = [0] * _pipe_len(tau)
    got = solve_mpc_exact(8, list(range(8, 20))[:12], pipe, 0, cfg)
    assert len(got.plan) == 12
    assert got.plan[0] == got.u
    assert got.solve_time_s >= 0.0
