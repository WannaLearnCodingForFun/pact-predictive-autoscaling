"""Phase 5.1 acceptance tests: dead-time MPC, no gates."""

from __future__ import annotations

from dataclasses import replace

from pact.config import cold_start_ticks
from pact.control.mpc import apply_plant, solve_mpc

from tests.helpers import make_config


def test_constant_demand_equal_to_n_gives_u_zero() -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
    n = 4
    demand = [n] * cfg.forecast.horizon
    pending = [0] * tau
    sol = solve_mpc(n, demand, pending, last_u=0, config=cfg)
    assert sol.u == 0


def test_step_increase_acts_tau_ticks_before_arrival() -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    # Zero actuation/churn so spreading a +4 step is not cheaper than waiting;
    # this test is only about the dead-time plant, not the quadratic u² term.
    cfg = replace(
        cfg,
        control=replace(cfg.control, r_act=0.0, s_churn=0.0, delta_max=8),
    )
    tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
    assert tau == 2
    t_step = 20
    n0, n_high = 4, 8
    horizon = cfg.forecast.horizon
    series = [n0 if t < t_step else n_high for t in range(80)]
    n = n0
    pending = [0] * tau
    last_u = 0
    actions: list[int] = []
    for t in range(40):
        forecast = [series[t + 1 + k] for k in range(horizon)]
        sol = solve_mpc(n, forecast, pending, last_u, cfg)
        actions.append(sol.u)
        n, pending = apply_plant(
            n, pending, sol.u, tau, cfg.control.n_min, cfg.control.n_max
        )
        last_u = sol.u
    first_up = next(t for t, u in enumerate(actions) if u > 0)
    assert first_up == t_step - tau
    assert first_up < t_step


def test_q_up_much_larger_than_q_down_yields_higher_replicas() -> None:
    series = [4] * 12 + [8] * 12 + [5] * 20
    n_asym = _mean_replicas(series, q_up=8.0, q_down=1.0)
    n_sym = _mean_replicas(series, q_up=1.0, q_down=1.0)
    assert n_asym > n_sym


def _mean_replicas(series: list[int], *, q_up: float, q_down: float) -> float:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    cfg = replace(
        cfg,
        control=replace(cfg.control, q_up=q_up, q_down=q_down),
    )
    tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
    horizon = cfg.forecast.horizon
    n = series[0]
    pending = [0] * tau
    last_u = 0
    replicas = [n]
    for t in range(len(series) - horizon - 1):
        forecast = [series[t + 1 + k] for k in range(horizon)]
        sol = solve_mpc(n, forecast, pending, last_u, cfg)
        n, pending = apply_plant(
            n, pending, sol.u, tau, cfg.control.n_min, cfg.control.n_max
        )
        last_u = sol.u
        replicas.append(n)
    return sum(replicas) / len(replicas)
