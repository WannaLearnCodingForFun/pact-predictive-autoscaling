"""Measurement harness for PACT, using only the repository's own components.

Runs on numpy alone. torch and statsmodels are unavailable in this environment,
so the learned forecaster (TCN) and the ARIMA/LSTM baselines cannot be executed
here. To isolate the *controller* contribution from the *forecaster*
contribution, every controller is driven by two bracketing oracles:

    OracleForecaster      perfect foresight  -> upper bound on forecast value
    PersistenceForecaster last value repeated -> lower bound (no model at all)

Any real forecaster lies between these two. Differences that hold under BOTH
brackets are properties of the controller, not of the predictor.

Every number this script writes is computed from a simulation run. Nothing is
hard-coded.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol, Sequence

import logging

import numpy as np

logging.getLogger("pact.control.safety").setLevel(logging.ERROR)

from pact.baselines.common import apply_pool_bounds
from pact.baselines.reactive import ReactiveHPA
from pact.capacity.mapper import map_to_demand
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
from pact.control.gates import apply_gates
from pact.control.mpc import apply_plant, solve_mpc
from pact.control.safety import evaluate_safety
from pact.drift.monitor import DriftMonitor
from pact.eval.metrics import (
    decision_classification,
    over_provision_ratio,
    scale_action_count,
    sla_violation_rate,
)
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import Observation

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


# ----------------------------------------------------------------- config
def base_config(
    *,
    dt: float = 5.0,
    mu: float = 40.0,
    tau_c_s: float = 45.0,
    n_min: int = 1,
    n_max: int = 16,
    horizon: int = 12,
) -> PactConfig:
    return PactConfig(
        telemetry=TelemetryConfig(dt=dt),
        forecast=ForecastConfig(horizon=horizon),
        capacity=CapacityConfig(mu=mu),
        control=ControlConfig(tau_c_s=tau_c_s, n_min=n_min, n_max=n_max),
        drift=DriftConfig(),
        simulator=SimulatorConfig(),
    )


# ----------------------------------------------------------------- workloads
def workload(kind: str, n: int, seed: int) -> np.ndarray:
    """Arrival-rate trace in requests/s. Deterministic given (kind, n, seed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "diurnal":
        base = 110 + 34 * np.sin(2 * np.pi * t / 300.0)
        sig = base + rng.normal(0, 7.0, n)
    elif kind == "burst":
        base = 90 + np.zeros(n)
        for centre, amp, width in ((n // 3, 150, 14), (2 * n // 3, 190, 10)):
            base = base + amp * np.exp(-0.5 * ((t - centre) / width) ** 2)
        sig = base + rng.normal(0, 7.0, n)
    elif kind == "ramp":
        up = np.clip((t - n * 0.2) / (n * 0.25), 0, 1)
        down = np.clip((n * 0.85 - t) / (n * 0.2), 0, 1)
        sig = 85 + 175 * up * down + rng.normal(0, 6.0, n)
    elif kind == "step":
        sig = np.where(t < n // 2, 85.0, 245.0) + rng.normal(0, 6.0, n)
    elif kind == "sawtooth":
        sig = 90 + 140 * ((t % 90) / 90.0) + rng.normal(0, 6.0, n)
    else:
        raise ValueError(f"unknown workload {kind!r}")
    return np.clip(sig, 30.0, None)


WORKLOADS = ("diurnal", "burst", "ramp", "step", "sawtooth")


# ----------------------------------------------------------------- forecasters
class Forecaster(Protocol):
    def predict(self, t: int, obs: Observation, n_ready: int) -> list[list[float]]:
        ...


@dataclass
class OracleForecaster:
    """Perfect foresight: converts true future arrivals to CPU/memory at N(t)."""

    lam: np.ndarray
    cfg: PactConfig

    def predict(self, t: int, obs: Observation, n_ready: int) -> list[list[float]]:
        H = self.cfg.forecast.horizon
        mu = self.cfg.capacity.mu
        n = max(n_ready, 1)
        out = []
        for h in range(1, H + 1):
            j = min(t + h, len(self.lam) - 1)
            u = float(self.lam[j]) / (mu * n)
            r = self.cfg.simulator.mem_baseline + self.cfg.simulator.mem_load_coeff * min(u, 1.0)
            out.append([u, r])
        return out


@dataclass
class PersistenceForecaster:
    """No model: the most recent observation repeated across the horizon."""

    cfg: PactConfig

    def predict(self, t: int, obs: Observation, n_ready: int) -> list[list[float]]:
        H = self.cfg.forecast.horizon
        return [[obs.u, obs.r] for _ in range(H)]


# ----------------------------------------------------------------- controllers
def _relaxed_rounded_u(
    n_current: int, demand: Sequence[int], cfg: PactConfig, last_u: int
) -> int:
    """OptScaler-style: solve the relaxed tracking problem, round the result.

    Reference: Zhang et al., OptScaler, PVLDB 17 (2024). That formulation has
    real-valued decision variables and states the first-step solution "can be
    rounded for implementation"; there is no integrality constraint. This
    reproduces that choice on an otherwise identical objective so the cost of
    rounding can be isolated. Deviations: single-step tracking of the first
    horizon element rather than a full relaxed QP, which upper-bounds the
    rounded controller's quality (a full relaxed solve cannot do better on the
    tracking term alone), and no chance constraint.
    """
    target = float(demand[0])
    u_relaxed = target - n_current
    u = int(round(u_relaxed))
    ctrl = cfg.control
    u = max(-ctrl.delta_max, min(ctrl.delta_max, u))
    n_next = n_current + u
    if n_next > ctrl.n_max:
        u -= n_next - ctrl.n_max
    elif n_next < ctrl.n_min:
        u += ctrl.n_min - n_next
    return u


# Kingman diverges at rho>=1. Saturated ticks are censored at this ceiling so
# that means are finite; the censoring rate is reported alongside every mean.
P95_CEILING_MS = 5000.0


@dataclass
class RunOutput:
    n_series: list[int]
    required: list[int]
    p95: list[float]
    util: list[float]
    requests: list[float]
    u_applied: list[int]
    gamma: list[float]
    solve_times: list[float]
    overrides: int
    rounding_disagreements: int
    ticks: int
    censored: int = 0


def run_controller(
    *,
    method: str,
    lam: np.ndarray,
    cfg: PactConfig,
    forecaster_kind: str,
    use_mpc: bool = True,
    use_churn: bool = True,
    use_coldstart: bool = True,
    use_margin: bool = True,
    use_safety: bool = True,
    n_initial: int = 4,
    lam_forecast: np.ndarray | None = None,
) -> RunOutput:
    """One closed-loop run against the repository's PoolSimulator."""
    run_cfg = cfg
    if not use_churn:
        run_cfg = replace(run_cfg, control=replace(run_cfg.control, s_churn=0.0))
    if not use_coldstart:
        run_cfg = replace(run_cfg, control=replace(run_cfg.control, tau_c_s=0.0))

    sim = PoolSimulator(cfg, n_initial=n_initial)      # plant keeps true tau_c
    tau = cold_start_ticks(run_cfg.control.tau_c_s, run_cfg.telemetry.dt)
    plant_tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)

    lam_fc = lam if lam_forecast is None else np.asarray(lam_forecast, dtype=float)
    fc: Forecaster = (
        OracleForecaster(lam=lam_fc, cfg=run_cfg)
        if forecaster_kind == "oracle"
        else PersistenceForecaster(cfg=run_cfg)
    )

    hpa = ReactiveHPA(cfg) if method == "reactive" else None
    monitor = DriftMonitor(run_cfg, tau=plant_tau) if use_margin else None

    mu, u_star = cfg.capacity.mu, cfg.capacity.u_target
    n_desired = n_initial
    pending: list[int] = [0] * plant_tau
    last_u = 0
    last_up_s = -1e9
    gamma = 0.0
    last_demand: list[int] = []

    out = RunOutput([], [], [], [], [], [], [], [], 0, 0, len(lam))

    static_level = int(math.ceil(float(lam.max()) / (mu * u_star)))
    static_level = max(cfg.control.n_min, min(cfg.control.n_max, static_level))

    for t in range(len(lam)):
        state = sim.step(float(lam[t]), n_desired)
        now_s = t * cfg.telemetry.dt
        obs = Observation(
            u=state.utilisation, r=state.memory_fraction, lam=float(lam[t]),
            ell_p95=state.p95_latency_ms, n=float(state.n_ready), t_s=now_s,
        )
        required = max(1, int(math.ceil(float(lam[t]) / (mu * u_star))))

        if method == "static":
            u = static_level - n_desired
            u = max(-cfg.control.delta_max, min(cfg.control.delta_max, u))
        elif method == "reactive":
            assert hpa is not None
            u = hpa.desired_replicas(obs, n_desired, now_s) - n_desired
        else:
            forecast = fc.predict(t, obs, state.n_ready)
            demand = map_to_demand(forecast, max(state.n_ready, 1), gamma, run_cfg)
            last_demand = list(demand)

            if method == "rounded":
                u = _relaxed_rounded_u(n_desired, demand, run_cfg, last_u)
            else:
                if use_mpc:
                    sol = solve_mpc(n_desired, demand, pending, last_u, run_cfg)
                    out.solve_times.append(sol.solve_time_s)
                    u_raw = float(sol.u)
                    r_u = _relaxed_rounded_u(n_desired, demand, run_cfg, last_u)
                    if r_u != sol.u:
                        out.rounding_disagreements += 1
                else:
                    u_raw = float(demand[0] - n_desired)
                gated = apply_gates(
                    u_raw, n_desired, run_cfg.control,
                    now_s=now_s, last_scale_up_s=last_up_s,
                )
                u = gated.u

            if use_safety:
                dec = evaluate_safety(state.utilisation, n_desired, run_cfg)
                if dec.active:
                    u = dec.u
                    out.overrides += 1

        u = apply_pool_bounds(n_desired, n_desired + u, cfg) - n_desired
        if u > 0:
            last_up_s = now_s
        n_desired += u
        last_u = u
        pending = (pending + [u])[-max(plant_tau, 1):]

        if monitor is not None:
            snap = monitor.step(
                n_required_observed=required,
                predicted_demand=last_demand if last_demand else [n_desired],
            )
            gamma = snap.gamma

        out.n_series.append(n_desired)
        out.required.append(required)
        p95 = state.p95_latency_ms
        if not math.isfinite(p95) or p95 > P95_CEILING_MS:
            p95 = P95_CEILING_MS
            out.censored += 1
        out.p95.append(p95)
        out.util.append(state.utilisation)
        out.requests.append(state.requests_served)
        out.u_applied.append(u)
        out.gamma.append(gamma)

    return out


# ----------------------------------------------------------------- metrics
def summarise(out: RunOutput, cfg: PactConfig) -> dict[str, float]:
    slo = cfg.capacity.slo_ms
    dt = cfg.telemetry.dt
    viol = sla_violation_rate(out.p95, slo) * 100.0
    over = over_provision_ratio(out.n_series, out.required) * 100.0
    actions = scale_action_count(out.n_series)
    cls = decision_classification(out.n_series, out.required)
    util = float(np.mean(out.util)) * 100.0
    p95 = float(np.mean(out.p95))
    served = float(np.sum(out.requests))
    rep_sec = float(np.sum(out.n_series)) * dt
    cost = (rep_sec / max(served, 1.0)) * 1000.0
    rec = cls.scale_up.recall
    return {
        "sla_violation_pct": viol,
        "over_provision_pct": over,
        "scale_actions": float(actions),
        "mean_util_pct": util,
        "mean_p95_ms": p95,
        "cost_per_1k_req": cost,
        "accuracy": float(cls.accuracy),
        "scaleup_precision": float(cls.scale_up.precision),
        "scaleup_recall": float(rec),
        "scaleup_f1": float(cls.scale_up.f1),
        "overrides": float(out.overrides),
        "mean_solve_ms": float(np.mean(out.solve_times) * 1000.0) if out.solve_times else float("nan"),
        "rounding_disagree_pct": 100.0 * out.rounding_disagreements / max(out.ticks, 1),
        "censored_pct": 100.0 * out.censored / max(out.ticks, 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")
