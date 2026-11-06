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
from typing import Any, Callable, Protocol, Sequence

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
from pact.control.mpc_exact import (
    ExactBudgetExceeded,
    exact_is_tractable,
    solve_mpc_exact,
)
from pact.control.safety import evaluate_safety
from pact.drift.monitor import DriftMonitor
from pact.eval.metrics import (
    decision_classification,
    oscillation_count,
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
    if kind in REAL_WORKLOADS:
        return real_window(kind, n, seed)[0]
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
    out: np.ndarray = np.clip(sig, 30.0, None)
    return out


SYNTHETIC_WORKLOADS = ("diurnal", "burst", "ramp", "step", "sawtooth")
# Recorded traces, converted by experiments/make_real_traces.py.
REAL_WORKLOADS = ("worldcup98", "worldcup98_testpeak")
WORKLOADS = SYNTHETIC_WORKLOADS + REAL_WORKLOADS
TRACES = RESULTS / "traces"
CHECKPOINTS = RESULTS / "checkpoints"
# Ticks of history kept before every real window so a trained forecaster has
# a full input window at the window's first tick (telemetry.window).
REAL_PREFIX = 48


_TRACE_CACHE: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}


def load_trace(name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """The full rescaled series and its metadata (results/traces/<name>.*)."""
    if name not in _TRACE_CACHE:
        npy, meta = TRACES / f"{name}.npy", TRACES / f"{name}.json"
        if not npy.exists() or not meta.exists():
            raise FileNotFoundError(
                f"{npy} is missing; download the archive (data/README.md) and "
                "run experiments/make_real_traces.py"
            )
        _TRACE_CACHE[name] = (np.load(npy), json.loads(meta.read_text()))
    return _TRACE_CACHE[name]


def real_blocks(name: str, n: int) -> tuple[int, int]:
    """(first block start, number of blocks) for n-tick windows in the test split.

    Windows are disjoint, start after REAL_PREFIX ticks of test-split history,
    and never reach back into the training or validation period, so the
    trained forecaster is only ever evaluated out of sample.
    """
    trace, meta = load_trace(name)
    first = int(meta["val_end"]) + REAL_PREFIX
    count = (len(trace) - first) // n
    if count < 1:
        raise ValueError(f"{name} test split cannot hold a {n}-tick window")
    return first, count


def real_window(name: str, n: int, seed: int) -> tuple[np.ndarray, int]:
    """An n-tick slice of a recorded trace and its absolute start index.

    ``seed`` picks one of the disjoint test-split blocks through a fixed
    permutation, so consecutive seeds give distinct windows.
    """
    trace, _ = load_trace(name)
    first, count = real_blocks(name, n)
    order = np.random.default_rng(0).permutation(count)
    start = first + int(order[seed % count]) * n
    return trace[start:start + n].copy(), start


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


_MODEL_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}


def load_frozen(name: str = "tcn") -> tuple[Any, dict[str, Any]]:
    """A frozen checkpoint from results/checkpoints. Nothing is trained here."""
    if name not in _MODEL_CACHE:
        from experiments.lambda_forecast import load_checkpoint
        path = CHECKPOINTS / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run experiments/train_forecasters.py"
            )
        _MODEL_CACHE[name] = load_checkpoint(path)
    return _MODEL_CACHE[name]


@dataclass
class TrainedForecaster:
    """A trained lambda model, converted to [u, r] exactly as OracleForecaster.

    The model sees only the trace up to and including tick t. Its forecasts
    depend on arrivals alone, which the controller cannot influence, so they
    are computed for the whole window in one batch at construction. predict()
    checks that the arrival rate it is given is the trace value it read, so a
    run on a rescaled or noised copy of the trace fails instead of silently
    forecasting a different series.
    """

    trace_name: str
    start: int
    n: int
    cfg: PactConfig
    model_name: str = "tcn"

    def __post_init__(self) -> None:
        from experiments.lambda_forecast import lambda_features, predict_lambda
        model, meta = load_frozen(self.model_name)
        trace, tmeta = load_trace(self.trace_name)
        # A derived family is its base series times one constant. The model
        # forecasts the base series it was trained on and the forecast is
        # scaled by the same constant; with train-statistics z-scoring this is
        # exactly what the model would output on the scaled series.
        base_name = tmeta.get("derived_from", self.trace_name)
        extra = float(tmeta.get("extra_scale", 1.0))
        base, bmeta = load_trace(base_name)
        if base.shape != trace.shape or not np.allclose(base * extra, trace,
                                                        rtol=1e-12, atol=0.0):
            raise ValueError(f"{self.trace_name} is not {base_name} x {extra}")
        if int(meta["horizon"]) != self.cfg.forecast.horizon:
            raise ValueError(
                f"checkpoint horizon {meta['horizon']} != config horizon "
                f"{self.cfg.forecast.horizon}"
            )
        if meta["trace_sha256"] != bmeta["sha256_scaled_npy"]:
            raise ValueError("checkpoint was trained on a different trace file")
        if self.start < int(meta["val_end"]):
            raise ValueError("window overlaps the training/validation period")
        feats = lambda_features(base, int(bmeta["t0_utc_epoch_s"]),
                                float(bmeta["dt_s"]), float(meta["lam_mean"]),
                                float(meta["lam_std"]))
        origins = np.arange(self.start, self.start + self.n)
        self._trace = trace
        self._lam_hat = extra * predict_lambda(model, meta, feats, origins)

    def predict(self, t: int, obs: Observation, n_ready: int) -> list[list[float]]:
        expected = float(self._trace[self.start + t])
        if abs(obs.lam - expected) > 1e-6 * max(1.0, abs(expected)):
            raise ValueError(
                f"observed arrival {obs.lam} at tick {t} is not trace value "
                f"{expected}; the trained arm only runs on unmodified windows"
            )
        mu = self.cfg.capacity.mu
        n = max(n_ready, 1)
        sim = self.cfg.simulator
        out = []
        for lam_hat in self._lam_hat[t]:
            u = max(float(lam_hat), 0.0) / (mu * n)
            out.append([u, sim.mem_baseline + sim.mem_load_coeff * min(u, 1.0)])
        return out


# ----------------------------------------------------------------- controllers
def _relaxed_rounded_u(
    n_current: int, demand: Sequence[int], cfg: PactConfig, last_u: int
) -> int:
    """OptScaler-style: solve the relaxed tracking problem, round the result.

    Reference: Zou et al., OptScaler, PVLDB 17(12):4090-4103 (2024),
    doi 10.14778/3685800.3685829. That formulation has
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
    solver: str = "exact"
    exact_ticks: int = 0
    fallback_ticks: int = 0


LEGACY_N_INITIAL = 4


def initial_replicas(lam: np.ndarray, cfg: PactConfig) -> int:
    """Four replicas, or the first tick's requirement under (2) if larger.

    The requirement is capped at ``n_max``; the floor of four is the start
    state every earlier run used, so runs whose first tick four replicas can
    already cover are unchanged.
    """
    need = int(math.ceil(float(lam[0]) / (cfg.capacity.mu * cfg.capacity.u_target)))
    return max(LEGACY_N_INITIAL, min(need, cfg.control.n_max))


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
    n_initial: int | None = None,
    lam_forecast: np.ndarray | None = None,
    solver: str = "auto",
    trace_name: str | None = None,
    window_start: int | None = None,
    model_name: str = "tcn",
) -> RunOutput:
    """One closed-loop run against the repository's PoolSimulator.

    ``solver`` selects the integer MPC implementation: ``"exact"`` is the
    globally optimal solver of :mod:`pact.control.mpc_exact`, ``"greedy"``
    the enumerate-then-complete-greedily solver of :mod:`pact.control.mpc`.

    ``forecaster_kind`` is ``"oracle"``, ``"persist"`` or ``"trained"``. The
    trained arm needs ``trace_name`` and ``window_start`` (from
    :func:`real_window`) so the model can read the history before the window.

    ``n_initial`` defaults to :func:`initial_replicas`, so a window cut from
    the middle of a recorded day does not open with a pool sized for a
    different load.
    """
    if solver not in ("exact", "greedy", "auto"):
        raise ValueError(f"unknown solver {solver!r}")
    if n_initial is None:
        n_initial = initial_replicas(lam, cfg)
    run_cfg = cfg
    if not use_churn:
        run_cfg = replace(run_cfg, control=replace(run_cfg.control, s_churn=0.0))
    if not use_coldstart:
        run_cfg = replace(run_cfg, control=replace(run_cfg.control, tau_c_s=0.0))

    sim = PoolSimulator(cfg, n_initial=n_initial)      # plant keeps true tau_c
    tau = cold_start_ticks(run_cfg.control.tau_c_s, run_cfg.telemetry.dt)

    # Pick the integer solver for this configuration. "auto" takes the exact
    # solver wherever the structural reduction makes it affordable and the
    # greedy completion where it does not, recording which was used so the
    # manifest can report the fraction of runs solved exactly.
    want = solver
    if want == "auto":
        want = "exact" if exact_is_tractable(
            run_cfg.forecast.horizon, tau, run_cfg.control.delta_max,
            run_cfg.control.n_min, run_cfg.control.n_max,
        ) else "greedy"
    mpc_solve = solve_mpc_exact if want == "exact" else solve_mpc
    solver_used = want
    exact_ticks = 0
    fallback_ticks = 0
    plant_tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)

    lam_fc = lam if lam_forecast is None else np.asarray(lam_forecast, dtype=float)
    fc: Forecaster
    if forecaster_kind == "oracle":
        fc = OracleForecaster(lam=lam_fc, cfg=run_cfg)
    elif forecaster_kind == "trained":
        if trace_name is None or window_start is None:
            raise ValueError("the trained forecaster needs trace_name and window_start")
        if lam_forecast is not None:
            raise ValueError("lam_forecast does not apply to the trained forecaster")
        fc = TrainedForecaster(trace_name=trace_name, start=window_start,
                               n=len(lam), cfg=run_cfg, model_name=model_name)
    else:
        fc = PersistenceForecaster(cfg=run_cfg)

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
    out.solver = solver_used

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
                    if mpc_solve is solve_mpc_exact:
                        try:
                            sol = solve_mpc_exact(
                                n_desired, demand, pending, last_u, run_cfg
                            )
                            exact_ticks += 1
                        except ExactBudgetExceeded:
                            # Safety net: the a-priori estimate said this
                            # configuration was affordable and it was not for
                            # this state. Fall back for this tick and count it.
                            sol = solve_mpc(
                                n_desired, demand, pending, last_u, run_cfg
                            )
                            fallback_ticks += 1
                    else:
                        sol = solve_mpc(n_desired, demand, pending, last_u, run_cfg)
                        fallback_ticks += 1
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

    out.exact_ticks = exact_ticks
    out.fallback_ticks = fallback_ticks
    return out


# ----------------------------------------------------------------- metrics
def summarise(out: RunOutput, cfg: PactConfig) -> dict[str, float]:
    slo = cfg.capacity.slo_ms
    dt = cfg.telemetry.dt
    viol = sla_violation_rate(out.p95, slo) * 100.0
    over = over_provision_ratio(out.n_series, out.required) * 100.0
    actions = scale_action_count(out.n_series)
    cls = decision_classification(out.n_series, out.required)
    osc = oscillation_count(out.n_series)
    osc_w = oscillation_count(out.n_series, window=12)
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
        "oscillations": float(osc),
        "oscillations_w12": float(osc_w),
        "mean_util_pct": util,
        "mean_p95_ms": p95,
        "cost_per_1k_req": cost,
        "accuracy": float(cls.accuracy),
        "hold_f1": float(cls.hold.f1),
        "scaledown_recall": float(cls.scale_down.recall),
        "scaleup_precision": float(cls.scale_up.precision),
        "scaleup_recall": float(rec),
        "scaleup_f1": float(cls.scale_up.f1),
        "overrides": float(out.overrides),
        "gamma_max": float(np.max(np.abs(out.gamma))) if out.gamma else 0.0,
        "gamma_moved_pct": (100.0 * float(np.mean(np.abs(np.asarray(out.gamma)) > 0.0))
                            if out.gamma else 0.0),
        "mean_solve_ms": float(np.mean(out.solve_times) * 1000.0) if out.solve_times else float("nan"),
        "rounding_disagree_pct": 100.0 * out.rounding_disagreements / max(out.ticks, 1),
        "censored_pct": 100.0 * out.censored / max(out.ticks, 1),
        "exact_solver": 1.0 if out.solver == "exact" else 0.0,
        "exact_tick_pct": (100.0 * out.exact_ticks
                           / max(out.exact_ticks + out.fallback_ticks, 1)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")
