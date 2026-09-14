"""Experiment runner: one plant, many decision mechanisms, many seeds."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pact.actuation.dry_run import DryRunBackend
from pact.baselines.arima_ctl import ArimaController
from pact.baselines.common import Scaler
from pact.baselines.lstm_ctl import LstmController
from pact.baselines.reactive import ReactiveHPA
from pact.baselines.static import StaticScaler
from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    PactConfig,
    load_config,
)
from pact.eval.datasets import ArrivalTrace, generate_burst
from pact.eval.metrics import (
    cost_per_1k_requests,
    decision_classification,
    over_provision_ratio,
    scale_action_count,
    sla_violation_rate,
)
from pact.eval.splits import SplitGuard, chronological_series_split
from pact.forecast.seq_models import RecurrentForecaster
from pact.forecast.train import seed_everything
from pact.loop import ControlLoop, RepeatObservationForecaster
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import SimCollector
from pact.telemetry.features import FEATURE_DIM

CONTROLLER_METHODS = ("pact", "reactive", "arima", "lstm", "static")
FORECAST_METHODS = ("pact", "reactive", "arima", "lstm")
N_SEEDS_DEFAULT = 5


@dataclass(frozen=True)
class MethodTrace:
    method: str
    seed: int
    dataset: str
    ts: tuple[float, ...]
    n: tuple[int, ...]
    n_required: tuple[int, ...]
    latency_ms: tuple[float, ...]
    utilisation: tuple[float, ...]
    action: tuple[int, ...]
    actual_u: tuple[float, ...]
    pred_h1: tuple[float, ...]
    pred_h12: tuple[float, ...]
    requests_served: tuple[float, ...]


@dataclass(frozen=True)
class RunMetrics:
    method: str
    seed: int
    dataset: str
    sla_violation: float
    over_provision: float
    action_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    cost_per_1k: float
    mean_utilisation: float


def required_replicas(arrival_rate: float, config: PactConfig) -> int:
    mu = config.capacity.mu
    u_star = config.capacity.u_target
    raw = math.ceil(arrival_rate / (mu * u_star)) if arrival_rate > 0.0 else 1
    return max(config.control.n_min, min(config.control.n_max, raw))


def run_scaler(
    config: PactConfig,
    arrivals: Sequence[float],
    scaler: Scaler,
    *,
    seed: int,
    dataset: str,
    n_initial: int | None = None,
    dt: float | None = None,
) -> MethodTrace:
    """Shared plant: same Δt, simulator, actuation, and pool bounds."""

    seed_everything(seed)
    n0 = config.control.n_min if n_initial is None else n_initial
    sim = PoolSimulator(config, n_initial=n0)
    collector = SimCollector(sim)
    backend = DryRunBackend(n_initial=n0)
    interval = config.telemetry.dt if dt is None else dt
    n = n0
    ts: list[float] = []
    ns: list[int] = []
    req: list[int] = []
    lat: list[float] = []
    util: list[float] = []
    actions: list[int] = []
    actual_u: list[float] = []
    served: list[float] = []
    for i, rate in enumerate(arrivals):
        now_s = i * interval
        sim.step(float(rate), n)
        obs = collector.collect(now_s)
        n_ready = max(int(round(obs.n)), config.control.n_min)
        desired = scaler.desired_replicas(obs, n_ready, now_s)
        action = desired - n
        backend.set_replicas(desired)
        n = desired
        ts.append(now_s)
        ns.append(n)
        req.append(required_replicas(float(rate), config))
        lat.append(obs.ell_p95)
        util.append(obs.u)
        actions.append(action)
        actual_u.append(obs.u)
        served.append(sim.last_state.requests_served if sim.last_state else 0.0)
    h12 = (0.0,) * len(ts)
    return MethodTrace(
        method=scaler.name,
        seed=seed,
        dataset=dataset,
        ts=tuple(ts),
        n=tuple(ns),
        n_required=tuple(req),
        latency_ms=tuple(lat),
        utilisation=tuple(util),
        action=tuple(actions),
        actual_u=tuple(actual_u),
        pred_h1=tuple(actual_u),
        pred_h12=h12,
        requests_served=tuple(served),
    )


def run_pact(
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    seed: int,
    dataset: str,
    n_initial: int | None = None,
    forecaster: RepeatObservationForecaster | None = None,
) -> MethodTrace:
    seed_everything(seed)
    n0 = config.control.n_min if n_initial is None else n_initial
    sim = PoolSimulator(config, n_initial=n0)
    fc = forecaster or RepeatObservationForecaster(config.forecast.horizon)
    loop = ControlLoop(
        config,
        collector=SimCollector(sim),
        backend=DryRunBackend(n_initial=n0),
        forecaster=fc,
        simulator=sim,
        arrival_rates=arrivals,
        n_initial=n0,
        sleep=lambda _s: None,
    )
    result = loop.run(len(arrivals))
    h = config.forecast.horizon
    ts = [rec.timestamp for rec in result.ticks]
    n = [rec.n_desired for rec in result.ticks]
    req = [
        required_replicas(float(arrivals[i]), config) for i in range(len(result.ticks))
    ]
    lat = [rec.observation.ell_p95 for rec in result.ticks]
    util = [rec.observation.u for rec in result.ticks]
    actions = [rec.applied_u for rec in result.ticks]
    actual_u = [rec.observation.u for rec in result.ticks]
    pred_h1 = [rec.forecast[0][0] for rec in result.ticks]
    idx12 = min(11, h - 1)
    pred_h12 = [rec.forecast[idx12][0] for rec in result.ticks]
    served = [
        min(rec.observation.lam, max(rec.observation.n, 1.0) * config.capacity.mu)
        * config.telemetry.dt
        for rec in result.ticks
    ]
    return MethodTrace(
        method="pact",
        seed=seed,
        dataset=dataset,
        ts=tuple(ts),
        n=tuple(n),
        n_required=tuple(req),
        latency_ms=tuple(lat),
        utilisation=tuple(util),
        action=tuple(actions),
        actual_u=tuple(actual_u),
        pred_h1=tuple(pred_h1),
        pred_h12=tuple(pred_h12),
        requests_served=tuple(served),
    )


def metrics_from_trace(trace: MethodTrace, config: PactConfig) -> RunMetrics:
    slo = config.capacity.slo_ms
    finite_lat = [x if math.isfinite(x) else slo * 10.0 for x in trace.latency_ms]
    clf = decision_classification(trace.n, trace.n_required)
    served = sum(trace.requests_served)
    if served <= 0.0:
        served = 1.0
    return RunMetrics(
        method=trace.method,
        seed=trace.seed,
        dataset=trace.dataset,
        sla_violation=sla_violation_rate(finite_lat, slo),
        over_provision=over_provision_ratio(
            [float(x) for x in trace.n],
            [float(max(r, 1)) for r in trace.n_required],
        ),
        action_count=scale_action_count([float(x) for x in trace.n]),
        accuracy=clf.accuracy,
        precision=clf.scale_up.precision,
        recall=clf.scale_up.recall,
        f1=clf.scale_up.f1,
        cost_per_1k=cost_per_1k_requests(
            [float(x) for x in trace.n],
            config.telemetry.dt,
            served,
            1.0,
        ),
        mean_utilisation=(
            sum(trace.utilisation) / len(trace.utilisation)
            if trace.utilisation
            else 0.0
        ),
    )


def run_method(
    method: str,
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    seed: int,
    dataset: str,
    lstm_model: RecurrentForecaster | None = None,
    arima_order: tuple[int, int, int] = (1, 0, 0),
) -> MethodTrace:
    if method == "pact":
        return run_pact(config, arrivals, seed=seed, dataset=dataset)
    if method == "reactive":
        scaler: Scaler = ReactiveHPA(config)
    elif method == "arima":
        scaler = ArimaController(config, order=arima_order)
    elif method == "lstm":
        model = lstm_model or RecurrentForecaster(
            kind="lstm",
            in_channels=FEATURE_DIM,
            window=config.telemetry.window,
            horizon=config.forecast.horizon,
            hidden_size=16,
        )
        scaler = LstmController(config, model)
    elif method == "static":
        scaler = StaticScaler.sized_to_peak(config, arrivals)
    else:
        raise ValueError(f"unknown method {method!r}")
    return run_scaler(
        config, arrivals, scaler, seed=seed, dataset=dataset
    )


def run_matrix(
    config: PactConfig,
    trace: ArrivalTrace,
    *,
    methods: Sequence[str] = FORECAST_METHODS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    split_guard: SplitGuard | None = None,
    evaluate_on: str = "test",
) -> list[MethodTrace]:
    """Run each (method, seed) on the same arrival series after a chronological split.

    ``evaluate_on='test'`` reads the test split through ``SplitGuard`` once.
    """

    split = chronological_series_split(trace.arrival_rates, trace.timestamps)
    if evaluate_on == "test":
        guard = split_guard or SplitGuard()
        arrivals = list(guard.read_test(split))
    elif evaluate_on == "val":
        arrivals = list(split.val)
    elif evaluate_on == "all":
        arrivals = list(trace.arrival_rates)
    else:
        raise ValueError(f"unknown evaluate_on {evaluate_on!r}")
    out: list[MethodTrace] = []
    for method in methods:
        for seed in seeds:
            out.append(
                run_method(
                    method, config, arrivals, seed=int(seed), dataset=trace.name
                )
            )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT experiment runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config, cold_start_path=args.cold_start)
    burst = generate_burst(
        "step",
        n_ticks=args.ticks,
        dt=config.telemetry.dt,
        rise_ticks=4,
        base=20.0,
        peak=60.0,
        name="d3_step_cli",
    )
    traces = run_matrix(
        config,
        burst,
        methods=FORECAST_METHODS,
        seeds=tuple(range(args.seeds)),
        evaluate_on="all",
    )
    from pact.eval.export import export_run

    export_run(Path("results"), traces, config)


if __name__ == "__main__":
    main()
