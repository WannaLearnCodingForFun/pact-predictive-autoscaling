"""Control loop: M1 → M2 → M3 → M4 → M5 on a fixed Δt tick (Phase 7).

Deadlines are ``start + (i+1)·Δt`` from ``time.monotonic()``. The loop never
``sleep(dt)`` blindly. τc is injected by ``load_config`` from
``results/cold_start.json``; ``from_yaml`` refuses to start if that file is
missing.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from pact.actuation.docker_backend import ActuationBackend
from pact.capacity.mapper import assert_horizon_actionable, map_to_demand
from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    PactConfig,
    cold_start_ticks,
    load_config,
)
from pact.control.gates import (
    GateDecision,
    apply_gates,
    clip_pool_bounds,
    clip_rate_limit,
)
from pact.control.mpc import apply_plant, solve_mpc
from pact.control.safety import SafetyDecision, evaluate_safety
from pact.drift.monitor import DriftMonitor, DriftSnapshot, RefitScheduler
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import Collector, Observation
from pact.telemetry.features import (
    EWMASmoother,
    MinMaxNormaliser,
    SlidingWindow,
    feature_dim,
    feature_vector,
)

logger = logging.getLogger(__name__)


class Forecaster(Protocol):
    def predict(self, window: NDArray[np.float32]) -> NDArray[np.float32]: ...


class RepeatObservationForecaster:
    """Copies the latest ``(u, r)`` across the horizon when no ONNX model is loaded."""

    def __init__(self, horizon: int) -> None:
        self._horizon = horizon
        self._last = np.zeros((horizon, 2), dtype=np.float32)

    def set_observation(self, u: float, r: float) -> None:
        self._last = np.tile(
            np.array([u, r], dtype=np.float32), (self._horizon, 1)
        )

    def predict(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        del window
        return self._last.copy()


@dataclass(frozen=True)
class TickRecord:
    tick: int
    timestamp: float
    wall_time_s: float
    observation: Observation
    forecast: tuple[tuple[float, float], ...]
    demand: tuple[int, ...]
    n_required: int
    mpc_u: int | None
    mpc_cost: float | None
    mpc_solve_time_s: float | None
    applied_u: int
    gate: GateDecision | None
    safety: SafetyDecision
    drift: DriftSnapshot
    n_desired: int
    deadline_slip_s: float


class TickObserver(Protocol):
    """Optional sink for completed ticks. Failures are swallowed by the loop."""

    def on_tick(self, record: TickRecord) -> None: ...


@dataclass(frozen=True)
class TickTiming:
    """Wall-clock seconds for one controller tick, excluding the plant step."""

    scrape_s: float
    features_s: float
    inference_s: float
    mapping_s: float
    mpc_s: float
    actuation_s: float


@dataclass(frozen=True)
class LoopResult:
    ticks: tuple[TickRecord, ...]
    deadline_slips_s: tuple[float, ...]
    timings: tuple[TickTiming, ...] = ()


class ControlLoop:
    """Collect, forecast, map, MPC/gates/safety, actuate, then drift."""

    def __init__(
        self,
        config: PactConfig,
        *,
        collector: Collector,
        backend: ActuationBackend,
        forecaster: Forecaster,
        simulator: PoolSimulator | None = None,
        arrival_rates: Sequence[float] | None = None,
        normaliser: MinMaxNormaliser | None = None,
        refit_scheduler: RefitScheduler | None = None,
        n_initial: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        observers: Sequence[TickObserver] | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        assert_horizon_actionable(config)
        self._config = config
        self._collector = collector
        self._backend = backend
        self._forecaster = forecaster
        self._simulator = simulator
        self._arrivals = None if arrival_rates is None else list(arrival_rates)
        self._normaliser = normaliser
        self._monotonic = monotonic
        self._sleep = sleep
        self._wall_clock = wall_clock
        self._observers: tuple[TickObserver, ...] = (
            tuple(observers) if observers is not None else ()
        )
        self._dt = config.telemetry.dt
        self._tau = cold_start_ticks(config.control.tau_c_s, self._dt)
        n0 = config.control.n_min if n_initial is None else n_initial
        self._n_desired = n0
        self._n_plant = n0
        self._pending: list[int] = []
        self._last_u = 0
        self._last_scale_up_s: float | None = None
        self._ewma = EWMASmoother(config.telemetry.alpha)
        n_channels = feature_dim(include_rho=config.ablation.include_rho)
        self._window = SlidingWindow(config.telemetry.window, n_channels)
        self._include_rho = config.ablation.include_rho
        self._skip_mpc = config.ablation.skip_mpc
        self._drift = DriftMonitor(
            config, tau=self._tau, scheduler=refit_scheduler
        )
        setter = getattr(forecaster, "set_observation", None)
        self._set_observation: Callable[[float, float], None] | None = (
            cast("Callable[[float, float], None]", setter) if callable(setter) else None
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path = DEFAULT_CONFIG_PATH,
        *,
        cold_start_path: Path = DEFAULT_COLD_START_PATH,
        **kwargs: Any,
    ) -> ControlLoop:
        """Load config (τc from ``cold_start.json``) then construct the loop.

        Raises ``MissingColdStartError`` if the measurement file is absent.
        """

        config = load_config(yaml_path, cold_start_path=cold_start_path)
        return cls(config, **kwargs)

    def run(self, n_ticks: int) -> LoopResult:
        if n_ticks < 1:
            raise ValueError(f"n_ticks must be >= 1, got {n_ticks}")
        if self._simulator is not None:
            if self._arrivals is None or len(self._arrivals) < n_ticks:
                raise ValueError(
                    "simulator mode requires arrival_rates of length n_ticks"
                )
        start = self._monotonic()
        records: list[TickRecord] = []
        slips: list[float] = []
        timings: list[TickTiming] = []
        for i in range(n_ticks):
            deadline = start + (i + 1) * self._dt
            record, timing = self._tick(i)
            now = self._monotonic()
            remaining = deadline - now
            slip = 0.0 if remaining >= 0.0 else -remaining
            slips.append(slip)
            record = replace(record, deadline_slip_s=slip)
            records.append(record)
            timings.append(timing)
            self._notify(record)
            if remaining > 0.0:
                self._sleep(remaining)
        return LoopResult(
            ticks=tuple(records),
            deadline_slips_s=tuple(slips),
            timings=tuple(timings),
        )

    def _notify(self, record: TickRecord) -> None:
        for observer in self._observers:
            try:
                observer.on_tick(record)
            except Exception:
                logger.exception("tick observer failed; control loop continues")

    def _tick(self, index: int) -> tuple[TickRecord, TickTiming]:
        now_s = index * self._dt
        if self._simulator is not None and self._arrivals is not None:
            self._simulator.step(self._arrivals[index], self._n_desired)
        t0 = time.perf_counter()
        obs = self._collector.collect(now_s)
        t1 = time.perf_counter()
        n_ready = max(int(round(obs.n)), self._config.control.n_min)
        raw = feature_vector(obs, include_rho=self._include_rho)
        smoothed = self._ewma.update(raw)
        features = (
            self._normaliser.transform(smoothed)
            if self._normaliser is not None
            else smoothed
        )
        self._window.push(features)
        if self._set_observation is not None:
            self._set_observation(obs.u, obs.r)
        t2 = time.perf_counter()

        forecast_pairs: tuple[tuple[float, float], ...]
        demand: tuple[int, ...]
        mpc_u: int | None = None
        mpc_cost: float | None = None
        mpc_solve_time_s: float | None = None
        gate: GateDecision | None = None
        t_inf0 = time.perf_counter()
        if self._window.full:
            window = np.asarray(self._window.tensor(), dtype=np.float32)
            raw_fc = self._forecaster.predict(window)
            forecast_pairs = _as_pairs(raw_fc)
        else:
            forecast_pairs = tuple(
                (float(obs.u), float(obs.r))
                for _ in range(self._config.forecast.horizon)
            )
        t3 = time.perf_counter()
        demand = tuple(
            map_to_demand(forecast_pairs, n_ready, self._drift.gamma, self._config)
        )
        n_required = map_to_demand(
            ((obs.u, obs.r),), n_ready, 0.0, self._config
        )[0]
        drift = self._drift.step(n_required, demand)
        t4 = time.perf_counter()

        safety = evaluate_safety(obs.u, n_ready, self._config)
        if self._skip_mpc:
            safety = SafetyDecision(active=False, u=0)
            applied = _direct_demand_action(n_ready, demand[0], self._config)
            if applied > 0:
                self._last_scale_up_s = now_s
        elif safety.active:
            applied = safety.u
            if applied > 0:
                self._last_scale_up_s = now_s
        else:
            sol = solve_mpc(
                n_ready, demand, self._pending, self._last_u, self._config
            )
            mpc_u = sol.u
            mpc_cost = sol.cost
            mpc_solve_time_s = sol.solve_time_s
            gate = apply_gates(
                sol.u,
                n_ready,
                self._config.control,
                now_s=now_s,
                last_scale_up_s=self._last_scale_up_s,
            )
            applied = gate.u
            self._last_scale_up_s = gate.last_scale_up_s
        t5 = time.perf_counter()

        self._n_plant, self._pending = apply_plant(
            self._n_plant,
            self._pending,
            applied,
            self._tau,
            self._config.control.n_min,
            self._config.control.n_max,
        )
        self._n_desired = _clip(
            self._n_desired + applied,
            self._config.control.n_min,
            self._config.control.n_max,
        )
        self._backend.set_replicas(self._n_desired)
        self._last_u = applied
        t6 = time.perf_counter()
        self._log_tick(
            now_s, obs, forecast_pairs, demand, mpc_u, mpc_cost, applied, gate, drift
        )
        timing = TickTiming(
            scrape_s=t1 - t0,
            features_s=t2 - t1,
            inference_s=t3 - t_inf0 if self._window.full else 0.0,
            mapping_s=t4 - t3,
            mpc_s=t5 - t4,
            actuation_s=t6 - t5,
        )
        record = TickRecord(
            tick=index,
            timestamp=now_s,
            wall_time_s=self._wall_clock(),
            observation=obs,
            forecast=forecast_pairs,
            demand=demand,
            n_required=n_required,
            mpc_u=mpc_u,
            mpc_cost=mpc_cost,
            mpc_solve_time_s=mpc_solve_time_s,
            applied_u=applied,
            gate=gate,
            safety=safety,
            drift=drift,
            n_desired=self._n_desired,
            deadline_slip_s=0.0,
        )
        return record, timing

    def _log_tick(
        self,
        timestamp: float,
        obs: Observation,
        forecast: tuple[tuple[float, float], ...],
        demand: tuple[int, ...],
        mpc_u: int | None,
        mpc_cost: float | None,
        applied: int,
        gate: GateDecision | None,
        drift: DriftSnapshot,
    ) -> None:
        payload = {
            "timestamp": timestamp,
            "u": obs.u,
            "r": obs.r,
            "lam": obs.lam,
            "ell_p95": obs.ell_p95,
            "n": obs.n,
            "forecast": [list(pair) for pair in forecast],
            "demand": list(demand),
            "mpc_u": mpc_u,
            "mpc_cost": mpc_cost,
            "applied_u": applied,
            "gate": None
            if gate is None
            else {
                "u": gate.u,
                "rounded": gate.rounded,
                "rate_clipped": gate.rate_clipped,
                "bound_clipped": gate.bound_clipped,
                "after_deadband": gate.after_deadband,
                "cooldown_blocked": gate.cooldown_blocked,
            },
            "gamma": drift.gamma,
            "ebar": drift.ebar,
            "error": drift.error,
        }
        logger.info("%s", json.dumps(payload, sort_keys=True))


def _direct_demand_action(
    n_current: int, n_hat_next: int, config: PactConfig
) -> int:
    """Apply ``Ñ(t+1)`` with shared pool bounds; skip MPC, gates, and safety."""

    ctrl = config.control
    u = int(n_hat_next) - n_current
    u = clip_rate_limit(u, ctrl.delta_max)
    return clip_pool_bounds(u, n_current, ctrl.n_min, ctrl.n_max)


def _as_pairs(forecast: NDArray[np.float32]) -> tuple[tuple[float, float], ...]:
    arr = np.asarray(forecast, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"forecast must be (H, 2), got {arr.shape}")
    return tuple((float(row[0]), float(row[1])) for row in arr)


def _clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def main(argv: Sequence[str] | None = None) -> None:
    """Live loop: Prometheus collector + compose backend. Requires τc on disk."""

    import argparse

    from pact.actuation.docker_backend import DockerComposeBackend
    from pact.telemetry.collector import PrometheusCollector

    parser = argparse.ArgumentParser(description="PACT live control loop")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument(
        "--compose-file", type=Path, default=Path("testbed/docker-compose.yml")
    )
    parser.add_argument("--service", default="service")
    parser.add_argument("--ticks", type=int, default=120)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config, cold_start_path=args.cold_start)
    timeout_s = max(config.control.tau_c_s + config.telemetry.dt, 30.0)
    loop = ControlLoop(
        config,
        collector=PrometheusCollector(config),
        backend=DockerComposeBackend(
            args.service,
            args.compose_file,
            timeout_s=timeout_s,
            poll_s=1.0,
        ),
        forecaster=RepeatObservationForecaster(config.forecast.horizon),
        n_initial=config.control.n_min,
    )
    loop.run(args.ticks)


if __name__ == "__main__":
    main()

