"""U4: forecast-degradation sweeps and the reactive crossing point.

Corruption is applied to the horizon vector the controller actually uses.
The reactive baseline does not see the forecast, so its SLA is a horizontal
line. Crossing magnitude is linearly interpolated from measured points; it
is omitted when PACT never exceeds reactive. D1→D2 shift rows are written
only from traces that were actually loaded — never invented FIFA/Bitbrains.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    MissingColdStartError,
    PactConfig,
    load_config,
)
from pact.eval.datasets import (
    ArrivalTrace,
    generate_burst,
    parse_bitbrains_faststorage,
    parse_fifa_requests,
)
from pact.eval.runner import metrics_from_trace, run_method, run_pact
from pact.forecast.seq_models import ModuleForecaster
from pact.forecast.tcn import DilatedTCN
from pact.forecast.train import chronological_window_splits, seed_everything
from pact.loop import Forecaster, RepeatObservationForecaster
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import SimCollector
from pact.telemetry.features import feature_dim, feature_vector

logger = logging.getLogger(__name__)

ROBUSTNESS_COLUMNS: tuple[str, ...] = (
    "row_type",
    "kind",
    "magnitude",
    "safety_enabled",
    "method",
    "seed",
    "dataset",
    "sla_violation",
    "over_provision",
)
CORRUPTION_KINDS: tuple[str, ...] = ("noise", "bias", "lag", "freeze")
DEFAULT_NOISE_SIGMAS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
DEFAULT_BIASES: tuple[float, ...] = (-0.40, -0.20, -0.10, 0.0, 0.10, 0.20, 0.40)
DEFAULT_LAGS: tuple[int, ...] = (0, 1, 2, 4, 8)
DEFAULT_D1_PATH = Path("data/processed/d1_fifa.log")
DEFAULT_D2_PATH = Path("data/processed/d2_bitbrains.csv")


@dataclass(frozen=True)
class RobustnessRow:
    row_type: str
    kind: str
    magnitude: float
    safety_enabled: bool
    method: str
    seed: int
    dataset: str
    sla_violation: float
    over_provision: float


class CorruptForecaster:
    """Wrap a ``Forecaster`` and inject one controlled corruption family."""

    def __init__(
        self,
        inner: Forecaster,
        *,
        kind: str,
        magnitude: float,
        seed: int = 0,
    ) -> None:
        if kind not in CORRUPTION_KINDS:
            raise ValueError(f"unknown corruption kind {kind!r}")
        if kind == "noise" and magnitude < 0.0:
            raise ValueError(f"noise sigma must be >= 0, got {magnitude}")
        if kind == "lag":
            lag = int(magnitude)
            if lag != magnitude or lag < 0:
                raise ValueError(f"lag must be a non-negative integer, got {magnitude}")
        self._inner = inner
        self.kind = kind
        self.magnitude = float(magnitude)
        self._rng = np.random.default_rng(int(seed))
        self._history: list[NDArray[np.float32]] = []
        self._frozen: NDArray[np.float32] | None = None

    def set_observation(self, u: float, r: float) -> None:
        setter = getattr(self._inner, "set_observation", None)
        if callable(setter):
            setter(u, r)

    def predict(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.kind == "freeze" and self._frozen is not None:
            return self._frozen.copy()
        raw = np.asarray(self._inner.predict(window), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError(f"forecast must be (H, 2), got {raw.shape}")
        if self.kind == "noise":
            if self.magnitude == 0.0:
                return raw.copy()
            noise = self._rng.normal(0.0, self.magnitude, size=raw.shape)
            return (raw + noise).astype(np.float32, copy=False)
        if self.kind == "bias":
            return (raw + np.float32(self.magnitude)).astype(np.float32, copy=False)
        if self.kind == "lag":
            self._history.append(raw.copy())
            lag = int(self.magnitude)
            idx = len(self._history) - 1 - lag
            if idx < 0:
                return self._history[0].copy()
            return self._history[idx].copy()
        self._frozen = raw.copy()
        return self._frozen.copy()


def disable_safety(config: PactConfig) -> PactConfig:
    """Raise ``u_emergency`` so the ceiling never fires."""

    return replace(config, control=replace(config.control, u_emergency=math.inf))


def crossing_magnitude(
    magnitudes: Sequence[float],
    pact_sla: Sequence[float],
    reactive_sla: float,
) -> float | None:
    """Smallest magnitude where PACT SLA exceeds reactive, linearly interpolated.

    ``None`` when PACT stays at or below reactive on every measured point.
    If PACT is already worse at the smallest magnitude, that magnitude is
    the crossing.
    """

    if len(magnitudes) != len(pact_sla) or not magnitudes:
        raise ValueError("magnitudes and pact_sla must be non-empty and aligned")
    order = sorted(
        zip(
            (float(m) for m in magnitudes),
            (float(s) for s in pact_sla),
            strict=True,
        )
    )
    mags = [m for m, _ in order]
    slas = [s for _, s in order]
    if slas[0] > reactive_sla:
        return mags[0]
    for i in range(1, len(mags)):
        prev_sla, sla = slas[i - 1], slas[i]
        if prev_sla <= reactive_sla < sla:
            if sla == prev_sla:
                return mags[i]
            frac = (reactive_sla - prev_sla) / (sla - prev_sla)
            return mags[i - 1] + frac * (mags[i] - mags[i - 1])
    return None


def mean_arrival(rates: Sequence[float]) -> float:
    if not rates:
        raise ValueError("mean_arrival of empty series")
    return sum(float(x) for x in rates) / len(rates)


def shift_magnitude(train: Sequence[float], test: Sequence[float]) -> float:
    """Absolute difference of mean arrival rates — a measured shift, not a label."""

    return abs(mean_arrival(test) - mean_arrival(train))


def persistence_forecaster(config: PactConfig) -> RepeatObservationForecaster:
    return RepeatObservationForecaster(config.forecast.horizon)


def run_corrupted_pact(
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    kind: str,
    magnitude: float,
    seed: int,
    dataset: str,
    inner: Forecaster | None = None,
) -> tuple[float, float]:
    wrapped = CorruptForecaster(
        inner if inner is not None else persistence_forecaster(config),
        kind=kind,
        magnitude=magnitude,
        seed=seed,
    )
    trace = run_pact(
        config, arrivals, seed=seed, dataset=dataset, forecaster=wrapped
    )
    metrics = metrics_from_trace(trace, config)
    return metrics.sla_violation, metrics.over_provision


def _row(
    *,
    row_type: str,
    kind: str,
    magnitude: float,
    safety_enabled: bool,
    method: str,
    seed: int,
    dataset: str,
    sla_violation: float,
    over_provision: float,
) -> RobustnessRow:
    return RobustnessRow(
        row_type=row_type,
        kind=kind,
        magnitude=magnitude,
        safety_enabled=safety_enabled,
        method=method,
        seed=seed,
        dataset=dataset,
        sla_violation=sla_violation,
        over_provision=over_provision,
    )


def sweep_family(
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    kind: str,
    magnitudes: Sequence[float],
    seeds: Sequence[int],
    dataset: str,
    reactive_by_seed: dict[int, tuple[float, float]],
) -> list[RobustnessRow]:
    rows: list[RobustnessRow] = []
    for safety_on in (True, False):
        cfg = config if safety_on else disable_safety(config)
        sla_by_mag: dict[float, list[float]] = {float(m): [] for m in magnitudes}
        for mag in magnitudes:
            for seed in seeds:
                sla, over = run_corrupted_pact(
                    cfg,
                    arrivals,
                    kind=kind,
                    magnitude=float(mag),
                    seed=int(seed),
                    dataset=dataset,
                )
                sla_by_mag[float(mag)].append(sla)
                rows.append(
                    _row(
                        row_type="point",
                        kind=kind,
                        magnitude=float(mag),
                        safety_enabled=safety_on,
                        method="pact",
                        seed=int(seed),
                        dataset=dataset,
                        sla_violation=sla,
                        over_provision=over,
                    )
                )
        pact_means = [
            sum(sla_by_mag[float(m)]) / len(sla_by_mag[float(m)])
            for m in magnitudes
        ]
        reactive_mean = sum(reactive_by_seed[int(s)][0] for s in seeds) / len(seeds)
        crossed = crossing_magnitude(
            [float(m) for m in magnitudes], pact_means, reactive_mean
        )
        if crossed is not None:
            rows.append(
                _row(
                    row_type="crossing",
                    kind=kind,
                    magnitude=crossed,
                    safety_enabled=safety_on,
                    method="pact",
                    seed=-1,
                    dataset=dataset,
                    sla_violation=reactive_mean,
                    over_provision=float("nan"),
                )
            )
    return rows


def collect_feature_series(
    config: PactConfig, arrivals: Sequence[float]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64]]:
    n = config.control.n_min
    sim = PoolSimulator(config, n_initial=n)
    collector = SimCollector(sim)
    features: list[list[float]] = []
    targets: list[list[float]] = []
    timestamps: list[float] = []
    dt = config.telemetry.dt
    mu = config.capacity.mu
    u_star = config.capacity.u_target
    include_rho = config.ablation.include_rho
    expected = feature_dim(include_rho=include_rho)
    for i, rate in enumerate(arrivals):
        now_s = i * dt
        sim.step(float(rate), n)
        obs = collector.collect(now_s)
        features.append(list(feature_vector(obs, include_rho=include_rho)))
        targets.append([float(obs.u), float(obs.r)])
        timestamps.append(now_s)
        raw = math.ceil(float(rate) / (mu * u_star)) if rate > 0.0 else 1
        n = max(config.control.n_min, min(config.control.n_max, raw))
    feats = np.asarray(features, dtype=np.float32)
    if feats.shape[1] != expected:
        raise RuntimeError(f"expected {expected} features, got {feats.shape}")
    return (
        feats,
        np.asarray(targets, dtype=np.float32),
        np.asarray(timestamps, dtype=np.float64),
    )


def train_on_arrivals(
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
) -> ModuleForecaster:
    """Fit a TCN on plant features from ``arrivals``. No ONNX — used for shift."""

    from pact.forecast.losses import AsymmetricHorizonHuber
    from pact.forecast.train import _loader, _run_epoch

    seed_everything(seed)
    features, targets, timestamps = collect_feature_series(config, arrivals)
    train, val, _test = chronological_window_splits(
        features,
        targets,
        timestamps,
        window=config.telemetry.window,
        horizon=config.forecast.horizon,
    )
    model = DilatedTCN.from_config(config, in_channels=features.shape[1])
    loss_fn = AsymmetricHorizonHuber.from_config(config.forecast)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.forecast.lr)
    train_loader = _loader(train, config.forecast.batch_size, shuffle=True, seed=seed)
    val_loader = _loader(val, config.forecast.batch_size, shuffle=False, seed=seed)
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0
    for _epoch in range(1, max_epochs + 1):
        _run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = _run_epoch(model, val_loader, loss_fn, None)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return ModuleForecaster(model)


def load_fifa_path(path: Path, *, dt: float) -> ArrivalTrace:
    if path.suffix == ".gz":
        import gzip

        text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return parse_fifa_requests(text.splitlines(), dt=dt)


def load_bitbrains_path(path: Path, *, dt: float) -> ArrivalTrace:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    return parse_bitbrains_faststorage(rows, target_dt_s=dt)


def run_distribution_shift(
    config: PactConfig,
    train_trace: ArrivalTrace,
    test_trace: ArrivalTrace,
    *,
    seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    reactive_on_test: dict[int, tuple[float, float]],
) -> list[RobustnessRow]:
    """Train on ``train_trace``, evaluate PACT on ``test_trace`` with no retraining."""

    mag = shift_magnitude(train_trace.arrival_rates, test_trace.arrival_rates)
    rows: list[RobustnessRow] = []
    dataset = f"{train_trace.name}->{test_trace.name}"
    for safety_on in (True, False):
        cfg = config if safety_on else disable_safety(config)
        slas: list[float] = []
        for seed in seeds:
            model = train_on_arrivals(
                cfg,
                train_trace.arrival_rates,
                seed=int(seed),
                max_epochs=max_epochs,
                patience=patience,
            )
            trace = run_pact(
                cfg,
                test_trace.arrival_rates,
                seed=int(seed),
                dataset=dataset,
                forecaster=model,
                method="pact",
            )
            metrics = metrics_from_trace(trace, cfg)
            slas.append(metrics.sla_violation)
            rows.append(
                _row(
                    row_type="point",
                    kind="shift",
                    magnitude=mag,
                    safety_enabled=safety_on,
                    method="pact",
                    seed=int(seed),
                    dataset=dataset,
                    sla_violation=metrics.sla_violation,
                    over_provision=metrics.over_provision,
                )
            )
        reactive_mean = (
            sum(reactive_on_test[int(s)][0] for s in seeds) / len(seeds)
        )
        pact_mean = sum(slas) / len(slas)
        crossed = crossing_magnitude((mag,), (pact_mean,), reactive_mean)
        if crossed is not None:
            rows.append(
                _row(
                    row_type="crossing",
                    kind="shift",
                    magnitude=crossed,
                    safety_enabled=safety_on,
                    method="pact",
                    seed=-1,
                    dataset=dataset,
                    sla_violation=reactive_mean,
                    over_provision=float("nan"),
                )
            )
    return rows


def run_reactive_baselines(
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    seeds: Sequence[int],
    dataset: str,
) -> tuple[list[RobustnessRow], dict[int, tuple[float, float]]]:
    rows: list[RobustnessRow] = []
    by_seed: dict[int, tuple[float, float]] = {}
    for seed in seeds:
        trace = run_method(
            "reactive", config, arrivals, seed=int(seed), dataset=dataset
        )
        metrics = metrics_from_trace(trace, config)
        by_seed[int(seed)] = (metrics.sla_violation, metrics.over_provision)
        rows.append(
            _row(
                row_type="baseline",
                kind="reactive",
                magnitude=0.0,
                safety_enabled=False,
                method="reactive",
                seed=int(seed),
                dataset=dataset,
                sla_violation=metrics.sla_violation,
                over_provision=metrics.over_provision,
            )
        )
    return rows, by_seed


def run_robustness(
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    dataset: str,
    seeds: Sequence[int],
    noise_sigmas: Sequence[float] = DEFAULT_NOISE_SIGMAS,
    biases: Sequence[float] = DEFAULT_BIASES,
    lags: Sequence[int] = DEFAULT_LAGS,
    d1: ArrivalTrace | None = None,
    d2: ArrivalTrace | None = None,
    max_epochs: int = 3,
    patience: int = 3,
) -> list[RobustnessRow]:
    rows, reactive = run_reactive_baselines(
        config, arrivals, seeds=seeds, dataset=dataset
    )
    rows.extend(
        sweep_family(
            config,
            arrivals,
            kind="noise",
            magnitudes=tuple(float(x) for x in noise_sigmas),
            seeds=seeds,
            dataset=dataset,
            reactive_by_seed=reactive,
        )
    )
    rows.extend(
        sweep_family(
            config,
            arrivals,
            kind="bias",
            magnitudes=tuple(float(x) for x in biases),
            seeds=seeds,
            dataset=dataset,
            reactive_by_seed=reactive,
        )
    )
    rows.extend(
        sweep_family(
            config,
            arrivals,
            kind="lag",
            magnitudes=tuple(float(k) for k in lags),
            seeds=seeds,
            dataset=dataset,
            reactive_by_seed=reactive,
        )
    )
    rows.extend(
        sweep_family(
            config,
            arrivals,
            kind="freeze",
            magnitudes=(1.0,),
            seeds=seeds,
            dataset=dataset,
            reactive_by_seed=reactive,
        )
    )
    if d1 is not None and d2 is not None:
        _, reactive_d2 = run_reactive_baselines(
            config, d2.arrival_rates, seeds=seeds, dataset=d2.name
        )
        rows.extend(
            run_distribution_shift(
                config,
                d1,
                d2,
                seeds=seeds,
                max_epochs=max_epochs,
                patience=patience,
                reactive_on_test=reactive_d2,
            )
        )
        rows.extend(reactive_d2_rows(reactive_d2, d2.name))
    return rows


def reactive_d2_rows(
    by_seed: dict[int, tuple[float, float]], dataset: str
) -> list[RobustnessRow]:
    return [
        _row(
            row_type="baseline",
            kind="reactive",
            magnitude=0.0,
            safety_enabled=False,
            method="reactive",
            seed=seed,
            dataset=dataset,
            sla_violation=sla,
            over_provision=over,
        )
        for seed, (sla, over) in sorted(by_seed.items())
    ]


def write_robustness_csv(path: Path, rows: Sequence[RobustnessRow]) -> None:
    if not rows:
        raise ValueError(f"refusing to write {path} with no computed rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(ROBUSTNESS_COLUMNS))
        for row in rows:
            safety = "true" if row.safety_enabled else "false"
            over = (
                ""
                if math.isnan(row.over_provision)
                else f"{row.over_provision:.10g}"
            )
            writer.writerow(
                [
                    row.row_type,
                    row.kind,
                    f"{row.magnitude:.10g}",
                    safety,
                    row.method,
                    row.seed,
                    row.dataset,
                    f"{row.sla_violation:.10g}",
                    over,
                ]
            )


def _optional_trace(
    path: Path | None, *, loader: str, dt: float
) -> ArrivalTrace | None:
    if path is None or not path.is_file():
        if path is not None:
            logger.warning("skipping %s shift: %s is missing", loader, path)
        return None
    if loader == "d1":
        return load_fifa_path(path, dt=dt)
    return load_bitbrains_path(path, dt=dt)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT forecast-robustness sweep")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--d1", type=Path, default=DEFAULT_D1_PATH)
    parser.add_argument("--d2", type=Path, default=DEFAULT_D2_PATH)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config, cold_start_path=args.cold_start)
    except MissingColdStartError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Measure τc on the Docker testbed, then re-run:\n"
            "  .venv/bin/python scripts/measure_cold_start.py "
            "--n-reps 1 --skip-cold\n"
            "  .venv/bin/python -m pact.eval.robustness --results results"
        ) from None
    n_seeds = args.seeds if args.seeds is not None else config.eval.n_seeds
    burst = generate_burst(
        "step",
        n_ticks=args.ticks,
        dt=config.telemetry.dt,
        rise_ticks=4,
        base=20.0,
        peak=60.0,
        name="d3_step_robustness",
    )
    d1 = _optional_trace(args.d1, loader="d1", dt=config.telemetry.dt)
    d2 = _optional_trace(args.d2, loader="d2", dt=config.telemetry.dt)
    if (d1 is None) ^ (d2 is None):
        logger.warning("D1→D2 shift needs both traces; skipping shift family")
        d1, d2 = None, None
    rows = run_robustness(
        config,
        burst.arrival_rates,
        dataset=burst.name,
        seeds=tuple(range(n_seeds)),
        d1=d1,
        d2=d2,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    write_robustness_csv(args.results / "robustness.csv", rows)


if __name__ == "__main__":
    main()
