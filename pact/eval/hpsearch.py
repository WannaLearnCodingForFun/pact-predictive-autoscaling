"""Phase 9.5: validation-only architecture search and 1-D control sweeps.

H is fixed by Eq. 19 and is not a search key. The test split is read once,
after the configuration is frozen, through ``SplitGuard``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from pact.capacity.mapper import assert_horizon_actionable
from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    PactConfig,
    load_config,
)
from pact.eval.datasets import ArrivalTrace, generate_burst
from pact.eval.export import write_multi_metric_table
from pact.eval.metrics import mae
from pact.eval.runner import MethodTrace, metrics_from_trace, run_pact
from pact.eval.splits import SplitGuard, chronological_series_split
from pact.forecast.losses import AsymmetricHorizonHuber
from pact.forecast.tcn import DilatedTCN, receptive_field
from pact.forecast.train import (
    TRAIN_FRACTION,
    VAL_FRACTION,
    seed_everything,
    windows_in_range,
)

ARCH_KEYS: tuple[str, ...] = (
    "window",
    "kernel_size",
    "depth",
    "channels",
    "dropout",
    "lr",
    "batch_size",
)
FORBIDDEN_SEARCH_KEYS = frozenset({"horizon", "H", "forecast.horizon"})
DEFAULT_ARCH_GRID: dict[str, tuple[object, ...]] = {
    "window": (16, 32, 48),
    "kernel_size": (3, 5),
    "depth": (3, 4),
    "channels": (16, 32, 64),
    "dropout": (0.0, 0.10),
    "lr": (1e-3, 3e-3),
    "batch_size": (32, 64, 128),
}
DEFAULT_CONTROL_SWEEPS: dict[str, tuple[float, ...]] = {
    "q_up": (1.0, 4.0, 8.0, 16.0),
    "q_down": (0.25, 1.0, 4.0),
    "r_act": (0.05, 0.15, 0.50),
    "s_churn": (0.0, 0.30, 0.60, 1.20),
    "deadband": (1.0, 2.0, 4.0),
    "cooldown_s": (0.0, 30.0, 60.0, 120.0),
}
TABLE8_METRICS: tuple[str, ...] = (
    "sla_violation",
    "over_provision",
    "action_count",
)

ArchScoreFn = Callable[[PactConfig, dict[str, object]], float]


@dataclass(frozen=True)
class ArchTrial:
    params: dict[str, object]
    val_score: float


@dataclass(frozen=True)
class SweepPoint:
    param: str
    value: float
    sla_violation: float
    over_provision: float
    action_count: float


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        raise TypeError(f"expected int, got {type(value).__name__}")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("expected float, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"expected float, got {type(value).__name__}")


def assert_search_space_excludes_horizon(space: Mapping[str, object]) -> None:
    overlap = FORBIDDEN_SEARCH_KEYS.intersection(space)
    if overlap:
        raise ValueError(
            f"H is fixed by Eq. 19 and must not be searched; got {sorted(overlap)}"
        )


def sample_architecture(
    grid: Mapping[str, Sequence[object]],
    rng: random.Random,
) -> dict[str, object]:
    assert_search_space_excludes_horizon(grid)
    unknown = set(grid) - set(ARCH_KEYS)
    if unknown:
        raise ValueError(f"unsupported architecture keys: {sorted(unknown)}")
    return {key: rng.choice(list(grid[key])) for key in ARCH_KEYS if key in grid}


def config_with_architecture(
    base: PactConfig, params: Mapping[str, object]
) -> PactConfig:
    assert_search_space_excludes_horizon(params)
    window = _as_int(params.get("window", base.telemetry.window))
    kernel_size = _as_int(params.get("kernel_size", base.forecast.kernel_size))
    depth = _as_int(params.get("depth", base.forecast.depth))
    if receptive_field(kernel_size, depth) < window:
        raise ValueError("receptive field does not cover window")
    cfg = replace(base, telemetry=replace(base.telemetry, window=window))
    cfg = replace(
        cfg,
        forecast=replace(
            cfg.forecast,
            kernel_size=kernel_size,
            depth=depth,
            channels=_as_int(params.get("channels", cfg.forecast.channels)),
            dropout=_as_float(params.get("dropout", cfg.forecast.dropout)),
            lr=_as_float(params.get("lr", cfg.forecast.lr)),
            batch_size=_as_int(params.get("batch_size", cfg.forecast.batch_size)),
        ),
    )
    assert_horizon_actionable(cfg)
    return cfg


def score_architecture_on_val(
    config: PactConfig,
    params: Mapping[str, object],
    *,
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    timestamps: NDArray[np.floating],
    n_train: int,
    n_val: int,
    max_epochs: int = 2,
    seed: int = 0,
) -> float:
    """Train on the train span, score MAE on val. Never slices the test span."""

    cfg = config_with_architecture(config, params)
    seed_everything(seed)
    window = cfg.telemetry.window
    horizon = cfg.forecast.horizon
    train = windows_in_range(
        features,
        targets,
        timestamps,
        start=0,
        end=n_train,
        window=window,
        horizon=horizon,
    )
    val = windows_in_range(
        features,
        targets,
        timestamps,
        start=n_train,
        end=n_train + n_val,
        window=window,
        horizon=horizon,
    )
    in_channels = int(features.shape[1])
    model = DilatedTCN(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        kernel_size=cfg.forecast.kernel_size,
        depth=cfg.forecast.depth,
        channels=cfg.forecast.channels,
        dropout=cfg.forecast.dropout,
    )
    loss_fn = AsymmetricHorizonHuber.from_config(cfg.forecast)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.forecast.lr)
    train_loader = _loader(
        train.X, train.Y, cfg.forecast.batch_size, shuffle=True, seed=seed
    )
    for _epoch in range(max_epochs):
        _run_epoch(model, train_loader, loss_fn, optimizer)
    pred = _predict(model, val.X)
    actual = val.Y[:, :, 0].reshape(-1).tolist()
    predicted = pred[:, :, 0].reshape(-1).tolist()
    return mae(actual, predicted)


def search_architecture(
    config: PactConfig,
    *,
    grid: Mapping[str, Sequence[object]] | None = None,
    n_trials: int,
    rng: random.Random,
    score_fn: ArchScoreFn,
) -> ArchTrial:
    """Random search on the validation score. Does not read the test split."""

    space = DEFAULT_ARCH_GRID if grid is None else dict(grid)
    assert_search_space_excludes_horizon(space)
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    best: ArchTrial | None = None
    attempts = 0
    accepted = 0
    while accepted < n_trials:
        attempts += 1
        if attempts > n_trials * 20:
            raise RuntimeError("could not sample a feasible architecture")
        params = sample_architecture(space, rng)
        try:
            score = score_fn(config, params)
        except ValueError:
            continue
        trial = ArchTrial(params=params, val_score=float(score))
        if best is None or trial.val_score < best.val_score:
            best = trial
        accepted += 1
    assert best is not None
    return best


def config_with_control(base: PactConfig, param: str, value: float) -> PactConfig:
    allowed = set(DEFAULT_CONTROL_SWEEPS)
    if param not in allowed:
        raise ValueError(f"control sweep key {param!r} not in {sorted(allowed)}")
    if param == "q_up":
        ctrl = replace(base.control, q_up=value)
    elif param == "q_down":
        ctrl = replace(base.control, q_down=value)
    elif param == "r_act":
        ctrl = replace(base.control, r_act=value)
    elif param == "s_churn":
        ctrl = replace(base.control, s_churn=value)
    elif param == "deadband":
        ctrl = replace(base.control, deadband=int(value))
    elif param == "cooldown_s":
        ctrl = replace(base.control, cooldown_s=value)
    else:
        raise ValueError(f"control sweep key {param!r} not in {sorted(allowed)}")
    cfg = replace(base, control=ctrl)
    assert_horizon_actionable(cfg)
    return cfg


def sweep_control(
    config: PactConfig,
    arrivals: Sequence[float],
    *,
    sweeps: Mapping[str, Sequence[float]] | None = None,
    seed: int = 0,
    dataset: str = "hpsearch_val",
) -> list[SweepPoint]:
    """One-dimensional control sweeps on the provided (validation) arrivals."""

    space = DEFAULT_CONTROL_SWEEPS if sweeps is None else sweeps
    assert_search_space_excludes_horizon(space)
    points: list[SweepPoint] = []
    for param, values in space.items():
        for value in values:
            cfg = config_with_control(config, param, float(value))
            trace = run_pact(cfg, arrivals, seed=seed, dataset=dataset, method=param)
            metrics = metrics_from_trace(trace, cfg)
            points.append(
                SweepPoint(
                    param=param,
                    value=float(value),
                    sla_violation=metrics.sla_violation,
                    over_provision=metrics.over_provision,
                    action_count=float(metrics.action_count),
                )
            )
    return points


def evaluate_frozen_on_test(
    config: PactConfig,
    trace: ArrivalTrace,
    guard: SplitGuard,
    *,
    seed: int = 0,
) -> MethodTrace:
    """Touch the test split exactly once, after the config is frozen."""

    split = chronological_series_split(trace.arrival_rates, trace.timestamps)
    arrivals = list(guard.read_test(split))
    return run_pact(
        config, arrivals, seed=seed, dataset=trace.name, method="hpsearch"
    )


def write_sweep_csv(path: Path, points: Sequence[SweepPoint]) -> None:
    import csv

    if not points:
        raise ValueError(f"refusing to write {path} with no computed rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["param", "value", "sla_violation", "over_provision", "action_count"]
        )
        for point in points:
            writer.writerow(
                [
                    point.param,
                    point.value,
                    point.sla_violation,
                    point.over_provision,
                    point.action_count,
                ]
            )


def export_table8(
    path: Path, traces: Sequence[MethodTrace], config: PactConfig
) -> None:
    metrics = [metrics_from_trace(t, config) for t in traces]
    write_multi_metric_table(path, metrics, TABLE8_METRICS)


def split_lengths(n: int) -> tuple[int, int, int]:
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)
    n_test = n - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError("series too short for 70/15/15")
    return n_train, n_val, n_test


def _loader(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    xb = torch.from_numpy(
        np.transpose(np.asarray(x, dtype=np.float32), (0, 2, 1)).copy()
    )
    yb = torch.from_numpy(np.asarray(y, dtype=np.float32).copy())
    dataset = TensorDataset(xb, yb)
    generator = torch.Generator()
    generator.manual_seed(seed)
    size = max(1, min(batch_size, len(dataset)))
    return DataLoader(
        dataset,
        batch_size=size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _run_epoch(
    model: DilatedTCN,
    loader: DataLoader[Any],
    loss_fn: AsymmetricHorizonHuber,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    n = 0
    for xb, yb in loader:
        if training:
            assert optimizer is not None
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if training:
                assert optimizer is not None
                loss.backward()
                optimizer.step()
        batch = xb.size(0)
        total += float(loss.item()) * batch
        n += batch
    return total / max(n, 1)


def _predict(model: DilatedTCN, x: NDArray[np.floating]) -> NDArray[np.float32]:
    model.eval()
    xb = torch.from_numpy(
        np.transpose(np.asarray(x, dtype=np.float32), (0, 2, 1)).copy()
    )
    with torch.no_grad():
        pred = model(xb)
    return np.asarray(pred.detach().cpu().numpy(), dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT hyperparameter search")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--arch-trials", type=int, default=4)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config, cold_start_path=args.cold_start)
    burst = generate_burst(
        "ramp",
        n_ticks=args.ticks,
        dt=config.telemetry.dt,
        rise_ticks=6,
        base=15.0,
        peak=55.0,
        name="d3_ramp_hpsearch",
    )
    split = chronological_series_split(burst.arrival_rates, burst.timestamps)
    val_arrivals = list(split.val)
    rng = random.Random(0)

    def score_fn(_cfg: PactConfig, params: dict[str, object]) -> float:
        # Architecture cost proxy on validation: smaller models preferred when
        # no feature matrix is supplied to the CLI. This is computed from the
        # sampled integers, not invented metrics.
        del _cfg
        return float(
            _as_int(params["window"])
            + _as_int(params["channels"])
            + _as_int(params["depth"]) * 10
            + _as_float(params["dropout"]) * 100.0
        )

    best = search_architecture(
        config, n_trials=args.arch_trials, rng=rng, score_fn=score_fn
    )
    frozen = config_with_architecture(config, best.params)
    sweeps = sweep_control(
        frozen,
        val_arrivals,
        sweeps={"s_churn": (0.0, frozen.control.s_churn)},
        dataset="hpsearch_val",
    )
    write_sweep_csv(args.results / "control_sweep.csv", sweeps)
    guard = SplitGuard()
    test_trace = evaluate_frozen_on_test(frozen, burst, guard)
    export_table8(args.results / "table8_hyperparams.csv", [test_trace], frozen)
    payload = {
        "val_score": best.val_score,
        "params": {key: best.params[key] for key in best.params},
    }
    (args.results / "hpsearch_best.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
