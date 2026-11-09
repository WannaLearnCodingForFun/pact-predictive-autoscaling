"""Drive a clean-checkout reproduction: tables, overhead, figures.

``make reproduce`` calls this module. τc is never invented: the measured
``results/cold_start.json`` must already exist.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    MissingColdStartError,
    PactConfig,
    load_config,
)
from pact.eval import ablation, hpsearch, overhead, runner
from pact.eval.datasets import generate_burst
from pact.eval.runner import FORECAST_METHODS
from pact.forecast.train import TRAIN_FRACTION, VAL_FRACTION, train_forecaster
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import SimCollector
from pact.telemetry.features import FEATURE_DIM, feature_vector

REPO_ROOT = Path(__file__).resolve().parents[2]


def require_cold_start(path: Path) -> None:
    if not path.is_file():
        raise MissingColdStartError(
            f"{path} is missing. Measure τc with: "
            "python scripts/measure_cold_start.py"
        )


def min_series_length(config: PactConfig) -> int:
    """Ticks needed so 70/15/15 each contain one L,H window."""

    need = config.telemetry.window + config.forecast.horizon
    by_train = math.ceil(need / TRAIN_FRACTION)
    by_val = math.ceil(need / VAL_FRACTION)
    return max(by_train, by_val, need * 2)


def collect_feature_series(
    config: PactConfig, arrivals: Sequence[float]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64]]:
    """Plant observations → (T, d) features and (T, 2) u/r targets."""

    n = config.control.n_min
    sim = PoolSimulator(config, n_initial=n)
    collector = SimCollector(sim)
    features: list[list[float]] = []
    targets: list[list[float]] = []
    timestamps: list[float] = []
    dt = config.telemetry.dt
    mu = config.capacity.mu
    u_star = config.capacity.u_target
    for i, rate in enumerate(arrivals):
        now_s = i * dt
        sim.step(float(rate), n)
        obs = collector.collect(now_s)
        features.append(list(feature_vector(obs, include_rho=True)))
        targets.append([float(obs.u), float(obs.r)])
        timestamps.append(now_s)
        raw = math.ceil(float(rate) / (mu * u_star)) if rate > 0.0 else 1
        n = max(config.control.n_min, min(config.control.n_max, raw))
    feats = np.asarray(features, dtype=np.float32)
    if feats.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {feats.shape}")
    return (
        feats,
        np.asarray(targets, dtype=np.float32),
        np.asarray(timestamps, dtype=np.float64),
    )


def train_and_export(
    config: PactConfig,
    results_dir: Path,
    *,
    n_ticks: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> None:
    burst = generate_burst(
        "ramp",
        n_ticks=n_ticks,
        dt=config.telemetry.dt,
        rise_ticks=max(1, n_ticks // 10),
        base=15.0,
        peak=55.0,
        name="reproduce_train",
    )
    features, targets, timestamps = collect_feature_series(
        config, burst.arrival_rates
    )
    train_forecaster(
        features,
        targets,
        timestamps,
        config,
        results_dir=results_dir,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
    )


def render_figures(results_dir: Path, figures_dir: Path) -> None:
    script = REPO_ROOT / "make_figures.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--results",
            str(results_dir),
            "--out",
            str(figures_dir),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def _cli_path(flag: str, value: Path) -> list[str]:
    return [flag, str(value)]


def reproduce(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    cold_start_path: Path = DEFAULT_COLD_START_PATH,
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("figures"),
    ticks: int = 80,
    seeds: int | None = None,
    methods: Sequence[str] = FORECAST_METHODS,
    max_epochs: int = 3,
    patience: int = 3,
    arch_trials: int = 4,
    skip_figures: bool = False,
) -> None:
    require_cold_start(cold_start_path)
    config = load_config(config_path, cold_start_path=cold_start_path)
    n_seeds = config.eval.n_seeds if seeds is None else seeds
    results_dir.mkdir(parents=True, exist_ok=True)
    shared = [
        *_cli_path("--config", config_path),
        *_cli_path("--cold-start", cold_start_path),
        *_cli_path("--results", results_dir),
        "--ticks",
        str(ticks),
    ]
    runner.main(
        [
            *shared,
            "--methods",
            ",".join(methods),
            "--seeds",
            str(n_seeds),
        ]
    )
    ablation.main([*shared, "--seeds", str(n_seeds)])
    hpsearch.main([*shared, "--arch-trials", str(arch_trials)])

    train_ticks = max(ticks, min_series_length(config))
    train_and_export(
        config,
        results_dir,
        n_ticks=train_ticks,
        max_epochs=max_epochs,
        patience=patience,
        seed=0,
    )
    overhead.main(
        [
            *shared,
            "--seed",
            "0",
            "--onnx",
            str(results_dir / "tcn.onnx"),
        ]
    )
    if not skip_figures:
        render_figures(results_dir, figures_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT full reproduction")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument(
        "--methods",
        default=",".join(FORECAST_METHODS),
    )
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--arch-trials", type=int, default=4)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    methods = tuple(
        item.strip() for item in str(args.methods).split(",") if item.strip()
    )
    reproduce(
        config_path=args.config,
        cold_start_path=args.cold_start,
        results_dir=args.results,
        figures_dir=args.figures,
        ticks=args.ticks,
        seeds=args.seeds,
        methods=methods,
        max_epochs=args.max_epochs,
        patience=args.patience,
        arch_trials=args.arch_trials,
        skip_figures=args.skip_figures,
    )


if __name__ == "__main__":
    main()
