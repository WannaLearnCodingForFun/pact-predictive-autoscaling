"""CLI: tune baselines on validation with a shared budget."""

from __future__ import annotations

import argparse
from pathlib import Path

from pact.baselines.tune import tune_baselines
from pact.config import DEFAULT_COLD_START_PATH, DEFAULT_CONFIG_PATH, load_config
from pact.eval.datasets import generate_burst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=Path("results/baseline_tuning.json")
    )
    parser.add_argument("--ticks", type=int, default=80)
    args = parser.parse_args(argv)
    config = load_config(args.config, cold_start_path=args.cold_start)
    trace = generate_burst(
        "ramp",
        n_ticks=args.ticks,
        dt=config.telemetry.dt,
        rise_ticks=8,
        base=15.0,
        peak=70.0,
        name="tune_val_ramp",
    )
    chosen = tune_baselines(config, trace, budget=args.budget, results_path=args.out)
    print(chosen)


if __name__ == "__main__":
    main()
