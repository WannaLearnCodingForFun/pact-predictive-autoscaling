"""Phase 9.4: six single-component knockouts against the full PACT config."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    PactConfig,
    load_config,
    load_config_overlay,
)
from pact.eval.datasets import ArrivalTrace, generate_burst
from pact.eval.export import write_multi_metric_table
from pact.eval.runner import MethodTrace, metrics_from_trace, run_pact
from pact.eval.splits import SplitGuard, chronological_series_split

ABLATION_DIR = Path("configs/ablations")
ABLATION_NAMES: tuple[str, ...] = (
    "no_asym_loss",
    "no_mpc",
    "no_churn",
    "no_coldstart",
    "no_adaptive_margin",
    "no_rho_channel",
)
TABLE7_METRICS: tuple[str, ...] = (
    "sla_violation",
    "over_provision",
    "action_count",
)


def ablation_yaml(name: str) -> Path:
    path = ABLATION_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"ablation config not found: {path}")
    return path


def load_ablation(
    name: str,
    *,
    base_path: Path = DEFAULT_CONFIG_PATH,
    cold_start_path: Path = DEFAULT_COLD_START_PATH,
    tau_c_s: float | None = None,
) -> PactConfig:
    """Load default.yaml, then the named overlay. τc still comes from measurement."""

    if name not in ABLATION_NAMES:
        raise ValueError(f"unknown ablation {name!r}; expected one of {ABLATION_NAMES}")
    return load_config_overlay(
        base_path,
        ablation_yaml(name),
        cold_start_path=cold_start_path,
        tau_c_s=tau_c_s,
    )


def apply_ablation(base: PactConfig, name: str) -> PactConfig:
    """Disable exactly one component on an already-loaded config."""

    if name == "no_asym_loss":
        return replace(base, forecast=replace(base.forecast, kappa=1.0))
    if name == "no_mpc":
        return replace(base, ablation=replace(base.ablation, skip_mpc=True))
    if name == "no_churn":
        return replace(base, control=replace(base.control, s_churn=0.0))
    if name == "no_coldstart":
        return replace(
            base,
            control=replace(base.control, tau_c_s=0.0),
            ablation=replace(base.ablation, zero_tau_c=True),
        )
    if name == "no_adaptive_margin":
        return replace(base, ablation=replace(base.ablation, freeze_gamma=True))
    if name == "no_rho_channel":
        return replace(base, ablation=replace(base.ablation, include_rho=False))
    raise ValueError(f"unknown ablation {name!r}; expected one of {ABLATION_NAMES}")


def run_ablation_matrix(
    base: PactConfig,
    arrivals: Sequence[float],
    *,
    seeds: Sequence[int] = (0,),
    dataset: str = "ablation",
    names: Sequence[str] = ABLATION_NAMES,
    include_full: bool = True,
) -> list[MethodTrace]:
    """Run the full controller and each knockout on identical arrivals."""

    traces: list[MethodTrace] = []
    if include_full:
        for seed in seeds:
            traces.append(
                run_pact(
                    base, arrivals, seed=int(seed), dataset=dataset, method="full"
                )
            )
    for name in names:
        cfg = apply_ablation(base, name)
        for seed in seeds:
            traces.append(
                run_pact(
                    cfg, arrivals, seed=int(seed), dataset=dataset, method=name
                )
            )
    return traces


def export_table7(
    path: Path, traces: Sequence[MethodTrace], config: PactConfig
) -> None:
    metrics = [metrics_from_trace(t, config) for t in traces]
    write_multi_metric_table(path, metrics, TABLE7_METRICS)


def split_then_ablate(
    base: PactConfig,
    trace: ArrivalTrace,
    *,
    guard: SplitGuard,
    seeds: Sequence[int] = (0,),
) -> list[MethodTrace]:
    """The test split is read once, after the knockout list is frozen."""

    split = chronological_series_split(trace.arrival_rates, trace.timestamps)
    arrivals = list(guard.read_test(split))
    return run_ablation_matrix(base, arrivals, seeds=seeds, dataset=trace.name)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT ablation runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config, cold_start_path=args.cold_start)
    burst = generate_burst(
        "step",
        n_ticks=args.ticks,
        dt=config.telemetry.dt,
        rise_ticks=4,
        base=20.0,
        peak=60.0,
        name="d3_step_ablation",
    )
    traces = run_ablation_matrix(
        config,
        burst.arrival_rates,
        seeds=tuple(range(args.seeds)),
        dataset=burst.name,
    )
    export_table7(args.results / "table7_ablation.csv", traces, config)


if __name__ == "__main__":
    main()
