"""Tune each baseline on the validation split with a shared trial budget."""

from __future__ import annotations

import json
from pathlib import Path

from pact.baselines.arima_ctl import ArimaController, select_arima_order
from pact.baselines.reactive import HpaConfig, ReactiveHPA
from pact.baselines.static import peak_replicas
from pact.config import PactConfig
from pact.eval.datasets import ArrivalTrace
from pact.eval.runner import MethodTrace, metrics_from_trace, run_scaler
from pact.eval.splits import chronological_series_split


def tune_baselines(
    config: PactConfig,
    trace: ArrivalTrace,
    *,
    budget: int,
    results_path: Path,
) -> dict[str, object]:
    """Search on validation only. Writes ``baseline_tuning.json`` from scores."""

    if budget < 1:
        raise ValueError("budget must be >= 1")
    split = chronological_series_split(trace.arrival_rates, trace.timestamps)
    val = list(split.val)
    train = list(split.train)
    chosen: dict[str, object] = {"budget": budget, "dataset": trace.name}

    hpa_grid = [
        HpaConfig(tolerance=0.05, scale_down_window_s=60.0),
        HpaConfig(tolerance=0.10, scale_down_window_s=60.0),
        HpaConfig(tolerance=0.10, scale_down_window_s=300.0),
        HpaConfig(tolerance=0.15, scale_down_window_s=300.0),
        HpaConfig(tolerance=0.10, scale_down_window_s=0.0),
        HpaConfig(tolerance=0.20, scale_down_window_s=120.0),
    ][:budget]
    best_hpa = hpa_grid[0]
    best_hpa_score = float("inf")
    for hpa in hpa_grid:
        scaler = ReactiveHPA(config, hpa=hpa)
        run = run_scaler(config, val, scaler, seed=0, dataset=trace.name)
        score = _score(run, config)
        if score < best_hpa_score:
            best_hpa_score = score
            best_hpa = hpa
    chosen["reactive"] = {
        "tolerance": best_hpa.tolerance,
        "scale_down_window_s": best_hpa.scale_down_window_s,
        "val_score": best_hpa_score,
    }

    order = select_arima_order(train, val, budget=budget)
    arima_run = run_scaler(
        config,
        val,
        ArimaController(config, order=order),
        seed=0,
        dataset=trace.name,
    )
    chosen["arima"] = {
        "order": list(order),
        "val_score": _score(arima_run, config),
    }

    chosen["static"] = {
        "n_fixed": peak_replicas(val, config),
        "sized_to": "validation_peak",
    }
    chosen["lstm"] = {
        "hidden_size": 16,
        "kappa": 1.0,
        "loss": "symmetric_huber",
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(chosen, indent=2) + "\n")
    return chosen


def _score(trace: MethodTrace, config: PactConfig) -> float:
    metrics = metrics_from_trace(trace, config)
    return metrics.sla_violation + metrics.over_provision
