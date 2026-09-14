"""Write the CSVs ``make_figures.py`` consumes. No invented values."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from pact.config import PactConfig
from pact.eval.runner import MethodTrace, RunMetrics, metrics_from_trace
from pact.forecast.train import (
    compute_horizon_error,
    write_horizon_error_csv,
    write_training_csv,
)

FORECAST_COLUMNS = ("ts", "actual", "pred_h1", "pred_h12")
HORIZON_ERROR_COLUMNS = ("horizon", "mae", "rmse")
REPLICAS_COLUMNS = ("ts", "required", "pact", "reactive", "arima", "lstm")
LATENCY_COLUMNS = ("ts", "pact", "reactive", "arima", "lstm", "slo")
DECISIONS_COLUMNS = ("method", "accuracy", "precision", "recall", "f1")
TRADEOFF_COLUMNS = ("method", "sla_violation", "over_provision")
ACTIONS_COLUMNS = ("ts", "pact", "reactive", "arima", "lstm")
UTILISATION_COLUMNS = ("method", "utilisation")
TRAINING_COLUMNS = ("epoch", "train_loss", "val_loss")
TABLE_COLUMNS = ("method", "metric", "mean_pm_std", "n_runs")
WIDE_METHODS = ("pact", "reactive", "arima", "lstm")


def format_mean_std(values: Sequence[float]) -> tuple[str, int]:
    series = [float(x) for x in values]
    n = len(series)
    if n < 1:
        raise ValueError("cannot summarise an empty series")
    mean = sum(series) / n
    if n == 1:
        std = 0.0
    else:
        std = (sum((x - mean) ** 2 for x in series) / (n - 1)) ** 0.5
    return f"{mean:.6g} ± {std:.6g}", n


def _write(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write {path} with no computed rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def _by_method(
    traces: Sequence[MethodTrace], seed: int | None = None
) -> dict[str, MethodTrace]:
    chosen: dict[str, MethodTrace] = {}
    for trace in traces:
        if seed is not None and trace.seed != seed:
            continue
        if trace.method not in chosen:
            chosen[trace.method] = trace
    return chosen


def write_forecast_csv(path: Path, traces: Sequence[MethodTrace]) -> None:
    pact = _by_method(traces).get("pact")
    if pact is None:
        raise ValueError("forecast.csv requires a computed pact trace")
    rows = [
        [ts, actual, h1, h12]
        for ts, actual, h1, h12 in zip(
            pact.ts, pact.actual_u, pact.pred_h1, pact.pred_h12, strict=True
        )
    ]
    _write(path, FORECAST_COLUMNS, rows)


def write_replicas_csv(path: Path, traces: Sequence[MethodTrace]) -> None:
    grouped = _require_wide(traces)
    pact = grouped["pact"]
    rows = []
    for i, ts in enumerate(pact.ts):
        rows.append(
            [
                ts,
                pact.n_required[i],
                grouped["pact"].n[i],
                grouped["reactive"].n[i],
                grouped["arima"].n[i],
                grouped["lstm"].n[i],
            ]
        )
    _write(path, REPLICAS_COLUMNS, rows)


def write_latency_csv(
    path: Path, traces: Sequence[MethodTrace], slo_ms: float
) -> None:
    grouped = _require_wide(traces)
    pact = grouped["pact"]
    rows = []
    for i, ts in enumerate(pact.ts):
        rows.append(
            [
                ts,
                grouped["pact"].latency_ms[i],
                grouped["reactive"].latency_ms[i],
                grouped["arima"].latency_ms[i],
                grouped["lstm"].latency_ms[i],
                slo_ms,
            ]
        )
    _write(path, LATENCY_COLUMNS, rows)


def write_actions_csv(path: Path, traces: Sequence[MethodTrace]) -> None:
    grouped = _require_wide(traces)
    pact = grouped["pact"]
    rows = []
    for i, ts in enumerate(pact.ts):
        rows.append(
            [
                ts,
                grouped["pact"].action[i],
                grouped["reactive"].action[i],
                grouped["arima"].action[i],
                grouped["lstm"].action[i],
            ]
        )
    _write(path, ACTIONS_COLUMNS, rows)


def write_utilisation_csv(path: Path, traces: Sequence[MethodTrace]) -> None:
    rows: list[list[object]] = []
    for trace in traces:
        for u in trace.utilisation:
            rows.append([trace.method, u])
    _write(path, UTILISATION_COLUMNS, rows)


def write_decisions_csv(path: Path, metrics: Sequence[RunMetrics]) -> None:
    aggregated = _mean_metrics(metrics)
    rows = [
        [m.method, m.accuracy, m.precision, m.recall, m.f1]
        for m in aggregated
    ]
    _write(path, DECISIONS_COLUMNS, rows)


def write_tradeoff_csv(path: Path, metrics: Sequence[RunMetrics]) -> None:
    aggregated = _mean_metrics(metrics)
    rows = [[m.method, m.sla_violation, m.over_provision] for m in aggregated]
    _write(path, TRADEOFF_COLUMNS, rows)


def write_table_csv(
    path: Path, metrics: Sequence[RunMetrics], *, metric_name: str
) -> None:
    grouped: dict[str, list[float]] = {}
    attr = {
        "sla_violation": "sla_violation",
        "over_provision": "over_provision",
        "recall": "recall",
        "action_count": "action_count",
    }[metric_name]
    for row in metrics:
        grouped.setdefault(row.method, []).append(float(getattr(row, attr)))
    out: list[list[object]] = []
    for method, values in grouped.items():
        mean_std, n_runs = format_mean_std(values)
        out.append([method, metric_name, mean_std, n_runs])
    _write(path, TABLE_COLUMNS, out)


def write_horizon_error_from_arrays(
    path: Path, actual: Sequence[Sequence[float]], pred: Sequence[Sequence[float]]
) -> None:
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
        p = p[:, :, None]
    rows = compute_horizon_error(a, p)
    write_horizon_error_csv(path, rows)


def export_run(
    results_dir: Path,
    traces: Sequence[MethodTrace],
    config: PactConfig,
    *,
    training_rows: list[tuple[int, float, float]] | None = None,
) -> None:
    """Write every 9.3 figure CSV that can be filled from ``traces``."""

    metrics = [metrics_from_trace(t, config) for t in traces]
    write_forecast_csv(results_dir / "forecast.csv", traces)
    write_replicas_csv(results_dir / "replicas.csv", traces)
    write_latency_csv(results_dir / "latency.csv", traces, config.capacity.slo_ms)
    write_actions_csv(results_dir / "actions.csv", traces)
    write_utilisation_csv(results_dir / "utilisation.csv", traces)
    write_decisions_csv(results_dir / "decisions.csv", metrics)
    write_tradeoff_csv(results_dir / "tradeoff.csv", metrics)
    write_table_csv(results_dir / "table4_proposed.csv", metrics, metric_name="recall")
    write_table_csv(
        results_dir / "table6_baselines.csv", metrics, metric_name="sla_violation"
    )
    if training_rows:
        write_training_csv(results_dir / "training.csv", training_rows)


def _require_wide(traces: Sequence[MethodTrace]) -> dict[str, MethodTrace]:
    grouped = _by_method(traces, seed=None)
    missing = [name for name in WIDE_METHODS if name not in grouped]
    if missing:
        raise ValueError(
            f"wide CSVs need computed traces for {WIDE_METHODS}, missing {missing}"
        )
    length = len(grouped["pact"].ts)
    for name in WIDE_METHODS:
        if len(grouped[name].ts) != length:
            raise ValueError("method traces are not aligned")
    return grouped


def _mean_metrics(metrics: Sequence[RunMetrics]) -> list[RunMetrics]:
    grouped: dict[str, list[RunMetrics]] = {}
    for row in metrics:
        grouped.setdefault(row.method, []).append(row)

    def _avg(rows: list[RunMetrics], field: str) -> float:
        return sum(float(getattr(r, field)) for r in rows) / len(rows)

    out: list[RunMetrics] = []
    for method, rows in grouped.items():
        out.append(
            RunMetrics(
                method=method,
                seed=-1,
                dataset=rows[0].dataset,
                sla_violation=_avg(rows, "sla_violation"),
                over_provision=_avg(rows, "over_provision"),
                action_count=int(round(_avg(rows, "action_count"))),
                accuracy=_avg(rows, "accuracy"),
                precision=_avg(rows, "precision"),
                recall=_avg(rows, "recall"),
                f1=_avg(rows, "f1"),
                cost_per_1k=_avg(rows, "cost_per_1k"),
                mean_utilisation=_avg(rows, "mean_utilisation"),
            )
        )
    return out
