"""Phase 9.1–9.3: datasets, SplitGuard, runner, export schema."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
from pact.eval.datasets import (
    generate_burst,
    generate_synthetic_suite,
    parse_bitbrains_faststorage,
    parse_fifa_requests,
    write_dataset_summary,
)
from pact.eval.export import (
    ACTIONS_COLUMNS,
    DECISIONS_COLUMNS,
    FORECAST_COLUMNS,
    HORIZON_ERROR_COLUMNS,
    LATENCY_COLUMNS,
    REPLICAS_COLUMNS,
    TABLE_COLUMNS,
    TRADEOFF_COLUMNS,
    TRAINING_COLUMNS,
    UTILISATION_COLUMNS,
    export_run,
    write_replicas_csv,
    write_table_csv,
)
from pact.eval.runner import run_matrix
from pact.eval.splits import (
    SplitGuard,
    TestSplitReusedError,
    chronological_series_split,
)
from pact.forecast.train import write_horizon_error_csv, write_training_csv

from tests.helpers import make_config


def test_fifa_parser_buckets_requests_per_dt() -> None:
    # timestamps 1, 6, 11 → three 5 s bins, one request each → rate 0.2 req/s
    trace = parse_fifa_requests(["1", "6", "11"], dt=5.0, subset_s=20.0)
    assert trace.interpolated is False
    assert trace.n_samples == 3
    assert trace.arrival_rates == pytest.approx((0.2, 0.2, 0.2))
    assert trace.duration_s == pytest.approx(15.0)


def test_bitbrains_resample_is_interpolated_not_measured() -> None:
    rows = [
        ["Timestamp", "CPUUsage [%]"],
        ["0", "0"],
        ["300", "100"],
    ]
    trace = parse_bitbrains_faststorage(rows, native_dt_s=300.0, target_dt_s=5.0)
    assert trace.interpolated is True
    assert "interpolat" in trace.notes.lower()
    assert trace.n_samples == 61  # 0..300 inclusive at 5 s
    # midpoint t=150: 50% CPU → rate 0.5
    mid = trace.timestamps.index(150.0)
    assert trace.arrival_rates[mid] == pytest.approx(0.5)


def test_synthetic_rise_time_is_a_parameter() -> None:
    slow = generate_burst(
        "ramp", n_ticks=20, dt=5.0, rise_ticks=8, base=10.0, peak=50.0
    )
    fast = generate_burst(
        "ramp", n_ticks=20, dt=5.0, rise_ticks=2, base=10.0, peak=50.0
    )
    mid = 20 // 4
    assert slow.arrival_rates[mid] == pytest.approx(10.0)
    assert slow.arrival_rates[mid + 4] < fast.arrival_rates[mid + 2]
    assert slow.arrival_rates[mid + 8] == pytest.approx(50.0)


def test_dataset_summary_matches_measured_counts(tmp_path: Path) -> None:
    traces = generate_synthetic_suite(dt=5.0, n_ticks=12, rise_ticks=(2,))
    path = tmp_path / "dataset_summary.csv"
    write_dataset_summary(path, traces)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(traces)
    by_name = {row["dataset"]: row for row in rows}
    for trace in traces:
        row = by_name[trace.name]
        assert int(row["n_samples"]) == trace.n_samples
        assert float(row["duration_s"]) == pytest.approx(trace.duration_s)
        assert row["interpolated"] == "false"


def test_dataset_summary_refuses_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no traces"):
        write_dataset_summary(tmp_path / "dataset_summary.csv", [])


def test_split_guard_raises_on_second_test_read() -> None:
    rates = [float(i) for i in range(40)]
    ts = [i * 5.0 for i in range(40)]
    split = chronological_series_split(rates, ts)
    assert float(max(split.train_ts)) < float(min(split.val_ts))
    assert float(max(split.val_ts)) < float(min(split.test_ts))
    guard = SplitGuard()
    first = list(guard.read_test(split))
    assert first == list(split.test)
    with pytest.raises(TestSplitReusedError):
        guard.read_test(split)


def test_export_schema_and_computed_values(tmp_path: Path) -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    burst = generate_burst(
        "step", n_ticks=40, dt=5.0, rise_ticks=2, base=20.0, peak=55.0
    )
    traces = run_matrix(
        cfg,
        burst,
        methods=("pact", "reactive", "arima", "lstm"),
        seeds=(0,),
        evaluate_on="all",
    )
    export_run(tmp_path, traces, cfg)
    headers = {
        "forecast.csv": FORECAST_COLUMNS,
        "replicas.csv": REPLICAS_COLUMNS,
        "latency.csv": LATENCY_COLUMNS,
        "actions.csv": ACTIONS_COLUMNS,
        "utilisation.csv": UTILISATION_COLUMNS,
        "decisions.csv": DECISIONS_COLUMNS,
        "tradeoff.csv": TRADEOFF_COLUMNS,
        "table4_proposed.csv": TABLE_COLUMNS,
        "table6_baselines.csv": TABLE_COLUMNS,
    }
    for name, expected in headers.items():
        with (tmp_path / name).open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader))
            rows = list(reader)
        assert header == expected
        assert rows, name
    replicas = list(csv.DictReader((tmp_path / "replicas.csv").open()))
    pact = next(t for t in traces if t.method == "pact")
    assert float(replicas[0]["pact"]) == pytest.approx(pact.n[0])
    assert float(replicas[0]["required"]) == pytest.approx(pact.n_required[0])
    table = list(csv.DictReader((tmp_path / "table6_baselines.csv").open()))
    assert all("±" in row["mean_pm_std"] for row in table)
    assert all(int(row["n_runs"]) == 1 for row in table)


def test_export_refuses_missing_methods(tmp_path: Path) -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    burst = generate_burst(
        "impulse", n_ticks=24, dt=5.0, rise_ticks=1, base=10.0, peak=40.0
    )
    traces = run_matrix(
        cfg, burst, methods=("pact",), seeds=(0,), evaluate_on="all"
    )
    with pytest.raises(ValueError, match="missing"):
        write_replicas_csv(tmp_path / "replicas.csv", traces)


def test_horizon_and_training_csv_headers(tmp_path: Path) -> None:
    write_horizon_error_csv(tmp_path / "horizon_error.csv", [(1, 0.2, 0.3)])
    write_training_csv(tmp_path / "training.csv", [(1, 0.5, 0.6)])
    with (tmp_path / "horizon_error.csv").open(newline="", encoding="utf-8") as handle:
        assert tuple(next(csv.reader(handle))) == HORIZON_ERROR_COLUMNS
    with (tmp_path / "training.csv").open(newline="", encoding="utf-8") as handle:
        assert tuple(next(csv.reader(handle))) == TRAINING_COLUMNS


def test_write_table_requires_computed_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no computed"):
        write_table_csv(tmp_path / "table5_vs_existing.csv", [], metric_name="recall")
