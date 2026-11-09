"""Phase 9.5: val-only search, H is fixed, test split once after freeze."""

from __future__ import annotations

import csv
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pact.eval.datasets import generate_burst
from pact.eval.export import write_multi_metric_table
from pact.eval.hpsearch import (
    assert_search_space_excludes_horizon,
    config_with_architecture,
    evaluate_frozen_on_test,
    export_table8,
    score_architecture_on_val,
    search_architecture,
    split_lengths,
    sweep_control,
    write_sweep_csv,
)
from pact.eval.splits import (
    SplitGuard,
    TestSplitReusedError,
    chronological_series_split,
)
from pact.telemetry.features import FEATURE_DIM

from tests.helpers import make_config


def test_horizon_is_not_a_search_key() -> None:
    with pytest.raises(ValueError, match="Eq. 19"):
        assert_search_space_excludes_horizon({"horizon": (8, 12)})
    with pytest.raises(ValueError, match="Eq. 19"):
        search_architecture(
            make_config(tau_c_s=10.0),
            grid={"horizon": (8, 12), "window": (16,)},
            n_trials=1,
            rng=random.Random(0),
            score_fn=lambda _c, _p: 0.0,
        )


def test_architecture_search_uses_injected_val_scores_only() -> None:
    cfg = make_config(tau_c_s=10.0)
    seen: list[dict[str, object]] = []

    def score_fn(_cfg: object, params: dict[str, object]) -> float:
        seen.append(params)
        channels = params["channels"]
        assert isinstance(channels, int)
        return float(channels)

    best = search_architecture(
        cfg,
        grid={
            "window": (8,),
            "kernel_size": (3,),
            "depth": (3,),
            "channels": (4, 8),
            "dropout": (0.0,),
            "lr": (0.01,),
            "batch_size": (4,),
        },
        n_trials=4,
        rng=random.Random(0),
        score_fn=score_fn,
    )
    assert seen
    assert best.val_score == min(
        float(p["channels"]) for p in seen if isinstance(p["channels"], int)
    )
    frozen = config_with_architecture(cfg, best.params)
    assert frozen.forecast.horizon == cfg.forecast.horizon


def test_score_architecture_on_val_does_not_need_test_rows() -> None:
    cfg = make_config(tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    cfg = replace(
        cfg,
        forecast=replace(cfg.forecast, horizon=4, kernel_size=3, depth=3, channels=8),
    )
    n = 80
    features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    targets = np.zeros((n, 2), dtype=np.float32)
    timestamps = np.arange(n, dtype=np.float64) * 5.0
    n_train, n_val, n_test = split_lengths(n)
    assert n_train + n_val + n_test == n
    score = score_architecture_on_val(
        cfg,
        {
            "window": 8,
            "kernel_size": 3,
            "depth": 3,
            "channels": 8,
            "dropout": 0.0,
            "lr": 0.01,
            "batch_size": 4,
        },
        features=features,
        targets=targets,
        timestamps=timestamps,
        n_train=n_train,
        n_val=n_val,
        max_epochs=1,
        seed=0,
    )
    assert isinstance(score, float)
    assert score >= 0.0


def test_control_sweep_on_val_then_test_once(tmp_path: Path) -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    burst = generate_burst(
        "step", n_ticks=40, dt=5.0, rise_ticks=2, base=15.0, peak=50.0
    )
    split = chronological_series_split(burst.arrival_rates, burst.timestamps)
    points = sweep_control(
        cfg,
        list(split.val),
        sweeps={"s_churn": (0.0, 0.60)},
        dataset="val",
    )
    assert [p.value for p in points] == [0.0, 0.60]
    path = tmp_path / "control_sweep.csv"
    write_sweep_csv(path, points)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert {row["param"] for row in rows} == {"s_churn"}

    guard = SplitGuard()
    trace = evaluate_frozen_on_test(cfg, burst, guard, seed=0)
    table = tmp_path / "table8_hyperparams.csv"
    export_table8(table, [trace], cfg)
    table_rows = list(csv.DictReader(table.open()))
    assert table_rows
    assert all("±" in row["mean_pm_std"] for row in table_rows)
    assert all(int(row["n_runs"]) == 1 for row in table_rows)
    with pytest.raises(TestSplitReusedError):
        evaluate_frozen_on_test(cfg, burst, guard, seed=0)


def test_table8_refuses_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no computed"):
        write_multi_metric_table(
            tmp_path / "table8_hyperparams.csv", [], ("sla_violation",)
        )


def test_sweep_csv_refuses_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no computed"):
        write_sweep_csv(tmp_path / "control_sweep.csv", [])
