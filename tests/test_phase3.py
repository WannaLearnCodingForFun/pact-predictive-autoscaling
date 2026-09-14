"""Phase 3 acceptance tests (CURSOR_SPEC.md §3)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from pact.config import (
    CapacityConfig,
    ControlConfig,
    DriftConfig,
    ForecastConfig,
    PactConfig,
    SimulatorConfig,
    TelemetryConfig,
)
from pact.forecast.infer import ForecastBudgetError, OnnxForecaster
from pact.forecast.losses import AsymmetricHorizonHuber
from pact.forecast.tcn import DilatedTCN, receptive_field
from pact.forecast.train import (
    ChronologicalSplitError,
    assert_chronological_splits,
    chronological_window_splits,
    compute_horizon_error,
    export_onnx_and_verify,
    train_forecaster,
    write_horizon_error_csv,
)

from tests.helpers import make_config


def _small_tcn(**overrides: object) -> DilatedTCN:
    kwargs: dict[str, object] = {
        "in_channels": 3,
        "window": 12,
        "horizon": 4,
        "kernel_size": 3,
        "depth": 2,
        "channels": 8,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return DilatedTCN(**kwargs)  # type: ignore[arg-type]


def _small_pact_config() -> PactConfig:
    return PactConfig(
        telemetry=TelemetryConfig(dt=5.0, window=12),
        forecast=ForecastConfig(
            horizon=3,
            kernel_size=3,
            depth=2,
            channels=8,
            dropout=0.0,
            batch_size=16,
            lr=3e-3,
        ),
        capacity=CapacityConfig(),
        control=ControlConfig(tau_c_s=10.0),
        drift=DriftConfig(),
        simulator=SimulatorConfig(),
    )


def test_receptive_field_covers_window() -> None:
    # Eq. 9: R = 1 + 2(K-1)(2^D - 1). K=3, D=4 → 1 + 4*15 = 61 >= L=48.
    assert receptive_field(3, 4) == 61
    DilatedTCN.from_config(make_config(), in_channels=8)
    with pytest.raises(ValueError, match="receptive field"):
        DilatedTCN(
            in_channels=2,
            window=16,
            horizon=2,
            kernel_size=3,
            depth=1,
            channels=4,
            dropout=0.0,
        )


def test_causality_future_sample_does_not_change_past() -> None:
    torch.manual_seed(0)
    model = _small_tcn().eval()
    x = torch.randn(2, 3, 12)
    t_future = 8
    x_shifted = x.clone()
    x_shifted[:, :, t_future] += 4.0
    with torch.no_grad():
        y = model.features(x)
        y_shifted = model.features(x_shifted)
    assert torch.allclose(
        y[:, :, :t_future], y_shifted[:, :, :t_future], atol=1e-5, rtol=0.0
    )
    assert not torch.allclose(
        y[:, :, t_future:], y_shifted[:, :, t_future:], atol=1e-5, rtol=0.0
    )


def test_forward_emits_all_horizon_steps_in_one_pass() -> None:
    model = _small_tcn(horizon=5).eval()
    x = torch.zeros(2, 3, 12)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5, 2)


def test_asymmetric_loss_penalizes_underprediction() -> None:
    loss_fn = AsymmetricHorizonHuber(kappa=2.5, beta=0.15, huber_delta=1.0)
    pred = torch.zeros(3, 6, 2)
    under = torch.ones(3, 6, 2)  # e = ŷ − y = −1
    over = -torch.ones(3, 6, 2)  # e = +1
    under_loss = float(loss_fn(pred, under).item())
    over_loss = float(loss_fn(pred, over).item())
    assert under_loss > over_loss


def test_chronological_split_asserts_ordering() -> None:
    n = 200
    t = np.arange(n, dtype=np.float64) * 5.0
    features = np.zeros((n, 2), dtype=np.float32)
    targets = np.zeros((n, 2), dtype=np.float32)
    train, val, test = chronological_window_splits(
        features, targets, t, window=12, horizon=3
    )
    assert_chronological_splits(train.timestamps, val.timestamps, test.timestamps)
    assert float(train.timestamps.max()) < float(val.timestamps.min())
    assert float(val.timestamps.min()) < float(test.timestamps.min())

    shuffled = np.random.default_rng(0).permutation(t)
    with pytest.raises(ChronologicalSplitError, match="strictly increasing"):
        chronological_window_splits(
            features, targets, shuffled, window=12, horizon=3
        )
    with pytest.raises(ChronologicalSplitError):
        assert_chronological_splits(
            np.array([1.0, 5.0]),
            np.array([4.0, 6.0]),
            np.array([7.0, 8.0]),
        )


def test_horizon_error_csv_increases_with_h(tmp_path: Path) -> None:
    actual = np.zeros((5, 4, 2), dtype=np.float32)
    pred = np.zeros_like(actual)
    for h in range(4):
        pred[:, h, :] = float(h + 1)
    rows = compute_horizon_error(actual, pred)
    maes = [row[1] for row in rows]
    rmses = [row[2] for row in rows]
    assert maes == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert rmses == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert maes == sorted(maes)
    assert rmses == sorted(rmses)
    path = tmp_path / "horizon_error.csv"
    write_horizon_error_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert [row["horizon"] for row in table] == ["1", "2", "3", "4"]
    assert list(table[0].keys()) == ["horizon", "mae", "rmse"]


def test_onnx_matches_pytorch(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = _small_tcn().eval()
    path = tmp_path / "tcn.onnx"
    export_onnx_and_verify(model, path)
    runtime = OnnxForecaster(path, dt=5.0)
    window = np.random.default_rng(0).standard_normal((12, 3)).astype(np.float32)
    onnx_out = runtime.predict(window)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(window.T[None, ...])).numpy()[0]
    assert np.max(np.abs(torch_out - onnx_out)) < 1e-5
    assert runtime.providers[0] == "CPUExecutionProvider"


def test_infer_logs_and_raises_when_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _small_tcn().eval()
    path = tmp_path / "tcn.onnx"
    export_onnx_and_verify(model, path)
    runtime = OnnxForecaster(path, dt=0.01)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(
        "pact.forecast.infer.time.perf_counter", lambda: next(ticks)
    )
    with pytest.raises(ForecastBudgetError):
        runtime.predict(np.zeros((12, 3), dtype=np.float32))
    assert runtime.over_budget_count == 1


def test_train_writes_training_and_horizon_csvs(tmp_path: Path) -> None:
    cfg = _small_pact_config()
    n = 240
    t = np.arange(n, dtype=np.float64)
    u = 0.5 + 0.1 * np.sin(2 * np.pi * t / 40.0)
    r = 0.4 + 0.05 * np.cos(2 * np.pi * t / 25.0)
    features = np.stack([u, r], axis=1).astype(np.float32)
    targets = features.copy()
    timestamps = t * cfg.telemetry.dt
    result = train_forecaster(
        features,
        targets,
        timestamps,
        cfg,
        results_dir=tmp_path,
        max_epochs=2,
        patience=2,
        seed=0,
        in_channels=2,
    )
    assert result.n_params > 0
    assert result.infer_latency_s >= 0.0
    with result.training_log_path.open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    assert list(train_rows[0].keys()) == ["epoch", "train_loss", "val_loss"]
    assert len(train_rows) >= 1
    with result.horizon_error_path.open(newline="", encoding="utf-8") as handle:
        err_rows = list(csv.DictReader(handle))
    assert list(err_rows[0].keys()) == ["horizon", "mae", "rmse"]
    assert [row["horizon"] for row in err_rows] == ["1", "2", "3"]
    assert result.onnx_path.is_file()
