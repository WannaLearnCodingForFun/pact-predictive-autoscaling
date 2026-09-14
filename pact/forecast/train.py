"""Training loop, chronological splits, ONNX export, horizon-error curve."""

from __future__ import annotations

import csv
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from pact.config import PactConfig
from pact.eval.metrics import mae, rmse
from pact.forecast.infer import OnnxForecaster
from pact.forecast.losses import AsymmetricHorizonHuber
from pact.forecast.tcn import DilatedTCN

logger = logging.getLogger(__name__)

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


class ChronologicalSplitError(ValueError):
    """Raised when a split would leak the future into the past."""


@dataclass(frozen=True)
class WindowSet:
    X: NDArray[np.float32]  # (N, L, d)
    Y: NDArray[np.float32]  # (N, H, 2)
    timestamps: NDArray[np.float64]  # (N,) forecast-origin times


@dataclass
class TrainResult:
    model: DilatedTCN
    onnx_path: Path
    n_params: int
    infer_latency_s: float
    training_log_path: Path
    horizon_error_path: Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_chronological_splits(
    train_ts: NDArray[np.floating],
    val_ts: NDArray[np.floating],
    test_ts: NDArray[np.floating],
) -> None:
    """``max(train) < min(val) < min(test)``. Never skip this check."""

    if train_ts.size == 0 or val_ts.size == 0 or test_ts.size == 0:
        raise ChronologicalSplitError("each split must contain at least one window")
    train_max = float(np.max(train_ts))
    val_min = float(np.min(val_ts))
    val_max = float(np.max(val_ts))
    test_min = float(np.min(test_ts))
    if not (train_max < val_min < test_min):
        raise ChronologicalSplitError(
            f"max(train)={train_max} < min(val)={val_min} < min(test)={test_min} "
            "failed — split is not chronological"
        )
    if val_max >= test_min:
        raise ChronologicalSplitError(
            f"max(val)={val_max} >= min(test)={test_min} — splits overlap"
        )


def _require_strictly_increasing(timestamps: NDArray[np.floating]) -> None:
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamps must be a 1-D series of length >= 2")
    if np.any(timestamps[1:] <= timestamps[:-1]):
        raise ChronologicalSplitError(
            "timestamps must be strictly increasing; shuffling before splitting "
            "leaks the future into training"
        )


def chronological_window_splits(
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    timestamps: NDArray[np.floating],
    *,
    window: int,
    horizon: int,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> tuple[WindowSet, WindowSet, WindowSet]:
    """70/15/15 index split on the raw series, then window *inside* each split.

    Does not shuffle. Windows never cross a split boundary.
    """

    feats = np.asarray(features, dtype=np.float32)
    targs = np.asarray(targets, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"features must be (T, d), got {feats.shape}")
    if targs.shape != (feats.shape[0], 2):
        raise ValueError(f"targets must be (T, 2), got {targs.shape}")
    if ts.shape != (feats.shape[0],):
        raise ValueError("timestamps must align with the series length")
    _require_strictly_increasing(ts)

    n = feats.shape[0]
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ChronologicalSplitError("series too short for 70/15/15")

    train = _windows(feats, targs, ts, 0, n_train, window, horizon)
    val = _windows(feats, targs, ts, n_train, n_train + n_val, window, horizon)
    test = _windows(feats, targs, ts, n_train + n_val, n, window, horizon)
    assert_chronological_splits(train.timestamps, val.timestamps, test.timestamps)
    return train, val, test


def _windows(
    features: NDArray[np.float32],
    targets: NDArray[np.float32],
    timestamps: NDArray[np.float64],
    start: int,
    end: int,
    window: int,
    horizon: int,
) -> WindowSet:
    # origin t uses features[t-window+1 : t+1] and targets[t+1 : t+1+horizon]
    first = start + window - 1
    last = end - horizon - 1
    if last < first:
        raise ChronologicalSplitError(
            f"split [{start}, {end}) cannot form a window with L={window}, H={horizon}"
        )
    xs: list[NDArray[np.float32]] = []
    ys: list[NDArray[np.float32]] = []
    t_origins: list[float] = []
    for t in range(first, last + 1):
        xs.append(features[t - window + 1 : t + 1])
        ys.append(targets[t + 1 : t + 1 + horizon])
        t_origins.append(float(timestamps[t]))
    return WindowSet(
        X=np.stack(xs, axis=0),
        Y=np.stack(ys, axis=0),
        timestamps=np.asarray(t_origins, dtype=np.float64),
    )


def write_training_csv(path: Path, rows: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(rows)


def compute_horizon_error(
    actual: NDArray[np.floating],
    pred: NDArray[np.floating],
) -> list[tuple[int, float, float]]:
    """Per-horizon MAE and RMSE. ``actual``/``pred`` are ``(N, H, 2)``."""

    if actual.shape != pred.shape or actual.ndim != 3:
        raise ValueError(
            f"expected matching (N, H, 2), got {actual.shape} vs {pred.shape}"
        )
    horizon = actual.shape[1]
    rows: list[tuple[int, float, float]] = []
    for h in range(horizon):
        a = actual[:, h, :].reshape(-1).tolist()
        p = pred[:, h, :].reshape(-1).tolist()
        rows.append((h + 1, mae(a, p), rmse(a, p)))
    return rows


def write_horizon_error_csv(
    path: Path, rows: list[tuple[int, float, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["horizon", "mae", "rmse"])
        writer.writerows(rows)


def export_onnx_and_verify(model: DilatedTCN, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, model.in_channels, model.window)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["window"],
        output_names=["forecast"],
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    runtime = OnnxForecaster(path, dt=1e9)
    window = np.zeros((model.window, model.in_channels), dtype=np.float32)
    onnx_out = runtime.predict(window)
    delta = float(np.max(np.abs(torch_out[0] - onnx_out)))
    if delta > 1e-5:
        raise RuntimeError(f"ONNX output differs from PyTorch by {delta} (> 1e-5)")


def train_forecaster(
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    timestamps: NDArray[np.floating],
    config: PactConfig,
    *,
    results_dir: Path,
    max_epochs: int = 100,
    patience: int = 10,
    seed: int = 0,
    in_channels: int | None = None,
) -> TrainResult:
    seed_everything(seed)
    channels = features.shape[1] if in_channels is None else in_channels
    if features.shape[1] != channels:
        raise ValueError("in_channels does not match features")

    train, val, test = chronological_window_splits(
        features,
        targets,
        timestamps,
        window=config.telemetry.window,
        horizon=config.forecast.horizon,
    )
    model = DilatedTCN(
        in_channels=channels,
        window=config.telemetry.window,
        horizon=config.forecast.horizon,
        kernel_size=config.forecast.kernel_size,
        depth=config.forecast.depth,
        channels=config.forecast.channels,
        dropout=config.forecast.dropout,
    )
    loss_fn = AsymmetricHorizonHuber.from_config(config.forecast)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.forecast.lr)

    train_loader = _loader(train, config.forecast.batch_size, shuffle=True, seed=seed)
    val_loader = _loader(val, config.forecast.batch_size, shuffle=False, seed=seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0
    log_rows: list[tuple[int, float, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = _run_epoch(model, val_loader, loss_fn, None)
        log_rows.append((epoch, train_loss, val_loss))
        logger.info("epoch %s train=%.6f val=%.6f", epoch, train_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    training_log_path = results_dir / "training.csv"
    write_training_csv(training_log_path, log_rows)

    onnx_path = results_dir / "tcn.onnx"
    export_onnx_and_verify(model, onnx_path)

    test_pred = _predict_set(model, test)
    horizon_rows = compute_horizon_error(test.Y, test_pred)
    horizon_error_path = results_dir / "horizon_error.csv"
    write_horizon_error_csv(horizon_error_path, horizon_rows)

    dummy = np.zeros((model.window, model.in_channels), dtype=np.float32)
    runtime = OnnxForecaster(onnx_path, dt=config.telemetry.dt)
    infer_latency_s = _measure_latency(lambda: runtime.predict(dummy))
    logger.info(
        "n_params=%s infer_latency_s=%.6f", model.parameter_count(), infer_latency_s
    )
    return TrainResult(
        model=model,
        onnx_path=onnx_path,
        n_params=model.parameter_count(),
        infer_latency_s=infer_latency_s,
        training_log_path=training_log_path,
        horizon_error_path=horizon_error_path,
    )


def _loader(
    data: WindowSet, batch_size: int, *, shuffle: bool, seed: int
) -> DataLoader[Any]:
    x = torch.from_numpy(np.transpose(data.X, (0, 2, 1)).copy())
    y = torch.from_numpy(data.Y.copy())
    dataset = TensorDataset(x, y)
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
        if not isinstance(xb, Tensor) or not isinstance(yb, Tensor):
            raise TypeError("DataLoader must yield tensors")
        if training:
            assert optimizer is not None
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if training:
                if optimizer is None:
                    raise RuntimeError("optimizer required when training")
                loss.backward()
                optimizer.step()
        batch = xb.size(0)
        total += float(loss.item()) * batch
        n += batch
    return total / max(n, 1)


def _predict_set(model: DilatedTCN, data: WindowSet) -> NDArray[np.float32]:
    model.eval()
    x = torch.from_numpy(np.transpose(data.X, (0, 2, 1)).copy())
    with torch.no_grad():
        pred = model(x)
    array = np.asarray(pred.detach().cpu().numpy(), dtype=np.float32)
    return array


def _measure_latency(
    fn: Callable[[], object], *, n_warmup: int = 5, n_runs: int = 20
) -> float:
    for _ in range(n_warmup):
        fn()
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - start) / n_runs
