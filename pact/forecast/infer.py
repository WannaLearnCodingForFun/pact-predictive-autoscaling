"""ONNX Runtime inference, CPU, single-threaded (Module 2)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ForecastBudgetError(RuntimeError):
    """Raised when a single-tick inference exceeds Δt."""


class OnnxForecaster:
    """Loads an exported TCN. Input is an ``(L, d)`` window; output is ``(H, 2)``."""

    def __init__(self, onnx_path: Path, *, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        shape = self._session.get_inputs()[0].shape
        # [batch, channels, L]
        self.in_channels = int(shape[1])
        self.window = int(shape[2])
        self._dt = dt
        self.over_budget_count = 0

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def predict(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        x = _as_nchw(window, self.window, self.in_channels)
        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: x})
        elapsed = time.perf_counter() - t0
        if elapsed > self._dt:
            self.over_budget_count += 1
            logger.error(
                "inference took %.6fs, exceeding dt=%.6fs", elapsed, self._dt
            )
            raise ForecastBudgetError(
                f"inference {elapsed:.6f}s exceeds dt={self._dt:.6f}s"
            )
        forecast = np.asarray(outputs[0], dtype=np.float32)
        return forecast.reshape(forecast.shape[1], forecast.shape[2])


def _as_nchw(
    window: NDArray[np.floating],
    length: int,
    channels: int,
) -> NDArray[np.float32]:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"window must be 2-D, got {arr.shape}")
    if arr.shape == (length, channels):
        arr = arr.T
    elif arr.shape != (channels, length):
        raise ValueError(
            f"window shape {arr.shape} does not match (L={length}, d={channels})"
        )
    return np.ascontiguousarray(arr[np.newaxis, ...])
