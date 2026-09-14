"""Chronological splits and a one-shot test-split guard."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pact.forecast.train import ChronologicalSplitError, assert_chronological_splits


class TestSplitReusedError(RuntimeError):
    """Raised when the test split is read more than once on a guard."""

    __test__ = False


@dataclass(frozen=True)
class SeriesSplit:
    train: NDArray[np.float64]
    val: NDArray[np.float64]
    test: NDArray[np.float64]
    train_ts: NDArray[np.float64]
    val_ts: NDArray[np.float64]
    test_ts: NDArray[np.float64]


class SplitGuard:
    """Raises if test data is read more than once on this object.

    The experiment process constructs one guard and passes it through tuning
    and evaluation. A second read of the test split is a protocol bug.
    """

    def __init__(self) -> None:
        self._reads = 0

    @property
    def test_reads(self) -> int:
        return self._reads

    def read_test(self, data: SeriesSplit) -> NDArray[np.float64]:
        self._reads += 1
        if self._reads > 1:
            raise TestSplitReusedError(
                "test split read more than once per process"
            )
        return data.test


def chronological_series_split(
    values: Sequence[float],
    timestamps: Sequence[float],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> SeriesSplit:
    """70/15/15 on a 1-D series. Never shuffles."""

    series = np.asarray(values, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if series.shape != ts.shape or series.ndim != 1:
        raise ValueError("values and timestamps must be 1-D and aligned")
    if series.size < 3:
        raise ChronologicalSplitError("series too short for 70/15/15")
    if np.any(ts[1:] <= ts[:-1]):
        raise ChronologicalSplitError("timestamps must be strictly increasing")
    n = series.size
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ChronologicalSplitError("series too short for 70/15/15")
    train = series[:n_train]
    val = series[n_train : n_train + n_val]
    test = series[n_train + n_val :]
    train_ts = ts[:n_train]
    val_ts = ts[n_train : n_train + n_val]
    test_ts = ts[n_train + n_val :]
    assert_chronological_splits(train_ts, val_ts, test_ts)
    return SeriesSplit(
        train=train,
        val=val,
        test=test,
        train_ts=train_ts,
        val_ts=val_ts,
        test_ts=test_ts,
    )
