"""Feature construction for Module 1 (Eq. 2–5).

Channels: EWMA smoothing, min-max normalisation fitted on the training split
only, an incremental L×d sliding window, per-replica intensity ρ, and
sin/cos time-of-day.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pact.telemetry.collector import Observation

SECONDS_PER_DAY = 86400.0
# Eq. 1 (5) + ρ (Eq. 5) + sin/cos time-of-day
FEATURE_DIM = 8


class FrozenNormaliserError(RuntimeError):
    """Raised when ``fit()`` is called after ``freeze()``."""


def rho(arrival_rate: float, n_replicas: float) -> float:
    """Per-replica arrival intensity ρ(t) = λ(t) / max(N(t), 1) (Eq. 5)."""

    return arrival_rate / max(n_replicas, 1.0)


def time_of_day_features(t_s: float) -> tuple[float, float]:
    """``sin(2π t / 86400)``, ``cos(2π t / 86400)``."""

    angle = 2.0 * math.pi * t_s / SECONDS_PER_DAY
    return math.sin(angle), math.cos(angle)


def feature_vector(obs: Observation) -> tuple[float, ...]:
    """Observation plus ρ and time-of-day. Length ``FEATURE_DIM``."""

    sin_t, cos_t = time_of_day_features(obs.t_s)
    return (
        obs.u,
        obs.r,
        obs.lam,
        obs.ell_p95,
        obs.n,
        rho(obs.lam, obs.n),
        sin_t,
        cos_t,
    )


class EWMASmoother:
    """Recursive EWMA, one stored state per channel (Eq. 2).

    ``s(t) = α x(t) + (1 − α) s(t − 1)``. The first sample initialises state.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha
        self._state: list[float] | None = None

    def update(self, x: Sequence[float]) -> list[float]:
        values = [float(v) for v in x]
        if self._state is None:
            self._state = values
            return list(self._state)
        if len(values) != len(self._state):
            raise ValueError(
                f"expected {len(self._state)} channels, got {len(values)}"
            )
        a = self._alpha
        self._state = [
            a * v + (1.0 - a) * s for v, s in zip(values, self._state, strict=True)
        ]
        return list(self._state)


class MinMaxNormaliser:
    """Per-channel min-max scale (Eq. 3). Fit on the training split, then freeze."""

    def __init__(self) -> None:
        self._min: list[float] | None = None
        self._max: list[float] | None = None
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def fit(self, samples: Sequence[Sequence[float]]) -> None:
        if self._frozen:
            raise FrozenNormaliserError("fit() called after freeze()")
        if not samples:
            raise ValueError("fit() requires a non-empty training split")
        n_channels = len(samples[0])
        if n_channels == 0:
            raise ValueError("samples must have at least one channel")
        mins = [float("inf")] * n_channels
        maxs = [float("-inf")] * n_channels
        for row in samples:
            if len(row) != n_channels:
                raise ValueError("ragged training matrix")
            for i, value in enumerate(row):
                v = float(value)
                if v < mins[i]:
                    mins[i] = v
                if v > maxs[i]:
                    maxs[i] = v
        self._min = mins
        self._max = maxs

    def freeze(self) -> None:
        if self._min is None or self._max is None:
            raise RuntimeError("freeze() before fit()")
        self._frozen = True

    def transform(self, x: Sequence[float]) -> list[float]:
        if self._min is None or self._max is None:
            raise RuntimeError("transform() before fit()")
        if len(x) != len(self._min):
            raise ValueError(f"expected {len(self._min)} channels, got {len(x)}")
        out: list[float] = []
        for v, lo, hi in zip(x, self._min, self._max, strict=True):
            span = hi - lo
            out.append(0.0 if span == 0.0 else (float(v) - lo) / span)
        return out

    def dump(self, path: Path) -> None:
        if self._min is None or self._max is None:
            raise RuntimeError("dump() before fit()")
        path.write_text(
            json.dumps({"min": self._min, "max": self._max, "frozen": self._frozen})
        )

    @classmethod
    def load(cls, path: Path) -> MinMaxNormaliser:
        raw: Any = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise TypeError("normaliser file must be a JSON object")
        obj = cls()
        obj._min = [float(v) for v in raw["min"]]
        obj._max = [float(v) for v in raw["max"]]
        obj._frozen = True
        return obj


class SlidingWindow:
    """L×d ring buffer. ``push`` is O(d) (independent of L) per tick (Eq. 4)."""

    def __init__(self, length: int, n_channels: int) -> None:
        if length < 1:
            raise ValueError(f"window length must be >= 1, got {length}")
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        self._length = length
        self._n_channels = n_channels
        self._buf = [[0.0] * n_channels for _ in range(length)]
        self._head = 0
        self._filled = 0

    @property
    def full(self) -> bool:
        return self._filled == self._length

    def push(self, x: Sequence[float]) -> None:
        if len(x) != self._n_channels:
            raise ValueError(f"expected {self._n_channels} channels, got {len(x)}")
        row = self._buf[self._head]
        for i, value in enumerate(x):
            row[i] = float(value)
        self._head = (self._head + 1) % self._length
        if self._filled < self._length:
            self._filled += 1

    def tensor(self) -> list[list[float]]:
        """Oldest-to-newest rows. Length ``min(ticks, L)``."""

        if self._filled < self._length:
            return [list(self._buf[i]) for i in range(self._filled)]
        start = self._head
        return [
            list(self._buf[(start + i) % self._length]) for i in range(self._length)
        ]
