"""Trace replay driver for the analytical pool simulator.

Time compression folds ``compression`` consecutive trace samples into one
control tick (mean arrival rate; last desired replica count in the group).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from pact.sim.queue_model import PoolSimulator, PoolState


@dataclass(frozen=True)
class ReplayResult:
    states: tuple[PoolState, ...]
    arrival_rates: tuple[float, ...]
    n_replicas: tuple[int, ...]


class TraceReplay:
    def __init__(
        self,
        arrival_rates: Sequence[float],
        *,
        compression: int = 1,
        n_replicas: int | Sequence[int] = 1,
    ) -> None:
        if compression < 1:
            raise ValueError(f"compression must be >= 1, got {compression}")
        if not arrival_rates:
            raise ValueError("arrival_rates must be non-empty")
        replica_series = _as_replica_series(n_replicas, len(arrival_rates))
        self._arrival_rates = _compress_mean(arrival_rates, compression)
        self._n_replicas = _compress_last(replica_series, compression)

    def run(self, simulator: PoolSimulator) -> ReplayResult:
        states: list[PoolState] = []
        for rate, n in zip(self._arrival_rates, self._n_replicas, strict=True):
            states.append(simulator.step(rate, n))
        return ReplayResult(
            states=tuple(states),
            arrival_rates=self._arrival_rates,
            n_replicas=self._n_replicas,
        )


def _as_replica_series(
    n_replicas: int | Sequence[int], length: int
) -> tuple[int, ...]:
    if isinstance(n_replicas, Sequence) and not isinstance(n_replicas, (str, bytes)):
        series = tuple(int(n) for n in n_replicas)
        if len(series) != length:
            raise ValueError(
                f"n_replicas length {len(series)} does not match "
                f"arrival_rates length {length}"
            )
        return series
    return (int(n_replicas),) * length


def _compress_mean(values: Sequence[float], compression: int) -> tuple[float, ...]:
    groups = _groups(values, compression)
    return tuple(sum(g) / len(g) for g in groups)


def _compress_last(values: Sequence[int], compression: int) -> tuple[int, ...]:
    groups = _groups(values, compression)
    return tuple(g[-1] for g in groups)


S = TypeVar("S")


def _groups(values: Sequence[S], compression: int) -> list[list[S]]:
    grouped: list[list[S]] = []
    for i in range(0, len(values), compression):
        grouped.append(list(values[i : i + compression]))
    return grouped
