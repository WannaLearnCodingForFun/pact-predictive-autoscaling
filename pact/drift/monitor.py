"""Drift monitor (Module 5): tracking error, margin, refit trigger (Eq. 26–28).

``e(t) = N_required_observed(t) − N_predicted(t | t−τ)``.

The ring buffer stores every issued horizon keyed by target tick and issue
tick, so the comparison is against the prediction that referred to *this*
instant and was made at ``t − τ``. An off-by-τ here silently breaks the
margin update.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pact.config import PactConfig

logger = logging.getLogger(__name__)


class RefitScheduler(Protocol):
    """Runs incremental refit off the control thread. Must never block the loop."""

    def schedule(self) -> None: ...


class NullRefitScheduler:
    def schedule(self) -> None:
        return None


@dataclass(frozen=True)
class DriftSnapshot:
    tick: int
    error: float | None
    ebar: float
    gamma: float
    refit_requested: bool
    consecutive_high: int


class DriftMonitor:
    """Clipped-integral margin with a τ-aligned prediction ring buffer."""

    def __init__(
        self,
        config: PactConfig,
        *,
        tau: int,
        scheduler: RefitScheduler | None = None,
        gamma: float | None = None,
        freeze_gamma: bool | None = None,
    ) -> None:
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau}")
        self._tau = tau
        self._drift = config.drift
        self._frozen = (
            config.ablation.freeze_gamma if freeze_gamma is None else freeze_gamma
        )
        self._scheduler: RefitScheduler = scheduler or NullRefitScheduler()
        self._tick = 0
        # target_tick -> {issued_at: predicted N}
        self._issued: dict[int, dict[int, int]] = {}
        lo, hi = self._drift.gamma_min, self._drift.gamma_max
        self.gamma = lo if gamma is None else _clip(gamma, lo, hi)
        self.ebar = 0.0
        self.last_error: float | None = None
        self.consecutive_high = 0
        self.refit_requested = False

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def tau(self) -> int:
        return self._tau

    def predicted_at(self, target_tick: int, issued_at: int) -> int | None:
        """Return the stored ``N_predicted(target | issued_at)``, if any."""

        slot = self._issued.get(target_tick)
        if slot is None:
            return None
        return slot.get(issued_at)

    def step(
        self,
        n_required_observed: int,
        predicted_demand: Sequence[int],
    ) -> DriftSnapshot:
        """Consume the observation at ``t``, then record the forecast issued at ``t``.

        Lookup is ``N_predicted(t | t−τ)``: the horizon element whose target is
        the current tick and whose issue tick is ``t − τ``.
        """

        issued_at = self._tick - self._tau
        slot = self._issued.get(self._tick, {})
        error: float | None
        if issued_at in slot:
            e = float(n_required_observed - slot[issued_at])
            error = e
            self.last_error = e
            if not self._frozen:
                self.ebar = self._drift.eta * e + (1.0 - self._drift.eta) * self.ebar
                self.gamma = _clip(
                    self.gamma + self._drift.kappa_gamma * self.ebar,
                    self._drift.gamma_min,
                    self._drift.gamma_max,
                )
                if abs(self.ebar) > self._drift.xi:
                    self.consecutive_high += 1
                else:
                    self.consecutive_high = 0
                if (
                    self.consecutive_high >= self._drift.drift_window
                    and not self.refit_requested
                ):
                    self.refit_requested = True
                    logger.warning(
                        "drift trigger |ebar|=%.4f > xi=%.4f for %s ticks; "
                        "scheduling refit",
                        abs(self.ebar),
                        self._drift.xi,
                        self.consecutive_high,
                    )
                    self._scheduler.schedule()
        else:
            error = None
            self.last_error = None

        for k, n_hat in enumerate(predicted_demand):
            target = self._tick + 1 + k
            self._issued.setdefault(target, {})[self._tick] = int(n_hat)
        stale = [key for key in self._issued if key < self._tick]
        for key in stale:
            del self._issued[key]

        snapshot = DriftSnapshot(
            tick=self._tick,
            error=error,
            ebar=self.ebar,
            gamma=self.gamma,
            refit_requested=self.refit_requested,
            consecutive_high=self.consecutive_high,
        )
        self._tick += 1
        return snapshot

    def clear_refit(self) -> None:
        self.refit_requested = False
        self.consecutive_high = 0


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
