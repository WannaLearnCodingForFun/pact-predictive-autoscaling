"""Reactive safety ceiling: forecast-miss fallback (Module 4, §5.3).

If observed utilisation exceeds ``u_emergency``, scale up immediately and skip
the MPC and every gate (deadband, cooldown, rate limit). Pool bounds still
hold — ``N`` cannot exceed ``n_max``. Every firing is logged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from pact.config import PactConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyDecision:
    active: bool
    u: int


def immediate_scale_up(
    n_current: int,
    observed_utilisation: float,
    config: PactConfig,
) -> int:
    """Replicas to add now so utilisation heads back toward ``u_target``."""

    ctrl = config.control
    if n_current >= ctrl.n_max:
        return 0
    n = max(n_current, 1)
    target = config.capacity.u_target
    if target <= 0.0:
        return ctrl.n_max - n_current
    n_needed = math.ceil(observed_utilisation * n / target)
    n_needed = max(n_needed, n_current + 1)
    n_needed = min(n_needed, ctrl.n_max)
    return n_needed - n_current


def evaluate_safety(
    observed_utilisation: float,
    n_current: int,
    config: PactConfig,
) -> SafetyDecision:
    """Return an override when utilisation is strictly above ``u_emergency``.

    A ``None``-like miss is ``active=False``. ``active=True`` with ``u=0`` still
    counts as an override (pool already at ``n_max``) and is logged.
    """

    u_emergency = config.control.u_emergency
    if not observed_utilisation > u_emergency:
        return SafetyDecision(active=False, u=0)
    u = immediate_scale_up(n_current, observed_utilisation, config)
    logger.warning(
        "safety override utilisation=%.4f > u_emergency=%.4f n=%s u=%s "
        "(bypassing MPC and gates)",
        observed_utilisation,
        u_emergency,
        n_current,
        u,
    )
    return SafetyDecision(active=True, u=u)
