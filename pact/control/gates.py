"""Actuation gates (Eq. 25). Order is mandatory: feasibility, then gating.

1. round the relaxed solution
2. clip to ``±Δ_max``
3. clip so ``N + u ∈ [n_min, n_max]``
4. deadband: if ``|u| < θ`` → 0
5. cooldown: block scale-down within ``T_cool`` of a scale-up

Steps 1–3 are feasibility; 4–5 are gating. Guard rails cannot be skipped by
an unusual forecast because the clips run first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pact.config import ControlConfig


@dataclass(frozen=True)
class GateDecision:
    u: int
    last_scale_up_s: float | None
    rounded: int
    rate_clipped: int
    bound_clipped: int
    after_deadband: int
    cooldown_blocked: bool


def round_action(u: float) -> int:
    """Round a relaxed action to an integer (half away from zero)."""

    if u >= 0.0:
        return int(math.floor(u + 0.5))
    return int(math.ceil(u - 0.5))


def clip_rate_limit(u: int, delta_max: int) -> int:
    """Clip to ``±Δ_max``."""

    return max(-delta_max, min(delta_max, u))


def clip_pool_bounds(u: int, n_current: int, n_min: int, n_max: int) -> int:
    """Clip so ``N + u ∈ [n_min, n_max]``."""

    n_next = n_current + u
    if n_next > n_max:
        u -= n_next - n_max
    elif n_next < n_min:
        u += n_min - n_next
    return u


def apply_deadband(u: int, theta: int) -> int:
    """Zero the action if ``|u| < θ``."""

    if abs(u) < theta:
        return 0
    return u


def apply_cooldown(
    u: int,
    *,
    now_s: float,
    last_scale_up_s: float | None,
    cooldown_s: float,
) -> int:
    """Block scale-down while still inside ``T_cool`` of the last scale-up."""

    if u >= 0 or last_scale_up_s is None:
        return u
    if now_s - last_scale_up_s < cooldown_s:
        return 0
    return u


def apply_gates(
    u: float,
    n_current: int,
    ctrl: ControlConfig,
    *,
    now_s: float,
    last_scale_up_s: float | None,
) -> GateDecision:
    """Apply feasibility then gating, in spec order. Returns the applied ``u``."""

    rounded = round_action(u)
    rate_clipped = clip_rate_limit(rounded, ctrl.delta_max)
    bound_clipped = clip_pool_bounds(
        rate_clipped, n_current, ctrl.n_min, ctrl.n_max
    )
    after_deadband = apply_deadband(bound_clipped, ctrl.deadband)
    gated = apply_cooldown(
        after_deadband,
        now_s=now_s,
        last_scale_up_s=last_scale_up_s,
        cooldown_s=ctrl.cooldown_s,
    )
    cooldown_blocked = after_deadband < 0 and gated == 0
    new_last = now_s if gated > 0 else last_scale_up_s
    return GateDecision(
        u=gated,
        last_scale_up_s=new_last,
        rounded=rounded,
        rate_clipped=rate_clipped,
        bound_clipped=bound_clipped,
        after_deadband=after_deadband,
        cooldown_blocked=cooldown_blocked,
    )
