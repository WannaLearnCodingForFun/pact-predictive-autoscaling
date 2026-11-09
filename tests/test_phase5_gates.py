"""Phase 5.2 gate unit tests: each projection, then the composition."""

from __future__ import annotations

from dataclasses import replace

from pact.control.gates import (
    apply_cooldown,
    apply_deadband,
    apply_gates,
    clip_pool_bounds,
    clip_rate_limit,
    round_action,
)

from tests.helpers import make_config


def test_round_action_half_away_from_zero() -> None:
    assert round_action(1.4) == 1
    assert round_action(1.5) == 2
    assert round_action(-1.5) == -2
    assert round_action(-1.4) == -1
    assert round_action(2.0) == 2


def test_clip_rate_limit() -> None:
    assert clip_rate_limit(10, 4) == 4
    assert clip_rate_limit(-10, 4) == -4
    assert clip_rate_limit(3, 4) == 3
    assert clip_rate_limit(0, 4) == 0


def test_clip_pool_bounds() -> None:
    assert clip_pool_bounds(4, n_current=15, n_min=1, n_max=16) == 1
    assert clip_pool_bounds(-4, n_current=2, n_min=1, n_max=16) == -1
    assert clip_pool_bounds(2, n_current=8, n_min=1, n_max=16) == 2
    assert clip_pool_bounds(0, n_current=1, n_min=1, n_max=16) == 0


def test_deadband_zeros_strictly_inside_theta() -> None:
    assert apply_deadband(1, theta=2) == 0
    assert apply_deadband(-1, theta=2) == 0
    assert apply_deadband(2, theta=2) == 2
    assert apply_deadband(1, theta=1) == 1
    assert apply_deadband(0, theta=1) == 0


def test_cooldown_blocks_scale_down_inside_window() -> None:
    assert (
        apply_cooldown(-3, now_s=30.0, last_scale_up_s=0.0, cooldown_s=60.0) == 0
    )
    assert apply_cooldown(3, now_s=30.0, last_scale_up_s=0.0, cooldown_s=60.0) == 3
    assert (
        apply_cooldown(-3, now_s=60.0, last_scale_up_s=0.0, cooldown_s=60.0) == -3
    )
    assert (
        apply_cooldown(-3, now_s=10.0, last_scale_up_s=None, cooldown_s=60.0) == -3
    )


def test_composition_round_then_rate_then_bounds() -> None:
    cfg = make_config(n_min=1, n_max=16)
    ctrl = replace(cfg.control, delta_max=4, deadband=1, cooldown_s=0.0)

    rate = apply_gates(
        10.4, n_current=8, ctrl=ctrl, now_s=0.0, last_scale_up_s=None
    )
    assert rate.rounded == 10
    assert rate.rate_clipped == 4
    assert rate.bound_clipped == 4
    assert rate.u == 4

    bounds = apply_gates(
        10.0, n_current=15, ctrl=ctrl, now_s=0.0, last_scale_up_s=None
    )
    assert bounds.rounded == 10
    assert bounds.rate_clipped == 4
    assert bounds.bound_clipped == 1
    assert bounds.u == 1


def test_composition_feasibility_before_deadband() -> None:
    """Rounding happens before the deadband, so 0.6 is not swallowed as 0."""

    cfg = make_config(n_min=1, n_max=16)
    ctrl = replace(cfg.control, delta_max=4, deadband=1, cooldown_s=0.0)
    decision = apply_gates(
        0.6, n_current=8, ctrl=ctrl, now_s=0.0, last_scale_up_s=None
    )
    assert decision.rounded == 1
    assert decision.after_deadband == 1
    assert decision.u == 1
    assert apply_deadband(round_action(0.4), theta=1) == 0


def test_composition_cooldown_after_feasibility() -> None:
    cfg = make_config(n_min=1, n_max=16)
    ctrl = replace(cfg.control, delta_max=4, deadband=1, cooldown_s=60.0)
    blocked = apply_gates(
        -3.2, n_current=10, ctrl=ctrl, now_s=10.0, last_scale_up_s=0.0
    )
    assert blocked.rounded == -3
    assert blocked.bound_clipped == -3
    assert blocked.after_deadband == -3
    assert blocked.cooldown_blocked is True
    assert blocked.u == 0
    assert blocked.last_scale_up_s == 0.0

    allowed = apply_gates(
        2.0, n_current=10, ctrl=ctrl, now_s=10.0, last_scale_up_s=0.0
    )
    assert allowed.u == 2
    assert allowed.last_scale_up_s == 10.0
