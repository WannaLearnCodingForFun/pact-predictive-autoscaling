"""Hand-computed unit tests for every metric in section 1.2."""

from __future__ import annotations

import math

import pytest
from pact.eval.metrics import (
    cost_per_1k_requests,
    decision_classification,
    mae,
    mape,
    oscillation_count,
    over_provision_ratio,
    r2,
    rmse,
    scale_action_count,
    sla_violation_rate,
)


def test_mae() -> None:
    # |1-1| + |2-3| + |3-5| = 0+1+2; mean = 1
    assert mae([1.0, 2.0, 3.0], [1.0, 3.0, 5.0]) == 1.0


def test_rmse() -> None:
    # sqrt((0 + 1 + 4) / 3) = sqrt(5/3)
    assert rmse([1.0, 2.0, 3.0], [1.0, 3.0, 5.0]) == pytest.approx(math.sqrt(5.0 / 3.0))


def test_mape() -> None:
    # (0 + 1/2 + 2/3) / 3 = (7/6) / 3 = 7/18
    assert mape([1.0, 2.0, 3.0], [1.0, 3.0, 5.0]) == pytest.approx(7.0 / 18.0)


def test_r2() -> None:
    # ss_tot = (1-2)^2 + 0 + (3-2)^2 = 2; ss_res = 0+1+4 = 5; 1 - 5/2 = -1.5
    assert r2([1.0, 2.0, 3.0], [1.0, 3.0, 5.0]) == pytest.approx(-1.5)
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_sla_violation_rate() -> None:
    # ticks over 200: only 250. 200 is not strictly over the SLO.
    assert sla_violation_rate([100.0, 200.0, 250.0], 200.0) == pytest.approx(1.0 / 3.0)


def test_over_provision_ratio() -> None:
    # max(N-Nreq,0)/Nreq = (2/2, 0/4, 0/2); mean = 1/3
    assert over_provision_ratio([4.0, 4.0, 2.0], [2.0, 4.0, 2.0]) == pytest.approx(
        1.0 / 3.0
    )


def test_scale_action_count() -> None:
    # changes at 1→2, 2→3, 3→2
    assert scale_action_count([1, 1, 2, 2, 3, 2]) == 3
    assert scale_action_count([4, 4, 4]) == 0


def test_oscillation_count() -> None:
    # ΔN = +1, -1, +1 → two sign reversals
    assert oscillation_count([1, 2, 1, 2]) == 2
    # ΔN = +1, +1, -1 → one reversal
    assert oscillation_count([1, 2, 3, 2]) == 1
    # holds skipped: +1 then -1 → one reversal
    assert oscillation_count([1, 2, 2, 1]) == 1
    # reversal 3 ticks apart; window=2 excludes it
    assert oscillation_count([1, 2, 2, 2, 1], window=2) == 0
    assert oscillation_count([1, 2, 2, 2, 1], window=3) == 1


def test_cost_per_1k_requests() -> None:
    # replica-seconds = 2*5 + 2*5 = 20; 20 * 1 * 1000 / 1000 = 20
    assert cost_per_1k_requests([2, 2], 5.0, 1000.0, 1.0) == pytest.approx(20.0)
    # served series sums to 200; 20 * 1000 / 200 = 100
    assert cost_per_1k_requests(
        [2, 2], 5.0, [100.0, 100.0], 1.0
    ) == pytest.approx(100.0)


def test_decision_classification() -> None:
    # applied Δ: up, hold, down, hold
    # required Δ: up, up, down, down
    result = decision_classification([2, 3, 3, 2, 2], [2, 3, 4, 3, 2])
    assert result.accuracy == pytest.approx(0.5)
    assert result.scale_up.precision == pytest.approx(1.0)
    assert result.scale_up.recall == pytest.approx(0.5)
    assert result.scale_up.f1 == pytest.approx(2.0 / 3.0)
    assert result.scale_up.support == 2
    assert result.hold.precision == pytest.approx(0.0)
    assert result.hold.recall == pytest.approx(0.0)
    assert result.hold.support == 0
    assert result.scale_down.precision == pytest.approx(1.0)
    assert result.scale_down.recall == pytest.approx(0.5)
    assert result.scale_down.f1 == pytest.approx(2.0 / 3.0)
    assert result.scale_down.support == 2
