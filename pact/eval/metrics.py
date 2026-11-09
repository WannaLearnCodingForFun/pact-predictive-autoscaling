"""Evaluation metrics (paper Equations 29–33).

Every function is pure: sequences in, scalars or a frozen dataclass out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Decision = Literal["scale_up", "hold", "scale_down"]


@dataclass(frozen=True)
class ClassScore:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class DecisionClassification:
    scale_up: ClassScore
    hold: ClassScore
    scale_down: ClassScore
    accuracy: float


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error (Eq. 29)."""

    a, p = _paired(actual, predicted)
    return sum(abs(x - y) for x, y in zip(a, p, strict=True)) / len(a)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error (Eq. 30)."""

    a, p = _paired(actual, predicted)
    mean_sq = sum((x - y) ** 2 for x, y in zip(a, p, strict=True)) / len(a)
    return float(mean_sq**0.5)


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error as a fraction of |actual| (Eq. 31)."""

    a, p = _paired(actual, predicted)
    if any(x == 0.0 for x in a):
        raise ValueError("mape is undefined when any actual value is 0")
    return sum(abs(x - y) / abs(x) for x, y in zip(a, p, strict=True)) / len(a)


def r2(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Coefficient of determination."""

    a, p = _paired(actual, predicted)
    mean_a = sum(a) / len(a)
    ss_tot = sum((x - mean_a) ** 2 for x in a)
    ss_res = sum((x - y) ** 2 for x, y in zip(a, p, strict=True))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def sla_violation_rate(p95_series: Sequence[float], slo: float) -> float:
    """Fraction of ticks whose p95 latency is strictly over the SLO (Eq. 32)."""

    series = _require_nonempty(p95_series)
    return sum(1 for x in series if x > slo) / len(series)


def over_provision_ratio(
    n_series: Sequence[float], n_required_series: Sequence[float]
) -> float:
    """Mean positive excess replica count relative to required (Eq. 33).

    ``(1/T) Σ max(N(t) − N_req(t), 0) / N_req(t)``.
    """

    n, n_req = _paired(n_series, n_required_series)
    if any(r == 0.0 for r in n_req):
        raise ValueError("over_provision_ratio is undefined when N_req is 0")
    return sum(max(x - r, 0.0) / r for x, r in zip(n, n_req, strict=True)) / len(n)


def scale_action_count(n_series: Sequence[float]) -> int:
    """Number of ticks at which the replica count changed."""

    series = _require_nonempty(n_series)
    return sum(1 for prev, cur in zip(series, series[1:], strict=False) if cur != prev)


def oscillation_count(n_series: Sequence[float], window: int | None = None) -> int:
    """Sign reversals in ΔN.

    Zero deltas (holds) are skipped. If ``window`` is set, a reversal counts
    only when the two nonzero actions are at most ``window`` ticks apart.
    """

    series = _require_nonempty(n_series)
    if window is not None and window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    actions: list[tuple[int, float]] = []
    for i, (prev, cur) in enumerate(zip(series, series[1:], strict=False)):
        delta = cur - prev
        if delta != 0.0:
            actions.append((i + 1, delta))

    reversals = 0
    for (t_prev, d_prev), (t_cur, d_cur) in zip(actions, actions[1:], strict=False):
        if d_prev * d_cur >= 0.0:
            continue
        if window is not None and (t_cur - t_prev) > window:
            continue
        reversals += 1
    return reversals


def cost_per_1k_requests(
    n_series: Sequence[float],
    dt: float,
    requests_served: Sequence[float] | float,
    unit_cost: float,
) -> float:
    """Replica-time cost per 1000 served requests."""

    series = _require_nonempty(n_series)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if isinstance(requests_served, Sequence) and not isinstance(
        requests_served, (str, bytes)
    ):
        served = sum(_require_nonempty(requests_served))
    else:
        served = float(requests_served)
    if served <= 0.0:
        raise ValueError("requests_served must be positive")
    replica_seconds = sum(n * dt for n in series)
    return unit_cost * replica_seconds * 1000.0 / served


def decision_classification(
    n_applied: Sequence[float], n_required: Sequence[float]
) -> DecisionClassification:
    """Per-class precision / recall / F1 of applied vs required scale decisions.

    Each tick t≥1 is bucketed into {scale_up, hold, scale_down} from ΔN.
    Recall on scale_up is the headline number.
    """

    applied, required = _paired(n_applied, n_required)
    if len(applied) < 2:
        raise ValueError("decision_classification needs at least two ticks")
    y_pred = _decisions(applied)
    y_true = _decisions(required)
    labels: tuple[Decision, ...] = ("scale_up", "hold", "scale_down")
    scores = {label: _class_score(y_true, y_pred, label) for label in labels}
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return DecisionClassification(
        scale_up=scores["scale_up"],
        hold=scores["hold"],
        scale_down=scores["scale_down"],
        accuracy=correct / len(y_true),
    )


def _decisions(series: Sequence[float]) -> list[Decision]:
    out: list[Decision] = []
    for prev, cur in zip(series, series[1:], strict=False):
        if cur > prev:
            out.append("scale_up")
        elif cur < prev:
            out.append("scale_down")
        else:
            out.append("hold")
    return out


def _class_score(
    y_true: Sequence[Decision], y_pred: Sequence[Decision], label: Decision
) -> ClassScore:
    tp = sum(
        1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label
    )
    fp = sum(
        1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label
    )
    fn = sum(
        1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label
    )
    support = sum(1 for t in y_true if t == label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return ClassScore(precision=precision, recall=recall, f1=f1, support=support)


def _paired(
    actual: Sequence[float], predicted: Sequence[float]
) -> tuple[list[float], list[float]]:
    a = [float(x) for x in actual]
    p = [float(x) for x in predicted]
    if len(a) != len(p):
        raise ValueError(f"series length mismatch: {len(a)} vs {len(p)}")
    if not a:
        raise ValueError("series must be non-empty")
    return a, p


def _require_nonempty(series: Sequence[float]) -> list[float]:
    values = [float(x) for x in series]
    if not values:
        raise ValueError("series must be non-empty")
    return values
