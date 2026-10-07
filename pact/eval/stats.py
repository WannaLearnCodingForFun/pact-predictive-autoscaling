"""Paired nonparametric tests for headline metrics (publication upgrade U1).

Wilcoxon signed-rank, BCa bootstrap CIs, Cliff's delta, Holm–Bonferroni.
Seed-level scores are paired across methods; nothing here invents a result.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, wilcoxon

BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95
ALPHA = 0.05
REFERENCE_CANDIDATES = ("pact", "full")

K = TypeVar("K", bound=Hashable)
Statistic = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True)
class BootstrapCI:
    low: float
    high: float
    method: str  # "bca", "percentile", or "degenerate"


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float  # W+, sum of ranks of positive differences
    w_minus: float
    p_value: float
    n: int
    n_nonzero: int


@dataclass(frozen=True)
class MethodComparison:
    method: str
    mean: float
    std: float
    n: int
    ci: BootstrapCI
    p_raw: float | None
    p_holm: float | None
    significant: bool
    cliffs_delta: float | None


def sample_mean_std(values: Sequence[float]) -> tuple[float, float, int]:
    series = [float(x) for x in values]
    n = len(series)
    if n < 1:
        raise ValueError("cannot summarise an empty series")
    mean = sum(series) / n
    if n == 1:
        return mean, 0.0, n
    std = (sum((x - mean) ** 2 for x in series) / (n - 1)) ** 0.5
    return mean, std, n


def average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """1-based average ranks. Ties share the mean of their ordinal positions."""

    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = 0.5 * (float(i + 1) + float(j))
        ranks[order[i:j]] = avg
        i = j
    return tuple(float(r) for r in ranks)


def signed_rank_sums(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float, int]:
    """Return ``(W+, W−, n_nonzero)`` after dropping zero differences."""

    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    diff = a - b
    nonzero = diff[diff != 0.0]
    if nonzero.size == 0:
        return 0.0, 0.0, 0
    ranks = np.asarray(average_ranks(np.abs(nonzero)), dtype=np.float64)
    w_plus = float(np.sum(ranks[nonzero > 0.0]))
    w_minus = float(np.sum(ranks[nonzero < 0.0]))
    return w_plus, w_minus, int(nonzero.size)


def wilcoxon_signed_rank(
    x: Sequence[float], y: Sequence[float]
) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank on paired ``x`` vs ``y``.

    Zeros are dropped (Wilcox 1945). ``statistic`` is W+, the sum of ranks of
    strictly positive differences. Exact p-values are used when SciPy allows.
    """

    a = [float(v) for v in x]
    b = [float(v) for v in y]
    w_plus, w_minus, n_nonzero = signed_rank_sums(a, b)
    n = len(a)
    if n_nonzero == 0:
        return WilcoxonResult(
            statistic=0.0, w_minus=0.0, p_value=1.0, n=n, n_nonzero=0
        )
    try:
        result = wilcoxon(
            a,
            b,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            method="exact",
        )
    except (TypeError, ValueError):
        result = wilcoxon(
            a,
            b,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
    return WilcoxonResult(
        statistic=w_plus,
        w_minus=w_minus,
        p_value=float(result.pvalue),
        n=n,
        n_nonzero=n_nonzero,
    )


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Dominance of ``x`` over ``y``: ``(#x>y − #x<y) / (n_x n_y)``.

    Unpaired. Range is ``[-1, 1]``. Zero when every pair ties.
    """

    a = [float(v) for v in x]
    b = [float(v) for v in y]
    if not a or not b:
        raise ValueError("Cliff's delta needs two non-empty samples")
    n_gt = 0
    n_lt = 0
    for xi in a:
        for yj in b:
            if xi > yj:
                n_gt += 1
            elif xi < yj:
                n_lt += 1
    return (n_gt - n_lt) / (len(a) * len(b))


def holm_bonferroni(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm–Bonferroni adjusted p-values, returned in the original order."""

    m = len(p_values)
    if m == 0:
        return ()
    order = sorted(range(m), key=lambda i: (float(p_values[i]), i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = (m - rank) * float(p_values[idx])
        running = min(1.0, max(running, raw))
        adjusted[idx] = running
    return tuple(adjusted)


def mean_statistic(sample: NDArray[np.float64]) -> float:
    return float(np.mean(sample))


def jackknife_acceleration(
    values: Sequence[float] | NDArray[np.float64],
    statistic: Statistic = mean_statistic,
) -> float:
    """Efron acceleration ``a`` from leave-one-out replicates."""

    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n < 2:
        return 0.0
    theta_i = np.empty(n, dtype=np.float64)
    for i in range(n):
        left = np.asarray(np.delete(x, i), dtype=np.float64)
        theta_i[i] = statistic(left)
    theta_dot = float(np.mean(theta_i))
    u = theta_dot - theta_i
    sum_sq = float(np.sum(u**2))
    if sum_sq == 0.0:
        return 0.0
    return float(np.sum(u**3) / (6.0 * sum_sq**1.5))


def bca_alpha_levels(
    z0: float, acceleration: float, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """BCa percentile levels ``(α1, α2)`` from bias and acceleration."""

    alpha = (1.0 - confidence) / 2.0
    z_lo = float(norm.ppf(alpha))
    z_hi = float(norm.ppf(1.0 - alpha))

    def _adj(z: float) -> float:
        denom = 1.0 - acceleration * (z0 + z)
        if denom == 0.0:
            return alpha if z < 0.0 else 1.0 - alpha
        return float(norm.cdf(z0 + (z0 + z) / denom))

    a1 = min(1.0, max(0.0, _adj(z_lo)))
    a2 = min(1.0, max(0.0, _adj(z_hi)))
    if a1 > a2:
        a1, a2 = a2, a1
    return a1, a2


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    rng: np.random.Generator | int | None = 0,
    statistic: Statistic = mean_statistic,
) -> BootstrapCI:
    """95% BCa CI for ``statistic`` (default: the mean). Percentile fallback."""

    x = np.asarray(list(values), dtype=np.float64)
    n = int(x.size)
    if n < 1:
        raise ValueError("cannot bootstrap an empty series")
    theta = statistic(x)
    if n == 1 or float(np.max(x)) == float(np.min(x)):
        return BootstrapCI(low=theta, high=theta, method="degenerate")
    generator = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )
    draws = generator.integers(0, n, size=(n_resamples, n))
    thetas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        thetas[i] = statistic(x[draws[i]])
    thetas.sort()
    prop_less = float(np.mean(thetas < theta))
    if prop_less <= 0.0 or prop_less >= 1.0:
        return _percentile_ci(thetas, confidence)
    z0 = float(norm.ppf(prop_less))
    if not np.isfinite(z0):
        return _percentile_ci(thetas, confidence)
    acc = jackknife_acceleration(x, statistic)
    a1, a2 = bca_alpha_levels(z0, acc, confidence)
    return BootstrapCI(
        low=_quantile(thetas, a1),
        high=_quantile(thetas, a2),
        method="bca",
    )


def pick_reference(methods: Sequence[str]) -> str | None:
    for name in REFERENCE_CANDIDATES:
        if name in methods:
            return name
    return None


def aligned_pairs(
    left: Mapping[K, float], right: Mapping[K, float]
) -> tuple[list[float], list[float]]:
    keys = sorted(set(left) & set(right), key=lambda k: str(k))
    return [float(left[k]) for k in keys], [float(right[k]) for k in keys]


def apply_holm(
    rows: Sequence[MethodComparison], *, alpha: float = ALPHA
) -> list[MethodComparison]:
    """Adjust every raw p-value in ``rows`` as one Holm family."""

    pending = [i for i, row in enumerate(rows) if row.p_raw is not None]
    if not pending:
        return list(rows)
    raws: list[float] = []
    for i in pending:
        p_raw = rows[i].p_raw
        assert p_raw is not None
        raws.append(p_raw)
    adjusted = holm_bonferroni(raws)
    out = list(rows)
    for idx, p_holm in zip(pending, adjusted, strict=True):
        row = out[idx]
        out[idx] = MethodComparison(
            method=row.method,
            mean=row.mean,
            std=row.std,
            n=row.n,
            ci=row.ci,
            p_raw=row.p_raw,
            p_holm=p_holm,
            significant=p_holm < alpha,
            cliffs_delta=row.cliffs_delta,
        )
    return out


def compare_methods(
    grouped: Mapping[str, Mapping[K, float]],
    *,
    reference: str | None = None,
    alpha: float = ALPHA,
    rng: np.random.Generator | int | None = 0,
    holm: bool = True,
) -> list[MethodComparison]:
    """Per-method mean, BCa CI, and tests against ``reference``.

    Pass ``holm=False`` when the caller will Holm-correct a larger family
    (several metrics in one table).
    """

    methods = list(grouped)
    ref_name = reference if reference is not None else pick_reference(methods)
    ref_map = grouped.get(ref_name) if ref_name is not None else None

    rows: list[MethodComparison] = []
    for method, mapping in grouped.items():
        values = [float(v) for v in mapping.values()]
        mean, std, n = sample_mean_std(values)
        ci = bootstrap_mean_ci(values, rng=rng)
        p_raw: float | None = None
        delta: float | None = None
        if ref_map is not None and method != ref_name and len(ref_map) > 0:
            other, ref_vals = aligned_pairs(mapping, ref_map)
            if other:
                p_raw = wilcoxon_signed_rank(other, ref_vals).p_value
                delta = cliffs_delta(other, ref_vals)
        rows.append(
            MethodComparison(
                method=method,
                mean=mean,
                std=std,
                n=n,
                ci=ci,
                p_raw=p_raw,
                p_holm=None,
                significant=False,
                cliffs_delta=delta,
            )
        )
    if holm:
        return apply_holm(rows, alpha=alpha)
    return rows


def _percentile_ci(
    sorted_thetas: NDArray[np.float64], confidence: float
) -> BootstrapCI:
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        low=_quantile(sorted_thetas, alpha),
        high=_quantile(sorted_thetas, 1.0 - alpha),
        method="percentile",
    )


def _quantile(sorted_thetas: NDArray[np.float64], q: float) -> float:
    return float(np.quantile(sorted_thetas, q, method="linear"))
