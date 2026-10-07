"""U1 statistics: hand-computed Wilcoxon, Cliff, Holm, and BCa pieces."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml
from pact.config import EvalConfig, config_from_mapping
from pact.eval.export import TABLE_COLUMNS, write_multi_metric_table, write_table_csv
from pact.eval.runner import N_SEEDS_DEFAULT, RunMetrics
from pact.eval.stats import (
    average_ranks,
    bca_alpha_levels,
    bootstrap_mean_ci,
    cliffs_delta,
    holm_bonferroni,
    jackknife_acceleration,
    signed_rank_sums,
    wilcoxon_signed_rank,
)


def test_default_seed_count_is_ten() -> None:
    raw = yaml.safe_load(Path("configs/default.yaml").read_text())
    assert raw["eval"]["n_seeds"] == 10
    assert EvalConfig().n_seeds == 10
    assert N_SEEDS_DEFAULT == 10
    cfg = config_from_mapping({"eval": {"n_seeds": 10}}, tau_c_s=1.0)
    assert cfg.eval.n_seeds == 10


def test_average_ranks_splits_ties() -> None:
    # Values 0.1, 0.2, 0.3, 0.4, 0.4 → ranks 1, 2, 3, 4.5, 4.5
    assert average_ranks([0.1, 0.2, 0.3, 0.4, 0.4]) == (1.0, 2.0, 3.0, 4.5, 4.5)


def test_wilcoxon_signed_rank_sums_with_one_tie() -> None:
    # d = x − y = 0.2, 0.4, 0.4, 0.1, −0.3
    # |d| ranks: 0.1→1, 0.2→2, 0.3→3, 0.4→4.5, 0.4→4.5
    # W+ = 2 + 4.5 + 4.5 + 1 = 12; W− = 3
    x = [1.2, 1.5, 1.8, 2.0, 2.2]
    y = [1.0, 1.1, 1.4, 1.9, 2.5]
    w_plus, w_minus, n_nonzero = signed_rank_sums(x, y)
    assert n_nonzero == 5
    assert w_plus == pytest.approx(12.0)
    assert w_minus == pytest.approx(3.0)


def test_wilcoxon_exact_p_all_positive_n5() -> None:
    # n=5, no zeros, no ties. Only 2 of 2^5 sign patterns put all mass on
    # one side, so two-sided p = 2/32.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.0, 0.0, 0.0, 0.0, 0.0]
    result = wilcoxon_signed_rank(x, y)
    assert result.statistic == pytest.approx(15.0)  # 1+2+3+4+5
    assert result.w_minus == pytest.approx(0.0)
    assert result.p_value == pytest.approx(2.0 / 32.0)


def test_wilcoxon_exact_p_all_positive_n10() -> None:
    # n=10 is the smallest n where 2/2^n < 0.05. W+ = 55.
    x = [float(i) for i in range(1, 11)]
    y = [0.0] * 10
    result = wilcoxon_signed_rank(x, y)
    assert result.statistic == pytest.approx(55.0)
    assert result.p_value == pytest.approx(2.0 / 1024.0)


def test_wilcoxon_all_zeros_is_p_one() -> None:
    result = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result.n_nonzero == 0
    assert result.p_value == pytest.approx(1.0)


def test_cliffs_delta_hand_counted() -> None:
    # 3×3 pairs: one tie (3 vs 3), eight x<y, zero x>y → (0−8)/9
    assert cliffs_delta([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]) == pytest.approx(-8.0 / 9.0)
    # Every x exceeds every y → 4/4
    assert cliffs_delta([5.0, 6.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert cliffs_delta([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.0)


def test_holm_bonferroni_three_pvalues() -> None:
    # Sorted: 0.01, 0.03, 0.04. Raw Holm: 3·0.01=0.03, 2·0.03=0.06, 1·0.04=0.04.
    # Monotone: 0.03, max(0.03,0.06)=0.06, max(0.06,0.04)=0.06.
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[2] == pytest.approx(0.06)


def test_holm_bonferroni_caps_at_one() -> None:
    adjusted = holm_bonferroni([0.9, 0.8])
    assert adjusted == (1.0, 1.0)


def test_jackknife_acceleration_for_the_mean() -> None:
    # x=(1,2,6). Leave-one-out means 4, 3.5, 1.5. θ̄=3.
    # U = (−1, −0.5, 1.5). a = ΣU³ / (6 (ΣU²)^{3/2})
    # ΣU³ = 2.25; ΣU² = 3.5.
    expected = 2.25 / (6.0 * (3.5**1.5))
    assert jackknife_acceleration([1.0, 2.0, 6.0]) == pytest.approx(expected)
    assert jackknife_acceleration([4.0, 4.0, 4.0]) == pytest.approx(0.0)


def test_bca_levels_reduce_to_percentile_when_unbiased() -> None:
    # z0=0, a=0 → α1=Φ(z_{α})=α, α2=1−α for a 95% interval.
    a1, a2 = bca_alpha_levels(0.0, 0.0, 0.95)
    assert a1 == pytest.approx(0.025)
    assert a2 == pytest.approx(0.975)


def test_bootstrap_ci_is_degenerate_when_values_are_constant() -> None:
    ci = bootstrap_mean_ci([3.0, 3.0, 3.0], n_resamples=50)
    assert ci.method == "degenerate"
    assert ci.low == pytest.approx(3.0)
    assert ci.high == pytest.approx(3.0)


def _metric(
    method: str, seed: int, sla: float, recall: float = 0.5
) -> RunMetrics:
    return RunMetrics(
        method=method,
        seed=seed,
        dataset="d1",
        sla_violation=sla,
        over_provision=0.1,
        action_count=1,
        accuracy=0.5,
        precision=0.5,
        recall=recall,
        f1=0.5,
        cost_per_1k=1.0,
        mean_utilisation=0.5,
    )


def test_table_csv_has_ci_and_significance(tmp_path: Path) -> None:
    # 10 paired seeds, LSTM SLA strictly above PACT → Wilcoxon p = 2/1024.
    # One comparison in the family, so Holm leaves p unchanged. Cliff's δ = 1.
    metrics = []
    for seed in range(10):
        metrics.append(_metric("pact", seed, 0.10 + 0.001 * seed))
        metrics.append(_metric("lstm", seed, 0.20 + 0.001 * seed))
    path = tmp_path / "table6_baselines.csv"
    write_table_csv(path, metrics, metric_name="sla_violation")
    rows = list(csv.DictReader(path.open()))
    assert tuple(rows[0].keys()) == TABLE_COLUMNS
    by_method = {row["method"]: row for row in rows}
    assert by_method["pact"]["p_holm"] == ""
    assert by_method["pact"]["significant"] == ""
    assert by_method["pact"]["cliffs_delta"] == ""
    assert float(by_method["lstm"]["p_holm"]) == pytest.approx(2.0 / 1024.0)
    assert by_method["lstm"]["significant"] == "*"
    assert float(by_method["lstm"]["cliffs_delta"]) == pytest.approx(1.0)
    assert by_method["lstm"]["ci_95"].startswith("[")
    assert int(by_method["lstm"]["n_runs"]) == 10


def test_multi_metric_holm_family_is_the_whole_table(tmp_path: Path) -> None:
    # Two metrics, each with p_raw = 2/1024. Holm m=2:
    # 2·(2/1024)=4/1024, then 1·(2/1024)=2/1024 → monotone 4/1024, 4/1024.
    metrics = []
    for seed in range(10):
        metrics.append(
            _metric("pact", seed, 0.10 + 0.001 * seed, recall=0.90 + 0.001 * seed)
        )
        metrics.append(
            _metric("lstm", seed, 0.20 + 0.001 * seed, recall=0.50 + 0.001 * seed)
        )
    path = tmp_path / "table.csv"
    write_multi_metric_table(path, metrics, ("sla_violation", "recall"))
    rows = [r for r in csv.DictReader(path.open()) if r["method"] == "lstm"]
    assert len(rows) == 2
    expected = 4.0 / 1024.0
    for row in rows:
        assert float(row["p_holm"]) == pytest.approx(expected)
        assert row["significant"] == "*"
