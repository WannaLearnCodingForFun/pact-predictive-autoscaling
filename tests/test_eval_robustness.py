"""U4 robustness: hand-computed crossing, corruption algebra, safety off."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pact.config import PactConfig
from pact.eval.datasets import parse_bitbrains_faststorage, parse_fifa_requests
from pact.eval.robustness import (
    ROBUSTNESS_COLUMNS,
    CorruptForecaster,
    crossing_magnitude,
    disable_safety,
    run_corrupted_pact,
    run_robustness,
    shift_magnitude,
    write_robustness_csv,
)
from pact.eval.robustness import (
    main as robustness_main,
)
from pact.loop import RepeatObservationForecaster

from tests.helpers import make_config


class CountingForecaster:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        del window
        self.calls += 1
        return np.full((3, 2), float(self.calls), dtype=np.float32)


class ConstForecaster:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def predict(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        del window
        return np.full((2, 2), self.value, dtype=np.float32)


def _small_cfg() -> PactConfig:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=6)
    return replace(
        cfg,
        telemetry=replace(cfg.telemetry, window=12),
        forecast=replace(
            cfg.forecast,
            horizon=4,
            kernel_size=3,
            depth=2,
            channels=8,
            dropout=0.0,
            batch_size=8,
        ),
    )


def test_crossing_interpolates_hand_computed() -> None:
    # PACT SLA 0.10, 0.20, 0.40 at mag 0, 1, 2. Reactive 0.25.
    # Crosses between 1 and 2: t = (0.25-0.20)/(0.40-0.20) = 0.25 → 1.25
    assert crossing_magnitude((0.0, 1.0, 2.0), (0.10, 0.20, 0.40), 0.25) == (
        pytest.approx(1.25)
    )


def test_crossing_is_first_magnitude_when_already_worse() -> None:
    assert crossing_magnitude((0.5, 1.0), (0.4, 0.5), 0.3) == pytest.approx(0.5)


def test_crossing_none_when_pact_never_exceeds_reactive() -> None:
    assert crossing_magnitude((0.0, 1.0, 2.0), (0.10, 0.12, 0.15), 0.20) is None


def test_noise_zero_is_identity() -> None:
    inner = ConstForecaster(0.4)
    wrapped = CorruptForecaster(inner, kind="noise", magnitude=0.0, seed=0)
    dummy = np.zeros((4, 2), dtype=np.float32)
    np.testing.assert_array_equal(wrapped.predict(dummy), inner.predict(dummy))


def test_noise_matches_seeded_normal() -> None:
    inner = ConstForecaster(1.0)
    wrapped = CorruptForecaster(inner, kind="noise", magnitude=0.2, seed=7)
    dummy = np.zeros((2, 2), dtype=np.float32)
    rng = np.random.default_rng(7)
    expected = 1.0 + rng.normal(0.0, 0.2, size=(2, 2))
    np.testing.assert_allclose(wrapped.predict(dummy), expected, rtol=0, atol=1e-7)


def test_bias_adds_constant_to_every_entry() -> None:
    inner = ConstForecaster(0.5)
    wrapped = CorruptForecaster(inner, kind="bias", magnitude=-0.25, seed=0)
    dummy = np.zeros((2, 2), dtype=np.float32)
    np.testing.assert_allclose(wrapped.predict(dummy), np.full((2, 2), 0.25))


def test_lag_returns_forecast_from_k_ticks_ago() -> None:
    inner = CountingForecaster()
    wrapped = CorruptForecaster(inner, kind="lag", magnitude=2, seed=0)
    dummy = np.zeros((3, 2), dtype=np.float32)
    # k=2: until 3 samples exist, emit the oldest (1)
    assert wrapped.predict(dummy)[0, 0] == pytest.approx(1.0)
    assert wrapped.predict(dummy)[0, 0] == pytest.approx(1.0)
    assert wrapped.predict(dummy)[0, 0] == pytest.approx(1.0)
    assert wrapped.predict(dummy)[0, 0] == pytest.approx(2.0)


def test_freeze_stops_calling_inner_after_first_predict() -> None:
    inner = CountingForecaster()
    wrapped = CorruptForecaster(inner, kind="freeze", magnitude=1.0, seed=0)
    dummy = np.zeros((3, 2), dtype=np.float32)
    first = wrapped.predict(dummy)
    second = wrapped.predict(dummy)
    np.testing.assert_array_equal(first, second)
    assert inner.calls == 1


def test_set_observation_forwards_to_inner() -> None:
    inner = RepeatObservationForecaster(3)
    wrapped = CorruptForecaster(inner, kind="bias", magnitude=0.0, seed=0)
    wrapped.set_observation(0.7, 0.2)
    dummy = np.zeros((4, 2), dtype=np.float32)
    out = wrapped.predict(dummy)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], 0.7)
    np.testing.assert_allclose(out[:, 1], 0.2)


def test_disable_safety_never_fires() -> None:
    cfg = disable_safety(make_config())
    assert math.isinf(cfg.control.u_emergency)


def test_shift_magnitude_is_abs_mean_difference() -> None:
    # mean(10, 20, 30) = 20; mean(0, 4) = 2; |20-2| = 18
    assert shift_magnitude((10.0, 20.0, 30.0), (0.0, 4.0)) == pytest.approx(18.0)


def test_corrupted_pact_is_deterministic() -> None:
    cfg = _small_cfg()
    arrivals = [20.0] * 20
    a, _ = run_corrupted_pact(
        cfg, arrivals, kind="noise", magnitude=0.3, seed=3, dataset="t"
    )
    b, _ = run_corrupted_pact(
        cfg, arrivals, kind="noise", magnitude=0.3, seed=3, dataset="t"
    )
    assert a == b


def test_robustness_csv_has_safety_on_and_off_and_reactive(
    tmp_path: Path,
) -> None:
    cfg = _small_cfg()
    arrivals = [15.0] * 10 + [50.0] * 14
    rows = run_robustness(
        cfg,
        arrivals,
        dataset="d3_step_test",
        seeds=(0,),
        noise_sigmas=(0.0, 0.4),
        biases=(-0.2, 0.2),
        lags=(0, 1),
    )
    kinds = {r.kind for r in rows}
    assert kinds >= {"noise", "bias", "lag", "freeze", "reactive"}
    assert "shift" not in kinds
    safety_flags = {
        (r.kind, r.safety_enabled)
        for r in rows
        if r.row_type == "point" and r.method == "pact"
    }
    for kind in ("noise", "bias", "lag", "freeze"):
        assert (kind, True) in safety_flags
        assert (kind, False) in safety_flags
    dest = tmp_path / "robustness.csv"
    write_robustness_csv(dest, rows)
    text = dest.read_text()
    header = text.splitlines()[0].split(",")
    assert header == list(ROBUSTNESS_COLUMNS)
    assert "reactive" in text
    assert "true" in text and "false" in text


def test_distribution_shift_uses_loaded_d1_d2_means(tmp_path: Path) -> None:
    cfg = _small_cfg()
    fifa_lines = [f"{i * 5} 25" for i in range(130)]
    d1 = parse_fifa_requests(fifa_lines, dt=5.0, subset_s=5.0 * 129)
    # Constant 25 requests / 5 s → 5 req/s
    assert shift_magnitude(d1.arrival_rates, d1.arrival_rates) == pytest.approx(0.0)
    bb_rows = [["Timestamp", "CPUUsage [%]"], ["0", "50"], ["600", "50"]]
    d2 = parse_bitbrains_faststorage(bb_rows, native_dt_s=300.0, target_dt_s=5.0)
    mag = shift_magnitude(d1.arrival_rates, d2.arrival_rates)
    # |5.0 - 0.5| = 4.5
    assert mag == pytest.approx(4.5)
    rows = run_robustness(
        cfg,
        d1.arrival_rates[:24],
        dataset=d1.name,
        seeds=(0,),
        noise_sigmas=(0.0,),
        biases=(0.0,),
        lags=(0,),
        d1=d1,
        d2=d2,
        max_epochs=2,
        patience=2,
    )
    shift_rows = [r for r in rows if r.kind == "shift" and r.row_type == "point"]
    assert shift_rows
    assert all(r.magnitude == pytest.approx(4.5) for r in shift_rows)
    assert {r.safety_enabled for r in shift_rows} == {True, False}
    write_robustness_csv(tmp_path / "robustness.csv", rows)


def test_cli_exits_clearly_without_cold_start(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="measure_cold_start") as caught:
        robustness_main(
            ["--cold-start", str(missing), "--results", str(tmp_path)]
        )
    assert caught.value.code != 0
