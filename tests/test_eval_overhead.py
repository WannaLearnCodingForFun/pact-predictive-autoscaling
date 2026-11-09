"""U6 overhead: hand-computed percentiles, miss rate, parameter bytes."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pact.actuation.dry_run import DryRunBackend
from pact.config import PactConfig
from pact.eval.overhead import (
    OVERHEAD_COLUMNS,
    ComposeSample,
    ProcessResources,
    count_parameters,
    deadline_miss_rate,
    float32_parameter_bytes,
    int8_packed_bytes,
    latency_rows,
    measure_controller_overhead,
    parse_docker_cpu_percent,
    parse_docker_mem_bytes,
    percentile,
    resource_rows,
    write_overhead_csv,
)
from pact.loop import ControlLoop, RepeatObservationForecaster, TickTiming
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import SimCollector

from tests.helpers import make_config


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
        ),
    )


def test_percentile_linear_interpolation_hand_computed() -> None:
    # n=5, sorted [0.10, 0.20, 0.30, 0.40, 0.50]
    # rank = q/100 * 4
    series = (0.10, 0.20, 0.30, 0.40, 0.50)
    assert percentile(series, 0) == pytest.approx(0.10)
    assert percentile(series, 100) == pytest.approx(0.50)
    assert percentile(series, 50) == pytest.approx(0.30)
    # 0.95 * 4 = 3.8 → 0.40 + 0.8 * 0.10 = 0.48
    assert percentile(series, 95) == pytest.approx(0.48)
    # 0.99 * 4 = 3.96 → 0.40 + 0.96 * 0.10 = 0.496
    assert percentile(series, 99) == pytest.approx(0.496)


def test_deadline_miss_rate_hand_computed() -> None:
    slips = (0.0, 0.0, 0.1, 0.0, 0.2)
    assert deadline_miss_rate(slips) == pytest.approx(2.0 / 5.0)
    assert deadline_miss_rate((0.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_linear_parameter_count_and_packed_sizes() -> None:
    # nn.Linear(5, 2) with bias: 5*2 + 2 = 12
    layer = torch.nn.Linear(5, 2, bias=True)
    assert count_parameters(layer) == 12
    assert float32_parameter_bytes(layer) == 12 * 4
    assert int8_packed_bytes(layer) == 12


def test_docker_stats_parsers() -> None:
    assert parse_docker_mem_bytes("12.5MiB / 2GiB") == int(12.5 * 1024**2)
    assert parse_docker_cpu_percent("12.50%") == pytest.approx(12.5)


def test_resource_share_hand_computed() -> None:
    before = ProcessResources(cpu_user_s=1.0, cpu_system_s=0.5, rss_bytes=1000)
    after = ProcessResources(cpu_user_s=1.2, cpu_system_s=0.6, rss_bytes=4000)
    testbed = ComposeSample(cpu_percent=45.0, rss_bytes=6000, n_containers=2)
    rows = { (r.name, r.stat): r for r in resource_rows(before, after, 2.0, testbed) }
    assert rows[("controller_cpu", "value")].value == pytest.approx(0.3)
    assert rows[("controller_cpu_percent", "value")].value == pytest.approx(15.0)
    # 4000 / (4000 + 6000) = 0.4
    assert rows[("controller_rss_share", "value")].value == pytest.approx(0.4)
    # 15 / (15 + 45) = 0.25
    assert rows[("controller_cpu_share", "value")].value == pytest.approx(0.25)


def test_share_omitted_when_testbed_unsampled() -> None:
    before = ProcessResources(0.0, 0.0, 100)
    after = ProcessResources(0.1, 0.0, 200)
    names = {r.name for r in resource_rows(before, after, 1.0, None)}
    assert "controller_rss_share" not in names
    assert "controller_cpu_share" not in names
    assert "controller_rss" in names


def test_latency_rows_percentiles() -> None:
    timings = tuple(
        TickTiming(
            scrape_s=0.10 + 0.10 * i,
            features_s=0.01,
            inference_s=0.0,
            mapping_s=0.01,
            mpc_s=0.01,
            actuation_s=0.01,
        )
        for i in range(5)
    )
    rows = [r for r in latency_rows(timings) if r.name == "scrape"]
    by_stat = {r.stat: r.value for r in rows}
    assert by_stat["p50"] == pytest.approx(0.30)
    assert by_stat["p95"] == pytest.approx(0.48)
    assert by_stat["p99"] == pytest.approx(0.496)


def test_measure_writes_overhead_csv_without_inventing_share(
    tmp_path: Path,
) -> None:
    cfg = _small_cfg()
    rows = measure_controller_overhead(
        cfg,
        n_ticks=20,
        seed=0,
        testbed_sampler=lambda: None,
    )
    names = {r.name for r in rows}
    for stage in (
        "scrape",
        "features",
        "inference",
        "mapping",
        "mpc",
        "actuation",
    ):
        assert stage in names
    assert "parameter_count" in names
    assert "float32_bytes" in names
    assert "int8_packed_bytes" in names
    assert "miss_rate" in names
    assert "controller_rss_share" not in names
    dest = tmp_path / "overhead.csv"
    write_overhead_csv(dest, rows)
    with dest.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert list(table[0].keys()) == list(OVERHEAD_COLUMNS)
    assert table
    miss = next(r for r in table if r["name"] == "miss_rate")
    assert float(miss["value"]) == pytest.approx(0.0)


def test_loop_timings_match_tick_count() -> None:
    cfg = _small_cfg()
    n_ticks = 12
    sim = PoolSimulator(cfg, n_initial=1)
    loop = ControlLoop(
        cfg,
        collector=SimCollector(sim),
        backend=DryRunBackend(n_initial=1),
        forecaster=RepeatObservationForecaster(cfg.forecast.horizon),
        simulator=sim,
        arrival_rates=[20.0] * n_ticks,
        n_initial=1,
        sleep=lambda _s: None,
    )
    result = loop.run(n_ticks)
    assert len(result.timings) == n_ticks
    assert len(result.deadline_slips_s) == n_ticks
    for timing in result.timings:
        assert timing.scrape_s >= 0.0
        assert timing.features_s >= 0.0
        assert timing.inference_s >= 0.0
        assert timing.mapping_s >= 0.0
        assert timing.mpc_s >= 0.0
        assert timing.actuation_s >= 0.0
    # Window is 12; the last push fills it and inference may run.
    assert all(t.inference_s == 0.0 for t in result.timings[:-1])
