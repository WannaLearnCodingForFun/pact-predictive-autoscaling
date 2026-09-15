"""Phase 9.4: single-component ablations and table 7 export."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from pact.actuation.dry_run import DryRunBackend
from pact.config import DEFAULT_CONFIG_PATH, load_config, load_config_overlay
from pact.eval.ablation import (
    ABLATION_NAMES,
    apply_ablation,
    export_table7,
    load_ablation,
    run_ablation_matrix,
    split_then_ablate,
)
from pact.eval.datasets import generate_burst
from pact.eval.export import write_multi_metric_table
from pact.eval.splits import SplitGuard, TestSplitReusedError
from pact.forecast.losses import AsymmetricHorizonHuber
from pact.loop import ControlLoop, RepeatObservationForecaster
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import Observation, SimCollector
from pact.telemetry.features import FEATURE_DIM, feature_dim, feature_vector

from tests.helpers import make_config


def test_six_ablation_configs_exist_and_are_overlays() -> None:
    for name in ABLATION_NAMES:
        path = Path("configs/ablations") / f"{name}.yaml"
        assert path.is_file(), name
        raw = yaml.safe_load(path.read_text())
        assert isinstance(raw, dict)
        dumped = yaml.dump(raw)
        assert "tau_c_s" not in dumped


def test_each_ablation_disables_exactly_one_component(tmp_path: Path) -> None:
    cold = tmp_path / "cold_start.json"
    cold.write_text('{"mean_s": 10.0, "std_s": 0.1}\n')
    full = load_config(DEFAULT_CONFIG_PATH, cold_start_path=cold)

    knocked = load_ablation("no_asym_loss", cold_start_path=cold)
    overlay = load_config_overlay(
        DEFAULT_CONFIG_PATH,
        Path("configs/ablations/no_asym_loss.yaml"),
        cold_start_path=cold,
    )
    assert overlay.forecast.kappa == 1.0
    assert knocked.forecast.kappa == 1.0
    assert full.forecast.kappa != 1.0
    assert knocked.forecast.beta == full.forecast.beta
    assert knocked.control.s_churn == full.control.s_churn
    assert knocked.ablation.skip_mpc is False

    mpc = load_ablation("no_mpc", cold_start_path=cold)
    assert mpc.ablation.skip_mpc is True
    assert mpc.forecast.kappa == full.forecast.kappa
    assert mpc.control.s_churn == full.control.s_churn

    churn = load_ablation("no_churn", cold_start_path=cold)
    assert churn.control.s_churn == 0.0
    assert churn.ablation.skip_mpc is False
    assert churn.forecast.kappa == full.forecast.kappa

    cold_cfg = load_ablation("no_coldstart", cold_start_path=cold)
    assert cold_cfg.control.tau_c_s == 0.0
    assert cold_cfg.ablation.zero_tau_c is True
    assert full.control.tau_c_s == 10.0
    assert cold_cfg.forecast.kappa == full.forecast.kappa

    margin = load_ablation("no_adaptive_margin", cold_start_path=cold)
    assert margin.ablation.freeze_gamma is True
    assert margin.ablation.skip_mpc is False
    assert margin.control.tau_c_s == full.control.tau_c_s

    rho_cfg = load_ablation("no_rho_channel", cold_start_path=cold)
    assert rho_cfg.ablation.include_rho is False
    assert rho_cfg.ablation.freeze_gamma is False
    assert rho_cfg.forecast.kappa == full.forecast.kappa


def test_no_asym_loss_is_symmetric_huber() -> None:
    cfg = apply_ablation(make_config(tau_c_s=10.0), "no_asym_loss")
    assert cfg.forecast.kappa == 1.0
    loss = AsymmetricHorizonHuber.from_config(cfg.forecast)
    pred = torch.tensor([[[0.0, 0.0]]])
    low = loss(pred, torch.tensor([[[1.0, 0.0]]]))
    high = loss(pred, torch.tensor([[[-1.0, 0.0]]]))
    assert float(low) == pytest.approx(float(high))


def test_no_mpc_skips_module_4() -> None:
    cfg = apply_ablation(make_config(dt=5.0, tau_c_s=10.0), "no_mpc")
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    assert cfg.ablation.skip_mpc is True
    n_ticks = 24
    sim = PoolSimulator(cfg, n_initial=2)
    loop = ControlLoop(
        cfg,
        collector=SimCollector(sim),
        backend=DryRunBackend(n_initial=2),
        forecaster=RepeatObservationForecaster(cfg.forecast.horizon),
        simulator=sim,
        arrival_rates=[20.0] * 12 + [80.0] * 12,
        n_initial=2,
        sleep=lambda _s: None,
    )
    result = loop.run(n_ticks)
    assert all(tick.mpc_u is None for tick in result.ticks)
    assert all(tick.gate is None for tick in result.ticks)
    assert all(tick.safety.active is False for tick in result.ticks)


def test_no_churn_and_no_mpc_increase_actions_vs_full() -> None:
    base = make_config(dt=5.0, tau_c_s=10.0)
    base = replace(base, telemetry=replace(base.telemetry, window=8))
    # Keep utilisation below the safety ceiling so Module 4 is doing the work.
    base = replace(base, control=replace(base.control, u_emergency=2.0))
    arrivals = [25.0] * 10 + [70.0] * 10 + [25.0] * 16
    traces = run_ablation_matrix(
        base,
        arrivals,
        seeds=(0,),
        names=("no_mpc", "no_churn"),
        include_full=True,
    )
    by_name = {t.method: t for t in traces}

    def n_changes(name: str) -> int:
        series = by_name[name].n
        return sum(1 for a, b in zip(series, series[1:], strict=False) if a != b)

    assert n_changes("no_mpc") >= n_changes("full")
    assert apply_ablation(base, "no_churn").control.s_churn == 0.0
    # no_churn can take larger steps (fewer ticks with ΔN ≠ 0). Compare
    # total absolute replica movement, which the churn penalty is meant to cut.
    def travel(name: str) -> int:
        series = by_name[name].n
        return sum(abs(b - a) for a, b in zip(series, series[1:], strict=False))

    assert travel("no_churn") >= travel("full")


def test_no_coldstart_zeroes_plant_delay() -> None:
    cfg = apply_ablation(make_config(dt=5.0, tau_c_s=10.0), "no_coldstart")
    assert cfg.control.tau_c_s == 0.0
    sim = PoolSimulator(cfg, n_initial=2)
    assert sim.delay_ticks == 0


def test_no_adaptive_margin_freezes_gamma() -> None:
    cfg = apply_ablation(make_config(dt=5.0, tau_c_s=5.0), "no_adaptive_margin")
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=4))
    n_ticks = 20
    sim = PoolSimulator(cfg, n_initial=2)
    loop = ControlLoop(
        cfg,
        collector=SimCollector(sim),
        backend=DryRunBackend(n_initial=2),
        forecaster=RepeatObservationForecaster(cfg.forecast.horizon),
        simulator=sim,
        arrival_rates=[30.0] * n_ticks,
        n_initial=2,
        sleep=lambda _s: None,
    )
    result = loop.run(n_ticks)
    gammas = [tick.drift.gamma for tick in result.ticks]
    assert gammas
    assert all(g == pytest.approx(gammas[0]) for g in gammas)


def test_no_rho_drops_intensity_channel() -> None:
    obs = Observation(u=0.4, r=0.5, lam=40.0, ell_p95=100.0, n=2.0, t_s=0.0)
    full = feature_vector(obs, include_rho=True)
    dropped = feature_vector(obs, include_rho=False)
    assert len(full) == FEATURE_DIM
    assert len(dropped) == feature_dim(include_rho=False) == 7
    assert full[5] == pytest.approx(20.0)
    assert 20.0 not in dropped
    cfg = apply_ablation(make_config(tau_c_s=10.0), "no_rho_channel")
    assert cfg.ablation.include_rho is False


def test_table7_writes_computed_mean_pm_std(tmp_path: Path) -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    burst = generate_burst(
        "impulse", n_ticks=28, dt=5.0, rise_ticks=1, base=15.0, peak=50.0
    )
    traces = run_ablation_matrix(
        cfg, burst.arrival_rates, seeds=(0, 1), names=("no_mpc",), include_full=True
    )
    path = tmp_path / "table7_ablation.csv"
    export_table7(path, traces, cfg)
    rows = list(csv.DictReader(path.open()))
    assert rows
    assert {row["method"] for row in rows} == {"full", "no_mpc"}
    assert all("±" in row["mean_pm_std"] for row in rows)
    assert all(int(row["n_runs"]) == 2 for row in rows)


def test_table7_refuses_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no computed"):
        write_multi_metric_table(
            tmp_path / "table7_ablation.csv", [], ("sla_violation",)
        )


def test_ablation_test_split_read_once() -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    burst = generate_burst(
        "ramp", n_ticks=40, dt=5.0, rise_ticks=4, base=10.0, peak=40.0
    )
    guard = SplitGuard()
    split_then_ablate(cfg, burst, guard=guard, seeds=(0,))
    with pytest.raises(TestSplitReusedError):
        split_then_ablate(cfg, burst, guard=guard, seeds=(0,))
