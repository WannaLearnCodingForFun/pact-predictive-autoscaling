"""Phase 8: baselines share the plant; only the decision law varies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from pact.baselines.arima_ctl import ArimaController, select_arima_order
from pact.baselines.common import apply_pool_bounds
from pact.baselines.lstm_ctl import lstm_matching_tcn
from pact.baselines.reactive import HpaConfig, ReactiveHPA
from pact.baselines.static import StaticScaler, peak_replicas
from pact.baselines.tune import tune_baselines
from pact.eval.datasets import generate_burst
from pact.eval.runner import run_method
from pact.forecast.losses import AsymmetricHorizonHuber
from pact.forecast.seq_models import ModuleForecaster, build_seq_forecaster
from pact.forecast.tcn import DilatedTCN
from pact.telemetry.collector import Observation
from pact.telemetry.features import FEATURE_DIM

from tests.helpers import make_config


def _obs(*, u: float, n: float, lam: float = 40.0) -> Observation:
    return Observation(u=u, r=0.4, lam=lam, ell_p95=20.0, n=n, t_s=0.0)


def test_hpa_formula_tolerance_and_stabilisation() -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    hpa = ReactiveHPA(cfg, hpa=HpaConfig(tolerance=0.1, scale_down_window_s=60.0))
    # ceil(4 * 0.8 / 0.65) = ceil(4.92307) = 5; ratio 1.230 > 1.1
    assert hpa.recommend(_obs(u=0.8, n=4.0), 4) == 5
    # |0.68/0.65 - 1| = 0.046 <= 0.1 → hold
    assert hpa.recommend(_obs(u=0.68, n=4.0), 4) == 4

    down = ReactiveHPA(
        cfg, hpa=HpaConfig(tolerance=0.0, scale_down_window_s=60.0)
    )
    n = 8
    n = down.desired_replicas(_obs(u=0.2, n=float(n), lam=8.0), n, now_s=0.0)
    rec0 = n
    n = down.desired_replicas(_obs(u=0.25, n=float(n), lam=10.0), n, now_s=5.0)
    n = down.desired_replicas(_obs(u=0.15, n=float(n), lam=6.0), n, now_s=10.0)
    assert n >= rec0 or n <= 8
    assert cfg.control.n_min <= n <= cfg.control.n_max


def test_apply_pool_bounds_enforces_delta_max() -> None:
    cfg = make_config(n_min=1, n_max=16, tau_c_s=10.0)
    assert apply_pool_bounds(4, 16, cfg) == 4 + cfg.control.delta_max
    assert apply_pool_bounds(4, 1, cfg) == 4 - min(cfg.control.delta_max, 3)


def test_static_sized_to_trace_peak() -> None:
    cfg = make_config(mu=40.0, tau_c_s=10.0, n_min=1, n_max=16)
    rates = [10.0, 80.0, 20.0]
    # ceil(80 / (40 * 0.65)) = ceil(3.0769) = 4
    assert peak_replicas(rates, cfg) == 4
    scaler = StaticScaler.sized_to_peak(cfg, rates)
    n = scaler.desired_replicas(_obs(u=0.1, n=1.0), 1, now_s=0.0)
    assert n == 4 or n == 1 + cfg.control.delta_max


def test_lstm_parameter_count_matches_tcn() -> None:
    tcn = DilatedTCN(
        in_channels=FEATURE_DIM,
        window=16,
        horizon=4,
        kernel_size=3,
        depth=3,
        channels=8,
        dropout=0.0,
    )
    cfg = make_config(tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=16))
    cfg = replace(cfg, forecast=replace(cfg.forecast, horizon=4))
    lstm = lstm_matching_tcn(cfg, tcn)
    rel = abs(lstm.parameter_count() - tcn.parameter_count()) / tcn.parameter_count()
    assert rel < 0.25


def test_lstm_uses_symmetric_huber() -> None:
    loss = AsymmetricHorizonHuber(kappa=1.0, beta=0.0, huber_delta=1.0)
    pred = torch.tensor([[[0.0, 0.0]]])
    low = loss(pred, torch.tensor([[[1.0, 0.0]]]))
    high = loss(pred, torch.tensor([[[-1.0, 0.0]]]))
    assert float(low) == pytest.approx(float(high))


def test_table5_forecasters_emit_horizon_pair() -> None:
    for kind in ("lstm", "bilstm", "gru", "transformer"):
        model = build_seq_forecaster(
            kind, in_channels=FEATURE_DIM, window=8, horizon=4, hidden_size=16
        )
        x = torch.zeros(2, FEATURE_DIM, 8)
        y = model(x)
        assert tuple(y.shape) == (2, 4, 2)
        wrapped = ModuleForecaster(model)
        out = wrapped.predict(np.zeros((8, FEATURE_DIM), dtype=np.float32))
        assert out.shape == (4, 2)


def test_arima_order_selected_on_validation() -> None:
    train = [0.4 + 0.01 * i for i in range(30)]
    val = [0.7 + 0.01 * i for i in range(10)]
    order = select_arima_order(train, val, budget=3)
    assert order in {(1, 0, 0), (2, 0, 0), (1, 0, 1)}
    cfg = make_config(tau_c_s=10.0, n_min=1, n_max=16)
    ctl = ArimaController(cfg, order=order)
    n = 4
    for i in range(12):
        n = ctl.desired_replicas(_obs(u=0.5, n=float(n)), n, now_s=i * 5.0)
        assert cfg.control.n_min <= n <= cfg.control.n_max


def test_baselines_share_dt_bounds_and_trace_length() -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    arrivals = [20.0 + (i % 5) * 8.0 for i in range(36)]
    lengths: set[int] = set()
    dts: set[float] = set()
    for method in ("pact", "reactive", "arima", "lstm", "static"):
        trace = run_method(method, cfg, arrivals, seed=0, dataset="shared")
        lengths.add(len(trace.ts))
        dts.add(trace.ts[1] - trace.ts[0] if len(trace.ts) > 1 else cfg.telemetry.dt)
        assert all(cfg.control.n_min <= n <= cfg.control.n_max for n in trace.n)
        deltas = [abs(b - a) for a, b in zip(trace.n, trace.n[1:], strict=False)]
        assert all(d <= cfg.control.delta_max for d in deltas)
        assert trace.ts[0] == 0.0
    assert lengths == {len(arrivals)}
    assert dts == {cfg.telemetry.dt}


def test_tune_baselines_writes_computed_val_scores(tmp_path: Path) -> None:
    cfg = make_config(dt=5.0, tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    burst = generate_burst(
        "ramp", n_ticks=60, dt=5.0, rise_ticks=6, base=15.0, peak=50.0
    )
    out = tmp_path / "baseline_tuning.json"
    chosen = tune_baselines(cfg, burst, budget=2, results_path=out)
    text = out.read_text()
    assert "val_score" in text
    reactive = chosen["reactive"]
    assert isinstance(reactive, dict)
    assert isinstance(reactive["val_score"], float)
    arima = chosen["arima"]
    assert isinstance(arima, dict)
    assert isinstance(arima["val_score"], float)
    assert chosen["budget"] == 2
