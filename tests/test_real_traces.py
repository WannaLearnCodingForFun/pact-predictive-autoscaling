"""Hand-computed checks for the WorldCup98 conversion and the trained arm.

No downloaded data is needed: record streams, traces and the model are built
here with values small enough to verify by hand.
"""

from __future__ import annotations

import gzip
import io
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
from experiments import harness as H
from experiments.lambda_forecast import lambda_features, predict_lambda
from experiments.make_real_traces import (
    RECORD,
    bin_counts,
    decode_and_validate,
    iter_records,
    parse_check,
    rescale_factor,
    split_indices,
)
from pact.telemetry.collector import Observation
from torch import nn


def _pack(rows: list[tuple[int, int, int, int, int, int, int, int]]) -> bytes:
    return b"".join(struct.pack(">IIIIBBBB", *r) for r in rows)


ROWS = [
    # timestamp, client, object, size, method, status, type, server
    (100, 1, 7, 500, 0, 2, 1, 3),
    (101, 2, 8, 0xFFFFFFFF, 0, 2, 1, 3),
    (99, 3, 9, 250, 1, 4, 2, 5),
]


def test_record_layout_is_twenty_big_endian_bytes() -> None:
    assert RECORD.itemsize == 20
    rec = np.frombuffer(_pack(ROWS[:1]), dtype=RECORD)
    assert int(rec["timestamp"][0]) == 100
    assert int(rec["size"][0]) == 500
    assert int(rec["server"][0]) == 3


def test_iter_records_chunks_and_rejects_partial_record() -> None:
    got = np.concatenate(list(iter_records(io.BytesIO(_pack(ROWS)))))
    assert got["timestamp"].tolist() == [100, 101, 99]
    try:
        list(iter_records(io.BytesIO(_pack(ROWS) + b"\x00\x01")))
    except ValueError:
        pass
    else:
        raise AssertionError("a trailing partial record must be rejected")


def test_decode_matches_checklog_semantics() -> None:
    # Bytes skip the 0xFFFFFFFF sentinel: 500 + 250 = 750. One out-of-order
    # record (99 after 101). Servers 3 (x2) and 5 (x1).
    ref = parse_check(
        "    Total Requests: 3\n       Total Bytes: 750\n          Min Time: 99\n"
        "          Max Time: 101\n      Out of Order: 1\n    Unique Servers: 2\n"
        "Server 3: 2\nServer 5: 1\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.gz"
        path.write_bytes(gzip.compress(_pack(ROWS)))
        stamps, info = decode_and_validate(path, ref)
        assert stamps.tolist() == [100, 101, 99]
        assert info["records"] == 3 and info["bytes"] == 750
        bad = dict(ref, **{"Total Bytes": 751})
        try:
            decode_and_validate(path, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a byte-count mismatch must be rejected")


def test_bin_counts_by_hand() -> None:
    # t0 = 100, dt = 5: bins [100,105) [105,110) [110,115).
    ts = np.array([100, 104, 105, 109, 109, 114], dtype=np.int64)
    assert bin_counts(ts, 100, 3, 5).tolist() == [2, 3, 1]
    try:
        bin_counts(np.array([115]), 100, 3, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("a timestamp past the last bin must be rejected")


def test_rescale_factor_by_hand() -> None:
    # Peak 1000 req/s onto 0.85 * 40 * 16 = 544 req/s.
    raw = np.array([10.0, 1000.0, 50.0])
    assert abs(rescale_factor(raw, mu=40.0, n_max=16, fraction=0.85) - 0.544) < 1e-12


def test_split_indices_match_window_split_rule() -> None:
    # int(103 * 0.6) = 61; int(103 * 0.2) = 20.
    assert split_indices(103, 0.6, 0.2) == (61, 81)


def test_lambda_features_by_hand() -> None:
    # t0 = 22:00 UTC = 00:00 Paris, so the first bin has phase 0.
    t0 = 22 * 3600
    f = lambda_features(np.array([3.0, 5.0]), t0, 21600.0, mean=1.0, std=2.0)
    assert f[:, 0].tolist() == [1.0, 2.0]
    assert abs(f[0, 1]) < 1e-7 and abs(f[0, 2] - 1.0) < 1e-7
    # +6 h = quarter day: sin = 1, cos = 0.
    assert abs(f[1, 1] - 1.0) < 1e-6 and abs(f[1, 2]) < 1e-6


class _LastValue(nn.Module):
    """Forecasts z(t+h) = z(t) + h * 0.5 on channel 0."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.h = torch.arange(1, horizon + 1, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, 0, -1:]
        z = last + 0.5 * self.h[None, :]
        return torch.stack([z, z], dim=-1)


def test_predict_lambda_reads_only_the_past() -> None:
    meta = {"window": 3, "horizon": 2, "lam_mean": 10.0, "lam_std": 4.0}
    feats = np.zeros((6, 3), dtype=np.float32)
    feats[:, 0] = [0, 1, 2, 3, 4, 5]
    pred = predict_lambda(_LastValue(2), meta, feats, np.array([2, 4]))
    # Origin 2: z = 2 -> lambda = 10 + 4 * (2 + 0.5) = 20, then 22.
    assert pred.tolist() == [[20.0, 22.0], [28.0, 30.0]]
    try:
        predict_lambda(_LastValue(2), meta, feats, np.array([1]))
    except ValueError:
        pass
    else:
        raise AssertionError("an origin without L ticks of history must be rejected")


def test_trained_forecaster_arithmetic_and_leakage_guard() -> None:
    cfg = H.base_config()
    horizon = cfg.forecast.horizon
    trace = np.arange(200, dtype=np.float64) * 2.0
    tmeta = {"t0_utc_epoch_s": 0, "dt_s": 5.0, "sha256_scaled_npy": "abc"}
    meta = {"window": 4, "horizon": horizon, "lam_mean": 0.0, "lam_std": 1.0,
            "trace_sha256": "abc", "val_end": 100}
    H._TRACE_CACHE["_unit"] = (trace, tmeta)
    H._MODEL_CACHE["_unit"] = (_LastValue(horizon), meta)
    try:
        fc = H.TrainedForecaster(trace_name="_unit", start=150, n=10, cfg=cfg,
                                 model_name="_unit")
        obs = Observation(u=0.0, r=0.0, lam=float(trace[153]), ell_p95=0.0, n=4.0)
        out = fc.predict(3, obs, n_ready=4)
        # lambda(153) = 306; lambda-hat(h=1) = 306.5 -> u = 306.5 / (40 * 4).
        u1 = 306.5 / 160.0
        assert abs(out[0][0] - u1) < 1e-9
        assert abs(out[0][1] - (0.20 + 0.50 * 1.0)) < 1e-12
        u_last = (306.0 + 0.5 * horizon) / 160.0
        assert abs(out[-1][0] - u_last) < 1e-9
        bad = Observation(u=0.0, r=0.0, lam=307.0, ell_p95=0.0, n=4.0)
        try:
            fc.predict(3, bad, n_ready=4)
        except ValueError:
            pass
        else:
            raise AssertionError("an arrival off the trace must be rejected")
        try:
            H.TrainedForecaster(trace_name="_unit", start=99, n=10, cfg=cfg,
                                model_name="_unit")
        except ValueError:
            pass
        else:
            raise AssertionError("a window before val_end must be rejected")
    finally:
        H._TRACE_CACHE.pop("_unit", None)
        H._MODEL_CACHE.pop("_unit", None)


def test_trained_forecaster_on_a_derived_family_scales_the_base_forecast() -> None:
    cfg = H.base_config()
    horizon = cfg.forecast.horizon
    base = np.arange(200, dtype=np.float64) * 2.0
    H._TRACE_CACHE["_base"] = (base, {"t0_utc_epoch_s": 0, "dt_s": 5.0,
                                      "sha256_scaled_npy": "abc"})
    H._TRACE_CACHE["_x3"] = (base * 3.0, {"t0_utc_epoch_s": 0, "dt_s": 5.0,
                                          "derived_from": "_base", "extra_scale": 3.0,
                                          "sha256_scaled_npy": "other"})
    H._MODEL_CACHE["_unit"] = (_LastValue(horizon), {
        "window": 4, "horizon": horizon, "lam_mean": 0.0, "lam_std": 1.0,
        "trace_sha256": "abc", "val_end": 100})
    try:
        fc = H.TrainedForecaster(trace_name="_x3", start=150, n=10, cfg=cfg,
                                 model_name="_unit")
        obs = Observation(u=0.0, r=0.0, lam=918.0, ell_p95=0.0, n=4.0)
        # Base forecast at origin 153: 306 + 0.5 = 306.5; x3 = 919.5.
        out = fc.predict(3, obs, n_ready=8)
        assert abs(out[0][0] - 919.5 / 320.0) < 1e-9
        # u > 1, so memory saturates at baseline + coefficient.
        assert abs(out[0][1] - 0.70) < 1e-12
    finally:
        for k in ("_base", "_x3"):
            H._TRACE_CACHE.pop(k, None)
        H._MODEL_CACHE.pop("_unit", None)


def test_arima_multi_origin_matches_hand_computed_ar1() -> None:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:  # statsmodels is optional; E16 records whether it ran
        return
    from experiments.train_forecasters import arima_multi_origin

    # AR(1) about a mean of 10 with phi = 0.5: y(t+h|t) = 10 + 0.5**h (y_t - 10).
    y = np.array([10.0, 12.0, 9.0, 14.0, 11.0, 8.0, 10.0, 13.0])
    res = ARIMA(y, order=(1, 0, 0), trend="c").filter(np.array([10.0, 0.5, 1.0]))
    got = arima_multi_origin(res, np.array([3, 5]), 3)
    assert np.allclose(got[0], [12.0, 11.0, 10.5])
    assert np.allclose(got[1], [9.0, 9.5, 9.75])


def test_real_window_blocks_are_disjoint_and_out_of_sample() -> None:
    trace = np.arange(1000, dtype=np.float64)
    H._TRACE_CACHE["_blocks"] = (trace, {"val_end": 800})
    try:
        first, count = H.real_blocks("_blocks", 50)
        # Test split starts at 800; 48 ticks of history -> first = 848;
        # (1000 - 848) // 50 = 3 blocks.
        assert (first, count) == (848, 3)
        starts = sorted(H.real_window("_blocks", 50, s)[1] for s in range(3))
        assert starts == [848, 898, 948]
        lam, start = H.real_window("_blocks", 50, 3)
        assert start == H.real_window("_blocks", 50, 0)[1]
        assert lam.tolist() == trace[start:start + 50].tolist()
    finally:
        H._TRACE_CACHE.pop("_blocks", None)


def test_initial_replicas_floor_requirement_and_cap() -> None:
    cfg = H.base_config()
    assert cfg.capacity.mu == 40.0
    assert cfg.capacity.u_target == 0.65
    assert cfg.control.n_max == 16
    # One replica covers 40 * 0.65 = 26 req/s at the target.

    def one(x: float) -> np.ndarray:
        return np.array([x, 0.0])

    assert H.initial_replicas(one(10.0), cfg) == 4     # ceil(0.38) = 1 -> floor 4
    assert H.initial_replicas(one(104.0), cfg) == 4    # 104 / 26 = 4 exactly
    assert H.initial_replicas(one(105.0), cfg) == 5    # ceil(4.04) = 5
    assert H.initial_replicas(one(427.4), cfg) == 16   # ceil(16.44) = 17 -> n_max
    small = H.base_config(n_max=2)
    assert H.initial_replicas(one(427.4), small) == 4  # floor kept, as before


def test_run_controller_uses_load_sized_start_by_default() -> None:
    cfg = H.base_config()
    lam = np.full(12, 300.0)                             # needs ceil(300/26) = 12
    sized = H.run_controller(method="reactive", lam=lam, cfg=cfg,
                             forecaster_kind="oracle")
    fixed = H.run_controller(method="reactive", lam=lam, cfg=cfg,
                             forecaster_kind="oracle", n_initial=4)
    # 12 ready replicas carry 300 req/s at u = 0.625, inside the SLO; four
    # replicas offer 160 req/s, so the first tick is saturated and censored.
    assert sized.n_series[0] == 12
    assert sized.p95[0] < cfg.capacity.slo_ms
    assert fixed.p95[0] == 5000.0
