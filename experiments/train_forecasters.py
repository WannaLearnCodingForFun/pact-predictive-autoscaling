"""E16: train lambda forecasters on WorldCup98 and measure per-horizon error.

Models: the PACT TCN (pact.forecast.tcn.DilatedTCN, paper configuration) and
the baselines named in the paper -- LSTM, GRU, Bi-LSTM and a small
Transformer (pact.forecast.seq_models) with hidden sizes matched to the TCN's
parameter count, plus ARIMA (statsmodels) and last-value persistence.

Data: results/traces/worldcup98.npy (make_real_traces.py). Splits are
chronological and contiguous, 60/20/20 by time, through
pact.forecast.train.chronological_window_splits, and checked again with
assert_chronological_splits. Normalisation statistics come from the training
split only. Early stopping uses the validation split; the test split is read
once, after training, for the error table.

Outputs:
    results/checkpoints/<model>.pt + .json      frozen weights and metadata
    results/checkpoints/<model>_training.csv     per-epoch losses
    results/e16_parts/<model>.csv                write_horizon_error_csv output
    results/e16_forecast_accuracy.csv            all models, one row per h
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.harness import (  # noqa: E402
    CHECKPOINTS,
    RESULTS,
    base_config,
    load_trace,
)
from experiments.lambda_forecast import (  # noqa: E402
    IN_CHANNELS,
    build_model,
    lambda_features,
    load_checkpoint,
    predict_lambda,
    save_checkpoint,
)
from pact.forecast.losses import AsymmetricHorizonHuber  # noqa: E402
from pact.forecast.seq_models import hidden_size_matching_params  # noqa: E402
from pact.forecast.train import (  # noqa: E402
    _loader,
    _run_epoch,
    assert_chronological_splits,
    chronological_window_splits,
    compute_horizon_error,
    seed_everything,
    write_horizon_error_csv,
    write_training_csv,
)

TRACE = "worldcup98"
NEURAL = ("tcn", "lstm", "gru", "bilstm", "transformer")
PARTS = RESULTS / "e16_parts"
E16_COLUMNS = ("model", "seed", "horizon", "mae_req_s", "rmse_req_s",
               "nmae_pct", "bias_req_s", "under_pct", "mae_raw_req_s",
               "n_windows", "n_params", "epochs", "train_s")


def prepare(seed: int) -> dict[str, Any]:
    cfg = base_config()
    trace, tmeta = load_trace(TRACE)
    n = len(trace)
    window, horizon = cfg.telemetry.window, cfg.forecast.horizon
    train_frac, val_frac = float(tmeta["train_fraction"]), float(tmeta["val_fraction"])
    train_end, val_end = int(tmeta["train_end"]), int(tmeta["val_end"])
    expected = (int(n * train_frac), int(n * train_frac) + int(n * val_frac))
    if expected != (train_end, val_end):
        raise RuntimeError("trace metadata split indices disagree with the split rule")
    mean = float(np.mean(trace[:train_end]))
    std = float(np.std(trace[:train_end]))
    t0, dt = int(tmeta["t0_utc_epoch_s"]), float(tmeta["dt_s"])
    feats = lambda_features(trace, t0, dt, mean, std)
    targets = np.repeat(feats[:, :1], 2, axis=1)
    stamps = t0 + np.arange(n, dtype=np.float64) * dt
    train, val, test = chronological_window_splits(
        feats, targets, stamps, window=window, horizon=horizon,
        train_fraction=train_frac, val_fraction=val_frac,
    )
    assert_chronological_splits(train.timestamps, val.timestamps, test.timestamps)
    test_origins = np.rint((test.timestamps - t0) / dt).astype(np.int64)
    if test_origins.min() < val_end:
        raise RuntimeError("a test window reaches into the validation period")
    seed_everything(seed)
    return {
        "cfg": cfg, "trace": trace, "tmeta": tmeta, "feats": feats,
        "train": train, "val": val, "test": test, "test_origins": test_origins,
        "mean": mean, "std": std, "window": window, "horizon": horizon,
        "train_end": train_end, "val_end": val_end,
    }


def base_meta(kind: str, d: dict[str, Any], seed: int) -> dict[str, Any]:
    f = d["cfg"].forecast
    return {
        "kind": kind, "seed": seed, "window": d["window"], "horizon": d["horizon"],
        "in_channels": IN_CHANNELS, "kernel_size": f.kernel_size, "depth": f.depth,
        "channels": f.channels, "dropout": f.dropout, "lr": f.lr,
        "batch_size": f.batch_size, "loss": "AsymmetricHorizonHuber",
        "kappa": f.kappa, "beta": f.beta, "huber_delta": f.huber_delta,
        "lam_mean": d["mean"], "lam_std": d["std"],
        "train_end": d["train_end"], "val_end": d["val_end"],
        "trace": TRACE, "trace_sha256": d["tmeta"]["sha256_scaled_npy"],
        "target": "lambda (exogenous arrival rate), z-scored with train-split stats",
    }


def _run_epoch_on(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_fn: AsymmetricHorizonHuber,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> float:
    """pact.forecast.train._run_epoch with batches moved to ``device``."""
    if device == "cpu":
        return _run_epoch(cast(Any, model), loader, loss_fn, optimizer)
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            loss = loss_fn(model(xb), yb)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def train_neural(
    kind: str, d: dict[str, Any], seed: int, max_epochs: int, patience: int,
    device: str = "cpu",
) -> tuple[str, np.ndarray, dict[str, Any]]:
    meta = base_meta(kind, d, seed)
    if kind != "tcn":
        tcn = build_model({**meta, "kind": "tcn"})
        tcn_params = int(sum(p.numel() for p in tcn.parameters()))
        meta["hidden_size"] = hidden_size_matching_params(
            cast(Any, kind), target_params=tcn_params, in_channels=IN_CHANNELS,
            window=d["window"], horizon=d["horizon"], hi=256,
        )
    seed_everything(seed)
    model = build_model(meta).to(device)
    cfgf = d["cfg"].forecast
    loss_fn = AsymmetricHorizonHuber.from_config(cfgf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfgf.lr)
    train_loader = _loader(d["train"], cfgf.batch_size, shuffle=True, seed=seed)
    val_loader = _loader(d["val"], cfgf.batch_size, shuffle=False, seed=seed)
    best, best_state, stale, log = float("inf"), None, 0, []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        tr = _run_epoch_on(model, train_loader, loss_fn, opt, device)
        va = _run_epoch_on(model, val_loader, loss_fn, None, device)
        log.append((epoch, tr, va))
        print(f"    {kind} epoch {epoch}: train {tr:.5f} val {va:.5f} "
              f"({time.time() - started:.0f}s)", flush=True)
        if va < best:
            best, stale = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    # Inference and every closed-loop run use the frozen weights on the CPU.
    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    meta.update({
        "train_device": device,
        "torch": torch.__version__,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": len(log), "best_val_loss": best,
        "train_seconds": time.time() - started,
        "max_epochs": max_epochs, "patience": patience,
    })
    tag = kind if seed == 0 else f"{kind}_s{seed}"
    save_checkpoint(CHECKPOINTS / f"{tag}.pt", model, meta)
    write_training_csv(CHECKPOINTS / f"{tag}_training.csv", log)
    pred = predict_lambda(model, meta, d["feats"], d["test_origins"])
    return tag, pred, meta


def fit_arima(d: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit on the training split, then filter forward with frozen parameters.

    Multi-step forecasts from every test origin come from the filtered state:
    a(t+1|t) = T a(t|t) + c,  y(t+h|t) = Z a(t+h|t) + d. Orders are chosen by
    AIC on the training split from a small grid.
    """
    from statsmodels.tsa.arima.model import ARIMA

    trace = d["trace"]
    train = trace[: d["train_end"]]
    grid = [((1, 0, 0), "c"), ((2, 0, 1), "c"), ((1, 1, 1), "n"), ((2, 1, 2), "n")]
    fits = []
    for order, trend in grid:
        started = time.time()
        res = ARIMA(train, order=order, trend=trend).fit()
        fits.append((res.aic, order, trend, res, time.time() - started))
        print(f"    ARIMA{order} trend={trend}: AIC {res.aic:.1f}", flush=True)
    aic, order, trend, res, secs = min(fits, key=lambda f: f[0])
    full = res.apply(trace)
    pred = arima_multi_origin(full, d["test_origins"], d["horizon"])
    meta = {"kind": "arima", "order": list(order), "trend": trend, "aic": aic,
            "fit_seconds": secs, "params": res.params.tolist(),
            "n_params": int(len(res.params)), "seed": 0,
            "grid": [[list(o), t, a] for a, o, t, _r, _s in fits]}
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    (CHECKPOINTS / "arima.json").write_text(json.dumps(meta, indent=2))
    return pred, meta


def _time_invariant(m: np.ndarray, ndim: int) -> np.ndarray:
    return m[..., 0] if m.ndim == ndim else m


def arima_multi_origin(res: Any, origins: np.ndarray, horizon: int) -> np.ndarray:
    ssm = res.model.ssm
    design = _time_invariant(ssm["design"], 3)
    trans = _time_invariant(ssm["transition"], 3)
    c = _time_invariant(ssm["state_intercept"], 2)
    dd = _time_invariant(ssm["obs_intercept"], 2)
    a = res.filtered_state[:, origins]
    out = np.empty((origins.size, horizon), dtype=np.float64)
    for h in range(horizon):
        a = trans @ a + c[:, None]
        out[:, h] = (design @ a)[0] + dd[0]
    return out


def errors(
    model: str, pred: np.ndarray, d: dict[str, Any], meta: dict[str, Any]
) -> list[dict[str, Any]]:
    trace, origins, H = d["trace"], d["test_origins"], d["horizon"]
    actual = trace[origins[:, None] + np.arange(1, H + 1)[None, :]]
    rows = compute_horizon_error(actual[:, :, None], pred[:, :, None])
    PARTS.mkdir(parents=True, exist_ok=True)
    write_horizon_error_csv(PARTS / f"{model}.csv", rows)
    factor = float(d["tmeta"]["scale_factor"])
    err = pred - actual
    out = []
    for h, mae_v, rmse_v in rows:
        out.append({
            "model": model, "seed": meta.get("seed", 0), "horizon": h,
            "mae_req_s": round(mae_v, 6), "rmse_req_s": round(rmse_v, 6),
            "nmae_pct": round(100.0 * mae_v / float(np.mean(actual[:, h - 1])), 6),
            # Positive bias is over-forecasting; under_pct is the share of
            # forecasts below the realised rate, which the loss penalises.
            "bias_req_s": round(float(np.mean(err[:, h - 1])), 6),
            "under_pct": round(100.0 * float(np.mean(err[:, h - 1] < 0.0)), 4),
            "mae_raw_req_s": round(mae_v / factor, 6),
            "n_windows": int(origins.size),
            "n_params": meta.get("n_params", 0),
            "epochs": meta.get("epochs_run", ""),
            "train_s": round(float(meta.get("train_seconds",
                                            meta.get("fit_seconds", 0.0))), 1),
        })
    return out


def write_e16(rows: list[dict[str, Any]]) -> None:
    path = RESULTS / "e16_forecast_accuracy.csv"
    existing = []
    if path.exists():
        done = {(r["model"], int(r["seed"])) for r in rows}
        existing = [r for r in csv.DictReader(path.open())
                    if (r["model"], int(r["seed"])) not in done]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(E16_COLUMNS))
        w.writeheader()
        w.writerows(existing + rows)
    print(f"  wrote {path.name} ({len(existing) + len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="persistence,arima," + ",".join(NEURAL))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-only", action="store_true",
                        help="score existing checkpoints on the test split; "
                             "no training")
    parser.add_argument("--device", default="cpu",
                        help="training device (cpu or mps); inference is always cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    d = prepare(args.seed)
    print(f"train {len(d['train'].X)} / val {len(d['val'].X)} / test "
          f"{len(d['test'].X)} windows; lambda mean {d['mean']:.2f} std {d['std']:.2f}",
          flush=True)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"== {model}", flush=True)
        if model == "persistence":
            last = d["trace"][d["test_origins"]]
            pred = np.repeat(last[:, None], d["horizon"], axis=1)
            write_e16(errors("persistence", pred, d, {"seed": 0, "n_params": 0}))
        elif model == "arima":
            pred, meta = fit_arima(d)
            write_e16(errors("arima", pred, d, meta))
        elif model in NEURAL and args.eval_only:
            tag = model if args.seed == 0 else f"{model}_s{args.seed}"
            net, meta = load_checkpoint(CHECKPOINTS / f"{tag}.pt")
            pred = predict_lambda(net, meta, d["feats"], d["test_origins"])
            write_e16(errors(tag, pred, d, meta))
        elif model in NEURAL:
            tag, pred, meta = train_neural(model, d, args.seed, args.max_epochs,
                                           args.patience, args.device)
            write_e16(errors(tag, pred, d, meta))
        else:
            raise SystemExit(f"unknown model {model!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
