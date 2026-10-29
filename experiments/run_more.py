"""Additional sweeps: control-parameter sensitivity, solve-time scaling,
per-workload breakdown and decision-class detail.

Writes e8_*.csv through e12_*.csv. Run after run_all.py.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.harness import (  # noqa: E402
    RESULTS, WORKLOADS, base_config, run_controller, summarise, workload,
)
from pact.eval.stats import bootstrap_mean_ci, wilcoxon_signed_rank  # noqa: E402

N_TICKS = 400
SEEDS = 8


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  SKIP {path.name}")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {path.name} ({len(rows)} rows)")


def sweep(label: str, mutate, values, n_seeds=SEEDS, method="pact"):
    """Run one control-parameter sweep across all workloads."""
    rows = []
    for v in values:
        cfg = mutate(base_config(), v)
        acc = {"sla": [], "over": [], "act": [], "osc": [], "cost": [], "util": []}
        for wl in WORKLOADS:
            for seed in range(n_seeds):
                lam = workload(wl, N_TICKS, 7000 + seed)
                s = summarise(run_controller(method=method, lam=lam, cfg=cfg,
                                             forecaster_kind="oracle"), cfg)
                acc["sla"].append(s["sla_violation_pct"])
                acc["over"].append(s["over_provision_pct"])
                acc["act"].append(s["scale_actions"])
                acc["osc"].append(s["oscillations"])
                acc["cost"].append(s["cost_per_1k_req"])
                acc["util"].append(s["mean_util_pct"])
        rec = {"parameter": label, "value": v, "n_runs": len(acc["sla"])}
        for k, vals in acc.items():
            a = np.asarray(vals, dtype=float)
            rec[f"{k}_mean"] = round(float(a.mean()), 4)
            rec[f"{k}_std"] = round(float(a.std(ddof=1)), 4)
        rows.append(rec)
    return rows


# ===================================================== E8 control sensitivity
def e8_sensitivity() -> None:
    rows = []
    rows += sweep("churn_weight_s",
                  lambda c, v: replace(c, control=replace(c.control, s_churn=v)),
                  [0.0, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0])
    rows += sweep("asymmetry_qup_over_qdown",
                  lambda c, v: replace(c, control=replace(c.control, q_up=v, q_down=1.0)),
                  [1.0, 2.0, 4.0, 8.0, 12.0, 20.0])
    rows += sweep("deadband_theta",
                  lambda c, v: replace(c, control=replace(c.control, deadband=v)),
                  [0, 1, 2, 3])
    rows += sweep("rate_limit_delta_max",
                  lambda c, v: replace(c, control=replace(c.control, delta_max=v)),
                  [1, 2, 4, 6, 8])
    rows += sweep("cooldown_s",
                  lambda c, v: replace(c, control=replace(c.control, cooldown_s=v)),
                  [0.0, 30.0, 60.0, 120.0])
    _csv(RESULTS / "e8_sensitivity.csv", rows)


# ===================================================== E9 solve-time scaling
def e9_solvetime() -> None:
    rows = []
    for dmax in (1, 2, 4, 8, 16):
        for H in (4, 8, 12, 20, 30):
            cfg = base_config(horizon=H)
            cfg = replace(cfg, control=replace(cfg.control, delta_max=dmax))
            ts, acts = [], []
            for wl in ("burst", "step"):
                for seed in range(4):
                    lam = workload(wl, N_TICKS, 7500 + seed)
                    s = summarise(run_controller(method="pact", lam=lam, cfg=cfg,
                                                 forecaster_kind="oracle"), cfg)
                    ts.append(s["mean_solve_ms"])
                    acts.append(s["scale_actions"])
            a = np.asarray(ts, dtype=float)
            rows.append({"delta_max": dmax, "H": H,
                         "candidates": 2 * dmax + 1,
                         "mean_solve_ms": round(float(a.mean()), 5),
                         "max_solve_ms": round(float(a.max()), 5),
                         "budget_frac_pct": round(float(a.mean()) / 5000.0 * 100.0, 6),
                         "n_runs": len(ts)})
    _csv(RESULTS / "e9_solvetime.csv", rows)


# ===================================================== E10 per-workload
def e10_per_workload() -> None:
    cfg = base_config()
    methods = [("pact", "oracle"), ("rounded", "oracle"), ("reactive", "oracle"),
               ("static", "oracle")]
    rows = []
    for wl in WORKLOADS:
        for m, fk in methods:
            acc = {"sla": [], "over": [], "act": [], "osc": [], "cost": [],
                   "p95": [], "util": []}
            for seed in range(10):
                lam = workload(wl, N_TICKS, 1000 + seed)
                s = summarise(run_controller(method=m, lam=lam, cfg=cfg,
                                             forecaster_kind=fk), cfg)
                acc["sla"].append(s["sla_violation_pct"])
                acc["over"].append(s["over_provision_pct"])
                acc["act"].append(s["scale_actions"])
                acc["osc"].append(s["oscillations"])
                acc["cost"].append(s["cost_per_1k_req"])
                acc["p95"].append(s["mean_p95_ms"])
                acc["util"].append(s["mean_util_pct"])
            rec = {"workload": wl,
                   "method": m if m in ("reactive", "static") else f"{m}-{fk}",
                   "n_runs": len(acc["sla"])}
            for k, vals in acc.items():
                a = np.asarray(vals, dtype=float)
                rec[f"{k}_mean"] = round(float(a.mean()), 4)
                rec[f"{k}_std"] = round(float(a.std(ddof=1)), 4)
            rows.append(rec)
    _csv(RESULTS / "e10_per_workload.csv", rows)


# ===================================================== E11 decision classes
def e11_decisions() -> None:
    cfg = base_config()
    methods = [("pact", "oracle"), ("pact", "persist"), ("rounded", "oracle"),
               ("reactive", "oracle")]
    rows = []
    for m, fk in methods:
        acc = {"accuracy": [], "scaleup_precision": [], "scaleup_recall": [],
               "scaleup_f1": [], "hold_f1": [], "scaledown_recall": []}
        for wl in WORKLOADS:
            for seed in range(10):
                lam = workload(wl, N_TICKS, 1000 + seed)
                s = summarise(run_controller(method=m, lam=lam, cfg=cfg,
                                             forecaster_kind=fk), cfg)
                for k in acc:
                    acc[k].append(s[k])
        rec = {"method": m if m in ("reactive", "static") else f"{m}-{fk}",
               "n_runs": len(acc["accuracy"])}
        for k, vals in acc.items():
            a = np.asarray(vals, dtype=float)
            ci = bootstrap_mean_ci(a, n_resamples=3000, rng=11)
            rec[f"{k}_mean"] = round(float(a.mean()), 4)
            rec[f"{k}_lo"] = round(float(ci.low), 4)
            rec[f"{k}_hi"] = round(float(ci.high), 4)
        rows.append(rec)
    _csv(RESULTS / "e11_decisions.csv", rows)


# ===================================================== E12 cold-start sensitivity
def e12_coldstart() -> None:
    """Fixed, adequate horizon; vary only the plant's cold start."""
    rows = []
    for tau in (5.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0):
        cfg = base_config(tau_c_s=tau, horizon=30)   # 150 s lookahead covers all
        for m in ("pact", "rounded", "reactive"):
            acc = {"sla": [], "over": [], "act": []}
            for wl in ("burst", "step", "ramp"):
                for seed in range(5):
                    lam = workload(wl, N_TICKS, 7800 + seed)
                    s = summarise(run_controller(method=m, lam=lam, cfg=cfg,
                                                 forecaster_kind="oracle"), cfg)
                    acc["sla"].append(s["sla_violation_pct"])
                    acc["over"].append(s["over_provision_pct"])
                    acc["act"].append(s["scale_actions"])
            rec = {"tau_c_s": tau, "method": m, "n_runs": len(acc["sla"])}
            for k, vals in acc.items():
                a = np.asarray(vals, dtype=float)
                rec[f"{k}_mean"] = round(float(a.mean()), 4)
                rec[f"{k}_std"] = round(float(a.std(ddof=1)), 4)
            rows.append(rec)
    _csv(RESULTS / "e12_coldstart.csv", rows)


if __name__ == "__main__":
    t0 = time.time()
    print("E8 control-parameter sensitivity")
    e8_sensitivity()
    print("E9 solve-time scaling")
    e9_solvetime()
    print("E10 per-workload breakdown")
    e10_per_workload()
    print("E11 decision classes")
    e11_decisions()
    print("E12 cold-start sensitivity at fixed horizon")
    e12_coldstart()
    print(f"done in {time.time() - t0:.1f}s")


# ===================================================== E13 predictor quality
def e13_predictor_quality() -> None:
    """Interpolate between the persistence and oracle brackets.

    A blended predictor returns (1-w)*persistence + w*oracle. w = 0 and w = 1
    recover the two brackets; intermediate values trace the controller's
    response to predictor quality, which the brackets alone leave unmeasured.
    """
    from experiments.harness import OracleForecaster, PersistenceForecaster
    import experiments.harness as H

    class BlendForecaster:
        def __init__(self, lam, cfg, w):
            self.o = OracleForecaster(lam=lam, cfg=cfg)
            self.p = PersistenceForecaster(cfg=cfg)
            self.w = w

        def predict(self, t, obs, n_ready):
            a = self.o.predict(t, obs, n_ready)
            b = self.p.predict(t, obs, n_ready)
            w = self.w
            return [[w * x[0] + (1 - w) * y[0], w * x[1] + (1 - w) * y[1]]
                    for x, y in zip(a, b)]

    cfg = base_config()
    rows = []
    orig = H.OracleForecaster
    for w in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0):
        for method in ("pact", "rounded"):
            acc = {"sla": [], "over": [], "act": [], "osc": []}
            for wl in WORKLOADS:
                for seed in range(6):
                    lam = workload(wl, N_TICKS, 8200 + seed)
                    # inject the blended predictor in place of the oracle
                    H.OracleForecaster = lambda lam=lam, cfg=cfg, w=w: BlendForecaster(lam, cfg, w)
                    try:
                        s = summarise(run_controller(method=method, lam=lam, cfg=cfg,
                                                     forecaster_kind="oracle"), cfg)
                    finally:
                        H.OracleForecaster = orig
                    acc["sla"].append(s["sla_violation_pct"])
                    acc["over"].append(s["over_provision_pct"])
                    acc["act"].append(s["scale_actions"])
                    acc["osc"].append(s["oscillations"])
            rec = {"oracle_weight": w, "method": method, "n_runs": len(acc["sla"])}
            for k, vals in acc.items():
                a = np.asarray(vals, dtype=float)
                rec[f"{k}_mean"] = round(float(a.mean()), 4)
                rec[f"{k}_std"] = round(float(a.std(ddof=1)), 4)
            rows.append(rec)
    _csv(RESULTS / "e13_predictor_quality.csv", rows)
