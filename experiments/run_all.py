"""Run every experiment that is executable in this environment and write CSVs.

Each experiment writes to results/. Nothing here is hard-coded: every number
comes from a simulation run against the repository's own PoolSimulator and
controller modules.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.harness import (  # noqa: E402
    RESULTS,
    WORKLOADS,
    base_config,
    run_controller,
    summarise,
    workload,
)
from pact.eval.stats import (  # noqa: E402
    bootstrap_mean_ci,
    cliffs_delta,
    holm_bonferroni,
    wilcoxon_signed_rank,
)

N_TICKS = 400
N_SEEDS = 10
METHODS = [
    ("pact", "oracle"),
    ("pact", "persist"),
    ("rounded", "oracle"),
    ("rounded", "persist"),
    ("reactive", "oracle"),
    ("static", "oracle"),
]
LABEL = {
    ("pact", "oracle"): "PACT (integer-exact MPC, oracle fc.)",
    ("pact", "persist"): "PACT (integer-exact MPC, persistence fc.)",
    ("rounded", "oracle"): "Relaxed+rounded MPC (oracle fc.)",
    ("rounded", "persist"): "Relaxed+rounded MPC (persistence fc.)",
    ("reactive", "oracle"): "Reactive threshold (HPA-style)",
    ("static", "oracle"): "Static over-provisioned",
}
KEY = {k: f"{m}-{f}" for k, (m, f) in zip(METHODS, METHODS)}


def mkey(m: str, f: str) -> str:
    return f"{m}-{f}" if m not in ("reactive", "static") else m


# ===================================================== E1 main comparison
def e1_main() -> dict:
    cfg = base_config()
    rows, per_method = [], {}
    for m, f in METHODS:
        k = mkey(m, f)
        per_method[k] = {"sla": [], "over": [], "act": [], "util": [], "cost": [], "p95": []}
    for wl in WORKLOADS:
        for seed in range(N_SEEDS):
            lam = workload(wl, N_TICKS, 1000 + seed)
            for m, f in METHODS:
                out = run_controller(method=m, lam=lam, cfg=cfg, forecaster_kind=f)
                s = summarise(out, cfg)
                k = mkey(m, f)
                rows.append({"workload": wl, "seed": seed, "method": k,
                             "label": LABEL[(m, f)], **{kk: round(vv, 6) for kk, vv in s.items()}})
                per_method[k]["sla"].append(s["sla_violation_pct"])
                per_method[k]["over"].append(s["over_provision_pct"])
                per_method[k]["act"].append(s["scale_actions"])
                per_method[k]["util"].append(s["mean_util_pct"])
                per_method[k]["cost"].append(s["cost_per_1k_req"])
                per_method[k]["p95"].append(s["mean_p95_ms"])
    _csv(RESULTS / "e1_runs.csv", rows)

    summary = []
    for k, d in per_method.items():
        rec = {"method": k, "n_runs": len(d["sla"])}
        for metric, vals in d.items():
            a = np.asarray(vals, dtype=float)
            ci = bootstrap_mean_ci(a, n_resamples=4000, rng=7)
            rec[f"{metric}_mean"] = round(float(a.mean()), 4)
            rec[f"{metric}_std"] = round(float(a.std(ddof=1)), 4)
            rec[f"{metric}_ci_lo"] = round(float(ci.low), 4)
            rec[f"{metric}_ci_hi"] = round(float(ci.high), 4)
        summary.append(rec)
    _csv(RESULTS / "e1_summary.csv", summary)
    return per_method


# ===================================================== E2 statistics
def e2_stats(per_method: dict) -> None:
    ref = "pact-oracle"
    rows, pvals, keys = [], [], []
    for k in per_method:
        if k == ref:
            continue
        for metric in ("sla", "over", "act", "cost"):
            x = np.asarray(per_method[ref][metric], dtype=float)
            y = np.asarray(per_method[k][metric], dtype=float)
            w = wilcoxon_signed_rank(x, y)
            d = cliffs_delta(x, y)
            rows.append({"reference": ref, "method": k, "metric": metric,
                         "ref_mean": round(float(x.mean()), 4),
                         "other_mean": round(float(y.mean()), 4),
                         "statistic": round(float(w.statistic), 4),
                         "p_raw": round(float(w.p_value), 6),
                         "cliffs_delta": round(float(d), 4),
                         "n_pairs": int(len(x))})
            pvals.append(float(w.p_value))
            keys.append(len(rows) - 1)
    adj = holm_bonferroni(pvals)
    for i, a in zip(keys, adj):
        rows[i]["p_holm"] = round(float(a), 6)
        rows[i]["significant_0.05"] = bool(a < 0.05)
    _csv(RESULTS / "e2_significance.csv", rows)


# ===================================================== E3 rounding gap
def e3_rounding() -> None:
    rows = []
    for n_max in (2, 4, 8, 16, 32):
        cfg = base_config(n_max=n_max)
        scale = n_max / 16.0
        for wl in WORKLOADS:
            for seed in range(5):
                lam = workload(wl, N_TICKS, 2000 + seed) * scale
                ex = summarise(run_controller(method="pact", lam=lam, cfg=cfg,
                                              forecaster_kind="oracle"), cfg)
                rd = summarise(run_controller(method="rounded", lam=lam, cfg=cfg,
                                              forecaster_kind="oracle"), cfg)
                rows.append({
                    "n_max": n_max, "workload": wl, "seed": seed,
                    "disagree_pct": round(ex["rounding_disagree_pct"], 4),
                    "exact_sla": round(ex["sla_violation_pct"], 4),
                    "rounded_sla": round(rd["sla_violation_pct"], 4),
                    "delta_sla": round(rd["sla_violation_pct"] - ex["sla_violation_pct"], 4),
                    "exact_over": round(ex["over_provision_pct"], 4),
                    "rounded_over": round(rd["over_provision_pct"], 4),
                    "delta_over": round(rd["over_provision_pct"] - ex["over_provision_pct"], 4),
                    "exact_actions": int(ex["scale_actions"]),
                    "rounded_actions": int(rd["scale_actions"]),
                    "exact_solve_ms": round(ex["mean_solve_ms"], 6),
                })
    _csv(RESULTS / "e3_rounding_gap.csv", rows)


# ===================================================== E4 ablation
def e4_ablation() -> None:
    cfg = base_config()
    variants = {
        "full": {},
        "no_mpc": {"use_mpc": False},
        "no_churn": {"use_churn": False},
        "no_coldstart": {"use_coldstart": False},
        "no_margin": {"use_margin": False},
        "no_safety": {"use_safety": False},
    }
    rows, agg = [], {}
    for name, kw in variants.items():
        agg[name] = {"sla": [], "over": [], "act": []}
        for wl in WORKLOADS:
            for seed in range(N_SEEDS):
                lam = workload(wl, N_TICKS, 3000 + seed)
                s = summarise(run_controller(method="pact", lam=lam, cfg=cfg,
                                             forecaster_kind="oracle", **kw), cfg)
                rows.append({"variant": name, "workload": wl, "seed": seed,
                             **{k: round(v, 6) for k, v in s.items()}})
                agg[name]["sla"].append(s["sla_violation_pct"])
                agg[name]["over"].append(s["over_provision_pct"])
                agg[name]["act"].append(s["scale_actions"])
    _csv(RESULTS / "e4_ablation_runs.csv", rows)

    base = {k: float(np.mean(v)) for k, v in agg["full"].items()}
    summ = []
    for name, d in agg.items():
        rec = {"variant": name, "n_runs": len(d["sla"])}
        for metric, vals in d.items():
            a = np.asarray(vals, dtype=float)
            rec[f"{metric}_mean"] = round(float(a.mean()), 4)
            rec[f"{metric}_std"] = round(float(a.std(ddof=1)), 4)
            rec[f"{metric}_delta_vs_full"] = round(float(a.mean() - base[metric]), 4)
        if name != "full":
            x = np.asarray(agg["full"]["sla"], dtype=float)
            y = np.asarray(d["sla"], dtype=float)
            rec["sla_p_raw"] = round(float(wilcoxon_signed_rank(x, y).p_value), 6)
        summ.append(rec)
    _csv(RESULTS / "e4_ablation_summary.csv", summ)


# ===================================================== E5 horizon law
def e5_horizon() -> None:
    rows = []
    for tau_c in (5.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        for H in (1, 2, 4, 6, 8, 12, 16, 20, 24, 30):
            cfg = base_config(tau_c_s=tau_c, horizon=H)
            tau_ticks = tau_c / cfg.telemetry.dt
            sla_p, sla_r, exact = [], [], []
            for wl in ("burst", "step", "ramp"):
                for seed in range(3):
                    lam = workload(wl, N_TICKS, 4000 + seed)
                    sp = summarise(run_controller(
                        method="pact", lam=lam, cfg=cfg, forecaster_kind="oracle"), cfg)
                    sla_p.append(sp["sla_violation_pct"])
                    exact.append(sp["exact_solver"])
                    sla_r.append(summarise(run_controller(
                        method="reactive", lam=lam, cfg=cfg, forecaster_kind="oracle"), cfg)["sla_violation_pct"])
            p, r = float(np.mean(sla_p)), float(np.mean(sla_r))
            rows.append({"tau_c_s": tau_c, "tau_ticks": round(tau_ticks, 2), "H": H,
                         "H_dt": H * cfg.telemetry.dt,
                         "covers_coldstart": bool(H * cfg.telemetry.dt >= tau_c),
                         "pact_sla": round(p, 4), "reactive_sla": round(r, 4),
                         "improvement_pp": round(r - p, 4),
                         "exact_frac": round(float(np.mean(exact)), 4),
                         "n_runs": len(sla_p)})
    _csv(RESULTS / "e5_horizon_law.csv", rows)


# ===================================================== E6 operating frontier
def e6_frontier() -> None:
    rows = []
    for u_target in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        for m, f in [("pact", "oracle"), ("rounded", "oracle"), ("reactive", "oracle")]:
            cfg = base_config()
            cfg = replace(cfg, capacity=replace(cfg.capacity, u_target=u_target))
            sla, over = [], []
            for wl in WORKLOADS:
                for seed in range(3):
                    lam = workload(wl, N_TICKS, 5000 + seed)
                    s = summarise(run_controller(method=m, lam=lam, cfg=cfg,
                                                 forecaster_kind=f), cfg)
                    sla.append(s["sla_violation_pct"])
                    over.append(s["over_provision_pct"])
            rows.append({"u_target": u_target, "method": mkey(m, f),
                         "sla_mean": round(float(np.mean(sla)), 4),
                         "over_mean": round(float(np.mean(over)), 4),
                         "n_runs": len(sla)})
    _csv(RESULTS / "e6_frontier.csv", rows)


# ===================================================== E7 robustness
def e7_robustness() -> None:
    cfg = base_config()
    rows = []
    for sigma in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        for guard in (True, False):
            sla, over, ovr = [], [], []
            for wl in ("burst", "step", "diurnal"):
                for seed in range(4):
                    lam = workload(wl, N_TICKS, 6000 + seed)
                    rng = np.random.default_rng(900 + seed)
                    noisy = np.clip(lam * (1.0 + rng.normal(0, sigma, len(lam))), 10.0, None)
                    out = run_controller(method="pact", lam=lam, cfg=cfg,
                                         forecaster_kind="oracle", use_safety=guard,
                                         lam_forecast=noisy)
                    s = summarise(out, cfg)
                    sla.append(s["sla_violation_pct"])
                    over.append(s["over_provision_pct"])
                    ovr.append(s["overrides"])
            rows.append({"forecast_noise_sigma": sigma, "safety_guard": guard,
                         "sla_mean": round(float(np.mean(sla)), 4),
                         "sla_std": round(float(np.std(sla, ddof=1)), 4),
                         "over_mean": round(float(np.mean(over)), 4),
                         "overrides_total": int(np.sum(ovr)),
                         "n_runs": len(sla)})
    # reactive reference at the same operating point
    sla_r = []
    for wl in ("burst", "step", "diurnal"):
        for seed in range(4):
            lam = workload(wl, N_TICKS, 6000 + seed)
            sla_r.append(summarise(run_controller(method="reactive", lam=lam, cfg=cfg,
                                                  forecaster_kind="oracle"), cfg)["sla_violation_pct"])
    rows.append({"forecast_noise_sigma": -1.0, "safety_guard": True,
                 "sla_mean": round(float(np.mean(sla_r)), 4),
                 "sla_std": round(float(np.std(sla_r, ddof=1)), 4),
                 "over_mean": float("nan"), "n_runs": len(sla_r)})
    _csv(RESULTS / "e7_robustness.csv", rows)


# ===================================================== manifest
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        return ""


def _executed_real() -> list[str]:
    names = ["e16_forecast_accuracy.csv", "e17_trained_runs.csv",
             "e17_trained_summary.csv", "e17_bracket_check.csv",
             "e18_horizon_law_real.csv", "e19_robustness_real.csv",
             "e20_initial_state.csv", "e21_bias_probe.csv",
             "e22_margin_ceiling_ablation.csv"]
    return [n for n in names if (RESULTS / n).exists()]


def _checkpoints() -> dict:
    out = {}
    for meta in sorted((RESULTS / "checkpoints").glob("*.json")):
        m = json.loads(meta.read_text())
        out[meta.stem] = {k: m.get(k) for k in (
            "kind", "seed", "n_params", "epochs_run", "best_val_loss",
            "train_device", "torch", "order", "aic") if k in m}
    return out


def manifest() -> None:
    sha = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain"))
    on_remote = bool(_git("branch", "-r", "--contains", sha))
    trace_meta = {}
    for name in ("worldcup98", "worldcup98_testpeak"):
        p = RESULTS / "traces" / f"{name}.json"
        if p.exists():
            t = json.loads(p.read_text())
            trace_meta[name] = {k: t.get(k) for k in (
                "scale_factor", "scaled_peak_req_s", "test_peak_req_s",
                "train_end", "val_end", "sha256_scaled_npy")}
    try:
        import statsmodels
        import torch
        libs = {"torch": torch.__version__, "statsmodels": statsmodels.__version__}
    except ImportError:
        libs = {}
    m = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sha,
        "git_dirty": dirty,
        "git_sha_on_remote": on_remote,
        "git_note": (
            "results were produced from uncommitted changes on top of git_sha; "
            "regenerate the manifest after committing and pushing"
            if dirty else "results match the committed tree at git_sha"),
        **libs,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "n_ticks_per_run": N_TICKS,
        "n_seeds": N_SEEDS,
        "workloads": list(WORKLOADS),
        "plant": "pact.sim.queue_model.PoolSimulator (analytical, Kingman p95)",
        "start_state": (
            "every run starts with max(4, ceil(lambda[0] / (mu * u_target))) "
            "replicas, capped at n_max (experiments.harness.initial_replicas); "
            "earlier result sets used a fixed 4, which on recorded windows put "
            "every violation in the start-up transient (e20_initial_state.csv)"),
        "mpc_solver": (
            "pact.control.mpc_exact.solve_mpc_exact -- globally optimal integer "
            "solve, validated against exhaustive enumeration in "
            "tests/test_mpc_exact.py and experiments/verify_exact.py"
        ),
        "superseded_solver": (
            "pact.control.mpc.solve_mpc enumerates first actions but completes "
            "the horizon greedily, so its first action is not globally optimal; "
            "it is retained only as the comparison arm of e15"
        ),
        "executed": [
            "e1_runs.csv", "e1_summary.csv", "e2_significance.csv",
            "e3_rounding_gap.csv", "e4_ablation_runs.csv",
            "e4_ablation_summary.csv", "e5_horizon_law.csv",
            "e6_frontier.csv", "e7_robustness.csv",
            "e8_sensitivity.csv", "e8_solver_compare.csv",
            "e9_solvetime.csv", "e10_per_workload.csv",
            "e11_decisions.csv", "e12_coldstart.csv",
            "e13_predictor_quality.csv", "e14_inference_trace.csv",
            "e15_exact_vs_greedy.json", "e15_validation.json",
            *_executed_real(),
        ],
        "real_traces": trace_meta,
        "checkpoints": _checkpoints(),
        "not_executed": {
            "Docker testbed and measured cold start": "no Docker daemon in this environment",
            "Bitbrains GWA-T-12": (
                "archives located and verified (data/provenance.json) but not used: "
                "5-minute per-VM utilisation, not request arrivals; driving a 5 s "
                "controller would need 60x interpolation"),
            "Trained forecaster in the horizon sweep (E18)": (
                "the checkpoint has a fixed horizon H = 12; E18 varies H and so "
                "uses the oracle forecaster, as E5 does"),
            "Training-variance estimate": (
                "every forecaster in E16 was trained from one seed; no spread "
                "across training runs was measured"),
            "Cause of the trained arm's churn reduction": (
                "E21 shows the TCN's mean bias does not reproduce it; the "
                "smoothing hypothesis suggested by E13 was not tested"),
            "Margin and ceiling ablation under imperfect predictors on synthetic traffic": (
                "E22 covers the recorded windows only; E4 covers the synthetic "
                "families under the oracle only"),
        },
        "forecaster_arms": (
            "Controllers are driven by a trained TCN forecasting the arrival rate "
            "(E17, E22, recorded trace only, frozen checkpoint, no training at "
            "evaluation time) and by two bracketing predictors, perfect foresight "
            "and persistence, which every experiment uses so that controller "
            "effects can be separated from forecaster quality."
        ),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(m, indent=2))
    print("  wrote manifest.json")


def _csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        print(f"  SKIP {path.name}: no rows")
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
    print(f"  wrote {path.name}  ({len(rows)} rows)")


if __name__ == "__main__":
    t0 = time.time()
    print("E1 main comparison")
    pm = e1_main()
    print("E2 statistics")
    e2_stats(pm)
    print("E3 rounding gap")
    e3_rounding()
    print("E4 ablation")
    e4_ablation()
    print("E5 horizon law")
    e5_horizon()
    print("E6 frontier")
    e6_frontier()
    print("E7 robustness")
    e7_robustness()
    manifest()
    print(f"done in {time.time() - t0:.1f}s")
