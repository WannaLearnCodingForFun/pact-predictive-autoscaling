"""Experiments that need the recorded WorldCup98 trace.

E17  trained forecaster against both brackets, every disjoint test-split window
E18  horizon-dead-time law on recorded traffic (E5 grid, oracle forecaster)
E19  forecast-corruption robustness and the safety ceiling on recorded traffic
E20  where violations fall under the fixed and the load-sized start state
E21  oracle shifted by the TCN's mean bias, against the trained arm
E22  margin and ceiling ablation under the persistence and trained predictors

Each experiment persists its CSV as soon as it finishes. Usage:
    python run_real.py e17 | e18 | e19
    python run_real.py e18:<trace>:<tau_c>   one E18 slice; `e18` merges them
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.harness import (  # noqa: E402
    LEGACY_N_INITIAL,
    REAL_WORKLOADS,
    RESULTS,
    WORKLOADS,
    base_config,
    initial_replicas,
    real_blocks,
    real_window,
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
E17_ARMS = [
    ("pact", "oracle"), ("pact", "persist"), ("pact", "trained"),
    ("rounded", "oracle"), ("rounded", "persist"), ("rounded", "trained"),
    ("reactive", "oracle"),
]
# Lower is better for every metric compared.
E17_METRICS = ("sla_violation_pct", "over_provision_pct", "scale_actions",
               "oscillations", "cost_per_1k_req", "mean_p95_ms")


def _csv(path: Path, rows: list[dict]) -> None:
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
    print(f"  wrote {path.name} ({len(rows)} rows)", flush=True)


def _key(m: str, f: str) -> str:
    return m if m == "reactive" else f"{m}-{f}"


E17_PARTS = RESULTS / "e17_parts"
E17_PER_KEYS = E17_METRICS + ("overrides", "gamma_max")


def e17_part(trace: str) -> None:
    """One family of E17; ``e17`` merges the parts and corrects across both."""
    E17_PARTS.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    _e17_family(trace, base_config(), runs)
    _csv(E17_PARTS / f"{trace}_runs.csv", runs)


def _e17_from_part(trace: str, runs: list[dict]) -> dict:
    per: dict[str, dict[str, list[float]]] = {
        _key(m, f): {k: [] for k in E17_PER_KEYS} for m, f in E17_ARMS
    }
    with (E17_PARTS / f"{trace}_runs.csv").open() as fh:
        part = list(csv.DictReader(fh))
    part.sort(key=lambda r: int(r["block"]))
    for r in part:
        runs.append(r)
        for metric in E17_PER_KEYS:
            per[r["method"]][metric].append(float(r[metric]))
    return per


def e17_trained() -> None:
    cfg = base_config()
    runs: list[dict] = []
    summary: list[dict] = []
    rows: list[dict] = []
    pvals: list[float] = []
    slots: list[tuple[int, str]] = []
    for trace in REAL_WORKLOADS:
        if (E17_PARTS / f"{trace}_runs.csv").exists():
            per = _e17_from_part(trace, runs)
        else:
            per = _e17_family(trace, cfg, runs)
        _e17_summarise(trace, per, summary, rows, pvals, slots)
        _csv(RESULTS / "e17_trained_runs.csv", runs)
    for (i, ref_name), adj in zip(slots, holm_bonferroni(pvals), strict=True):
        rows[i][f"p_holm_vs_{ref_name}"] = round(float(adj), 6)
    _csv(RESULTS / "e17_trained_summary.csv", summary)
    _csv(RESULTS / "e17_bracket_check.csv", rows)


def _e17_family(trace: str, cfg, runs: list[dict]) -> dict:
    _, n_blocks = real_blocks(trace, N_TICKS)
    per: dict[str, dict[str, list[float]]] = {
        _key(m, f): {k: [] for k in E17_PER_KEYS} for m, f in E17_ARMS
    }
    started = time.time()
    for block in range(n_blocks):
        lam, start = real_window(trace, N_TICKS, block)
        for m, f in E17_ARMS:
            kw = {"trace_name": trace, "window_start": start} if f == "trained" else {}
            s = summarise(run_controller(method=m, lam=lam, cfg=cfg,
                                         forecaster_kind=f, **kw), cfg)
            k = _key(m, f)
            runs.append({"workload": trace, "block": block, "window_start": start,
                         "method": k, **{kk: round(vv, 6) for kk, vv in s.items()}})
            for metric in per[k]:
                per[k][metric].append(s[metric])
        if block % 10 == 9:
            print(f"  E17 {trace} {block + 1}/{n_blocks} windows "
                  f"({time.time() - started:.0f}s)", flush=True)
    return per


def _e17_summarise(trace: str, per: dict, summary: list[dict], rows: list[dict],
                   pvals: list[float], slots: list[tuple[int, str]]) -> None:
    for k, d in per.items():
        rec: dict = {"workload": trace, "method": k,
                     "n_runs": len(d["sla_violation_pct"])}
        for metric, vals in d.items():
            a = np.asarray(vals, dtype=float)
            ci = bootstrap_mean_ci(a, n_resamples=4000, rng=17)
            rec[f"{metric}_mean"] = round(float(a.mean()), 4)
            rec[f"{metric}_ci_lo"] = round(float(ci.low), 4)
            rec[f"{metric}_ci_hi"] = round(float(ci.high), 4)
        summary.append(rec)

    for ctrl in ("pact", "rounded"):
        o, p, t = per[f"{ctrl}-oracle"], per[f"{ctrl}-persist"], per[f"{ctrl}-trained"]
        for metric in E17_METRICS:
            mo, mp, mt = (float(np.mean(x[metric])) for x in (o, p, t))
            lo, hi = min(mo, mp), max(mo, mp)
            if lo <= mt <= hi:
                where = "between"
            elif mt < lo:
                where = "below_both"
            else:
                where = "above_both"
            span = mp - mo
            rec = {
                "workload": trace, "controller": ctrl, "metric": metric,
                "oracle_mean": round(mo, 4), "persist_mean": round(mp, 4),
                "trained_mean": round(mt, 4),
                "bracket_lo": round(lo, 4), "bracket_hi": round(hi, 4),
                "position": where,
                # 0 = at the oracle, 1 = at persistence.
                "fraction_oracle_to_persist": (round((mt - mo) / span, 4)
                                               if span != 0 else ""),
                "n_pairs": len(t[metric]),
            }
            for ref_name, ref in (("oracle", o), ("persist", p)):
                x = np.asarray(t[metric], dtype=float)
                y = np.asarray(ref[metric], dtype=float)
                if np.all(x == y):
                    rec[f"p_vs_{ref_name}"] = 1.0
                    rec[f"delta_vs_{ref_name}"] = 0.0
                else:
                    w = wilcoxon_signed_rank(x, y)
                    rec[f"p_vs_{ref_name}"] = round(float(w.p_value), 6)
                    rec[f"delta_vs_{ref_name}"] = round(float(cliffs_delta(x, y)), 4)
                    pvals.append(float(w.p_value))
                    slots.append((len(rows), ref_name))
            rows.append(rec)


E18_TAUS = (5.0, 15.0, 30.0, 45.0, 60.0, 90.0)
E18_PARTS = RESULTS / "e18_parts"


def _e18_part_path(trace: str, tau_c: float) -> Path:
    return E18_PARTS / f"{trace}_tau{tau_c:g}.csv"


def e18_part(trace: str, tau_c: float) -> None:
    """One (family, cold-start) slice of E18, so the grid can run in parallel."""
    E18_PARTS.mkdir(parents=True, exist_ok=True)
    marker = _e18_part_path(trace, tau_c).with_suffix(".running")
    marker.write_text(str(os.getpid()))
    try:
        rows: list[dict] = []
        _e18_cell(trace, tau_c, rows, time.time())
        _csv(_e18_part_path(trace, tau_c), rows)
    finally:
        marker.unlink(missing_ok=True)


def _running_elsewhere(part: Path) -> bool:
    marker = part.with_suffix(".running")
    if not marker.exists():
        return False
    try:
        os.kill(int(marker.read_text()), 0)
    except (ValueError, ProcessLookupError):
        return False
    return True


def e18_horizon() -> None:
    """Compute any missing slices, then merge every slice in grid order.

    A slice being computed by another live process is waited for rather than
    recomputed."""
    rows: list[dict] = []
    for trace in REAL_WORKLOADS:
        for tau_c in E18_TAUS:
            part = _e18_part_path(trace, tau_c)
            while not part.exists() and _running_elsewhere(part):
                time.sleep(30)
            if not part.exists():
                e18_part(trace, tau_c)
            with part.open() as fh:
                rows.extend(csv.DictReader(fh))
    _csv(RESULTS / "e18_horizon_law_real.csv", rows)


def _e18_cell(trace: str, tau_c: float, rows: list[dict], started: float) -> None:
    for H in (1, 2, 4, 6, 8, 12, 16, 20, 24, 30):
        cfg = base_config(tau_c_s=tau_c, horizon=H)
        sla_p, sla_r, exact = [], [], []
        for seed in range(9):
            lam, _ = real_window(trace, N_TICKS, 4000 + seed)
            sp = summarise(run_controller(method="pact", lam=lam, cfg=cfg,
                                          forecaster_kind="oracle"), cfg)
            sla_p.append(sp["sla_violation_pct"])
            exact.append(sp["exact_solver"])
            sla_r.append(summarise(run_controller(
                method="reactive", lam=lam, cfg=cfg,
                forecaster_kind="oracle"), cfg)["sla_violation_pct"])
        p, r = float(np.mean(sla_p)), float(np.mean(sla_r))
        rows.append({"workload": trace, "tau_c_s": tau_c,
                     "tau_ticks": round(tau_c / cfg.telemetry.dt, 2),
                     "H": H, "H_dt": H * cfg.telemetry.dt,
                     "covers_coldstart": bool(H * cfg.telemetry.dt >= tau_c),
                     "pact_sla": round(p, 4), "reactive_sla": round(r, 4),
                     "improvement_pp": round(r - p, 4),
                     "exact_frac": round(float(np.mean(exact)), 4),
                     "n_runs": len(sla_p)})
        print(f"  E18 {trace} tau={tau_c} H={H} ({time.time() - started:.0f}s)",
              flush=True)


def e19_robustness() -> None:
    rows: list[dict] = []
    for trace in REAL_WORKLOADS:
        _e19_family(trace, rows)
        _csv(RESULTS / "e19_robustness_real.csv", rows)


def _e19_family(trace: str, rows: list[dict]) -> None:
    cfg = base_config()
    windows = [real_window(trace, N_TICKS, 6000 + seed)[0] for seed in range(12)]
    for sigma in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        for guard in (True, False):
            sla, over, ovr = [], [], []
            for seed, lam in enumerate(windows):
                rng = np.random.default_rng(900 + seed)
                noise = rng.normal(0, sigma, len(lam))
                noisy = np.clip(lam * (1.0 + noise), 10.0, None)
                s = summarise(run_controller(method="pact", lam=lam, cfg=cfg,
                                             forecaster_kind="oracle", use_safety=guard,
                                             lam_forecast=noisy), cfg)
                sla.append(s["sla_violation_pct"])
                over.append(s["over_provision_pct"])
                ovr.append(s["overrides"])
            rows.append({"workload": trace, "forecast_noise_sigma": sigma,
                         "safety_guard": guard,
                         "sla_mean": round(float(np.mean(sla)), 4),
                         "sla_std": round(float(np.std(sla, ddof=1)), 4),
                         "over_mean": round(float(np.mean(over)), 4),
                         "overrides_total": int(np.sum(ovr)),
                         "n_runs": len(sla)})
    sla_r = [summarise(run_controller(method="reactive", lam=lam, cfg=cfg,
                                      forecaster_kind="oracle"),
                       cfg)["sla_violation_pct"]
             for lam in windows]
    rows.append({"workload": trace, "forecast_noise_sigma": -1.0, "safety_guard": True,
                 "sla_mean": round(float(np.mean(sla_r)), 4),
                 "sla_std": round(float(np.std(sla_r, ddof=1)), 4),
                 "over_mean": float("nan"), "overrides_total": "",
                 "n_runs": len(sla_r)})


E21_PARTS = RESULTS / "e21_parts"


def tcn_bias_req_s() -> float:
    """Mean TCN forecast bias over every horizon in E16, WC98/trace units."""
    with (RESULTS / "e16_forecast_accuracy.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["model"] == "tcn"]
    if not rows:
        raise SystemExit("e16_forecast_accuracy.csv has no tcn rows")
    return float(np.mean([float(r["bias_req_s"]) for r in rows]))


def e21_part(trace: str) -> None:
    """Oracle shifted by the TCN's mean bias, on every E17 window of one family.

    Tests whether the bias alone moves the closed loop the way the trained arm
    did. The shift is additive and scaled with the family, as the TCN's
    output is."""
    from experiments.harness import load_trace  # noqa: PLC0415
    E21_PARTS.mkdir(parents=True, exist_ok=True)
    cfg = base_config()
    meta = load_trace(trace)[1]
    shift = tcn_bias_req_s() * float(meta.get("extra_scale", 1.0))
    _, n_blocks = real_blocks(trace, N_TICKS)
    runs: list[dict] = []
    for block in range(n_blocks):
        lam, start = real_window(trace, N_TICKS, block)
        for m in ("pact", "rounded"):
            s = summarise(run_controller(method=m, lam=lam, cfg=cfg,
                                         forecaster_kind="oracle",
                                         lam_forecast=lam + shift), cfg)
            runs.append({"workload": trace, "block": block, "window_start": start,
                         "method": f"{m}-oracle_biased", "shift_req_s": round(shift, 6),
                         **{k: round(v, 6) for k, v in s.items()}})
    _csv(E21_PARTS / f"{trace}_runs.csv", runs)


def e21_bias_probe() -> None:
    """Merge E21 parts against E17's oracle and trained arms, per metric."""
    with (RESULTS / "e17_trained_runs.csv").open() as fh:
        e17 = list(csv.DictReader(fh))
    rows: list[dict] = []
    for trace in REAL_WORKLOADS:
        part = E21_PARTS / f"{trace}_runs.csv"
        if not part.exists():
            e21_part(trace)
        with part.open() as fh:
            biased = list(csv.DictReader(fh))
        for ctrl in ("pact", "rounded"):
            arm = {
                "oracle": [r for r in e17 if r["workload"] == trace
                           and r["method"] == f"{ctrl}-oracle"],
                "trained": [r for r in e17 if r["workload"] == trace
                            and r["method"] == f"{ctrl}-trained"],
                "biased": [r for r in biased if r["method"] == f"{ctrl}-oracle_biased"],
            }
            for metric in E17_METRICS:
                mo, mt, mb = (float(np.mean([float(r[metric]) for r in arm[k]]))
                              for k in ("oracle", "trained", "biased"))
                rows.append({
                    "workload": trace, "controller": ctrl, "metric": metric,
                    "shift_req_s": biased[0]["shift_req_s"],
                    "oracle_mean": round(mo, 4), "biased_oracle_mean": round(mb, 4),
                    "trained_mean": round(mt, 4),
                    # +1 / -1 / 0: direction each arm moved away from the oracle.
                    "trained_dir": int(np.sign(round(mt - mo, 4))),
                    "biased_dir": int(np.sign(round(mb - mo, 4))),
                    "n_runs": len(arm["biased"]),
                })
    _csv(RESULTS / "e21_bias_probe.csv", rows)


E22_PARTS = RESULTS / "e22_parts"
E22_VARIANTS = {"full": {}, "no_margin": {"use_margin": False},
                "no_safety": {"use_safety": False}}
E22_FORECASTERS = ("persist", "trained")
E22_METRICS = ("sla_violation_pct", "over_provision_pct", "scale_actions")


def e22_part(trace: str) -> None:
    """Margin and ceiling ablation for exact MPC on every E17 window of one family,
    under the two imperfect predictors. E4 ablates them only under the oracle."""
    E22_PARTS.mkdir(parents=True, exist_ok=True)
    cfg = base_config()
    _, n_blocks = real_blocks(trace, N_TICKS)
    runs: list[dict] = []
    started = time.time()
    for block in range(n_blocks):
        lam, start = real_window(trace, N_TICKS, block)
        for fc in E22_FORECASTERS:
            kw = {"trace_name": trace, "window_start": start} if fc == "trained" else {}
            for variant, flags in E22_VARIANTS.items():
                s = summarise(run_controller(method="pact", lam=lam, cfg=cfg,
                                             forecaster_kind=fc, **flags, **kw), cfg)
                runs.append({"workload": trace, "block": block, "window_start": start,
                             "forecaster": fc, "variant": variant,
                             **{k: round(v, 6) for k, v in s.items()}})
        if block % 10 == 9:
            print(f"  E22 {trace} {block + 1}/{n_blocks} windows "
                  f"({time.time() - started:.0f}s)", flush=True)
    _csv(E22_PARTS / f"{trace}_runs.csv", runs)


def e22_ablation() -> None:
    """Paired comparison of each ablated variant against the full controller."""
    rows: list[dict] = []
    pvals: list[float] = []
    for trace in REAL_WORKLOADS:
        part = E22_PARTS / f"{trace}_runs.csv"
        if not part.exists():
            e22_part(trace)
        with part.open() as fh:
            runs = list(csv.DictReader(fh))
        arms: dict[tuple[str, str], list[dict]] = {}
        for r in sorted(runs, key=lambda r: int(r["block"])):
            arms.setdefault((r["forecaster"], r["variant"]), []).append(r)
        for fc in E22_FORECASTERS:
            full = arms[(fc, "full")]
            for variant in E22_VARIANTS:
                cur = arms[(fc, variant)]
                overrides = int(sum(float(r["overrides"]) for r in cur))
                for metric in E22_METRICS:
                    a = np.array([float(r[metric]) for r in full])
                    b = np.array([float(r[metric]) for r in cur])
                    row = {"workload": trace, "forecaster": fc, "variant": variant,
                           "metric": metric, "full_mean": round(float(a.mean()), 4),
                           "variant_mean": round(float(b.mean()), 4),
                           "delta": round(float((b - a).mean()), 4),
                           "n_windows_changed": int(np.sum(np.abs(b - a) > 1e-9)),
                           "overrides_total": overrides,
                           "gamma_moved_runs": int(sum(float(r["gamma_max"]) > 0
                                                       for r in cur)),
                           "n_runs": len(cur)}
                    if variant != "full":
                        p = (1.0 if not np.any(np.abs(b - a) > 1e-9)
                             else wilcoxon_signed_rank(b.tolist(), a.tolist()).p_value)
                        row["p_raw"] = round(float(p), 6)
                        pvals.append(float(p))
                    rows.append(row)
    adj = iter(holm_bonferroni(pvals))
    for row in rows:
        if "p_raw" in row:
            row["p_holm"] = round(float(next(adj)), 6)
    _csv(RESULTS / "e22_margin_ceiling_ablation.csv", rows)


EARLY_TICKS = 30


def e20_initial_state() -> None:
    """Where violations fall in a window under the fixed and the load-sized start.

    Every family, nine windows each, reactive and exact MPC with the oracle.
    This is the measurement behind the start-state rule in the harness."""
    cfg = base_config()
    rows: list[dict] = []
    for wl in WORKLOADS:
        lams = [workload(wl, N_TICKS, 4000 + seed) for seed in range(9)]
        for rule in ("fixed_4", "load_sized"):
            for method in ("reactive", "pact"):
                tot = early = 0
                n0s = []
                for lam in lams:
                    n0 = (LEGACY_N_INITIAL if rule == "fixed_4"
                          else initial_replicas(lam, cfg))
                    out = run_controller(method=method, lam=lam, cfg=cfg,
                                         forecaster_kind="oracle", n_initial=n0)
                    v = np.asarray(out.p95, dtype=float) > cfg.capacity.slo_ms
                    tot += int(v.sum())
                    early += int(v[:EARLY_TICKS].sum())
                    n0s.append(n0)
                rows.append({"workload": wl, "start_rule": rule, "method": method,
                             "n_initial_mean": round(float(np.mean(n0s)), 3),
                             "n_initial_max": int(max(n0s)),
                             "n_windows_start_changed": sum(
                                 int(n != LEGACY_N_INITIAL) for n in n0s),
                             "violation_ticks": tot,
                             f"violation_ticks_first_{EARLY_TICKS}": early,
                             "violation_pct": round(
                                 100.0 * tot / (len(lams) * N_TICKS), 4),
                             "n_runs": len(lams)})
                print(f"  E20 {wl} {rule} {method}: {tot} ({early} early)", flush=True)
    _csv(RESULTS / "e20_initial_state.csv", rows)


STAGES = {"e17": e17_trained, "e18": e18_horizon, "e19": e19_robustness,
          "e20": e20_initial_state, "e21": e21_bias_probe, "e22": e22_ablation}

if __name__ == "__main__":
    names = sys.argv[1:] or list(STAGES)
    for name in names:
        t0 = time.time()
        print(f"=== {name}", flush=True)
        if name.startswith(("e17:", "e21:", "e22:")):
            trace = name.split(":", 1)[1]
            if trace not in REAL_WORKLOADS:
                raise SystemExit(f"unknown family {name}")
            {"e17": e17_part, "e21": e21_part, "e22": e22_part}[name[:3]](trace)
        elif name.startswith("e18:"):
            _, trace, tau = name.split(":")
            if trace not in REAL_WORKLOADS or float(tau) not in E18_TAUS:
                raise SystemExit(f"unknown E18 slice {name}")
            e18_part(trace, float(tau))
        else:
            STAGES[name]()
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
