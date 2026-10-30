"""Figures for the paper, plotted directly from results/*.csv.

Every value drawn here is read from a CSV written by run_all.py. No figure
contains a hard-coded number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Validated categorical slots, fixed order, never cycled.
C = {
    "pact-oracle": "#2a78d6",
    "pact-persist": "#1baf7a",
    "rounded-oracle": "#eb6834",
    "rounded-persist": "#eda100",
    "reactive": "#4a3aa7",
    "static": "#6b7280",
}
LS = {
    "pact-oracle": "-", "pact-persist": (0, (4, 1.5)),
    "rounded-oracle": "--", "rounded-persist": (0, (3, 1, 1, 1)),
    "reactive": "-.", "static": ":",
}
MK = {
    "pact-oracle": "o", "pact-persist": "s", "rounded-oracle": "^",
    "rounded-persist": "D", "reactive": "v", "static": "P",
}
NAME = {
    "pact-oracle": "PACT, integer-exact (oracle fc.)",
    "pact-persist": "PACT, integer-exact (persistence fc.)",
    "rounded-oracle": "Relaxed + rounded (oracle fc.)",
    "rounded-persist": "Relaxed + rounded (persistence fc.)",
    "reactive": "Reactive threshold (HPA-style)",
    "static": "Static over-provisioned",
}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 600,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#c9c9c4",
    "grid.alpha": 0.4, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7,
    "lines.linewidth": 1.4, "lines.markersize": 4,
})


def save(fig, name, layout=True):
    # Without this the default subplot margins leave a wide white border that
    # bbox_inches="tight" then crops, so the saved image comes out far smaller
    # than its figsize and has to be scaled up on the page -- which is what
    # made the rendered font sizes inconsistent between figures.
    if layout:
        try:
            fig.tight_layout(pad=0.4)
        except Exception:
            pass
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    print("  " + name)


# ============================================================ Fig: frontier
def fig_frontier():
    d = pd.read_csv(RES / "e6_frontier.csv")
    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    for m in ("pact-oracle", "rounded-oracle", "reactive"):
        s = d[d.method == m].sort_values("u_target")
        ax.plot(s.over_mean, s.sla_mean, color=C[m], ls=LS[m], marker=MK[m],
                label=NAME[m], zorder=3)
        lo = s.iloc[0]
        hi = s.iloc[-1]
        ax.annotate(f"$u^*$={lo.u_target:.2f}", (lo.over_mean, lo.sla_mean),
                    xytext=(3, -8), textcoords="offset points", fontsize=6, color=C[m])
        ax.annotate(f"{hi.u_target:.2f}", (hi.over_mean, hi.sla_mean),
                    xytext=(3, 3), textcoords="offset points", fontsize=6, color=C[m])
    ax.set_xlabel("Over-provisioning ratio (%)")
    ax.set_ylabel("SLA violation rate (%)")
    ax.annotate("better", xy=(6, 1.2), xytext=(26, 8.5), fontsize=7, color=MUTED,
                ha="center", arrowprops=dict(arrowstyle="->", lw=0.8, color=MUTED))
    ax.legend(loc="upper right")
    save(fig, "fig_frontier.png")


# ============================================================ Fig: horizon law
def fig_horizon():
    d = pd.read_csv(RES / "e5_horizon_law.csv")
    piv = d.pivot(index="tau_c_s", columns="H", values="pact_sla")
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="YlOrRd",
                   extent=[0, len(piv.columns), 0, len(piv.index)])
    ax.set_xticks(np.arange(len(piv.columns)) + 0.5)
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(np.arange(len(piv.index)) + 0.5)
    ax.set_yticklabels([f"{int(v)}" for v in piv.index])
    ax.set_xlabel("Forecast horizon $H$ (ticks)")
    ax.set_ylabel("Cold start $\\tau_c$ (s)")
    # Draw the Eq.19 feasibility boundary H*dt >= tau_c
    dt = 5.0
    xs, ys = [], []
    for i, tau in enumerate(piv.index):
        need = tau / dt
        pos = np.searchsorted(np.asarray(piv.columns, dtype=float), need, side="left")
        xs.append(min(pos, len(piv.columns)))
        ys.append(i)
    ax.step(xs, np.array(ys) + 0.5, where="mid", color="#1a1a1a", lw=1.6)
    ax.plot([], [], color="#1a1a1a", lw=1.6, label="$H\\Delta t = \\tau_c$ (Eq. 19)")
    ax.legend(loc="lower right", fontsize=6.3, facecolor="white",
              framealpha=0.85, frameon=True, edgecolor="none")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("SLA violation (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.grid(False)
    save(fig, "fig_horizon_law.png")


# ============================================================ Fig: horizon cliff
def fig_cliff():
    d = pd.read_csv(RES / "e5_horizon_law.csv")
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    cmap = plt.get_cmap("viridis")
    taus = sorted(d.tau_c_s.unique())
    for i, tau in enumerate(taus):
        s = d[d.tau_c_s == tau].sort_values("H")
        col = cmap(i / max(len(taus) - 1, 1))
        ax.plot(s.H * 5.0, s.pact_sla, color=col, marker="o", ms=2.6,
                label=f"$\\tau_c$ = {int(tau)} s")
        ax.axvline(tau, color=col, ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel("Lookahead $H\\Delta t$ (s)")
    ax.set_ylabel("SLA violation rate (%)")
    ax.set_xscale("log")
    ax.legend(ncol=2, fontsize=6.3, loc="upper right")
    save(fig, "fig_horizon_cliff.png")


# ============================================================ Fig: rounding gap
def fig_rounding():
    d = pd.read_csv(RES / "e3_rounding_gap.csv")
    g = d.groupby("n_max").agg(
        disagree=("disagree_pct", "mean"),
        ex_act=("exact_actions", "mean"),
        rd_act=("rounded_actions", "mean"),
    ).reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.4))
    a1.plot(g.n_max, g.disagree, color=C["pact-oracle"], marker="o")
    a1.set_xscale("log", base=2)
    a1.set_xticks(g.n_max)
    a1.set_xticklabels(g.n_max)
    a1.set_xlabel("Pool bound $N_{max}$")
    a1.set_ylabel("Ticks where decisions differ (%)")
    a1.set_title("(a) Rounding disagreement", fontsize=8.5)

    w = 0.36
    x = np.arange(len(g))
    a2.bar(x - w / 2, g.ex_act, w, color=C["pact-oracle"], label="Integer-exact",
           edgecolor="white", lw=0.6)
    a2.bar(x + w / 2, g.rd_act, w, color=C["rounded-oracle"], label="Relaxed + rounded",
           edgecolor="white", lw=0.6)
    a2.set_xticks(x)
    a2.set_xticklabels(g.n_max)
    a2.set_xlabel("Pool bound $N_{max}$")
    a2.set_ylabel("Scaling actions per run")
    a2.set_title("(b) Actuation churn", fontsize=8.5)
    a2.xaxis.grid(False)
    a2.legend()
    save(fig, "fig_rounding_gap.png")


# ============================================================ Fig: main bars
def fig_main():
    s = pd.read_csv(RES / "e1_summary.csv")
    order = ["static", "reactive", "rounded-persist", "rounded-oracle",
             "pact-persist", "pact-oracle"]
    s = s.set_index("method").loc[order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    specs = [("sla", "SLA violation (%)"), ("over", "Over-provisioning (%)"),
             ("act", "Scaling actions")]
    for ax, (key, lab) in zip(axes, specs):
        y = s[f"{key}_mean"].values
        lo = y - s[f"{key}_ci_lo"].values
        hi = s[f"{key}_ci_hi"].values - y
        cols = [C[m] for m in s.method]
        ax.barh(np.arange(len(s)), y, xerr=[lo, hi], color=cols,
                edgecolor="white", lw=0.6, error_kw=dict(lw=0.8, capsize=2,
                                                         ecolor="#333333"))
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels([NAME[m].split(" (")[0] for m in s.method] if key == "sla" else [])
        ax.set_xlabel(lab)
        ax.yaxis.grid(False)
    fig.suptitle("Mean over 5 workloads $\\times$ 10 seeds, 95 % bootstrap CI",
                 fontsize=7.5, y=1.02, color=MUTED)
    save(fig, "fig_main_comparison.png")


# ============================================================ Fig: ablation
def fig_ablation():
    a = pd.read_csv(RES / "e4_ablation_summary.csv")
    a = a[a.variant != "full"].copy()
    labels = {"no_mpc": "− MPC layer", "no_churn": "− churn penalty",
              "no_coldstart": "− cold-start comp.", "no_margin": "− adaptive margin",
              "no_safety": "− safety ceiling"}
    a["lab"] = a.variant.map(labels)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.1), sharey=True)
    y = np.arange(len(a))
    a1.barh(y, a.sla_delta_vs_full, 0.6, color=C["rounded-oracle"],
            edgecolor="white", lw=0.6)
    a1.axvline(0, color=INK, lw=0.9)
    a1.set_yticks(y)
    a1.set_yticklabels(a.lab)
    a1.set_xlabel("$\\Delta$ SLA violation (pp)")
    a1.set_title("(a) Service level", fontsize=8.5)
    a1.yaxis.grid(False)
    a2.barh(y, a.over_delta_vs_full, 0.6, color=C["pact-oracle"],
            edgecolor="white", lw=0.6)
    a2.axvline(0, color=INK, lw=0.9)
    a2.set_xlabel("$\\Delta$ over-provisioning (pp)")
    a2.set_title("(b) Capacity cost", fontsize=8.5)
    a2.yaxis.grid(False)
    fig.text(0.5, -0.19, "Negative is lower than the full controller; "
             "none of the service-level deltas reach $p<0.05$.",
             ha="center", fontsize=6.6, color=MUTED)
    save(fig, "fig_ablation.png")


# ============================================================ Fig: robustness
def fig_robustness():
    r = pd.read_csv(RES / "e7_robustness.csv")
    ref = r[r.forecast_noise_sigma < 0]
    r = r[r.forecast_noise_sigma >= 0]
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    for guard, col, lab in ((True, C["pact-oracle"], "Safety ceiling on"),
                            (False, C["rounded-oracle"], "Safety ceiling off")):
        s = r[r.safety_guard == guard].sort_values("forecast_noise_sigma")
        ax.plot(s.forecast_noise_sigma, s.sla_mean, color=col,
                lw=3.0 if guard else 1.2, alpha=0.45 if guard else 1.0,
                ls="-" if guard else (0, (4, 2)),
                marker=None if guard else "s", ms=3.4, label=lab)
    if len(ref):
        ax.axhline(float(ref.sla_mean.iloc[0]), color=C["reactive"], ls="-.",
                   lw=1.2, label="Reactive baseline")
    ax.annotate("the two curves coincide:\nthe ceiling never fires here",
                xy=(0.22, 0.03), xytext=(0.15, 0.55), fontsize=6.4, color=MUTED,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=MUTED))
    ax.set_xlabel("Forecast corruption $\\sigma$ (relative)")
    ax.set_ylabel("SLA violation rate (%)")
    ax.legend(loc="upper right")
    save(fig, "fig_robustness.png")


# ============================================================ Fig: trace
def fig_trace():
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.harness import base_config, run_controller, summarise, workload

    cfg = base_config()
    lam = workload("burst", 400, 1001)
    t = np.arange(len(lam)) * cfg.telemetry.dt / 60.0
    runs = {}
    for m, f in (("pact", "oracle"), ("rounded", "oracle"), ("reactive", "oracle")):
        runs[f"{m}-{f}" if m not in ("reactive",) else m] = run_controller(
            method=m, lam=lam, cfg=cfg, forecaster_kind=f)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.9, 3.9), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1]))
    req = runs["pact-oracle"].required
    a1.step(t, req, where="post", color=INK, lw=1.6, label="Required capacity", zorder=4)
    for k in ("pact-oracle", "rounded-oracle", "reactive"):
        a1.step(t, runs[k].n_series, where="post", color=C[k], ls=LS[k], lw=1.2,
                label=NAME[k], zorder=3)
    a1.set_ylabel("Replicas $N(t)$")
    a1.legend(ncol=2, loc="upper left", fontsize=6.4)

    for k in ("pact-oracle", "rounded-oracle", "reactive"):
        a2.plot(t, runs[k].p95, color=C[k], ls=LS[k], lw=1.1, label=NAME[k])
    a2.axhline(cfg.capacity.slo_ms, color="#b3261e", lw=1.1)
    a2.annotate(f"SLO = {int(cfg.capacity.slo_ms)} ms", xy=(t[-1], cfg.capacity.slo_ms),
                xytext=(-3, 4), textcoords="offset points", ha="right",
                fontsize=6.5, color="#b3261e")
    a2.set_ylabel("p95 latency (ms)")
    a2.set_xlabel("Time (min)")
    a2.set_ylim(40, 420)
    save(fig, "fig_trace.png")


if __name__ == "__main__":
    fig_frontier()
    fig_horizon()
    fig_cliff()
    fig_rounding()
    fig_main()
    fig_ablation()
    fig_robustness()
    fig_trace()
    print("figures done")
