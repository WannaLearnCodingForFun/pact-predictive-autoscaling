"""Figures for the expanded sections, plotted from results/e8..e12."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

C = {"pact": "#2a78d6", "pact-oracle": "#2a78d6", "pact-persist": "#1baf7a",
     "rounded": "#eb6834", "rounded-oracle": "#eb6834", "reactive": "#4a3aa7",
     "static": "#6b7280"}
LS = {"pact": "-", "rounded": "--", "reactive": "-.", "static": ":"}
MK = {"pact": "o", "rounded": "^", "reactive": "v", "static": "P"}
NM = {"pact": "Integer-exact MPC", "rounded": "Relaxed + rounded",
      "reactive": "Reactive threshold", "static": "Static"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 400,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.6, "axes.labelsize": 8,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#c9c9c4", "grid.alpha": 0.4,
    "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7,
    "lines.linewidth": 1.4, "lines.markersize": 3.8,
})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    print("  " + name)


# ===================================================== control sensitivity
def fig_sensitivity():
    s = pd.read_csv(RES / "e8_sensitivity.csv")
    panels = [
        ("churn_weight_s", "Churn weight $s$", False),
        ("cooldown_s", "Cooldown $T_{cool}$ (s)", False),
        ("deadband_theta", "Deadband $\\theta$ (replicas)", False),
        ("asymmetry_qup_over_qdown", "Asymmetry $q^+/q^-$", True),
    ]
    fig, axgrid = plt.subplots(2, 2, figsize=(6.7, 3.5))
    axes = axgrid.ravel()
    fig.subplots_adjust(wspace=0.55, hspace=0.62)
    for ax, (key, lab, logx) in zip(axes, panels):
        d = s[s.parameter == key].sort_values("value")
        ax.plot(d.value, d.sla_mean, color=C["pact"], marker="o",
                label="SLA violation (%)")
        ax.set_xlabel(lab)
        if logx:
            ax.set_xscale("log")
        ax2 = ax.twinx()
        ax2.plot(d.value, d.osc_mean, color=C["rounded"], ls="--", marker="^",
                 ms=3.2, label="Oscillations")
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(labelsize=6.6, colors=C["rounded"])
        ax2.grid(False)
        ax.set_ylabel("SLA viol. (%)", color=C["pact"], fontsize=7.2)
        ax.tick_params(axis="y", colors=C["pact"])
        ax2.set_ylabel("Oscillations", color=C["rounded"], fontsize=7.2)
    fig.text(0.5, -0.045, "Left axis (blue circles): service-level violation. "
             "Right axis (orange triangles): oscillation count. "
             "Synthetic traces, 5 families × 8 seeds per point.",
             ha="center", fontsize=6.6, color=MUTED)
    save(fig, "fig_sensitivity.png")


# ===================================================== cooldown trade-off
def fig_cooldown():
    s = pd.read_csv(RES / "e8_sensitivity.csv")
    d = s[s.parameter == "cooldown_s"].sort_values("value")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(d.act_mean, d.sla_mean, color=C["pact"], marker="o", zorder=3)
    for r in d.itertuples():
        ax.annotate(f"{int(r.value)} s", (r.act_mean, r.sla_mean),
                    xytext=(5, 4), textcoords="offset points", fontsize=6.6,
                    color=MUTED)
    ax.set_xlabel("Scaling actions per run")
    ax.set_ylabel("SLA violation rate (%)")
    ax.set_xscale("log")
    ax.annotate("better", xy=(31, 0.30), xytext=(70, 0.95), fontsize=7,
                color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=MUTED))
    save(fig, "fig_cooldown.png")


# ===================================================== solve time
def fig_solvetime():
    s = pd.read_csv(RES / "e9_solvetime.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.3))
    cmap = plt.get_cmap("viridis")
    ds = sorted(s.delta_max.unique())
    for i, dm in enumerate(ds):
        d = s[s.delta_max == dm].sort_values("H")
        a1.plot(d.H, d.mean_solve_ms, color=cmap(i / max(len(ds) - 1, 1)),
                marker="o", label=f"$\\Delta_{{max}}$ = {dm}")
    a1.set_xlabel("Horizon $H$ (ticks)")
    a1.set_ylabel("Mean solve time (ms)")
    a1.set_title("(a) Scaling with horizon and action set", fontsize=8.4)
    a1.legend(ncol=2, fontsize=6.5)

    s = s.assign(work=s.candidates * s.H)
    a2.scatter(s.work, s.mean_solve_ms, s=18, color=C["pact"],
               edgecolor="white", lw=0.5, zorder=3)
    k = np.polyfit(s.work, s.mean_solve_ms, 1)
    xs = np.linspace(s.work.min(), s.work.max(), 50)
    a2.plot(xs, np.polyval(k, xs), color=MUTED, lw=1.0, ls="--",
            label=f"linear fit, slope {k[0]*1000:.2f} µs per unit")
    a2.set_xlabel("Work proxy $(2\\Delta_{max}+1) \\times H$")
    a2.set_ylabel("Mean solve time (ms)")
    a2.set_title("(b) Cost is linear in the enumerated work", fontsize=8.4)
    a2.legend(fontsize=6.8)
    save(fig, "fig_solvetime.png")


# ===================================================== cold-start robustness
def fig_coldstart():
    c = pd.read_csv(RES / "e12_coldstart.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.4))
    for m in ("pact", "rounded", "reactive"):
        d = c[c.method == m].sort_values("tau_c_s")
        a1.plot(d.tau_c_s, d.sla_mean, color=C[m], ls=LS[m], marker=MK[m],
                label=NM[m])
        a2.plot(d.tau_c_s, d.act_mean, color=C[m], ls=LS[m], marker=MK[m],
                label=NM[m])
    a1.set_xlabel("Cold start $\\tau_c$ (s)")
    a1.set_ylabel("SLA violation rate (%)")
    a1.set_title("(a) Service level", fontsize=8.4)
    a1.legend()
    a2.set_xlabel("Cold start $\\tau_c$ (s)")
    a2.set_ylabel("Scaling actions per run")
    a2.set_title("(b) Actuation churn", fontsize=8.4)
    fig.text(0.5, -0.1, "Horizon fixed at $H$ = 30 (150 s lookahead), which "
             "satisfies Eq. (6) at every $\\tau_c$ shown. Synthetic traces, "
             "3 families × 5 seeds.", ha="center", fontsize=6.6, color=MUTED)
    save(fig, "fig_coldstart.png")


# ===================================================== per-workload
def fig_per_workload():
    d = pd.read_csv(RES / "e10_per_workload.csv")
    wls = list(dict.fromkeys(d.workload))
    methods = ["pact-oracle", "rounded-oracle", "reactive", "static"]
    key = {"pact-oracle": "pact", "rounded-oracle": "rounded",
           "reactive": "reactive", "static": "static"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    x = np.arange(len(wls))
    w = 0.78 / len(methods)
    for i, m in enumerate(methods):
        sub = d[d.method == m].set_index("workload").loc[wls]
        pos = x + i * w - 0.39 + w / 2
        a1.bar(pos, sub.sla_mean, w * 0.9, yerr=sub.sla_std,
               color=C[key[m]], edgecolor="white", lw=0.5,
               error_kw=dict(lw=0.7, capsize=1.6, ecolor="#444"),
               label=NM[key[m]])
        a2.bar(pos, sub.over_mean, w * 0.9, yerr=sub.over_std,
               color=C[key[m]], edgecolor="white", lw=0.5,
               error_kw=dict(lw=0.7, capsize=1.6, ecolor="#444"))
    for ax, lab in ((a1, "SLA violation (%)"), (a2, "Over-provisioning (%)")):
        ax.set_xticks(x)
        ax.set_xticklabels([w_.capitalize() for w_ in wls], fontsize=7.2)
        ax.set_ylabel(lab)
        ax.xaxis.grid(False)
    a1.legend(ncol=2, fontsize=6.4, loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.text(0.5, -0.07, "Synthetic traces, 10 seeds per bar; whiskers are one "
             "standard deviation.", ha="center", fontsize=6.6, color=MUTED)
    save(fig, "fig_per_workload.png")


# ===================================================== decision classes
def fig_decisions():
    d = pd.read_csv(RES / "e11_decisions.csv")
    order = ["reactive", "rounded-oracle", "pact-persist", "pact-oracle"]
    lab = {"reactive": "Reactive", "rounded-oracle": "Rounded (oracle)",
           "pact-persist": "Exact (persistence)", "pact-oracle": "Exact (oracle)"}
    col = {"reactive": C["reactive"], "rounded-oracle": C["rounded"],
           "pact-persist": C["pact-persist"], "pact-oracle": C["pact"]}
    d = d.set_index("method").loc[order].reset_index()
    metrics = [("scaleup_recall", "Scale-up recall"),
               ("scaleup_precision", "Scale-up precision"),
               ("hold_f1", "Hold F1"), ("accuracy", "Overall accuracy")]
    fig, ax = plt.subplots(figsize=(6.7, 2.3))
    x = np.arange(len(metrics))
    w = 0.78 / len(d)
    for i, r in d.iterrows():
        pos = x + i * w - 0.39 + w / 2
        vals = [r[f"{k}_mean"] for k, _ in metrics]
        lo = [r[f"{k}_mean"] - r[f"{k}_lo"] for k, _ in metrics]
        hi = [r[f"{k}_hi"] - r[f"{k}_mean"] for k, _ in metrics]
        ax.bar(pos, vals, w * 0.9, yerr=[lo, hi], color=col[r.method],
               edgecolor="white", lw=0.5, label=lab[r.method],
               error_kw=dict(lw=0.7, capsize=1.8, ecolor="#444"))
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.xaxis.grid(False)
    ax.legend(ncol=4, fontsize=6.8, loc="upper left", bbox_to_anchor=(0, 1.19))
    fig.text(0.5, -0.09, "Each tick classified into scale-up / hold / scale-down "
             "against the observed requirement. Synthetic traces, 50 runs per bar, "
             "95 % bootstrap intervals.", ha="center", fontsize=6.6, color=MUTED)
    save(fig, "fig_decisions.png")


if __name__ == "__main__":
    fig_sensitivity()
    fig_cooldown()
    fig_solvetime()
    fig_coldstart()
    fig_per_workload()
    fig_decisions()
    print("done")


def fig_predictor_quality():
    d = pd.read_csv(RES / "e13_predictor_quality.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.4))
    for m, lab in (("pact", "Integer-exact MPC"), ("rounded", "Relaxed + rounded")):
        s = d[d.method == m].sort_values("oracle_weight")
        a1.errorbar(s.oracle_weight, s.sla_mean, yerr=s.sla_std / np.sqrt(s.n_runs),
                    color=C[m], ls=LS[m], marker=MK[m], capsize=2, lw=1.4, label=lab)
        a2.plot(s.oracle_weight, s.act_mean, color=C[m], ls=LS[m], marker=MK[m],
                label=lab)
        if m == "pact":
            b = s.loc[s.sla_mean.idxmin()]
            a1.scatter([b.oracle_weight], [b.sla_mean], s=70, facecolor="none",
                       edgecolor=C[m], lw=1.4, zorder=5)
            a1.annotate(f"minimum at $w$ = {b.oracle_weight:g}",
                        (b.oracle_weight, b.sla_mean), xytext=(-58, 18),
                        textcoords="offset points", fontsize=6.8, color=MUTED,
                        ha="center",
                        arrowprops=dict(arrowstyle="->", lw=0.7, color=MUTED))
    for ax, lab in ((a1, "SLA violation rate (%)"), (a2, "Scaling actions per run")):
        ax.set_xlabel("Predictor quality $w$", fontsize=7.6)
        ax.set_ylabel(lab)
    a1.set_title("(a) Service level", fontsize=8.4)
    a2.set_title("(b) Actuation churn", fontsize=8.4)
    a1.legend(fontsize=6.8)
    fig.text(0.5, -0.07, "$w$ = 0 is persistence, $w$ = 1 is perfect foresight. "
             "persistence. Synthetic traces, 5 families × 6 seeds per point; "
             "bars are the standard error.", ha="center", fontsize=6.6, color=MUTED)
    save(fig, "fig_predictor_quality.png")


if True:
    fig_predictor_quality()
