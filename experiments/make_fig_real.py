"""Figures for the recorded-trace experiments, plotted from results/ only.

fig_wc98_trace.png         the measured series, splits shaded, one test window
fig_forecast_accuracy.png  E16 per-horizon error, every model
fig_trained_brackets.png   E17 trained arm against both brackets
fig_horizon_law_real.png   E18 cold start x horizon grid on recorded traffic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import FIG, MUTED, RES, save  # noqa: E402

TR = RES / "traces"
FAMILY = {"worldcup98": "WorldCup98, whole-series peak at 85 %",
          "worldcup98_testpeak": "WorldCup98, test-period peak at 85 %"}
MODEL = {"tcn": "TCN (PACT)", "lstm": "LSTM", "gru": "GRU", "bilstm": "Bi-LSTM",
         "transformer": "Transformer", "arima": "ARIMA", "persistence": "Persistence"}
MCOL = {"tcn": "#2a78d6", "lstm": "#eb6834", "gru": "#eda100", "bilstm": "#4a3aa7",
        "transformer": "#b8418f", "arima": "#1baf7a", "persistence": "#6b7280"}


def fig_trace():
    raw = np.load(TR / "worldcup98_raw.npy")
    meta = json.loads((TR / "worldcup98.json").read_text())
    tp = np.load(TR / "worldcup98_testpeak.npy")
    base = np.load(TR / "worldcup98.npy")
    dt = float(meta["dt_s"])
    days = np.arange(raw.size) * dt / 86400.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.3),
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    a1.plot(days, raw, color="#2a78d6", lw=0.25)
    te, ve = meta["train_end"] * dt / 86400.0, meta["val_end"] * dt / 86400.0
    a1.axvspan(te, ve, color="#eda100", alpha=0.12, lw=0)
    a1.axvspan(ve, days[-1], color="#1baf7a", alpha=0.12, lw=0)
    top = raw.max()
    for x, lab in ((te / 2, "train"), ((te + ve) / 2, "validation"),
                   ((ve + days[-1]) / 2, "test")):
        a1.text(x, top * 0.97, lab, ha="center", va="top", fontsize=6.8, color=MUTED)
    a1.set_xticks(np.arange(0, 11))
    a1.set_xticklabels([f"{d}" for d in range(5, 16)])
    a1.set_xlabel("June 1998 (Paris local time)")
    a1.set_ylabel("Measured requests/s (5 s bins)")
    a1.set_xlim(0, days[-1])
    a1.set_title("(a) Measured arrival rate, days 41-50", fontsize=8.4)

    from experiments.harness import real_window  # noqa: PLC0415
    lam_b, start = real_window("worldcup98", 400, 1000)
    lam_t, _ = real_window("worldcup98_testpeak", 400, 1000)
    assert np.allclose(lam_b, base[start:start + 400])
    assert np.allclose(lam_t, tp[start:start + 400])
    t = np.arange(400) * dt / 60.0
    a2.plot(t, lam_t, color="#eb6834", lw=0.8, label=FAMILY["worldcup98_testpeak"])
    a2.plot(t, lam_b, color="#2a78d6", lw=0.8, label=FAMILY["worldcup98"])
    a2.axhline(640.0, color=MUTED, ls=":", lw=0.8)
    a2.text(t[0], 648.0, "pool capacity", ha="left", va="bottom", fontsize=6.4,
            color=MUTED)
    a2.set_ylim(0, 720)
    a2.set_xlabel("Minutes into window")
    a2.set_ylabel("Rate fed to the pool (req/s)")
    a2.set_title("(b) One 400-tick test window", fontsize=8.4)
    a2.legend(fontsize=5.8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    save(fig, "fig_wc98_trace.png")


def fig_accuracy():
    d = pd.read_csv(RES / "e16_forecast_accuracy.csv")
    d = d[d.seed == 0]
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    order = [m for m in MODEL if m in set(d.model)]
    for m in order:
        s = d[d.model == m].sort_values("horizon")
        ax.plot(s.horizon, s.nmae_pct, color=MCOL[m], marker="o", ms=2.4,
                lw=1.6 if m == "tcn" else 1.0,
                ls="--" if m in ("persistence", "arima") else "-", label=MODEL[m])
    ax.set_xlabel("Forecast step $h$ (5 s ticks)")
    ax.set_ylabel("MAE / mean arrival rate (%)")
    ax.set_xticks(range(1, 13))
    ax.legend(ncol=2, fontsize=6.0)
    save(fig, "fig_forecast_accuracy.png")


def fig_brackets():
    s = pd.read_csv(RES / "e17_trained_summary.csv")
    fams = [f for f in FAMILY if f in set(s.workload)]
    metrics = [("sla_violation_pct", "SLA violation (%)"),
               ("over_provision_pct", "Over-provisioning (%)"),
               ("scale_actions", "Scaling actions per run")]
    arms = [("oracle", "Oracle", "#2a78d6"), ("trained", "Trained TCN", "#b8418f"),
            ("persist", "Persistence", "#1baf7a")]
    fig, axes = plt.subplots(len(fams), 3, figsize=(6.9, 1.9 * len(fams)),
                             squeeze=False)
    for i, fam in enumerate(fams):
        sf = s[s.workload == fam].set_index("method")
        for j, (key, lab) in enumerate(metrics):
            ax = axes[i][j]
            x = np.arange(2)
            w = 0.26
            for k, (fk, flab, col) in enumerate(arms):
                rows = sf.loc[[f"pact-{fk}", f"rounded-{fk}"]]
                y = rows[f"{key}_mean"].to_numpy()
                err = [y - rows[f"{key}_ci_lo"].to_numpy(),
                       rows[f"{key}_ci_hi"].to_numpy() - y]
                ax.bar(x + (k - 1) * w, y, w * 0.92, yerr=err, color=col,
                       edgecolor="white", lw=0.5, label=flab,
                       error_kw=dict(lw=0.7, capsize=1.6, ecolor="#444"))
            ax.set_xticks(x)
            ax.set_xticklabels(["Integer-exact", "Relaxed + rounded"], fontsize=6.8)
            ax.xaxis.grid(False)
            ax.set_ylabel(lab, fontsize=6.8)
            if j == 0:
                ax.set_title(FAMILY[fam], fontsize=7.4, loc="left")
            if i == 0 and j == 2:
                ax.legend(fontsize=6.0, loc="upper right")
    n = int(s.n_runs.iloc[0])
    fig.text(0.5, -0.02, f"{n} disjoint 400-tick test windows per bar; "
             "95 % bootstrap intervals.", ha="center", fontsize=6.6, color=MUTED)
    save(fig, "fig_trained_brackets.png")


def fig_horizon_real():
    d = pd.read_csv(RES / "e18_horizon_law_real.csv")
    fams = [f for f in FAMILY if f in set(d.workload)]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.35 * len(fams), 2.5),
                             squeeze=False)
    vmax = float(d.pact_sla.max()) or 1.0
    for ax, fam in zip(axes[0], fams, strict=True):
        piv = d[d.workload == fam].pivot(index="tau_c_s", columns="H",
                                         values="pact_sla")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="YlOrRd",
                       vmin=0.0, vmax=vmax,
                       extent=[0, len(piv.columns), 0, len(piv.index)])
        ax.set_xticks(np.arange(len(piv.columns)) + 0.5)
        ax.set_xticklabels(piv.columns, fontsize=6.6)
        ax.set_yticks(np.arange(len(piv.index)) + 0.5)
        ax.set_yticklabels([f"{int(v)}" for v in piv.index])
        ax.set_xlabel("Forecast horizon $H$ (ticks)")
        ax.set_ylabel("Cold start $\\tau_c$ (s)")
        cols = np.asarray(piv.columns, dtype=float)
        xs = [min(np.searchsorted(cols, tau / 5.0, side="left"), len(cols))
              for tau in piv.index]
        ax.step(xs, np.arange(len(piv.index)) + 0.5, where="mid", color="#1a1a1a",
                lw=1.6)
        ax.set_title(FAMILY[fam], fontsize=7.4)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label("SLA violation (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "fig_horizon_law_real.png", layout=False)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    FIG.mkdir(exist_ok=True)
    for name, fn, need in (
        ("trace", fig_trace, TR / "worldcup98.json"),
        ("accuracy", fig_accuracy, RES / "e16_forecast_accuracy.csv"),
        ("brackets", fig_brackets, RES / "e17_trained_summary.csv"),
        ("horizon", fig_horizon_real, RES / "e18_horizon_law_real.csv"),
    ):
        if need.exists():
            fn()
        else:
            print(f"  SKIP {name}: {need.name} missing")
