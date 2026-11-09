"""Figure for Section V-I: exact solve against greedy horizon completion.

Three panels, all drawn from results/e15_ticks.csv:
  (a) where the two solvers disagree, as a function of the exact action
  (b) the distribution of the greedy objective excess, which is one-sided
  (c) solve time for both, on a log axis because they differ by ~100x
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical slots: blue = exact, orange = greedy completion.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#101418", "#55606c", "#d7dce2"

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 600,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 7.6, "axes.labelsize": 7.8, "axes.titlesize": 8.4,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7.0,
})


def load():
    rows = list(csv.DictReader(open(RES / "e15_ticks.csv")))
    out = []
    for r in rows:
        if not r["greedy_cost_true"]:
            continue
        out.append({
            "eu": int(r["exact_u"]), "gu": int(r["greedy_u"]),
            "agree": int(r["agree"]) == 1,
            "rel": float(r["rel_excess_pct"]),
            "abs": float(r["abs_excess"]),
            "et": float(r["exact_ms"]), "gt": float(r["greedy_ms"]),
        })
    return out


def main() -> None:
    d = load()
    meta = json.loads((RES / "e15_exact_vs_greedy.json").read_text())
    pc = meta["paper_config"]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.9, 2.25))

    # ---- (a) disagreement rate by the exact action
    us = sorted({r["eu"] for r in d})
    tot = [sum(1 for r in d if r["eu"] == u) for u in us]
    dis = [sum(1 for r in d if r["eu"] == u and not r["agree"]) for u in us]
    pct = [100.0 * x / max(n, 1) for x, n in zip(dis, tot)]
    a1.bar(us, pct, color=ORANGE, width=0.72, edgecolor="none")
    a1.set_xlabel("exact action $u^{*}(t)$ (replicas)")
    a1.set_ylabel("ticks where greedy differs (%)")
    a1.set_title("(a) Where they disagree")
    a1.set_xticks(us)
    a1.axhline(100.0 - pc["agree_pct"], color=MUTED, lw=0.8, ls=(0, (4, 3)))
    a1.annotate(f"overall {100.0 - pc['agree_pct']:.1f}%",
                xy=(us[0], 100.0 - pc["agree_pct"]),
                xytext=(0, 4), textcoords="offset points",
                fontsize=6.4, color=MUTED)

    # ---- (b) objective excess, one-sided by construction
    rel = np.array([r["rel"] for r in d])
    pos = rel[rel > 1e-9]
    a2.hist(pos, bins=np.logspace(np.log10(max(pos.min(), 1e-3)),
                                  np.log10(pos.max()), 26),
            color=ORANGE, edgecolor="none")
    a2.set_xscale("log")
    a2.set_xlabel("greedy objective excess (%)")
    a2.set_ylabel("ticks")
    a2.set_title("(b) Cost of the approximation")
    a2.annotate(f"{100.0 * len(pos) / len(rel):.0f}% of ticks worse\n"
                f"0% better\nmedian {np.median(pos):.2f}%",
                xy=(0.97, 0.94), xycoords="axes fraction",
                ha="right", va="top", fontsize=6.4, color=MUTED)

    # ---- (c) solve time
    et = np.array([r["et"] for r in d])
    gt = np.array([r["gt"] for r in d])
    for arr, c, lab in ((gt, ORANGE, "greedy completion"), (et, BLUE, "exact")):
        xs = np.sort(arr)
        a3.plot(xs, 100.0 * np.arange(1, len(xs) + 1) / len(xs),
                color=c, lw=1.5, label=lab)
    a3.set_xscale("log")
    a3.set_xlabel("solve time per tick (ms)")
    a3.set_ylabel("ticks at or below (%)")
    a3.set_title("(c) What exactness costs")
    a3.axvline(5000.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    a3.annotate("control\ninterval\n5000 ms", xy=(5000.0, 52),
                xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=6.2, color=MUTED)
    a3.set_xlim(0.1, 9000)
    a3.legend(loc="center left", bbox_to_anchor=(0.02, 0.76))

    for ax in (a1, a2, a3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    path = OUT / "fig_exact_vs_greedy.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    print(f"  {path.name}")


if __name__ == "__main__":
    main()
