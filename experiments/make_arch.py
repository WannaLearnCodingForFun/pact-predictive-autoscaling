"""Architecture figure matching the code that was actually executed.

Component names are bold. Every box corresponds to a module in the repository
and is annotated with its import path, so the figure can be checked against the
source. Modules that exist but were NOT exercised in this evaluation are drawn
with a dashed outline and labelled as such.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

INK, GREY, LINE = "#101418", "#55606c", "#9aa2ad"
BAND, BOXF = "#f4f6f8", "#ffffff"
PLANT, CTRL, ENF, OFF = "#b35a21", "#2a78d6", "#3f4650", "#9aa2ad"

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})

fig, ax = plt.subplots(figsize=(11.4, 5.9), dpi=260)
ax.set_xlim(0, 118)
ax.set_ylim(0, 60.5)
ax.axis("off")


def band(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=1.1",
                                fc=BAND, ec=LINE, lw=0.8, ls=(0, (5, 3))))
    ax.text(x + 1.3, y + h - 1.9, label, fontsize=8.0, color=GREY,
            fontweight="bold", va="center")


def box(x, y, w, h, title, sub, path, accent, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
        fc=BOXF, ec=accent, lw=1.25, ls=(0, (4, 2)) if dashed else "-"))
    ax.add_patch(Rectangle((x, y), 0.55, h, fc=accent, ec="none"))
    cx = x + w / 2 + 0.3
    ax.text(cx, y + h - 2.1, title, fontsize=8.3, color=INK,
            fontweight="bold", ha="center", va="center")
    ax.text(cx, y + h - 4.7, sub, fontsize=6.6, color=GREY, ha="center",
            va="center", linespacing=1.35)
    ax.text(cx, y + 1.4, path, fontsize=5.9, color=OFF, ha="center",
            va="center", style="italic", family="monospace")


def arrow(p, q, color=INK, lw=1.15, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=color, lw=lw,
                                 linestyle=ls, mutation_scale=11,
                                 shrinkA=1.4, shrinkB=1.4))


def tag(x, y, s, color=GREY, fs=6.6, ha="center"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center", style="italic")


band(2, 41.0, 114, 17.5, "PLANT")
band(2, 18.5, 114, 20.5, "PREDICTIVE CONTROL PLANE")
band(2, 2.0, 114, 14.5, "ENFORCEMENT")

# ---------------- plant
box(5, 45.0, 23, 10.5, "Workload Generator",
    "five synthetic arrival traces\ndiurnal, burst, ramp, step, sawtooth",
    "experiments/harness.py", PLANT)
box(31, 45.0, 25, 10.5, "Pool Simulator",
    "M/G/c queue, cold-start delay\nKingman p95 (Eq. 18)",
    "pact/sim/queue_model.py", PLANT)
box(59, 45.0, 25, 10.5, "M1  Telemetry",
    "observation vector (Eq. 1)\nu, r, λ, p95, N",
    "pact/telemetry/collector.py", PLANT)
box(87, 45.0, 26, 10.5, "M2  Forecaster",
    "oracle / persistence brackets\nTCN not run: torch unavailable",
    "pact/forecast/tcn.py", PLANT, dashed=True)

arrow((28, 50.2), (31, 50.2), color=PLANT)
arrow((56, 50.2), (59, 50.2), color=PLANT)
arrow((84, 50.2), (87, 50.2), color=PLANT)
tag(29.5, 56.0, "λ(t)")
tag(57.5, 52.0, "state")
tag(85.5, 52.0, "X(t)")

arrow((93, 45.0), (88, 33.5), color=CTRL)
tag(93.6, 39.5, "Ŷ(t)", color=CTRL, fs=7.2, ha="left")

# ---------------- control plane
box(62, 22.5, 26, 11.0, "M3  Capacity Mapper",
    "queueing law, margin γ,\nlatency feasibility (Eq. 14–18)",
    "pact/capacity/mapper.py", CTRL)
box(33, 22.5, 26, 11.0, "M4  Integer MPC",
    "exact search over [−Δmax, Δmax]\ndead time τ, churn penalty (Eq. 20–24)",
    "pact/control/mpc.py", CTRL)
box(5, 22.5, 25, 11.0, "Gates and Ceiling",
    "deadband, cooldown (Eq. 25)\nsafety override",
    "pact/control/gates.py", CTRL)

arrow((62, 28.0), (59, 28.0), color=CTRL)
arrow((33, 28.0), (30, 28.0), color=CTRL)
tag(60.5, 34.6, "Ñ(t+h)", color=CTRL, fs=7.0)
tag(31.5, 34.6, "u*(t)", color=CTRL, fs=7.0)

ax.add_patch(FancyBboxPatch((62, 18.6), 26, 3.6,
                            boxstyle="round,pad=0,rounding_size=0.8",
                            fc=BOXF, ec=CTRL, lw=1.25))
ax.add_patch(Rectangle((62, 18.6), 0.55, 3.6, fc=CTRL, ec="none"))
ax.text(75.3, 21.0, "M5  Drift Monitor · margin γ", fontsize=7.4,
        color=INK, fontweight="bold", ha="center", va="center")
ax.text(75.3, 19.4, "pact/drift/monitor.py", fontsize=5.8, color=OFF,
        ha="center", va="center", style="italic", family="monospace")
arrow((11, 22.5), (11, 20.4), color=CTRL, ls=(0, (3, 2)))
arrow((11, 20.4), (62, 20.4), color=CTRL, ls=(0, (3, 2)))
arrow((88, 20.4), (97, 20.4), color=CTRL, ls=(0, (3, 2)))
arrow((97, 20.4), (97, 22.5), color=CTRL, ls=(0, (3, 2)))
tag(103.0, 28.0, "γ(t+1)", color=CTRL, fs=6.8)

# ---------------- enforcement
box(5, 4.2, 25, 9.4, "Actuator",
    "rate limit ±Δmax\npool bounds",
    "pact/actuation/dry_run.py", ENF)
box(33, 4.2, 26, 9.4, "Metrics",
    "violation, over-provision,\nchurn, decision classes",
    "pact/eval/metrics.py", ENF)
box(62, 4.2, 26, 9.4, "Statistics",
    "Wilcoxon, Holm, BCa CI,\nCliff's δ",
    "pact/eval/stats.py", ENF)
box(91, 4.2, 22, 9.4, "Docker Backend",
    "not run: no daemon\nin this environment",
    "pact/actuation/docker_backend.py", ENF, dashed=True)

arrow((11, 22.5), (11, 13.6), color=ENF)
arrow((30, 8.9), (33, 8.9), color=ENF)
arrow((59, 8.9), (62, 8.9), color=ENF)

# feedback to the plant
arrow((11, 4.2), (11, 2.9), color=PLANT)
arrow((11, 2.9), (116.6, 2.9), color=PLANT)
arrow((116.6, 2.9), (116.6, 57.2), color=PLANT)
arrow((116.6, 57.2), (43.5, 57.2), color=PLANT)
arrow((43.5, 57.2), (43.5, 55.5), color=PLANT)
tag(112.0, 30.0, "N(t)+u", color=PLANT, fs=6.8, ha="right")

ax.text(2, 0.4, "Dashed outline: implemented in the repository but not exercised "
                "in this evaluation (see Section on limitations).",
        fontsize=6.4, color=GREY, style="italic")

fig.savefig(OUT / "fig_architecture.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.05, dpi=260)
print("  fig_architecture.png")
