"""Build results/e8_solver_compare.csv: the churn sweep under both solvers.

The E8 churn-weight sweep is run twice, unchanged in every respect except the
integer solver: once with the exact solver (results/e8_parts/) and once with
the enumerate-then-complete-greedily solver (results/e8_solver_greedy/, from
``run_stage.py e8:churn_weight_s@greedy``). Same workloads, same seeds, same
start state and configuration, so the pair is a controlled comparison of what
approximating the solve does to a tuning conclusion.

The mechanism the comparison probes is the one Section III-E describes.
Greedy completion zeroes the tail of the horizon, and a zero tail misprices
exactly the term the churn weight scales, so the two solvers can be expected
to diverge most where s is large.

Writes nothing if either input is absent.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GREEDY = RESULTS / "e8_solver_greedy" / "churn_weight_s.json"
PART = RESULTS / "e8_parts" / "churn_weight_s.json"
OUT = RESULTS / "e8_solver_compare.csv"

FIELDS = ["parameter", "value",
          "exact_sla", "greedy_sla",
          "exact_over", "greedy_over",
          "exact_osc", "greedy_osc",
          "exact_act", "greedy_act", "n_runs"]


def main() -> int:
    if not GREEDY.exists() or not PART.exists():
        print(f"missing input: {GREEDY if not GREEDY.exists() else PART}")
        return 1

    greedy = {float(r["value"]): r for r in json.loads(GREEDY.read_text())}
    exact = json.loads(PART.read_text())

    rows = []
    for e in sorted(exact, key=lambda r: float(r["value"])):
        v = float(e["value"])
        g = greedy.get(v)
        if g is None:
            continue
        if int(g["n_runs"]) != int(e["n_runs"]):
            print(f"run counts differ at s = {v}: {g['n_runs']} vs {e['n_runs']}")
            return 1
        rows.append({
            "parameter": "churn_weight_s", "value": v,
            "exact_sla": round(float(e["sla_mean"]), 4),
            "greedy_sla": round(float(g["sla_mean"]), 4),
            "exact_over": round(float(e["over_mean"]), 4),
            "greedy_over": round(float(g["over_mean"]), 4),
            "exact_osc": round(float(e["osc_mean"]), 4),
            "greedy_osc": round(float(g["osc_mean"]), 4),
            "exact_act": round(float(e["act_mean"]), 4),
            "greedy_act": round(float(g["act_mean"]), 4),
            "n_runs": int(e["n_runs"]),
        })

    if not rows:
        print("no overlapping values; nothing written")
        return 1

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {OUT.name} ({len(rows)} rows)")

    worst = max(rows, key=lambda r: abs(r["greedy_sla"] - r["exact_sla"]))
    print(f"  largest divergence at s = {worst['value']}: "
          f"greedy {worst['greedy_sla']} % vs exact {worst['exact_sla']} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
