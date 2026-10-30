"""Build results/e8_solver_compare.csv: the churn sweep under both solvers.

The E8 sweep was first run with the greedy-completion solver and then again,
unchanged in every other respect, with the exact one. Same workloads, same
seeds, same configuration; only the solver differs. That makes the pair a
controlled comparison of what approximating the solve does to a tuning
conclusion, which is worth reporting separately from the sweep itself.

It matters here because the greedy numbers say something the exact numbers
do not: that raising the churn weight to unity drives violations to
0.013 %, and holds them there. Under the exact solver the same sweep sits
near 0.58 %. The qualitative reading is unchanged -- raising s improves
service level, capacity and oscillation together -- but the size of the
effect was an artifact of the solver, not a property of the controller.

The mechanism is the one Section III-E describes. Greedy completion zeroes
the tail of the horizon, and a zero tail misprices exactly the term the
churn weight scales: from u(t+m-1) = 4 the all-zero tail costs 9.60 against
4.35 for a ramp-down. Raising s magnifies that mispricing, so the greedy
solver's behaviour diverges from the objective it is nominally minimising
precisely where s is large.

Reads the archived greedy run from results_greedy_backup/ and the exact
sweep from results/e8_parts/. Writes nothing if either is absent.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BACKUP = ROOT / "results_greedy_backup" / "e8_sensitivity.csv"
PART = RESULTS / "e8_parts" / "churn_weight_s.json"
OUT = RESULTS / "e8_solver_compare.csv"

FIELDS = ["parameter", "value",
          "exact_sla", "greedy_sla",
          "exact_over", "greedy_over",
          "exact_osc", "greedy_osc",
          "exact_act", "greedy_act", "n_runs"]


def main() -> int:
    if not BACKUP.exists() or not PART.exists():
        print(f"missing input: {BACKUP if not BACKUP.exists() else PART}")
        return 1

    greedy = {
        float(r["value"]): r
        for r in csv.DictReader(BACKUP.open())
        if r["parameter"] == "churn_weight_s"
    }
    exact = json.loads(PART.read_text())

    rows = []
    for e in sorted(exact, key=lambda r: float(r["value"])):
        v = float(e["value"])
        g = greedy.get(v)
        if g is None:
            continue
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
    raise SystemExit(main())
