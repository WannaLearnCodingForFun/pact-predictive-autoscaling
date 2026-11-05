"""Collect per-stage experiment runtimes into results/runtime_stages.csv.

The manuscript quoted "approximately six minutes" for the experiment stage.
That was true of the greedy-completion solver and is wrong by two orders of
magnitude for the exact one, so the figure needs to come from measurement
like every other number in the paper rather than from prose.

Durations are parsed from the run logs the experiments wrote as they went.
Only stages that printed a duration are included; E1-E4 and E15 were run
before the logging was added and are listed as not instrumented rather than
estimated, so the total reported is a lower bound on the real cost and says
so.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "runtime_stages.csv"

# log file -> (stage label, how durations appear in it)
CUMULATIVE = re.compile(r"=== (.+?) done at (\d+)s")
PER_SWEEP = re.compile(r"=== (.+?): \d+ rows in (\d+)s")


def main() -> int:
    rows: list[dict] = []

    # Logs whose markers are cumulative since the job started: take the last.
    for log, stage in (("rerun_rest.log", "E5 horizon law + E6 frontier"),
                       ("rerun_e7.log", "E7 robustness"),
                       ("rerun_e9_14.log", "E9-E14")):
        p = RESULTS / log
        if not p.exists():
            continue
        vals = [int(m.group(2)) for m in CUMULATIVE.finditer(p.read_text(errors="ignore"))]
        if vals:
            rows.append({"stage": stage, "seconds": max(vals),
                         "instrumented": "yes", "source": log})

    # E8 ran as five separate sweeps, each printing its own duration.
    total8 = 0
    found8 = False
    for log in ("rerun_e8a.log", "rerun_e8b.log", "rerun_e8c.log"):
        p = RESULTS / log
        if not p.exists():
            continue
        for m in PER_SWEEP.finditer(p.read_text(errors="ignore")):
            total8 += int(m.group(2))
            found8 = True
    if found8:
        rows.append({"stage": "E8 control-parameter sweeps (5)", "seconds": total8,
                     "instrumented": "yes", "source": "rerun_e8[abc].log"})

    for stage in ("E1-E4 main comparison, statistics, rounding gap, ablation",
                  "E15 exactness verification"):
        rows.append({"stage": stage, "seconds": "", "instrumented": "no",
                     "source": "run predates duration logging"})

    if not any(r["instrumented"] == "yes" for r in rows):
        print("no instrumented stages found; nothing written")
        return 1

    rows.sort(key=lambda r: (r["instrumented"] != "yes", r["stage"]))
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "seconds", "instrumented", "source"])
        w.writeheader()
        w.writerows(rows)

    measured = sum(int(r["seconds"]) for r in rows if r["instrumented"] == "yes")
    print(f"  wrote {OUT.name} ({len(rows)} stages)")
    print(f"  instrumented total: {measured} s = {measured / 3600:.1f} h "
          f"(lower bound; two stages not instrumented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
