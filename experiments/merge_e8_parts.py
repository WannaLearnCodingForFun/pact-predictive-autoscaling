"""Assemble results/e8_parts/*.json into results/e8_sensitivity.csv.

Rows keep the order e8_sensitivity() produced, so the merged file is
interchangeable with the one the monolithic function wrote. Sweeps that have
not been run yet are reported rather than silently omitted: a partial
sensitivity table that looks complete is worse than one that says what is
missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_e8_part import ORDER, PARTS  # noqa: E402
from run_more import _csv  # noqa: E402

RESULTS = ROOT / "results"


def main() -> int:
    rows, missing = [], []
    for label in ORDER:
        p = PARTS / f"{label}.json"
        if not p.exists():
            missing.append(label)
            continue
        rows.extend(json.loads(p.read_text()))

    if not rows:
        print("no parts present; nothing written")
        return 1

    if missing:
        # Writing a short e8_sensitivity.csv would be worse than writing
        # nothing: the document looks up specific parameter values in it, so
        # a missing sweep becomes an undefined lookup rendered as a blank or
        # a NaN rather than a visible failure. Park it beside the real file.
        out = RESULTS / "e8_sensitivity.partial.csv"
        _csv(out, rows)
        print(f"merged {len(ORDER) - len(missing)}/{len(ORDER)} sweeps, "
              f"{len(rows)} rows -> {out.name}")
        print(f"NOT written to e8_sensitivity.csv; still missing: {missing}")
        return 1

    _csv(RESULTS / "e8_sensitivity.csv", rows)
    print(f"merged all {len(ORDER)} sweeps, {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
