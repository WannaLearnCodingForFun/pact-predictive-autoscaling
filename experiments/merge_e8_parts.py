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

    _csv(RESULTS / "e8_sensitivity.csv", rows)
    print(f"merged {len(ORDER) - len(missing)}/{len(ORDER)} sweeps, "
          f"{len(rows)} rows")
    if missing:
        print(f"MISSING (still carrying no exact-solver data): {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
