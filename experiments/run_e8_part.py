"""Run one or more E8 sensitivity sweeps and persist each as it finishes.

e8_sensitivity() ran all five sweeps and wrote the CSV only at the end, so a
run that did not finish left nothing behind. With the exact solver the whole
set is around four hours -- the rate-limit sweep alone is most of it, because
the enumeration is (2*delta_max+1)^m and delta_max reaches 8, which is 17^4
prefixes a tick against 9^4 at the default.

Each sweep is written to results/e8_parts/<label>.json as soon as it
completes. merge_e8_parts.py assembles whatever is present into
results/e8_sensitivity.csv in the original row order, so progress survives a
run being cut short.

Usage:  python run_e8_part.py cooldown_s deadband_theta
        python run_e8_part.py --all
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_more as M  # noqa: E402

PARTS = ROOT / "results" / "e8_parts"
PARTS.mkdir(parents=True, exist_ok=True)

# Same definitions as e8_sensitivity(), in the same order, so the merged CSV
# is identical to what the monolithic function produced.
SWEEPS = [
    ("churn_weight_s",
     lambda c, v: replace(c, control=replace(c.control, s_churn=v)),
     [0.0, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0]),
    ("asymmetry_qup_over_qdown",
     lambda c, v: replace(c, control=replace(c.control, q_up=v, q_down=1.0)),
     [1.0, 2.0, 4.0, 8.0, 12.0, 20.0]),
    ("deadband_theta",
     lambda c, v: replace(c, control=replace(c.control, deadband=v)),
     [0, 1, 2, 3]),
    ("rate_limit_delta_max",
     lambda c, v: replace(c, control=replace(c.control, delta_max=v)),
     [1, 2, 4, 6, 8]),
    ("cooldown_s",
     lambda c, v: replace(c, control=replace(c.control, cooldown_s=v)),
     [0.0, 30.0, 60.0, 120.0]),
]
ORDER = [s[0] for s in SWEEPS]


def main(argv: list[str]) -> int:
    wanted = ORDER if (not argv or argv[0] == "--all") else argv
    unknown = [w for w in wanted if w not in ORDER]
    if unknown:
        print(f"unknown sweep(s): {unknown}\nknown: {ORDER}")
        return 2

    for label, mutate, values in SWEEPS:
        if label not in wanted:
            continue
        out = PARTS / f"{label}.json"
        if out.exists():
            print(f"=== {label}: already present, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"=== {label}: {len(values)} values", flush=True)
        rows = M.sweep(label, mutate, values)
        out.write_text(json.dumps(rows, indent=1))
        print(f"=== {label}: {len(rows)} rows in {time.time() - t0:.0f}s"
              f" -> {out.name}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
