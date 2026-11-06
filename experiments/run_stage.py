"""Run one named experiment stage in its own process.

The suite functions in run_all.py and run_more.py each write their own CSV,
so independent stages can run in parallel. E2 needs E1's in-memory results
and runs with it. E8 sweeps go through run_e8_part.py so each persists alone.

Usage:  python run_stage.py e1 | e3 | ... | e13 | e8:<sweep label>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

import run_all  # noqa: E402
import run_more  # noqa: E402


def _e1() -> None:
    run_all.e2_stats(run_all.e1_main())


STAGES = {
    "e1": _e1,
    "e3": run_all.e3_rounding,
    "e4": run_all.e4_ablation,
    "e5": run_all.e5_horizon,
    "e6": run_all.e6_frontier,
    "e7": run_all.e7_robustness,
    "e9": run_more.e9_solvetime,
    "e10": run_more.e10_per_workload,
    "e11": run_more.e11_decisions,
    "e12": run_more.e12_coldstart,
    "e13": run_more.e13_predictor_quality,
}


def main(argv: list[str]) -> int:
    for name in argv:
        t0 = time.time()
        print(f"=== {name}", flush=True)
        if name.startswith("e8:"):
            import run_e8_part
            rc = run_e8_part.main([name[3:]])
            if rc:
                return rc
        else:
            STAGES[name]()
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
