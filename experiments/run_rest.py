"""Resume the experiment suite from E5 with the guarded solver selection.

run_all.py was started before the tractability guard existed, so it is left
to finish E1-E4 (all of which are in the regime where the exact solve is
cheap) and is then stopped. This script runs the remainder -- E5 to E7, then
the extended experiments and the decision trace -- with ``solver="auto"``,
which takes the exact solver wherever the structural reduction makes it
affordable and the greedy completion where it does not, recording which was
used for every run.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_all  # noqa: E402


def stage(name: str, fn) -> None:
    t0 = time.perf_counter()
    print(f"\n=== {name}", flush=True)
    fn()
    print(f"=== {name} done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> int:
    stage("E5 horizon law", run_all.e5_horizon)
    stage("E6 operating frontier", run_all.e6_frontier)
    stage("E7 robustness", run_all.e7_robustness)
    stage("manifest", run_all.manifest)

    for script in ("run_more.py", "run_trace.py"):
        print(f"\n=== {script}", flush=True)
        r = subprocess.run(
            [sys.executable, "-u", str(HERE / script)],
            cwd=str(HERE),
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        )
        if r.returncode != 0:
            print(f"!!! {script} exited {r.returncode}", flush=True)
            return r.returncode
    print("\nall remaining experiments complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
