"""Minimal test runner for environments without pytest.

pytest is not installable here (no package-index access), so this runner
imports each module under ``tests/`` and calls every zero-argument function
whose name starts with ``test_``. Tests that take arguments are fixture-based
and are reported as skipped rather than silently ignored, so the pass count
is never inflated by tests that did not run.

This is not a replacement for pytest: it does not do assertion rewriting,
parametrisation, or fixtures. It is enough to establish that the modules
import and that the fixture-free assertions hold.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TESTS = ROOT / "tests"

import _pytest_shim  # noqa: E402
_pytest_shim.install()
SHIMMED = sys.modules["pytest"].__version__ == "0.0-shim"


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"t_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import os
    os.chdir(ROOT)       # several tests open repo files by relative path
    files = sorted(p for p in TESTS.glob("test_*.py"))
    passed = failed = skipped = 0
    import_errors = []
    failures = []

    for path in files:
        try:
            mod = load(path)
        except Exception as exc:  # noqa: BLE001
            import_errors.append((path.name, repr(exc)))
            print(f"  {path.name}: IMPORT ERROR {exc!r}")
            continue
        fns = [
            (n, f) for n, f in vars(mod).items()
            if n.startswith("test_") and callable(f)
        ]
        p = f_ = s = 0
        for name, fn in sorted(fns):
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                s += 1
                continue
            required = [
                prm for prm in sig.parameters.values()
                if prm.default is inspect.Parameter.empty
                and prm.kind in (prm.POSITIONAL_OR_KEYWORD, prm.KEYWORD_ONLY)
            ]
            cases = getattr(fn, "__parametrize__", None)
            if cases and len(cases) == 1:
                argnames, argvalues = cases[0]
                names = ([a.strip() for a in argnames.split(",")]
                         if isinstance(argnames, str) else list(argnames))
                # only expand when the fixtures are exactly the parametrised args
                if {prm.name for prm in required} == set(names):
                    for vals in argvalues:
                        tup = vals if isinstance(vals, (tuple, list)) else (vals,)
                        try:
                            fn(**dict(zip(names, tup)))
                            p += 1
                        except _pytest_shim.Skipped:
                            s += 1
                        except Exception as exc:  # noqa: BLE001
                            f_ += 1
                            failures.append((path.name, f"{name}{tup}",
                                             traceback.format_exc(limit=4)))
                            print(f"  {path.name}::{name}{tup}: FAIL {exc!r}")
                    continue
            if required:
                s += 1          # needs a fixture this runner cannot supply
                continue
            try:
                fn()
                p += 1
            except _pytest_shim.Skipped:
                s += 1
            except Exception as exc:  # noqa: BLE001
                f_ += 1
                failures.append((path.name, name, traceback.format_exc(limit=4)))
                print(f"  {path.name}::{name}: FAIL {exc!r}")
        passed += p
        failed += f_
        skipped += s
        print(f"  {path.name:34s} pass={p:3d} fail={f_:3d} skip(fixture)={s:3d}")

    print()
    print(f"files={len(files)}  passed={passed}  failed={failed}  "
          f"skipped={skipped}  import_errors={len(import_errors)}")

    out = ROOT / "results" / "test_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "runner": "experiments/run_tests.py",
        "pytest": ("real pytest" if not SHIMMED else
                   "experiments/_pytest_shim.py (pytest not installable here)"),
        "files": len(files),
        "passed": passed,
        "failed": failed,
        "skipped_fixture_based": skipped,
        "import_errors": import_errors,
        "failures": [{"file": a, "test": b} for a, b, _ in failures],
    }, indent=2))
    print(f"wrote {out}")
    return 1 if (failed or import_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
