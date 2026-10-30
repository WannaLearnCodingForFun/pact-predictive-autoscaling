"""Cross-check the numbers printed in PACT_paper.docx against results/*.csv.

Most values in the manuscript are interpolated from the result files at build
time, so the risk is not arithmetic but wiring: a wrong column name, a stale
cached file, a figure quoted from an older run. This script recomputes the
headline quantities straight from the CSVs and checks that each one appears
in the rendered document text.

A check can fail in two ways and they mean different things:
  MISSING  - the recomputed value does not appear anywhere in the document
  OK       - it appears

Formatting is matched loosely (the document uses a narrow no-break space
before % and ms, and rounds to a fixed number of decimals), so the check is
on the digits.
"""

from __future__ import annotations

import csv
import html
import json
import re
import statistics
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCX = ROOT / "PACT_paper.docx"


def doc_text() -> str:
    x = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf8")
    t = re.sub(r"</w:p>", "\n", x)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    # normalise the exotic spaces the document uses around units
    return t.replace(" ", " ").replace(" ", " ")


def rows(name: str) -> list[dict]:
    return list(csv.DictReader(open(RES / name)))


def fmt(v: float, nd: int) -> str:
    return f"{v:.{nd}f}"


CHECKS: list[tuple[str, str]] = []


def check(label: str, value: str) -> None:
    CHECKS.append((label, value))


def main() -> int:
    text = doc_text()

    # ---- E1 main comparison
    e1 = {r["method"]: r for r in rows("e1_summary.csv")}
    for m, nd in (("pact-oracle", 2), ("rounded-oracle", 2), ("reactive", 2)):
        if m in e1:
            check(f"E1 {m} violation", fmt(float(e1[m]["sla_mean"]), nd))
            check(f"E1 {m} over-provision", fmt(float(e1[m]["over_mean"]), 2))

    # ---- E3 rounding gap: disagreement at the smallest and largest pool
    e3 = rows("e3_rounding_gap.csv")
    if e3:
        by_n: dict[int, list[float]] = {}
        for r in e3:
            by_n.setdefault(int(r["n_max"]), []).append(float(r["disagree_pct"]))
        ns = sorted(by_n)
        check("E3 disagreement at smallest pool",
              fmt(statistics.mean(by_n[ns[0]]), 1))
        check("E3 disagreement at largest pool",
              fmt(statistics.mean(by_n[ns[-1]]), 1))

    # ---- E5 horizon law: the two sides of the boundary
    e5 = rows("e5_horizon_law.csv")
    if e5:
        taus = sorted({float(r["tau_c_s"]) for r in e5})
        below, above = [], []
        for tau in taus:
            rs = [r for r in e5 if float(r["tau_c_s"]) == tau]
            b = [float(r["pact_sla"]) for r in rs
                 if r["covers_coldstart"] not in ("True", "true")]
            a = [float(r["pact_sla"]) for r in rs
                 if r["covers_coldstart"] in ("True", "true")]
            if b and a:
                below.append(statistics.mean(b))
                above.append(statistics.mean(a))
        if below:
            check("E5 mean violation below boundary",
                  fmt(statistics.mean(below), 2))
            check("E5 mean violation above boundary",
                  fmt(statistics.mean(above), 2))
            check("E5 settings sampling both sides", str(len(below)))
        worst = min(float(r["improvement_pp"]) for r in e5)
        best = max(float(r["improvement_pp"]) for r in e5)
        check("E5 worst deficit vs reactive (pp)", fmt(abs(worst), 2))
        check("E5 best gain vs reactive (pp)", fmt(best, 2))

    # ---- E6 operating frontier at the most aggressive target
    e6 = rows("e6_frontier.csv")
    if e6:
        u = max(float(r["u_target"]) for r in e6)
        for m in ("pact-oracle", "rounded-oracle"):
            sel = [r for r in e6
                   if float(r["u_target"]) == u and r.get("method") == m]
            if sel:
                check(f"E6 {m} violation at u*={u:g}",
                      fmt(float(sel[0]["sla_mean"]), 2))

    # ---- E15 exact vs greedy
    g = json.loads((RES / "e15_exact_vs_greedy.json").read_text())
    v, pc = g["validation"], g["paper_config"]
    check("E15 brute-forced states", str(v["cases"]))
    check("E15 exact attained optimum", str(v["exact_attains_optimum"]))
    check("E15 greedy optimal first actions",
          str(v["greedy_first_action_optimal"]))
    check("E15 plant states compared", str(pc["ticks"]))
    check("E15 agreement (%)", fmt(pc["agree_pct"], 2))
    check("E15 disagreement (%)", fmt(100.0 - pc["agree_pct"], 2))
    check("E15 greedy worse (%)", fmt(pc["greedy_worse_pct"], 2))
    check("E15 exact max solve (ms)", fmt(pc["exact_max_ms"], 2))
    check("E15 tracking-relevant actions m", str(pc["m"]))
    # guards against the units slip that once printed "118.9 %" of a
    # 5 s interval for a 59 ms solve
    check("E15 exact max as share of 5 s interval (%)",
          fmt(pc["exact_max_ms"] / 50.0, 2))

    # ---- report
    width = max(len(a) for a, _ in CHECKS)
    missing = []
    for label, value in CHECKS:
        ok = value in text
        print(f"  {'OK ' if ok else 'MISS'}  {label:<{width}}  {value}")
        if not ok:
            missing.append((label, value))

    print()
    print(f"{len(CHECKS) - len(missing)}/{len(CHECKS)} values located "
          f"in the rendered document")
    out = RES / "numerical_verification.json"
    out.write_text(json.dumps({
        "checked": len(CHECKS),
        "located": len(CHECKS) - len(missing),
        "missing": [{"label": a, "value": b} for a, b in missing],
    }, indent=2))
    print(f"wrote {out}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
