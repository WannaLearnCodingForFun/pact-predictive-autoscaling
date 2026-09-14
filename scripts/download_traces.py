"""Download / generate PACT evaluation traces (D1, D2, D3)."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from pact.eval.datasets import (
    ArrivalTrace,
    generate_synthetic_suite,
    parse_bitbrains_faststorage,
    parse_fifa_requests,
    write_dataset_summary,
)

PROCESSED_DIR = Path("data/processed")
SUMMARY_PATH = Path("results/dataset_summary.csv")

FIFA_URL = "https://ita.ee.lbl.gov/html/contrib/WorldCup.html"


def load_fifa_file(path: Path, *, dt: float = 5.0) -> ArrivalTrace:
    if path.suffix == ".gz":
        text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return parse_fifa_requests(text.splitlines(), dt=dt)


def load_bitbrains_file(path: Path, *, target_dt_s: float = 5.0) -> ArrivalTrace:
    import csv

    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    return parse_bitbrains_faststorage(rows, target_dt_s=target_dt_s)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fifa", type=Path, default=None)
    parser.add_argument("--bitbrains", type=Path, default=None)
    parser.add_argument("--synthetic-ticks", type=int, default=200)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)

    traces: list[ArrivalTrace] = []
    traces.extend(
        generate_synthetic_suite(dt=args.dt, n_ticks=args.synthetic_ticks)
    )
    if args.fifa is not None:
        traces.append(load_fifa_file(args.fifa, dt=args.dt))
    if args.bitbrains is not None:
        traces.append(load_bitbrains_file(args.bitbrains, target_dt_s=args.dt))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset_summary(args.summary, traces)
    print(f"wrote {args.summary} ({len(traces)} traces)")
    print(
        "D1 FIFA: pass --fifa PATH to a World Cup request log "
        f"(see {FIFA_URL}). D2: pass --bitbrains PATH to a fastStorage CSV. "
        "D2 resampling is interpolated, not measured."
    )


if __name__ == "__main__":
    main()
