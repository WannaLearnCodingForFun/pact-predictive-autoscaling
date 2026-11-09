"""Convert the WorldCup98 access logs into arrival-rate traces for the harness.

Input:  data/raw/worldcup98/wc_day{41..50}_*.gz (binary, big-endian, 20-byte
        records; see data/README.md) and the per-file reference statistics
        shipped in WorldCup_tools.tar.gz (check/wc_dayX_Y.check).
Output: results/traces/worldcup98_raw.npy   requests/s, measured, dt = 5 s
        results/traces/worldcup98.npy       the same series times one constant
        results/traces/worldcup98.json      binning, scale factor, splits
        data/provenance.json                checksums and validation results

Every file is decoded in full and checked against the archive's own checklog
output (request count, byte total, first/last timestamp, per-server counts)
before any of its requests are binned. A mismatch aborts the conversion.

Scale. The harness pool is mu = 40 req/s per replica and n_max = 16, so it
saturates near 640 req/s; the WorldCup98 peak is far above that. Option (a) of
the revision brief is used: the whole series is multiplied by one constant
so that its peak 5-second rate equals PEAK_FRACTION of mu * n_max. A single
multiplicative constant preserves the shape exactly -- every burst keeps its
relative height and timing, and nothing is smoothed. The factor is recorded
in worldcup98.json and stated in the paper as a limitation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.harness import base_config  # noqa: E402

RAW = ROOT / "data" / "raw" / "worldcup98"
TRACES = ROOT / "results" / "traces"
PROVENANCE = ROOT / "data" / "provenance.json"

BASE_URL = "https://ita.ee.lbl.gov/traces/WorldCup/"
LANDING = "https://ita.ee.lbl.gov/html/contrib/WorldCup.html"
DAYS = range(41, 51)
DT_S = 5
PEAK_FRACTION = 0.85
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
PARIS_OFFSET_S = 7200

# Quoted from LANDING (retrieved with the logs; see data/README.md).
ATTRIBUTION = ('M. Arlitt and T. Jin, "1998 World Cup Web Site Access Logs", '
               "August 1998. Available at http://www.acm.org/sigcomm/ITA/.")
LICENCE = (
    "Copyright (C) 1997, 1998, 1999 Hewlett-Packard Company. The software and "
    "documentation may be used, reproduced, modified and distributed for "
    "non-commercial purposes only. The access logs may be used and "
    "redistributed freely as long as the Copyright and Disclaimer is "
    "distributed unmodified (data/WORLDCUP98_COPYRIGHT.txt). Publications "
    "must give the attribution above, and the ITA asks for a copy of any "
    "publication based on the logs."
)
DAY_DATES = {41: "1998-06-05", 42: "1998-06-06", 43: "1998-06-07", 44: "1998-06-08",
             45: "1998-06-09", 46: "1998-06-10", 47: "1998-06-11", 48: "1998-06-12",
             49: "1998-06-13", 50: "1998-06-14"}
BITBRAINS = {
    "status": "located and verified, not used",
    "landing_page": "https://atlarge-research.com/gwa-t-12/",
    "archives": {
        "https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip": 134047207,
        "https://atlarge-research.com/gwa-traces/gwa_t_12_rnd.zip": 163099172,
    },
    "archive_sizes": "Content-Length from HTTP 200 responses, bytes",
    "citation": ("S. Shen, V. van Beek, and A. Iosup, \"Statistical characterization "
                 "of business-critical workloads hosted in cloud datacenters,\" in "
                 "Proc. 15th IEEE/ACM Int. Symp. Cluster, Cloud and Grid Computing "
                 "(CCGrid), 2015."),
    "reason_not_used": (
        "per-VM CPU and memory utilisation sampled every 5 minutes, not request "
        "arrivals. Driving the 5 s harness would need 60x temporal interpolation, "
        "which invents every sub-5-minute dynamic the controller reacts to, and an "
        "inversion of utilisation through the capacity model to an implied "
        "arrival rate, which assumes the very model under test."
    ),
}

RECORD = np.dtype(
    [
        ("timestamp", ">u4"),
        ("client", ">u4"),
        ("object", ">u4"),
        ("size", ">u4"),
        ("method", "u1"),
        ("status", "u1"),
        ("type", "u1"),
        ("server", "u1"),
    ]
)
assert RECORD.itemsize == 20
NO_SIZE = 0xFFFFFFFF  # checklog excludes this sentinel from the byte total
_CHUNK_RECORDS = 1 << 20


def iter_records(stream: io.BufferedIOBase) -> Iterator[np.ndarray]:
    """Yield structured arrays of whole 20-byte records from a binary stream."""

    tail = b""
    while True:
        block = stream.read(_CHUNK_RECORDS * RECORD.itemsize)
        if not block:
            break
        block = tail + block
        whole = len(block) - len(block) % RECORD.itemsize
        tail = block[whole:]
        if whole:
            yield np.frombuffer(block[:whole], dtype=RECORD)
    if tail:
        raise ValueError(f"stream ended with a partial record of {len(tail)} bytes")


def parse_check(text: str) -> dict[str, Any]:
    """Parse checklog output (the reference statistics in the tools archive)."""

    out: dict[str, Any] = {"servers": {}}
    for line in text.splitlines():
        m = re.match(r"\s*Server (\d+): (\d+)", line)
        if m:
            out["servers"][int(m.group(1))] = int(m.group(2))
            continue
        m = re.match(r"\s*([A-Za-z ]+): ([0-9.]+)\s*$", line)
        if m and m.group(1).strip() not in out:
            out[m.group(1).strip()] = m.group(2)
        if line.startswith("Check:"):
            break
    return out


def bin_counts(timestamps: np.ndarray, t0: int, n_bins: int, dt: int) -> np.ndarray:
    """Requests per ``dt``-second bin over ``[t0, t0 + n_bins·dt)``."""

    ts = timestamps.astype(np.int64)
    idx = (ts - t0) // dt
    if idx.size and (idx.min() < 0 or idx.max() >= n_bins):
        raise ValueError("timestamp outside the binning range")
    return np.bincount(idx, minlength=n_bins).astype(np.int64)


def rescale_factor(
    raw_rate: np.ndarray, mu: float, n_max: int, fraction: float
) -> float:
    peak = float(np.max(raw_rate))
    if peak <= 0.0:
        raise ValueError("trace has no requests")
    return fraction * mu * n_max / peak


def split_indices(n: int, train: float, val: float) -> tuple[int, int]:
    """Same arithmetic as pact.forecast.train.chronological_window_splits."""

    n_train = int(n * train)
    n_val = int(n * val)
    return n_train, n_train + n_val


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_for_days(days: range) -> list[Path]:
    paths = []
    for day in days:
        parts = sorted(
            RAW.glob(f"wc_day{day}_*.gz"),
            key=lambda p: int(p.stem.split("_")[-1].split(".")[0]),
        )
        if not parts:
            raise FileNotFoundError(f"no files for day {day} under {RAW}")
        paths.extend(parts)
    return paths


def load_checks(tools_tar: Path) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    with tarfile.open(tools_tar) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name.startswith("wc_day") and name.endswith(".check"):
                fh = tar.extractfile(member)
                assert fh is not None
                checks[name[: -len(".check")]] = parse_check(fh.read().decode())
    return checks


def decode_and_validate(
    path: Path, ref: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    stem = path.name[: -len(".gz")]
    n = 0
    total_bytes = 0
    t_min: int | None = None
    t_max: int | None = None
    out_of_order = 0
    prev_last: int | None = None
    servers: dict[int, int] = {}
    stamps: list[np.ndarray] = []
    with gzip.open(path, "rb") as fh:
        for rec in iter_records(fh):
            ts = rec["timestamp"]
            n += rec.size
            size = rec["size"]
            total_bytes += int(size[size != NO_SIZE].astype(np.uint64).sum())
            lo, hi = int(ts.min()), int(ts.max())
            t_min = lo if t_min is None else min(t_min, lo)
            t_max = hi if t_max is None else max(t_max, hi)
            out_of_order += int(np.sum(ts[1:] < ts[:-1]))
            if prev_last is not None and int(ts[0]) < prev_last:
                out_of_order += 1
            prev_last = int(ts[-1])
            ids, counts = np.unique(rec["server"], return_counts=True)
            for s, c in zip(ids.tolist(), counts.tolist(), strict=True):
                servers[s] = servers.get(s, 0) + c
            stamps.append(ts.astype(np.uint32))
    if t_min is None or t_max is None:
        raise ValueError(f"{stem} contains no records")
    got: dict[str, int] = {
        "Total Requests": n,
        "Total Bytes": total_bytes,
        "Min Time": t_min,
        "Max Time": t_max,
        "Out of Order": out_of_order,
        "Unique Servers": len(servers),
    }
    problems = []
    for key, value in got.items():
        if int(ref[key]) != int(value):
            problems.append(f"{key}: decoded {value}, reference {ref[key]}")
    if servers != ref["servers"]:
        problems.append("per-server request counts differ from reference")
    if problems:
        raise ValueError(f"{stem} failed validation: " + "; ".join(problems))
    return np.concatenate(stamps), {"records": n, "bytes": total_bytes,
                                    "first_ts": t_min, "last_ts": t_max}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-sha", action="store_true",
                        help="reuse checksums already in data/provenance.json")
    args = parser.parse_args(argv)

    cfg = base_config()
    if int(cfg.telemetry.dt) != DT_S:
        raise SystemExit(f"harness dt is {cfg.telemetry.dt}, expected {DT_S}")
    tools = RAW / "WorldCup_tools.tar.gz"
    checks = load_checks(tools)
    files = files_for_days(DAYS)
    first_ref = checks[files[0].name[: -len(".gz")]]
    last_ref = checks[files[-1].name[: -len(".gz")]]
    # The archive's day files cover (local midnight, next local midnight]:
    # day 41 starts at 00:00:01 Paris time and day 50 ends at exactly
    # 00:00:00 on 15 June, with day 51 starting one second later. Bins are
    # aligned to that convention so the ten days fill them with nothing
    # dropped and nothing borrowed from day 40 or day 51.
    t0 = int(first_ref["Start Time"])
    t_end = int(last_ref["Finish Time"]) + 1
    if (t0 - 1 + PARIS_OFFSET_S) % 86400 or (t_end - 1 + PARIS_OFFSET_S) % 86400:
        raise SystemExit(
            "subset does not start and end on the archive's day boundaries"
        )
    n_bins = (t_end - t0) // DT_S
    if n_bins * DT_S != t_end - t0:
        raise SystemExit("subset length is not a whole number of bins")

    counts = np.zeros(n_bins, dtype=np.int64)
    file_rows = []
    started = time.time()
    for path in files:
        stem = path.name[: -len(".gz")]
        stamps, info = decode_and_validate(path, checks[stem])
        counts += bin_counts(stamps, t0, n_bins, DT_S)
        file_rows.append({"file": path.name, "url": BASE_URL + path.name, **info})
        print(f"  {path.name}: {info['records']} records validated "
              f"({time.time() - started:.0f}s)", flush=True)

    raw_rate = counts.astype(np.float64) / DT_S
    factor = rescale_factor(raw_rate, cfg.capacity.mu, cfg.control.n_max, PEAK_FRACTION)
    scaled = raw_rate * factor
    train_end, val_end = split_indices(n_bins, TRAIN_FRACTION, VAL_FRACTION)

    TRACES.mkdir(parents=True, exist_ok=True)
    np.save(TRACES / "worldcup98_raw.npy", raw_rate)
    np.save(TRACES / "worldcup98.npy", scaled)
    meta = {
        "name": "worldcup98",
        "source": LANDING,
        "days": [DAYS.start, DAYS.stop - 1],
        "t0_utc_epoch_s": t0,
        "t0_paris_local": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime(t0 + PARIS_OFFSET_S)),
        "dt_s": DT_S,
        "n_bins": int(n_bins),
        "total_requests": int(counts.sum()),
        "raw_peak_req_s": float(raw_rate.max()),
        "raw_mean_req_s": float(raw_rate.mean()),
        "raw_peak_bin": int(np.argmax(raw_rate)),
        "pool_capacity_req_s": cfg.capacity.mu * cfg.control.n_max,
        "peak_fraction_of_capacity": PEAK_FRACTION,
        "scale_factor": factor,
        "scaled_peak_req_s": float(scaled.max()),
        "scaled_mean_req_s": float(scaled.mean()),
        "transform": "lambda = (requests per 5 s bin / 5) * scale_factor; no smoothing",
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "train_end": train_end,
        "val_end": val_end,
        "sha256_scaled_npy": sha256(TRACES / "worldcup98.npy"),
        "sha256_raw_npy": sha256(TRACES / "worldcup98_raw.npy"),
    }
    (TRACES / "worldcup98.json").write_text(json.dumps(meta, indent=2))

    # Second family. The whole-trace peak falls on 10 June, in the training
    # period, and leaves the test period (13-14 June) below a quarter of pool
    # capacity, where no controller is stressed. This copy multiplies the same
    # series by one further constant so the test-period peak is at the same
    # fraction of capacity. A constant factor cancels in the forecaster's
    # train-statistics z-scoring, so no test-period information reaches it.
    test_peak = float(scaled[val_end:].max())
    extra = PEAK_FRACTION * cfg.capacity.mu * cfg.control.n_max / test_peak
    np.save(TRACES / "worldcup98_testpeak.npy", scaled * extra)
    meta_tp = {
        **{k: v for k, v in meta.items() if not k.startswith("sha256")},
        "name": "worldcup98_testpeak",
        "derived_from": "worldcup98",
        "extra_scale": extra,
        "scale_factor": factor * extra,
        "scaled_peak_req_s": float((scaled * extra).max()),
        "scaled_mean_req_s": float((scaled * extra).mean()),
        "test_peak_req_s": float((scaled * extra)[val_end:].max()),
        "transform": ("lambda = worldcup98 lambda * extra_scale, chosen so the "
                      "test-split peak is peak_fraction_of_capacity of the pool"),
        "sha256_scaled_npy": sha256(TRACES / "worldcup98_testpeak.npy"),
    }
    (TRACES / "worldcup98_testpeak.json").write_text(json.dumps(meta_tp, indent=2))
    meta["test_peak_req_s"] = test_peak
    (TRACES / "worldcup98.json").write_text(json.dumps(meta, indent=2))

    previous = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}
    old_sha = {
        f["file"]: f.get("sha256")
        for f in previous.get("worldcup98", {}).get("files", [])
    }
    for row in file_rows:
        path = RAW / row["file"]
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = old_sha.get(row["file"]) if args.skip_sha else None
        if not row["sha256"]:
            row["sha256"] = sha256(path)
    entry = previous.get("worldcup98", {})
    retrieved = (RAW / "RETRIEVED_UTC.txt").read_text().strip()
    entry.update({
        "source": LANDING,
        "download_base_url": BASE_URL,
        "retrieved_utc": retrieved,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "subset": {
            "days": [DAYS.start, DAYS.stop - 1],
            "dates_paris_local": [DAY_DATES[DAYS.start], DAY_DATES[DAYS.stop - 1]],
            "day_dates": {str(k): v for k, v in DAY_DATES.items()},
            "n_files": len(file_rows),
            "day_boundary_observed": (
                "each day's files run from 00:00:01 Paris time to 00:00:00 the next "
                "day inclusive (checklog Start/Finish Time); bins follow that"),
        },
        "files": file_rows,
        "tools_archive": {
            "file": tools.name,
            "url": BASE_URL + tools.name,
            "size_bytes": tools.stat().st_size,
            "sha256": sha256(tools),
        },
        "validation": (
            "every file decoded in full and matched against the checklog "
            "reference in the tools archive: total requests, total bytes, "
            "min/max timestamp, out-of-order count, unique servers and "
            "per-server counts"
        ),
        "subset_requests": int(counts.sum()),
        "derived": meta,
        "derived_testpeak": meta_tp,
    })
    previous["worldcup98"] = entry
    previous["bitbrains_gwa_t_12"] = BITBRAINS
    PROVENANCE.write_text(json.dumps(previous, indent=2))
    print(f"wrote {TRACES / 'worldcup98.npy'}: {n_bins} bins, "
          f"raw peak {raw_rate.max():.1f} req/s, factor {factor:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
