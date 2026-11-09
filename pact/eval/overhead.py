"""U6: controller overhead — stage latencies, CPU/RSS share, model size.

Share of testbed consumption is written only when Docker stats for the
compose project actually succeed. Missing docker is not recorded as 0%.
Int8 size is the packed byte count of the same float32 tensors, not a
trained quantised model.
"""

from __future__ import annotations

import argparse
import csv
import math
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn

from pact.actuation.dry_run import DryRunBackend
from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    PactConfig,
    load_config,
)
from pact.eval.datasets import generate_burst
from pact.forecast.tcn import DilatedTCN
from pact.forecast.train import export_onnx_and_verify, seed_everything
from pact.loop import ControlLoop, RepeatObservationForecaster, TickTiming
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import SimCollector

STAGES: tuple[str, ...] = (
    "scrape",
    "features",
    "inference",
    "mapping",
    "mpc",
    "actuation",
)
PERCENTILES: tuple[int, ...] = (50, 95, 99)
OVERHEAD_COLUMNS: tuple[str, ...] = ("kind", "name", "stat", "value", "unit")
COMPOSE_PROJECT = "pact"


@dataclass(frozen=True)
class OverheadRow:
    kind: str
    name: str
    stat: str
    value: float
    unit: str


@dataclass(frozen=True)
class ProcessResources:
    cpu_user_s: float
    cpu_system_s: float
    rss_bytes: int


@dataclass(frozen=True)
class ComposeSample:
    cpu_percent: float
    rss_bytes: int
    n_containers: int


TestbedSampler = Callable[[], ComposeSample | None]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile. ``q`` is in ``[0, 100]``."""

    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    xs = sorted(float(x) for x in values)
    if not xs:
        raise ValueError("percentile of empty series")
    if len(xs) == 1:
        return xs[0]
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    weight = rank - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def deadline_miss_rate(slips_s: Sequence[float]) -> float:
    if not slips_s:
        raise ValueError("deadline miss rate needs at least one tick")
    missed = sum(1 for slip in slips_s if float(slip) > 0.0)
    return missed / len(slips_s)


def rss_bytes_from_rusage(usage: resource.struct_rusage | None = None) -> int:
    """``ru_maxrss`` is kilobytes on Linux and bytes on macOS."""

    snap = resource.getrusage(resource.RUSAGE_SELF) if usage is None else usage
    rss = int(snap.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def read_process_resources() -> ProcessResources:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ProcessResources(
        cpu_user_s=float(usage.ru_utime),
        cpu_system_s=float(usage.ru_stime),
        rss_bytes=rss_bytes_from_rusage(usage),
    )


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def float32_parameter_bytes(module: nn.Module) -> int:
    total = 0
    for param in module.parameters():
        total += int(param.detach().cpu().contiguous().numpy().nbytes)
    return total


def int8_packed_bytes(module: nn.Module) -> int:
    total = 0
    for param in module.parameters():
        packed = param.detach().cpu().contiguous().numpy().astype(np.int8, copy=False)
        total += int(packed.nbytes)
    return total


def parse_docker_mem_bytes(text: str) -> int:
    """Parse the used half of docker stats ``MemUsage`` (``12.5MiB / 2GiB``)."""

    used = text.split("/", 1)[0].strip()
    if not used:
        raise ValueError(f"unparseable docker MemUsage: {text!r}")
    digits = ""
    unit = ""
    for char in used:
        if char.isdigit() or char in ".-":
            digits += char
        elif not char.isspace():
            unit += char
    if not digits:
        raise ValueError(f"unparseable docker MemUsage: {text!r}")
    scale = {
        "b": 1.0,
        "kib": 1024.0,
        "kb": 1000.0,
        "mib": 1024.0**2,
        "mb": 1000.0**2,
        "gib": 1024.0**3,
        "gb": 1000.0**3,
        "tib": 1024.0**4,
        "tb": 1000.0**4,
    }.get(unit.lower(), None)
    if scale is None:
        raise ValueError(f"unknown docker memory unit in {text!r}")
    return int(float(digits) * scale)


def parse_docker_cpu_percent(text: str) -> float:
    stripped = text.strip().rstrip("%").strip()
    if not stripped:
        raise ValueError(f"unparseable docker CPUPerc: {text!r}")
    return float(stripped)


def sample_compose_testbed(
    project: str = COMPOSE_PROJECT, timeout_s: float = 8.0
) -> ComposeSample | None:
    """Instantaneous docker stats for compose project ``pact``, or None."""

    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode != 0:
        return None
    ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not ids:
        return None
    try:
        stats = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.CPUPerc}}\t{{.MemUsage}}",
                *ids,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if stats.returncode != 0 or not stats.stdout.strip():
        return None
    cpu_total = 0.0
    rss_total = 0
    rows = 0
    for line in stats.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            return None
        try:
            cpu_total += parse_docker_cpu_percent(parts[0])
            rss_total += parse_docker_mem_bytes(parts[1])
        except ValueError:
            return None
        rows += 1
    if rows == 0:
        return None
    return ComposeSample(
        cpu_percent=cpu_total, rss_bytes=rss_total, n_containers=rows
    )


def _stage_seconds(timings: Sequence[TickTiming], stage: str) -> list[float]:
    attr = f"{stage}_s"
    return [float(getattr(tick, attr)) for tick in timings]


def model_size_rows(
    model: nn.Module, *, onnx_path: Path | None = None
) -> list[OverheadRow]:
    rows = [
        OverheadRow(
            "model", "parameter_count", "value", float(count_parameters(model)), "count"
        ),
        OverheadRow(
            "model",
            "float32_bytes",
            "value",
            float(float32_parameter_bytes(model)),
            "bytes",
        ),
        OverheadRow(
            "model",
            "int8_packed_bytes",
            "value",
            float(int8_packed_bytes(model)),
            "bytes",
        ),
    ]
    if onnx_path is not None and onnx_path.is_file():
        rows.append(
            OverheadRow(
                "model",
                "onnx_bytes",
                "value",
                float(onnx_path.stat().st_size),
                "bytes",
            )
        )
    return rows


def latency_rows(timings: Sequence[TickTiming]) -> list[OverheadRow]:
    if not timings:
        raise ValueError("refusing to write overhead latency with no ticks")
    rows: list[OverheadRow] = []
    for stage in STAGES:
        series = _stage_seconds(timings, stage)
        for q in PERCENTILES:
            rows.append(
                OverheadRow(
                    "latency",
                    stage,
                    f"p{q}",
                    percentile(series, float(q)),
                    "s",
                )
            )
    return rows


def resource_rows(
    before: ProcessResources,
    after: ProcessResources,
    wall_s: float,
    testbed: ComposeSample | None,
) -> list[OverheadRow]:
    cpu_user = max(0.0, after.cpu_user_s - before.cpu_user_s)
    cpu_system = max(0.0, after.cpu_system_s - before.cpu_system_s)
    cpu_s = cpu_user + cpu_system
    rss = after.rss_bytes
    rows = [
        OverheadRow("resource", "controller_cpu_user", "value", cpu_user, "s"),
        OverheadRow("resource", "controller_cpu_system", "value", cpu_system, "s"),
        OverheadRow("resource", "controller_cpu", "value", cpu_s, "s"),
        OverheadRow("resource", "controller_rss", "value", float(rss), "bytes"),
        OverheadRow("resource", "controller_wall", "value", wall_s, "s"),
    ]
    if wall_s > 0.0:
        cpu_percent = 100.0 * cpu_s / wall_s
        rows.append(
            OverheadRow(
                "resource", "controller_cpu_percent", "value", cpu_percent, "percent"
            )
        )
    else:
        cpu_percent = 0.0
    if testbed is None:
        return rows
    rows.append(
        OverheadRow(
            "resource", "testbed_cpu_percent", "value", testbed.cpu_percent, "percent"
        )
    )
    rows.append(
        OverheadRow(
            "resource", "testbed_rss", "value", float(testbed.rss_bytes), "bytes"
        )
    )
    rows.append(
        OverheadRow(
            "resource",
            "testbed_containers",
            "value",
            float(testbed.n_containers),
            "count",
        )
    )
    rss_denom = float(rss + testbed.rss_bytes)
    if rss_denom > 0.0:
        rows.append(
            OverheadRow(
                "resource",
                "controller_rss_share",
                "value",
                float(rss) / rss_denom,
                "fraction",
            )
        )
    cpu_denom = cpu_percent + testbed.cpu_percent
    if cpu_denom > 0.0:
        rows.append(
            OverheadRow(
                "resource",
                "controller_cpu_share",
                "value",
                cpu_percent / cpu_denom,
                "fraction",
            )
        )
    return rows


def measure_controller_overhead(
    config: PactConfig,
    *,
    n_ticks: int,
    seed: int = 0,
    onnx_path: Path | None = None,
    testbed_sampler: TestbedSampler | None = sample_compose_testbed,
) -> list[OverheadRow]:
    """Run the control loop once and return measured overhead rows."""

    seed_everything(seed)
    arrivals = generate_burst(
        "step",
        n_ticks=n_ticks,
        dt=config.telemetry.dt,
        rise_ticks=max(1, min(4, n_ticks // 4)),
        base=20.0,
        peak=60.0,
        name="overhead_step",
    ).arrival_rates
    n0 = config.control.n_min
    sim = PoolSimulator(config, n_initial=n0)
    loop = ControlLoop(
        config,
        collector=SimCollector(sim),
        backend=DryRunBackend(n_initial=n0),
        forecaster=RepeatObservationForecaster(config.forecast.horizon),
        simulator=sim,
        arrival_rates=arrivals,
        n_initial=n0,
        sleep=lambda _s: None,
    )
    before = read_process_resources()
    wall0 = time.perf_counter()
    result = loop.run(n_ticks)
    wall_s = time.perf_counter() - wall0
    after = read_process_resources()
    testbed = testbed_sampler() if testbed_sampler is not None else None
    if len(result.timings) != n_ticks:
        raise RuntimeError(
            f"expected {n_ticks} timings, got {len(result.timings)}"
        )
    rows: list[OverheadRow] = []
    rows.extend(latency_rows(result.timings))
    rows.extend(resource_rows(before, after, wall_s, testbed))
    rows.append(
        OverheadRow(
            "deadline",
            "miss_rate",
            "value",
            deadline_miss_rate(result.deadline_slips_s),
            "fraction",
        )
    )
    rows.append(
        OverheadRow("deadline", "n_ticks", "value", float(n_ticks), "count")
    )
    model = DilatedTCN.from_config(config)
    rows.extend(model_size_rows(model, onnx_path=onnx_path))
    return rows


def write_overhead_csv(path: Path, rows: Sequence[OverheadRow]) -> None:
    if not rows:
        raise ValueError(f"refusing to write {path} with no computed rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(OVERHEAD_COLUMNS))
        for row in rows:
            writer.writerow(
                [row.kind, row.name, row.stat, f"{row.value:.10g}", row.unit]
            )


def ensure_onnx_for_size(config: PactConfig, dest: Path) -> Path:
    """Export the architecture graph if ``dest`` is not already on disk."""

    if dest.is_file():
        return dest
    model = DilatedTCN.from_config(config)
    model.eval()
    export_onnx_and_verify(model, dest)
    return dest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT controller overhead")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="ONNX file whose size is recorded (exported if missing)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config, cold_start_path=args.cold_start)
    onnx_arg = args.onnx if args.onnx is not None else args.results / "tcn.onnx"
    onnx_path: Path | None
    if onnx_arg.is_file():
        onnx_path = onnx_arg
    else:
        try:
            onnx_path = ensure_onnx_for_size(config, onnx_arg)
        except (OSError, RuntimeError, ValueError, ImportError):
            onnx_path = None
    rows = measure_controller_overhead(
        config,
        n_ticks=args.ticks,
        seed=args.seed,
        onnx_path=onnx_path,
    )
    write_overhead_csv(args.results / "overhead.csv", rows)


if __name__ == "__main__":
    main()
