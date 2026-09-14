"""Dataset loaders for D1 (FIFA), D2 (Bitbrains), D3 (synthetic bursts)."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FIFA_DT_S = 5.0
BITBRAINS_NATIVE_DT_S = 300.0
FIFA_SUBSET_S = 7 * 86400.0


@dataclass(frozen=True)
class ArrivalTrace:
    name: str
    timestamps: tuple[float, ...]
    arrival_rates: tuple[float, ...]
    dt: float
    interpolated: bool
    notes: str

    @property
    def n_samples(self) -> int:
        return len(self.arrival_rates)

    @property
    def duration_s(self) -> float:
        if not self.timestamps:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0] + self.dt


def parse_fifa_requests(
    lines: Sequence[str],
    *,
    dt: float = FIFA_DT_S,
    subset_s: float = FIFA_SUBSET_S,
    name: str = "d1_fifa",
) -> ArrivalTrace:
    """Bucket one-request-per-line unix timestamps (or ``t count``) into Δt bins."""

    events: list[tuple[float, int]] = []
    for raw in lines:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.replace(",", " ").split()
        t = float(parts[0])
        count = int(parts[1]) if len(parts) > 1 else 1
        events.append((t, count))
    if not events:
        raise ValueError("FIFA trace is empty")
    events.sort(key=lambda item: item[0])
    t0 = events[0][0]
    t_end = min(events[-1][0], t0 + subset_s)
    n_bins = max(1, int(math.floor((t_end - t0) / dt)) + 1)
    counts = [0.0] * n_bins
    for t, count in events:
        if t > t_end:
            break
        idx = min(int((t - t0) / dt), n_bins - 1)
        counts[idx] += float(count)
    rates = tuple(c / dt for c in counts)
    timestamps = tuple(t0 + i * dt for i in range(n_bins))
    return ArrivalTrace(
        name=name,
        timestamps=timestamps,
        arrival_rates=rates,
        dt=dt,
        interpolated=False,
        notes="1998 FIFA World Cup HTTP; 7-day subset, counted per 5 s",
    )


def parse_bitbrains_faststorage(
    rows: Sequence[Sequence[str]],
    *,
    native_dt_s: float = BITBRAINS_NATIVE_DT_S,
    target_dt_s: float = FIFA_DT_S,
    name: str = "d2_bitbrains",
) -> ArrivalTrace:
    """Resample 5-minute CPU-usage samples to 5 s by linear interpolation.

    Interpolated values are **not** measured samples. The ``interpolated``
    flag and notes record that fact.
    """

    times: list[float] = []
    usage: list[float] = []
    header = [c.strip().lower() for c in rows[0]] if rows else []
    body = rows[1:] if header and not _is_float(header[0]) else rows
    t_idx, u_idx = _bitbrains_columns(header if body is not rows else [])
    for row in body:
        if not row or (len(row) > 0 and str(row[0]).startswith("#")):
            continue
        times.append(float(row[t_idx]))
        usage.append(float(row[u_idx]))
    if len(times) < 2:
        raise ValueError("Bitbrains trace needs at least two samples")
    # Timestamps may be milliseconds.
    if times[1] - times[0] > 10_000:
        times = [t / 1000.0 for t in times]
    t0, t1 = times[0], times[-1]
    n_bins = int(math.floor((t1 - t0) / target_dt_s)) + 1
    stamps = [t0 + i * target_dt_s for i in range(n_bins)]
    interpolated = _linear_interpolate(times, usage, stamps)
    # CPU % → a request-rate proxy: usage fraction times a unit capacity.
    rates = tuple(max(v, 0.0) / 100.0 for v in interpolated)
    return ArrivalTrace(
        name=name,
        timestamps=tuple(stamps),
        arrival_rates=rates,
        dt=target_dt_s,
        interpolated=True,
        notes=(
            f"GWA-T-12 fastStorage resampled {native_dt_s:.0f}s → "
            f"{target_dt_s:.0f}s by linear interpolation; "
            f"not measured at {target_dt_s:.0f}s"
        ),
    )


def generate_burst(
    kind: str,
    *,
    n_ticks: int,
    dt: float,
    rise_ticks: int,
    base: float,
    peak: float,
    name: str | None = None,
) -> ArrivalTrace:
    """Synthetic burst: step, ramp, sawtooth, or impulse. Rise time is a parameter."""

    if n_ticks < 2:
        raise ValueError("n_ticks must be >= 2")
    if rise_ticks < 1:
        raise ValueError("rise_ticks must be >= 1")
    rates: list[float] = []
    mid = n_ticks // 4
    for t in range(n_ticks):
        if kind == "step":
            rates.append(peak if t >= mid else base)
        elif kind == "ramp":
            if t < mid:
                rates.append(base)
            elif t < mid + rise_ticks:
                frac = (t - mid) / rise_ticks
                rates.append(base + frac * (peak - base))
            else:
                rates.append(peak)
        elif kind == "sawtooth":
            period = max(rise_ticks * 2, 2)
            phase = t % period
            frac = phase / (period - 1) if period > 1 else 0.0
            rates.append(base + frac * (peak - base))
        elif kind == "impulse":
            rates.append(peak if mid <= t < mid + rise_ticks else base)
        else:
            raise ValueError(f"unknown burst kind {kind!r}")
    stamps = tuple(i * dt for i in range(n_ticks))
    label = name or f"d3_{kind}_rise{rise_ticks}"
    return ArrivalTrace(
        name=label,
        timestamps=stamps,
        arrival_rates=tuple(rates),
        dt=dt,
        interpolated=False,
        notes=f"synthetic {kind} burst, rise_ticks={rise_ticks}",
    )


def generate_synthetic_suite(
    *,
    dt: float = 5.0,
    n_ticks: int = 200,
    rise_ticks: Sequence[int] = (1, 4, 12),
    base: float = 20.0,
    peak: float = 80.0,
) -> list[ArrivalTrace]:
    traces: list[ArrivalTrace] = []
    for kind in ("step", "ramp", "sawtooth", "impulse"):
        for rise in rise_ticks:
            traces.append(
                generate_burst(
                    kind,
                    n_ticks=n_ticks,
                    dt=dt,
                    rise_ticks=int(rise),
                    base=base,
                    peak=peak,
                )
            )
    return traces


def write_dataset_summary(path: Path, traces: Sequence[ArrivalTrace]) -> None:
    """Write measured duration and sample counts. No placeholder rows."""

    if not traces:
        raise ValueError("refusing to write dataset_summary.csv with no traces")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "n_samples",
                "duration_s",
                "dt_s",
                "interpolated",
                "notes",
            ]
        )
        for trace in traces:
            writer.writerow(
                [
                    trace.name,
                    trace.n_samples,
                    f"{trace.duration_s:.6g}",
                    f"{trace.dt:.6g}",
                    str(trace.interpolated).lower(),
                    trace.notes,
                ]
            )


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _bitbrains_columns(header: Sequence[str]) -> tuple[int, int]:
    if not header:
        return 0, 1
    t_idx = 0
    u_idx = 1
    for i, name in enumerate(header):
        if "timestamp" in name or name == "time":
            t_idx = i
        if "cpuusage" in name.replace(" ", "") or name in {"cpu", "cpu_usage"}:
            u_idx = i
        elif "cpu" in name and "usage" in name:
            u_idx = i
    return t_idx, u_idx


def _linear_interpolate(
    x: Sequence[float], y: Sequence[float], query: Sequence[float]
) -> list[float]:
    out: list[float] = []
    j = 0
    for q in query:
        while j + 1 < len(x) and x[j + 1] < q:
            j += 1
        if q <= x[0]:
            out.append(y[0])
            continue
        if q >= x[-1]:
            out.append(y[-1])
            continue
        x0, x1 = x[j], x[j + 1]
        y0, y1 = y[j], y[j + 1]
        if x1 == x0:
            out.append(y0)
        else:
            frac = (q - x0) / (x1 - x0)
            out.append(y0 + frac * (y1 - y0))
    return out
