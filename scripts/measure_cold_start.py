"""Measure cold-start delay: scale command → first request on the new replica.

Warm and cold image-cache repetitions are first-class. ``mean_s`` written to
``results/cold_start.json`` is computed from the samples that actually ran.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml
from pact.config import DEFAULT_COLD_START_PATH, DEFAULT_CONFIG_PATH

N_REPS_DEFAULT = 20
DEFAULT_COMPOSE_FILE = Path("testbed/docker-compose.yml")
DEFAULT_SERVICE = "service"
DEFAULT_PROBE_URL = "http://127.0.0.1:8080/id"


class ReplicaProbe(Protocol):
    def replica_ids(self) -> set[str]: ...


class ReplicaScaler(Protocol):
    def scale(self, n: int) -> None: ...

    def drop_image_cache(self) -> None: ...


@dataclass(frozen=True)
class ColdStartSample:
    kind: str
    elapsed_s: float
    replica_id: str


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    series = [float(x) for x in values]
    n = len(series)
    if n < 1:
        raise ValueError("mean_std requires at least one sample")
    mean = sum(series) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in series) / (n - 1)
    return mean, var**0.5


def summarise_samples(samples: Sequence[ColdStartSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("refusing to write cold_start.json with no samples")
    elapsed = [s.elapsed_s for s in samples]
    overall_mean, overall_std = mean_std(elapsed)
    payload: dict[str, object] = {
        "mean_s": overall_mean,
        "std_s": overall_std,
        "n": len(samples),
        "samples_s": elapsed,
    }
    for kind in ("warm", "cold"):
        group = [s.elapsed_s for s in samples if s.kind == kind]
        if not group:
            continue
        mean, std = mean_std(group)
        payload[kind] = {
            "mean_s": mean,
            "std_s": std,
            "n": len(group),
            "samples_s": group,
        }
    return payload


def write_cold_start_json(
    path: Path,
    samples: Sequence[ColdStartSample],
    *,
    n_max: int,
    host_cpu_count: int,
) -> dict[str, object]:
    payload = summarise_samples(samples)
    payload["n_max"] = n_max
    payload["host_cpu_count"] = host_cpu_count
    payload["n_max_below_host_cores"] = n_max < host_cpu_count
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def measure_one(
    *,
    scaler: ReplicaScaler,
    probe: ReplicaProbe,
    n0: int,
    kind: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    timeout_s: float,
    poll_s: float,
    drop_cache: bool,
) -> ColdStartSample:
    if drop_cache:
        scaler.drop_image_cache()
    scaler.scale(n0)
    _wait_count(probe, n0, clock, sleep, timeout_s, poll_s)
    known = probe.replica_ids()
    started = clock()
    scaler.scale(n0 + 1)
    new_id = _wait_new(probe, known, clock, sleep, timeout_s, poll_s)
    elapsed = clock() - started
    if elapsed < 0.0:
        raise ValueError("clock moved backwards")
    return ColdStartSample(kind=kind, elapsed_s=elapsed, replica_id=new_id)


def run_measurement(
    *,
    scaler: ReplicaScaler,
    probe: ReplicaProbe,
    n0: int,
    n_reps: int,
    timeout_s: float,
    poll_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    include_cold: bool = True,
) -> list[ColdStartSample]:
    if n_reps < 1:
        raise ValueError(f"n_reps must be >= 1, got {n_reps}")
    samples: list[ColdStartSample] = []
    for _ in range(n_reps):
        samples.append(
            measure_one(
                scaler=scaler,
                probe=probe,
                n0=n0,
                kind="warm",
                clock=clock,
                sleep=sleep,
                timeout_s=timeout_s,
                poll_s=poll_s,
                drop_cache=False,
            )
        )
    if include_cold:
        for _ in range(n_reps):
            samples.append(
                measure_one(
                    scaler=scaler,
                    probe=probe,
                    n0=n0,
                    kind="cold",
                    clock=clock,
                    sleep=sleep,
                    timeout_s=timeout_s,
                    poll_s=poll_s,
                    drop_cache=True,
                )
            )
    return samples


def _wait_count(
    probe: ReplicaProbe,
    n: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    timeout_s: float,
    poll_s: float,
) -> None:
    deadline = clock() + timeout_s
    while clock() < deadline:
        if len(probe.replica_ids()) >= n:
            return
        sleep(poll_s)
    raise TimeoutError(f"timed out waiting for {n} replica ids")


def _wait_new(
    probe: ReplicaProbe,
    known: set[str],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    timeout_s: float,
    poll_s: float,
) -> str:
    deadline = clock() + timeout_s
    while clock() < deadline:
        extra = probe.replica_ids() - known
        if extra:
            return sorted(extra)[0]
        sleep(poll_s)
    raise TimeoutError("timed out waiting for a new replica to serve a request")


class HttpReplicaProbe:
    def __init__(self, url: str, *, timeout_s: float = 2.0) -> None:
        self._url = url
        self._timeout_s = timeout_s

    def replica_ids(self) -> set[str]:
        seen: set[str] = set()
        for _ in range(8):
            try:
                request = Request(self._url, method="GET")
                with urlopen(request, timeout=self._timeout_s) as response:
                    seen.add(response.read().decode().strip())
            except (OSError, URLError):
                continue
        return seen


class ComposeScaler:
    def __init__(
        self,
        service: str,
        compose_file: Path,
        *,
        runner: Callable[[list[str]], object] | None = None,
    ) -> None:
        self._service = service
        self._compose_file = compose_file
        self._runner = runner

    def scale(self, n: int) -> None:
        cmd = [
            "docker",
            "compose",
            "-f",
            str(self._compose_file),
            "up",
            "-d",
            "--scale",
            f"{self._service}={n}",
        ]
        self._run(cmd)

    def drop_image_cache(self) -> None:
        cmd = [
            "docker",
            "compose",
            "-f",
            str(self._compose_file),
            "build",
            "--no-cache",
            self._service,
        ]
        self._run(cmd)

    def _run(self, cmd: list[str]) -> None:
        if self._runner is not None:
            self._runner(cmd)
            return
        import subprocess

        subprocess.run(cmd, check=True, capture_output=True, text=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure τc on the Docker testbed")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_COLD_START_PATH)
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--probe-url", default=DEFAULT_PROBE_URL)
    parser.add_argument("--n0", type=int, default=1)
    parser.add_argument("--n-reps", type=int, default=N_REPS_DEFAULT)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-s", type=float, default=0.2)
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args(argv)
    raw = yaml.safe_load(args.config.read_text())
    if not isinstance(raw, dict) or "control" not in raw:
        raise ValueError(f"{args.config} must contain a control.n_max field")
    n_max = int(raw["control"]["n_max"])
    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise RuntimeError("os.cpu_count() returned None; cannot record Table 2")
    samples = run_measurement(
        scaler=ComposeScaler(args.service, args.compose_file),
        probe=HttpReplicaProbe(args.probe_url),
        n0=args.n0,
        n_reps=args.n_reps,
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
        include_cold=not args.skip_cold,
    )
    write_cold_start_json(
        args.output,
        samples,
        n_max=n_max,
        host_cpu_count=int(cpu_count),
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
