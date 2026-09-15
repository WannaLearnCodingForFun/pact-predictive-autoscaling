"""Phase 10: testbed stack and measured cold-start JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from subprocess import CompletedProcess

import pytest
import yaml
from pact.actuation.docker_backend import DockerComposeBackend
from scripts.measure_cold_start import (
    N_REPS_DEFAULT,
    mean_std,
    run_measurement,
    write_cold_start_json,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class _FakeBed:
    def __init__(self, clock: _Clock, *, warm_s: float, cold_s: float) -> None:
        self.clock = clock
        self.warm_s = warm_s
        self.cold_s = cold_s
        self.n = 1
        self.cold_next = False
        self.ids = {"r1"}

    def replica_ids(self) -> set[str]:
        return set(self.ids)

    def scale(self, n: int) -> None:
        if n > self.n:
            self.clock.t += self.cold_s if self.cold_next else self.warm_s
            self.cold_next = False
        self.n = n
        self.ids = {f"r{i}" for i in range(1, n + 1)}

    def drop_image_cache(self) -> None:
        self.cold_next = True


def test_compose_declares_required_services() -> None:
    raw = yaml.safe_load(Path("testbed/docker-compose.yml").read_text())
    services = raw["services"]
    assert {"service", "nginx", "cadvisor", "prometheus", "pact"} <= set(services)
    assert services["pact"]["cpus"] == "0.25"
    env = services["service"]["environment"]
    assert str(env["CONCURRENCY"]) == "1"
    assert "WORK_MS" in env


def test_nginx_uses_least_conn() -> None:
    conf = Path("testbed/nginx/nginx.conf").read_text()
    entry = Path("testbed/nginx/docker-entrypoint.sh").read_text()
    assert "least_conn" in conf or "least_conn" in entry
    assert "least_conn" in entry


def test_workload_is_single_threaded_with_concurrency_cap() -> None:
    app = Path("testbed/service/app.py").read_text()
    assert "CONCURRENCY" in app
    assert "Semaphore" in app
    assert "WORK_MS" in app


def test_swarm_stack_exists() -> None:
    raw = yaml.safe_load(Path("testbed/swarm/stack.yml").read_text())
    assert "service" in raw["services"]
    assert raw["networks"]["pact"]["driver"] == "overlay"
    pact = raw["services"]["pact"]["deploy"]["resources"]["reservations"]
    assert pact["cpus"] == "0.25"


def test_cold_start_json_mean_std_are_computed(tmp_path: Path) -> None:
    warm_s = 1.0
    cold_s = 3.0
    n_reps = N_REPS_DEFAULT
    clock = _Clock()
    bed = _FakeBed(clock, warm_s=warm_s, cold_s=cold_s)
    samples = run_measurement(
        scaler=bed,
        probe=bed,
        n0=1,
        n_reps=n_reps,
        timeout_s=10.0,
        poll_s=0.0,
        clock=clock,
        sleep=clock.sleep,
        include_cold=True,
    )
    assert len(samples) == 2 * n_reps
    assert sum(1 for s in samples if s.kind == "warm") == n_reps
    assert sum(1 for s in samples if s.kind == "cold") == n_reps
    elapsed = [s.elapsed_s for s in samples]
    mean, std = mean_std(elapsed)
    expected_mean = (warm_s + cold_s) / 2.0
    expected_std = math.sqrt(
        (
            n_reps * (warm_s - expected_mean) ** 2
            + n_reps * (cold_s - expected_mean) ** 2
        )
        / (2 * n_reps - 1)
    )
    assert mean == pytest.approx(expected_mean)
    assert std == pytest.approx(expected_std)
    path = tmp_path / "cold_start.json"
    payload = write_cold_start_json(
        path, samples, n_max=16, host_cpu_count=32
    )
    loaded = json.loads(path.read_text())
    assert loaded["mean_s"] == pytest.approx(mean)
    assert loaded["std_s"] == pytest.approx(std)
    assert loaded["n"] == 2 * n_reps
    assert loaded["n_max"] == 16
    assert loaded["host_cpu_count"] == 32
    assert loaded["n_max_below_host_cores"] is True
    assert payload["warm"]["n"] == n_reps  # type: ignore[index]


def test_cold_start_json_refuses_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no samples"):
        write_cold_start_json(
            tmp_path / "cold_start.json",
            [],
            n_max=16,
            host_cpu_count=8,
        )


def test_docker_backend_targets_testbed_compose() -> None:
    runs: list[list[str]] = []

    def runner(cmd: list[str]) -> CompletedProcess[str]:
        runs.append(cmd)
        return CompletedProcess(cmd, 0)

    backend = DockerComposeBackend(
        "service",
        Path("testbed/docker-compose.yml"),
        timeout_s=0.2,
        poll_s=0.01,
        count_healthy=lambda: 2,
        runner=runner,
        monotonic=lambda: 0.0,
        sleep=lambda _s: None,
    )
    backend.set_replicas(2)
    thread = backend._thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert runs
    assert "--scale" in runs[0]
    assert "service=2" in runs[0]
    assert "testbed/docker-compose.yml" in runs[0][3] or runs[0][3].endswith(
        "docker-compose.yml"
    )
