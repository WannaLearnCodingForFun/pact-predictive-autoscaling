"""Phase 7 acceptance tests: actuation backends and the control loop."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from pact.actuation.docker_backend import DockerComposeBackend
from pact.actuation.dry_run import DryRunBackend
from pact.config import MissingColdStartError
from pact.loop import ControlLoop, RepeatObservationForecaster
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import SimCollector

from tests.helpers import make_config


class _FakeClock:
    def __init__(self, t0: float = 100.0) -> None:
        self.t = t0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError("sleep remaining is negative")
        self.sleeps.append(seconds)
        self.t += seconds


def test_loop_refuses_to_start_without_cold_start_file(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(Path("configs/default.yaml").read_text())
    with pytest.raises(MissingColdStartError):
        ControlLoop.from_yaml(
            yaml_path, cold_start_path=tmp_path / "missing.json"
        )


def test_loop_200_ticks_sim_collector_dry_run_stable_timing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    n_ticks = 200
    cfg = make_config(dt=5.0, tau_c_s=10.0, n_min=1, n_max=16)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=8))
    sim = PoolSimulator(cfg, n_initial=2)
    backend = DryRunBackend(n_initial=2)
    clock = _FakeClock()
    loop = ControlLoop(
        cfg,
        collector=SimCollector(sim),
        backend=backend,
        forecaster=RepeatObservationForecaster(cfg.forecast.horizon),
        simulator=sim,
        arrival_rates=[20.0] * n_ticks,
        n_initial=2,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    with caplog.at_level(logging.INFO, logger="pact.loop"):
        result = loop.run(n_ticks)
    assert len(result.ticks) == n_ticks
    assert len(clock.sleeps) == n_ticks
    assert all(s == pytest.approx(cfg.telemetry.dt) for s in clock.sleeps)
    assert all(slip == 0.0 for slip in result.deadline_slips_s)
    logged = [
        json.loads(rec.message)
        for rec in caplog.records
        if rec.message.startswith("{")
    ]
    assert len(logged) == n_ticks
    sample = logged[-1]
    for key in (
        "timestamp",
        "u",
        "r",
        "lam",
        "ell_p95",
        "n",
        "forecast",
        "demand",
        "mpc_u",
        "applied_u",
        "gate",
        "gamma",
    ):
        assert key in sample


def test_dry_run_set_replicas_is_idempotent() -> None:
    backend = DryRunBackend(n_initial=2)
    backend.set_replicas(2)
    assert backend.commands == []
    backend.set_replicas(4)
    backend.set_replicas(4)
    assert backend.commands == [4]
    assert backend.in_sync()


def test_docker_backend_scale_is_async_and_idempotent() -> None:
    started = threading.Event()
    release = threading.Event()
    runs: list[list[str]] = []

    def runner(cmd: list[str]) -> CompletedProcess[str]:
        runs.append(cmd)
        started.set()
        assert release.wait(timeout=2.0)
        return CompletedProcess(cmd, 0)

    def count() -> int:
        return 3 if release.is_set() else 1

    backend = DockerComposeBackend(
        "service",
        Path("testbed/docker-compose.yml"),
        timeout_s=2.0,
        poll_s=0.01,
        count_healthy=count,
        runner=runner,
        monotonic=lambda: 0.0 if not release.is_set() else 0.5,
        sleep=lambda _s: None,
    )
    backend.set_replicas(3)
    assert started.wait(timeout=2.0)
    assert backend.desired_replicas == 3
    assert not backend.in_sync()
    backend.set_replicas(3)
    assert len(runs) == 1
    release.set()
    thread = backend._thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert backend.in_sync()
    assert runs[0][-1] == "service=3"
    assert "--scale" in runs[0]


@pytest.mark.skipif(
    os.environ.get("PACT_TESTBED") != "1",
    reason="set PACT_TESTBED=1 with a running compose stack to exercise live scale",
)
def test_loop_real_testbed_scales_service_up_and_down() -> None:
    compose = Path("testbed/docker-compose.yml")
    assert compose.is_file()
    backend = DockerComposeBackend(
        "service",
        compose,
        timeout_s=30.0,
        poll_s=0.5,
    )
    backend.set_replicas(2)
    thread = backend._thread
    assert thread is not None
    thread.join(timeout=35.0)
    assert backend.desired_replicas == 2
    backend.set_replicas(1)
    thread = backend._thread
    assert thread is not None
    thread.join(timeout=35.0)
    assert backend.desired_replicas == 1

