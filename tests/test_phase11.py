"""Phase 11: bind, control gate, launcher, docker ops, replay."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from pact.actuation.dry_run import DryRunBackend
from pact.loop import ControlLoop, RepeatObservationForecaster, TickObserver, TickRecord
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import SimCollector

from tests.helpers import make_config

MUTATING_PATHS = (
    "/api/experiment/start",
    "/api/experiment/cancel",
    "/api/docker/up",
    "/api/docker/down",
    "/api/docker/scale",
    "/api/docker/measure_cold_start",
)


def _loop(
    n_ticks: int, observers: Sequence[TickObserver] | None = None
) -> ControlLoop:
    cfg = make_config(dt=5.0, tau_c_s=10.0)
    cfg = replace(cfg, telemetry=replace(cfg.telemetry, window=4))
    sim = PoolSimulator(cfg, n_initial=2)
    return ControlLoop(
        cfg,
        collector=SimCollector(sim),
        backend=DryRunBackend(n_initial=2),
        forecaster=RepeatObservationForecaster(cfg.forecast.horizon),
        simulator=sim,
        arrival_rates=[20.0] * n_ticks,
        n_initial=2,
        sleep=lambda _s: None,
        observers=observers,
    )


def test_observer_exception_does_not_interrupt_loop() -> None:
    class Boom:
        def on_tick(self, record: TickRecord) -> None:
            del record
            raise RuntimeError("dashboard bus is down")

    n_ticks = 8
    result = _loop(n_ticks, observers=[Boom()]).run(n_ticks)
    assert len(result.ticks) == n_ticks
    assert result.ticks[0].tick == 0
    assert result.ticks[-1].tick == n_ticks - 1


def test_loop_source_does_not_import_dashboard() -> None:
    text = Path("pact/loop.py").read_text(encoding="utf-8")
    assert "dashboard" not in text
    assert "pact.dashboard" not in text


def test_observers_are_optional() -> None:
    n_ticks = 4
    result = _loop(n_ticks).run(n_ticks)
    assert len(result.ticks) == n_ticks


def test_bind_address_is_loopback_only() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard import BIND_HOST, BIND_PORT
    from pact.dashboard.app import create_app, run_server
    from pact.dashboard.security import assert_local_bind

    assert BIND_HOST == "127.0.0.1"
    assert BIND_PORT == 8000
    assert assert_local_bind("127.0.0.1") == "127.0.0.1"
    with pytest.raises(ValueError, match="127.0.0.1"):
        assert_local_bind("0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        assert_local_bind("::")
    with pytest.raises(ValueError, match="127.0.0.1"):
        run_server(host="0.0.0.0")
    client = TestClient(create_app())
    body = client.get("/api/bind").json()
    assert body["host"] == "127.0.0.1"


def test_mutating_endpoints_forbidden_without_control_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard.app import create_app

    monkeypatch.delenv("PACT_DASHBOARD_ALLOW_CONTROL", raising=False)
    client = TestClient(create_app())
    for path in MUTATING_PATHS:
        response = client.post(path, json={})
        assert response.status_code == 403, path


def test_config_path_outside_repo_is_rejected() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard.app import create_app
    from pact.dashboard.security import (
        ConfigPathRejected,
        list_allowed_configs,
        resolve_allowed_config,
    )

    with pytest.raises(ConfigPathRejected):
        resolve_allowed_config("/etc/passwd")
    with pytest.raises(ConfigPathRejected):
        resolve_allowed_config("../../etc/passwd")
    with pytest.raises(ConfigPathRejected):
        resolve_allowed_config("README.md")
    allowed = list_allowed_configs()
    assert "configs/default.yaml" in allowed
    assert any(p.startswith("configs/ablations/") for p in allowed)
    resolved = resolve_allowed_config("configs/default.yaml")
    assert resolved.name == "default.yaml"

    client = TestClient(create_app())
    denied = client.get("/api/config", params={"path": "/tmp/evil.yaml"})
    assert denied.status_code == 400
    escaped = client.get("/api/config", params={"path": "../pyproject.toml"})
    assert escaped.status_code == 400
    ok = client.get("/api/config", params={"path": "configs/default.yaml"})
    assert ok.status_code == 200
    assert ok.json()["path"] == "configs/default.yaml"


def test_experiment_start_rejects_outside_path_when_control_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard.app import create_app

    monkeypatch.setenv("PACT_DASHBOARD_ALLOW_CONTROL", "1")
    client = TestClient(create_app())
    denied = client.post(
        "/api/experiment/start", json={"config": "/etc/passwd"}
    )
    assert denied.status_code == 400
    escaped = client.post(
        "/api/experiment/start", json={"config": "../../etc/passwd"}
    )
    assert escaped.status_code == 400


def test_bus_observer_publishes_ticks_to_websocket() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard.app import create_app
    from pact.dashboard.bus import BusObserver, TickBus, encode_tick

    n_ticks = 6
    bus = TickBus()
    result = _loop(n_ticks, observers=[BusObserver(bus)]).run(n_ticks)
    assert len(bus.buffer) == n_ticks
    payload = encode_tick(result.ticks[-1])
    assert payload["tick"] == n_ticks - 1
    assert "observation" in payload
    assert "demand" in payload
    assert "drift" in payload
    assert "wall_time_s" in payload
    assert "n_required" in payload
    assert "mpc_solve_time_s" in payload
    client = TestClient(create_app(bus=bus))
    with client.websocket_connect("/ws/ticks") as ws:
        first = ws.receive_json()
        assert first["tick"] == 0
    index = client.get("/")
    html = index.text
    assert "Replicas vs demand" in html
    assert "Latency" in html
    assert "Forecast overlay" in html
    assert "Control internals" in html
    assert "Action log" in html
    assert "Metrics" in html
    assert "Experiment" in html
    assert "Replay" in html
    assert "Comparison" in html


def _write_cold_start(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mean_s": 2.0,
                "std_s": 0.0,
                "n": 1,
                "samples_s": [2.0],
                "n_max": 6,
                "host_cpu_count": 8,
                "n_max_below_host_cores": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_measure_cold_start_writes_results_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard.app import create_app
    from pact.dashboard.docker_ctl import PROJECT_NAME, DockerCtl
    from scripts.measure_cold_start import ColdStartSample, write_cold_start_json

    out = tmp_path / "cold_start.json"

    def measure(path: Path) -> dict[str, object]:
        return write_cold_start_json(
            path,
            [ColdStartSample(kind="warm", elapsed_s=1.5, replica_id="r2")],
            n_max=6,
            host_cpu_count=8,
        )

    ctl = DockerCtl(runner=lambda cmd: None, measure_fn=measure, cold_start_path=out)
    monkeypatch.setenv("PACT_DASHBOARD_ALLOW_CONTROL", "1")
    client = TestClient(
        create_app(docker=ctl, results_root=tmp_path, cold_start_path=out)
    )
    response = client.post("/api/docker/measure_cold_start", json={"n_reps": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["mean_s"] == pytest.approx(1.5)
    written = json.loads(out.read_text())
    assert written["mean_s"] == pytest.approx(1.5)
    assert written["n"] == 1
    argv = ctl.compose_argv(("up", "-d"))
    assert argv[:5] == ["docker", "compose", "-p", PROJECT_NAME, "-f"]
    assert PROJECT_NAME == "pact"


def test_docker_ops_are_hardcoded_and_reject_client_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard.app import create_app
    from pact.dashboard.docker_ctl import HARDCODED_SERVICE, PROJECT_NAME, DockerCtl

    cmds: list[list[str]] = []

    class _FakeContainer:
        labels = {
            "com.docker.compose.project": PROJECT_NAME,
            "com.docker.compose.service": HARDCODED_SERVICE,
        }
        status = "running"
        attrs = {
            "State": {"Health": {"Status": "healthy"}, "StartedAt": "t0"}
        }

        def logs(self, tail: int = 100, timestamps: bool = False) -> bytes:
            del timestamps
            return b"line1\nline2\n" * max(tail, 1)

    class _FakeClient:
        def ping(self) -> bool:
            return True

        class containers:
            @staticmethod
            def list(all: bool = True, filters: object = None) -> list[_FakeContainer]:
                del all, filters
                return [_FakeContainer()]

    ctl = DockerCtl(
        runner=lambda cmd: cmds.append(list(cmd)),
        client_factory=_FakeClient,
        cold_start_path=tmp_path / "cold_start.json",
    )
    monkeypatch.setenv("PACT_DASHBOARD_ALLOW_CONTROL", "1")
    client = TestClient(create_app(docker=ctl, results_root=tmp_path))
    assert client.post("/api/docker/up", json={}).status_code == 200
    assert cmds[0][3] == PROJECT_NAME
    denied = client.post(
        "/api/docker/scale", json={"n": 2, "command": "rm -rf /", "image": "evil"}
    )
    assert denied.status_code == 400
    scaled = client.post("/api/docker/scale", json={"n": 3})
    assert scaled.status_code == 200
    assert f"{HARDCODED_SERVICE}=3" in cmds[-1]
    assert "evil" not in " ".join(cmds[-1])
    logs = client.get("/api/docker/logs", params={"service": "service", "tail": 2})
    assert logs.status_code == 200
    forbidden = client.get("/api/docker/logs", params={"service": "not-a-service"})
    assert forbidden.status_code == 400
    status = client.get("/api/docker/status").json()
    assert status["daemon_reachable"] is True
    assert status["replica_count"] == 1


def test_simulator_start_matches_cli_csvs_and_feeds_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pact.dashboard.app import create_app
    from pact.dashboard.docker_ctl import DockerCtl
    from pact.dashboard.replay import (
        PANEL_FIELDS,
        load_replay_ticks,
        replay_has_six_panels,
    )
    from pact.eval.runner import main as runner_main

    cs = tmp_path / "cold_start.json"
    _write_cold_start(cs)
    cli_dir = tmp_path / "cli"
    ui_root = tmp_path / "ui"
    runner_main(
        [
            "--config",
            "configs/default.yaml",
            "--cold-start",
            str(cs),
            "--results",
            str(cli_dir),
            "--seeds",
            "1",
            "--ticks",
            "16",
            "--methods",
            "pact,reactive,arima,lstm",
        ]
    )
    monkeypatch.setenv("PACT_DASHBOARD_ALLOW_CONTROL", "1")
    client = TestClient(
        create_app(
            docker=DockerCtl(runner=lambda cmd: None, cold_start_path=cs),
            results_root=ui_root,
            cold_start_path=cs,
            sync_experiments=True,
        )
    )
    started = client.post(
        "/api/experiment/start",
        json={
            "config": "configs/default.yaml",
            "trace": "synthetic:step",
            "methods": ["pact", "reactive", "arima", "lstm"],
            "seeds": 1,
            "ticks": 16,
            "target": "simulator",
        },
    )
    assert started.status_code == 202
    run = started.json()
    assert run["status"] == "completed", run.get("error")
    live = client.get("/api/ticks").json()["ticks"]
    assert live
    assert all(key in live[0] for key in PANEL_FIELDS)
    with client.websocket_connect("/ws/ticks") as ws:
        first = ws.receive_json()
        assert first["tick"] == 0
        assert "observation" in first
        assert "demand" in first
        assert "drift" in first
    ui_dir = ui_root / run["results_dir"]
    for name in ("replicas.csv", "latency.csv", "forecast.csv", "actions.csv"):
        assert (cli_dir / name).read_text(encoding="utf-8") == (
            ui_dir / name
        ).read_text(encoding="utf-8")
    replay = client.get("/api/replay/ticks", params={"dir": run["results_dir"]})
    assert replay.status_code == 200
    payload = replay.json()
    assert payload["six_panels"] is True
    jsonl_path = ui_dir / "ticks.jsonl"
    recorded = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert payload["ticks"] == recorded
    jsonl_path.unlink()
    reconstructed = load_replay_ticks(ui_dir)
    assert replay_has_six_panels(reconstructed)
    assert reconstructed[0]["n_required"] == recorded[0]["n_required"]
    assert reconstructed[0]["observation"]["ell_p95"] == pytest.approx(
        recorded[0]["observation"]["ell_p95"]
    )
    outside = client.get("/api/replay/ticks", params={"dir": "/etc/passwd"})
    assert outside.status_code == 400

