"""Phase 11.0–11.3: bind, control gate, config allow-list, observer isolation."""

from __future__ import annotations

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
    not_built = client.post(
        "/api/experiment/start", json={"config": "configs/default.yaml"}
    )
    assert not_built.status_code == 501


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
