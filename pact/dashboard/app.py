"""FastAPI dashboard: live telemetry WebSocket and gated command REST."""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue as queue_mod
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from pact.config import (
    DEFAULT_COLD_START_PATH,
    DEFAULT_CONFIG_PATH,
    MissingColdStartError,
    load_cold_start_s,
)
from pact.dashboard import BIND_HOST, BIND_PORT, CONTROL_ENV
from pact.dashboard.bus import TickBus
from pact.dashboard.docker_ctl import (
    HARDCODED_SERVICE,
    PROJECT_NAME,
    DockerCtl,
    DockerUnavailable,
    ServiceNotAllowed,
)
from pact.dashboard.experiments import ALLOWED_METHODS, ExperimentManager
from pact.dashboard.replay import load_replay_ticks, replay_has_six_panels
from pact.dashboard.security import (
    REPO_ROOT,
    RESULTS_DIR,
    ConfigPathRejected,
    assert_local_bind,
    control_allowed,
    list_allowed_configs,
    list_allowed_traces,
    resolve_allowed_config,
    resolve_results_dir,
)
from pact.dashboard.state import TickBuffer

logger = logging.getLogger(__name__)


def create_app(
    bus: TickBus | None = None,
    *,
    docker: DockerCtl | None = None,
    experiments: ExperimentManager | None = None,
    results_root: Path | None = None,
    cold_start_path: Path | None = None,
    sync_experiments: bool = False,
) -> FastAPI:
    tick_bus = bus if bus is not None else TickBus(TickBuffer())
    static_dir = Path(__file__).resolve().parent / "static"
    results = results_root if results_root is not None else RESULTS_DIR
    cold_path = (
        cold_start_path if cold_start_path is not None else DEFAULT_COLD_START_PATH
    )
    docker_ctl = docker if docker is not None else DockerCtl(cold_start_path=cold_path)
    manager = experiments or ExperimentManager(
        bus=tick_bus,
        results_root=results,
        cold_start_path=cold_path,
        docker=docker_ctl,
        sync=sync_experiments,
    )

    app = FastAPI(title="PACT dashboard", docs_url=None, redoc_url=None)
    app.state.bus = tick_bus
    app.state.docker = docker_ctl
    app.state.experiments = manager

    class ControlGate(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Any:
            path = request.url.path
            mutating = path.startswith("/api/experiment/") or path.startswith(
                "/api/docker/"
            )
            if mutating and request.method not in {"GET", "HEAD", "OPTIONS"}:
                if not control_allowed():
                    return JSONResponse(
                        {
                            "detail": (
                                f"mutating endpoints require {CONTROL_ENV}=1"
                            )
                        },
                        status_code=403,
                    )
            return await call_next(request)

    app.add_middleware(ControlGate)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "bind": BIND_HOST}

    @app.get("/api/bind")
    def bind_info() -> dict[str, object]:
        return {"host": BIND_HOST, "port": BIND_PORT}

    @app.get("/api/configs")
    def configs() -> dict[str, object]:
        return {"configs": list(list_allowed_configs())}

    @app.get("/api/workloads")
    def workloads() -> dict[str, object]:
        return {"traces": list(list_allowed_traces())}

    @app.get("/api/methods")
    def methods() -> dict[str, object]:
        return {"methods": list(ALLOWED_METHODS)}

    @app.get("/api/config")
    def config_lookup(path: str) -> Any:
        try:
            resolved = resolve_allowed_config(path)
        except ConfigPathRejected as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        rel = resolved.relative_to(REPO_ROOT).as_posix()
        return {"path": rel}

    @app.get("/api/ticks")
    def ticks() -> dict[str, object]:
        return {"ticks": tick_bus.buffer.snapshot()}

    @app.get("/api/meta")
    def meta() -> dict[str, object]:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
        capacity = raw.get("capacity", {}) if isinstance(raw, dict) else {}
        telemetry = raw.get("telemetry", {}) if isinstance(raw, dict) else {}
        forecast = raw.get("forecast", {}) if isinstance(raw, dict) else {}
        drift = raw.get("drift", {}) if isinstance(raw, dict) else {}
        tau_c: float | None
        try:
            tau_c = load_cold_start_s(cold_path)
        except (MissingColdStartError, ValueError, OSError):
            tau_c = None
        docker_status = docker_ctl.status()
        return {
            "slo_ms": float(capacity.get("slo_ms", 200.0)),
            "dt": float(telemetry.get("dt", 5.0)),
            "horizon": int(forecast.get("horizon", 12)),
            "gamma_max": float(drift.get("gamma_max", 0.5)),
            "tau_c_s": tau_c,
            "compose_project": PROJECT_NAME,
            "bind": BIND_HOST,
            "daemon_reachable": bool(docker_status.get("daemon_reachable")),
            "project_up": bool(docker_status.get("project_up")),
            "replica_count": _as_int(docker_status.get("replica_count"), 0),
            "controller_running": manager.controller_running(),
        }

    @app.get("/api/docker/status")
    def docker_status() -> dict[str, object]:
        return docker_ctl.status()

    @app.get("/api/docker/logs")
    def docker_logs(
        service: str = HARDCODED_SERVICE, tail: int = 100
    ) -> Any:
        try:
            text = docker_ctl.logs(service=service, tail=tail)
        except ServiceNotAllowed as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except DockerUnavailable as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        return {"service": service, "tail": min(max(tail, 1), 500), "log": text}

    @app.post("/api/docker/up")
    def docker_up() -> Any:
        return _docker_mutate(docker_ctl.up, "up")

    @app.post("/api/docker/down")
    def docker_down() -> Any:
        return _docker_mutate(docker_ctl.down, "down")

    @app.post("/api/docker/scale")
    async def docker_scale(request: Request) -> Any:
        _require_control()
        body = await _json_body(request)
        if "command" in body or "image" in body or "container" in body:
            return JSONResponse(
                {"detail": "client-supplied docker command/image/id is rejected"},
                status_code=400,
            )
        n = int(body.get("n", 0))
        _log_mutation("docker/scale", {"n": n, "project": PROJECT_NAME})
        try:
            return docker_ctl.scale(n)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except (DockerUnavailable, OSError, subprocess.CalledProcessError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.post("/api/docker/measure_cold_start")
    async def docker_measure(request: Request) -> Any:
        _require_control()
        body = await _json_body(request)
        n_reps = int(body.get("n_reps", 1))
        _log_mutation(
            "docker/measure_cold_start",
            {"n_reps": n_reps, "output": str(docker_ctl.cold_start_path)},
        )
        try:
            payload = docker_ctl.measure_cold_start(n_reps=n_reps)
        except (DockerUnavailable, OSError, ValueError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        return payload

    @app.post("/api/experiment/start")
    async def experiment_start(request: Request) -> Any:
        _require_control()
        body = await _json_body(request)
        try:
            record = manager.start(body)
        except ConfigPathRejected as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except (ValueError, DockerUnavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        _log_mutation("experiment/start", body)
        return JSONResponse(record.as_dict(), status_code=202)

    @app.post("/api/experiment/cancel")
    async def experiment_cancel(request: Request) -> Any:
        _require_control()
        body = await _json_body(request)
        run_id = str(body.get("id", ""))
        _log_mutation("experiment/cancel", {"id": run_id})
        try:
            record = manager.cancel(run_id)
        except KeyError:
            return JSONResponse({"detail": "unknown run"}, status_code=404)
        return record.as_dict()

    @app.get("/api/experiment/runs")
    def experiment_runs() -> dict[str, object]:
        return {"runs": manager.list_runs()}

    @app.get("/api/experiment/{run_id}")
    def experiment_one(run_id: str) -> Any:
        record = manager.get(run_id)
        if record is None:
            return JSONResponse({"detail": "unknown run"}, status_code=404)
        return record.as_dict()

    @app.get("/api/replay/ticks")
    def replay_ticks(dir: str) -> Any:
        try:
            resolved = resolve_results_dir(dir, root=results)
        except ConfigPathRejected as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        try:
            loaded = load_replay_ticks(resolved)
        except (ValueError, OSError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return {
            "dir": dir,
            "ticks": loaded,
            "six_panels": replay_has_six_panels(loaded),
        }

    @app.websocket("/ws/ticks")
    async def ticks_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        q = tick_bus.subscribe()

        def _take() -> dict[str, Any] | None:
            try:
                return q.get(timeout=0.5)
            except queue_mod.Empty:
                return None

        try:
            for item in tick_bus.buffer.snapshot():
                await websocket.send_json(item)
            while True:
                payload = await asyncio.to_thread(_take)
                if payload is None:
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            return
        finally:
            tick_bus.unsubscribe(q)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def _docker_mutate(op: Any, name: str) -> Any:
    _require_control()
    _log_mutation(f"docker/{name}", {"project": PROJECT_NAME, "op": name})
    try:
        return op()
    except (DockerUnavailable, OSError, subprocess.CalledProcessError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return default


def _require_control() -> None:
    if not control_allowed():
        raise HTTPException(
            status_code=403,
            detail=f"mutating endpoints require {CONTROL_ENV}=1",
        )


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        raw = await request.json()
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _log_mutation(action: str, params: dict[str, Any]) -> None:
    logger.info(
        "dashboard mutate ts=%.3f action=%s params=%s",
        time.time(),
        action,
        params,
    )


def run_server(host: str = BIND_HOST, port: int = BIND_PORT) -> None:
    assert_local_bind(host)
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT local dashboard")
    parser.add_argument("--host", default=BIND_HOST)
    parser.add_argument("--port", type=int, default=BIND_PORT)
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port)
