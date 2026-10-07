"""FastAPI dashboard: live telemetry WebSocket and gated command REST."""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue as queue_mod
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
from pact.dashboard.docker_ctl import PROJECT_NAME
from pact.dashboard.security import (
    REPO_ROOT,
    ConfigPathRejected,
    assert_local_bind,
    control_allowed,
    list_allowed_configs,
    resolve_allowed_config,
)
from pact.dashboard.state import TickBuffer

logger = logging.getLogger(__name__)


def create_app(bus: TickBus | None = None) -> FastAPI:
    tick_bus = bus if bus is not None else TickBus(TickBuffer())
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="PACT dashboard", docs_url=None, redoc_url=None)
    app.state.bus = tick_bus

    class ControlGate(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Any:
            path = request.url.path
            mutating = path == "/api/experiment/start" or path.startswith(
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
            tau_c = load_cold_start_s(DEFAULT_COLD_START_PATH)
        except (MissingColdStartError, ValueError, OSError):
            tau_c = None
        return {
            "slo_ms": float(capacity.get("slo_ms", 200.0)),
            "dt": float(telemetry.get("dt", 5.0)),
            "horizon": int(forecast.get("horizon", 12)),
            "gamma_max": float(drift.get("gamma_max", 0.5)),
            "tau_c_s": tau_c,
            "compose_project": PROJECT_NAME,
            "bind": BIND_HOST,
        }

    @app.post("/api/experiment/start")
    async def experiment_start(request: Request) -> Any:
        _require_control()
        body = await _json_body(request)
        raw_path = str(body.get("config", ""))
        try:
            resolve_allowed_config(raw_path)
        except ConfigPathRejected as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        _log_mutation("experiment/start", body)
        return JSONResponse(
            {"detail": "experiment launcher is Phase 11.4"},
            status_code=501,
        )

    @app.post("/api/docker/up")
    async def docker_up() -> Any:
        return _docker_stub("up")

    @app.post("/api/docker/down")
    async def docker_down() -> Any:
        return _docker_stub("down")

    @app.post("/api/docker/scale")
    async def docker_scale() -> Any:
        return _docker_stub("scale")

    @app.post("/api/docker/measure_cold_start")
    async def docker_measure() -> Any:
        return _docker_stub("measure_cold_start")

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


def _require_control() -> None:
    if not control_allowed():
        raise HTTPException(
            status_code=403,
            detail=f"mutating endpoints require {CONTROL_ENV}=1",
        )


def _docker_stub(op: str) -> JSONResponse:
    _require_control()
    _log_mutation(f"docker/{op}", {"project": PROJECT_NAME, "op": op})
    return JSONResponse(
        {"detail": "docker control is Phase 11.5", "op": op},
        status_code=501,
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
