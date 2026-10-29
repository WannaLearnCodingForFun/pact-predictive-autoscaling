"""Constrained Docker operations (Phase 11.5).

The compose project name, compose file, and service names are hardcoded.
There is no API that accepts a client command, image, or container id.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pact.dashboard import COMPOSE_PROJECT
from pact.dashboard.security import REPO_ROOT

logger = logging.getLogger(__name__)

PROJECT_NAME = COMPOSE_PROJECT
COMPOSE_FILE = REPO_ROOT / "testbed" / "docker-compose.yml"
HARDCODED_SERVICE = "service"
ALLOWED_OPS = ("up", "down", "scale", "ps", "logs")
ALLOWED_SERVICES = (
    "service",
    "nginx",
    "nginx-exporter",
    "cadvisor",
    "prometheus",
    "pact",
)
MAX_LOG_TAIL = 500
MAX_SCALE = 32
COLD_START_OUTPUT = REPO_ROOT / "results" / "cold_start.json"
ComposeRunner = Callable[[list[str]], object]
MeasureFn = Callable[[Path], Mapping[str, object]]


class DockerUnavailable(RuntimeError):
    """Raised when the daemon is unreachable or the SDK is missing."""


class ServiceNotAllowed(ValueError):
    """Raised when a client names a service outside the hardcoded set."""


class DockerCtl:
    """Fixed-op compose controller. Client input never reaches a shell."""

    def __init__(
        self,
        *,
        runner: ComposeRunner | None = None,
        client_factory: Callable[[], Any] | None = None,
        measure_fn: MeasureFn | None = None,
        cold_start_path: Path | None = None,
    ) -> None:
        self._runner = runner
        self._client_factory = client_factory
        self._measure_fn = measure_fn
        self._cold_start_path = cold_start_path or COLD_START_OUTPUT

    @property
    def cold_start_path(self) -> Path:
        return self._cold_start_path

    def compose_argv(self, extra: Sequence[str]) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            PROJECT_NAME,
            "-f",
            str(COMPOSE_FILE),
            *list(extra),
        ]

    def up(self) -> dict[str, object]:
        self._run(self.compose_argv(("up", "-d")))
        return {"ok": True, "op": "up", "project": PROJECT_NAME}

    def down(self) -> dict[str, object]:
        self._run(self.compose_argv(("down",)))
        return {"ok": True, "op": "down", "project": PROJECT_NAME}

    def scale(self, n: int) -> dict[str, object]:
        replicas = int(n)
        if replicas < 1 or replicas > MAX_SCALE:
            raise ValueError(f"scale n must be in 1..{MAX_SCALE}, got {replicas}")
        extra = ("up", "-d", "--scale", f"{HARDCODED_SERVICE}={replicas}")
        self._run(self.compose_argv(extra))
        return {
            "ok": True,
            "op": "scale",
            "project": PROJECT_NAME,
            "service": HARDCODED_SERVICE,
            "n": replicas,
        }

    def status(self) -> dict[str, object]:
        try:
            client = self._client()
        except DockerUnavailable:
            return {
                "daemon_reachable": False,
                "project": PROJECT_NAME,
                "project_up": False,
                "replica_count": 0,
                "services": [],
            }
        services = self._service_rows(client)
        replicas = 0
        for row in services:
            if row["name"] == HARDCODED_SERVICE:
                replicas = _as_int(row["running"])
        return {
            "daemon_reachable": True,
            "project": PROJECT_NAME,
            "project_up": any(_as_int(s["running"]) > 0 for s in services),
            "replica_count": replicas,
            "services": services,
        }

    def logs(self, *, service: str = HARDCODED_SERVICE, tail: int = 100) -> str:
        name = self._require_service(service)
        n = min(max(int(tail), 1), MAX_LOG_TAIL)
        client = self._client()
        matches = [
            c
            for c in self._project_containers(client, all_containers=True)
            if c.labels.get("com.docker.compose.service") == name
        ]
        if not matches:
            return ""
        chunks: list[str] = []
        for container in matches:
            raw = container.logs(tail=n, timestamps=False)
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(raw)
            chunks.append(text)
        joined = "\n".join(chunks)
        lines = joined.splitlines()
        return "\n".join(lines[-n:])

    def measure_cold_start(self, *, n_reps: int = 1) -> dict[str, object]:
        """Write ``results/cold_start.json``. Output path is not client-supplied."""

        reps = min(max(int(n_reps), 1), 20)
        path = self._cold_start_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._measure_fn is not None:
            payload = dict(self._measure_fn(path))
            if "mean_s" not in payload:
                raise ValueError("measure_fn must write a payload with mean_s")
            if not path.is_file():
                path.write_text(_json_dump(payload))
            return payload
        return _run_live_measurement(path, n_reps=reps)

    def _run(self, cmd: list[str]) -> None:
        logger.info("docker argv=%s", cmd)
        if self._runner is not None:
            self._runner(cmd)
            return
        subprocess.run(cmd, check=True, capture_output=True, text=True, shell=False)

    def _client(self) -> Any:
        if self._client_factory is not None:
            try:
                return self._client_factory()
            except Exception as exc:
                raise DockerUnavailable("docker client factory failed") from exc
        try:
            import docker
        except ImportError as exc:
            raise DockerUnavailable("docker SDK is not installed") from exc
        try:
            client = docker.from_env()
            client.ping()
        except Exception as exc:
            raise DockerUnavailable("docker daemon is not reachable") from exc
        return client

    def _project_containers(
        self, client: Any, *, all_containers: bool
    ) -> list[Any]:
        return list(
            client.containers.list(
                all=all_containers,
                filters={"label": f"com.docker.compose.project={PROJECT_NAME}"},
            )
        )

    def _service_rows(self, client: Any) -> list[dict[str, object]]:
        by_name: dict[str, dict[str, object]] = {
            name: {
                "name": name,
                "desired": 0,
                "running": 0,
                "health": "unknown",
                "uptime": "",
            }
            for name in ALLOWED_SERVICES
        }
        for container in self._project_containers(client, all_containers=True):
            labels = getattr(container, "labels", {}) or {}
            name = str(labels.get("com.docker.compose.service", ""))
            if name not in by_name:
                continue
            row = by_name[name]
            row["desired"] = _as_int(row["desired"]) + 1
            status = str(getattr(container, "status", ""))
            if status == "running":
                row["running"] = _as_int(row["running"]) + 1
            attrs = getattr(container, "attrs", {}) or {}
            state = attrs.get("State", {}) if isinstance(attrs, dict) else {}
            health = state.get("Health", {}) if isinstance(state, dict) else {}
            if isinstance(health, dict) and health.get("Status"):
                row["health"] = str(health["Status"])
            elif status:
                row["health"] = status
            started = state.get("StartedAt", "") if isinstance(state, dict) else ""
            if started:
                row["uptime"] = str(started)
        return [by_name[name] for name in ALLOWED_SERVICES]

    @staticmethod
    def _require_service(service: str) -> str:
        if service not in ALLOWED_SERVICES:
            raise ServiceNotAllowed(f"service not allowed: {service!r}")
        return service


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"cannot coerce {type(value).__name__} to int")


def _json_dump(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(dict(payload), indent=2) + "\n"


def _run_live_measurement(path: Path, *, n_reps: int) -> dict[str, object]:
    from scripts.measure_cold_start import (
        DEFAULT_PROBE_URL,
        ComposeScaler,
        HttpReplicaProbe,
        run_measurement,
        write_cold_start_json,
    )

    n_max = 6
    try:
        import yaml

        raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
        if isinstance(raw, dict):
            n_max = int((raw.get("control") or {}).get("n_max", 6))
    except (OSError, ValueError, TypeError):
        n_max = 6
    cpu = os.cpu_count() or 1
    samples = run_measurement(
        scaler=ComposeScaler(HARDCODED_SERVICE, COMPOSE_FILE),
        probe=HttpReplicaProbe(DEFAULT_PROBE_URL),
        n0=1,
        n_reps=n_reps,
        timeout_s=120.0,
        poll_s=0.2,
        include_cold=False,
    )
    return write_cold_start_json(
        path, samples, n_max=n_max, host_cpu_count=int(cpu)
    )
