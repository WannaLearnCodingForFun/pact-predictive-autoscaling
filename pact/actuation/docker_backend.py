"""Docker Compose actuation with non-blocking reconciliation (Phase 7)."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

ComposeRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
ReplicaCounter = Callable[[], int]


class ActuationBackend(Protocol):
    def set_replicas(self, n: int) -> None: ...

    @property
    def desired_replicas(self) -> int: ...

    @property
    def observed_replicas(self) -> int: ...

    def in_sync(self) -> bool: ...


class DockerComposeBackend:
    """``docker compose up -d --scale service=n``, reconciled on a side thread.

    ``set_replicas`` never waits for healthy count. Calling with the current
    desired count is a no-op.
    """

    def __init__(
        self,
        service: str,
        compose_file: Path,
        *,
        timeout_s: float,
        poll_s: float,
        count_healthy: ReplicaCounter | None = None,
        runner: ComposeRunner | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        if poll_s <= 0.0:
            raise ValueError(f"poll_s must be positive, got {poll_s}")
        self._service = service
        self._compose_file = compose_file
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._count_healthy = count_healthy
        self._runner = runner or _default_runner
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._desired = 0
        self._observed = 0
        self._generation = 0
        self._thread: threading.Thread | None = None

    @property
    def desired_replicas(self) -> int:
        with self._lock:
            return self._desired

    @property
    def observed_replicas(self) -> int:
        with self._lock:
            return self._observed

    def in_sync(self) -> bool:
        with self._lock:
            return self._observed == self._desired

    def set_replicas(self, n: int) -> None:
        with self._lock:
            if n == self._desired:
                return
            self._desired = n
            self._generation += 1
            generation = self._generation
        thread = threading.Thread(
            target=self._reconcile, args=(n, generation), daemon=True
        )
        self._thread = thread
        thread.start()

    def _reconcile(self, n: int, generation: int) -> None:
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
        try:
            self._runner(cmd)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("compose scale failed: %s", exc)
            return
        deadline = self._monotonic() + self._timeout_s
        while self._monotonic() < deadline:
            with self._lock:
                if generation != self._generation:
                    return
            observed = self._read_count()
            with self._lock:
                self._observed = observed
                if generation != self._generation:
                    return
                if observed == n:
                    logger.info("reconciled service=%s n=%s", self._service, n)
                    return
            self._sleep(self._poll_s)
        logger.warning(
            "reconcile timeout service=%s desired=%s observed=%s",
            self._service,
            n,
            self.observed_replicas,
        )

    def _read_count(self) -> int:
        if self._count_healthy is not None:
            return int(self._count_healthy())
        return self._desired


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)
