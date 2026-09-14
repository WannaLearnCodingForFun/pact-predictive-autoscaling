"""No-op actuation backend for simulation (Phase 7)."""

from __future__ import annotations


class DryRunBackend:
    """Records scale commands and treats them as immediately healthy."""

    def __init__(self, n_initial: int = 1) -> None:
        self._desired = n_initial
        self._observed = n_initial
        self.commands: list[int] = []

    @property
    def desired_replicas(self) -> int:
        return self._desired

    @property
    def observed_replicas(self) -> int:
        return self._observed

    def in_sync(self) -> bool:
        return self._observed == self._desired

    def set_replicas(self, n: int) -> None:
        if n == self._desired:
            return
        self._desired = n
        self._observed = n
        self.commands.append(n)
