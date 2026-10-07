"""Ring buffer of recent serialised ticks."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

TICK_BUFFER_MAX = 2000


class TickBuffer:
    def __init__(self, maxlen: int = TICK_BUFFER_MAX) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def append(self, payload: Mapping[str, Any]) -> None:
        self._buf.append(dict(payload))

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()
