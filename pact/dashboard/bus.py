"""In-process pub/sub from the control thread to WebSocket clients."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from pact.dashboard.state import TickBuffer
from pact.loop import TickRecord


def encode_tick(record: TickRecord) -> dict[str, Any]:
    """``dataclasses.asdict`` then orjson (json fallback) so tuples become lists."""

    payload: dict[str, Any] = asdict(record)
    try:
        import orjson

        loaded: Any = orjson.loads(orjson.dumps(payload))
    except ImportError:
        loaded = json.loads(json.dumps(payload))
    if not isinstance(loaded, dict):
        raise TypeError("tick payload must be an object")
    return loaded


class TickBus:
    """Thread-safe fan-out. The loop thread publishes; WS tasks subscribe."""

    def __init__(self, buffer: TickBuffer | None = None) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self.buffer = buffer if buffer is not None else TickBuffer()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish_payload(self, payload: Mapping[str, Any]) -> None:
        item = dict(payload)
        self.buffer.append(item)
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put(item)

    def publish(self, record: TickRecord) -> None:
        self.publish_payload(encode_tick(record))


class BusObserver:
    """``TickObserver`` that pushes onto the bus. Lives in the dashboard package."""

    def __init__(self, bus: TickBus) -> None:
        self._bus = bus

    def on_tick(self, record: TickRecord) -> None:
        self._bus.publish(record)
