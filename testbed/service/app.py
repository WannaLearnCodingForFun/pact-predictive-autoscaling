"""Single-threaded HTTP workload with a concurrency cap and CPU burn.

Environment:
  PORT          listen port (default 8080)
  WORK_MS       busy-loop milliseconds per request (default 20)
  CONCURRENCY   in-flight request cap (default 1)
"""

from __future__ import annotations

import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORK_MS = float(os.environ.get("WORK_MS", "20"))
CONCURRENCY = max(1, int(os.environ.get("CONCURRENCY", "1")))
PORT = int(os.environ.get("PORT", "8080"))
REPLICA_ID = os.environ.get("HOSTNAME", socket.gethostname())
_GATE = threading.Semaphore(CONCURRENCY)
_REQUESTS = 0
_LOCK = threading.Lock()


def _burn(work_ms: float) -> None:
    deadline = time.perf_counter() + work_ms / 1000.0
    while time.perf_counter() < deadline:
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._respond(200, b"ok")
            return
        if self.path.startswith("/metrics"):
            with _LOCK:
                count = _REQUESTS
            body = (
                "# TYPE http_requests_total counter\n"
                f"http_requests_total {count}\n"
                "# TYPE nginx_http_requests_total counter\n"
                f"nginx_http_requests_total {count}\n"
            ).encode()
            self._respond(200, body, content_type="text/plain; version=0.0.4")
            return
        acquired = _GATE.acquire(timeout=5.0)
        if not acquired:
            self._respond(503, b"busy")
            return
        try:
            _burn(WORK_MS)
            with _LOCK:
                global _REQUESTS
                _REQUESTS += 1
            if self.path.startswith("/id"):
                self._respond(200, REPLICA_ID.encode())
            else:
                self._respond(200, b"ok")
        finally:
            _GATE.release()

    def _respond(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Replica-Id", REPLICA_ID)
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
