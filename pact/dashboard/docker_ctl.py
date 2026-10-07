"""Constrained Docker operations (Phase 11.5 — not implemented in 11.0–11.3).

The compose project name is hardcoded. There is no API that accepts a client
command, image, or container id.
"""

from __future__ import annotations

from pact.dashboard import COMPOSE_PROJECT

# Hardcoded compose project. Scale/up/down in Phase 11.5 use this name only.
PROJECT_NAME = COMPOSE_PROJECT
ALLOWED_OPS = ("up", "down", "scale", "ps", "logs")
