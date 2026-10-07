"""Bind address, control-gate, and config allow-list (Phase 11.0)."""

from __future__ import annotations

import os
from pathlib import Path

from pact.dashboard import BIND_HOST, CONTROL_ENV

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


class ConfigPathRejected(ValueError):
    """Raised when a client-supplied config path is not on the allow-list."""


def assert_local_bind(host: str) -> str:
    """Refuse any bind address other than loopback IPv4."""

    if host != BIND_HOST:
        raise ValueError(
            f"dashboard must bind {BIND_HOST} only; refused {host!r}"
        )
    return host


def control_allowed() -> bool:
    return os.environ.get(CONTROL_ENV) == "1"


def list_allowed_configs() -> tuple[str, ...]:
    """Relative paths under ``configs/`` and ``configs/ablations/``."""

    found: list[Path] = []
    found.extend(sorted(CONFIGS_DIR.glob("*.yaml")))
    ablation_dir = CONFIGS_DIR / "ablations"
    if ablation_dir.is_dir():
        found.extend(sorted(ablation_dir.glob("*.yaml")))
    out: list[str] = []
    for path in found:
        out.append(path.resolve().relative_to(REPO_ROOT).as_posix())
    return tuple(out)


def resolve_allowed_config(raw: str) -> Path:
    """Return a config path only if it is on the scanned allow-list.

    Absolute paths, ``..`` escapes, and anything outside the repo are rejected.
    """

    if not raw or raw.startswith("/") or raw.startswith("\\"):
        raise ConfigPathRejected(f"config path not allowed: {raw!r}")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ConfigPathRejected(f"config path not allowed: {raw!r}")
    resolved = (REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ConfigPathRejected(f"config path not allowed: {raw!r}") from exc
    allowed = {(REPO_ROOT / rel).resolve() for rel in list_allowed_configs()}
    if resolved not in allowed:
        raise ConfigPathRejected(f"config path not allowed: {raw!r}")
    return resolved
