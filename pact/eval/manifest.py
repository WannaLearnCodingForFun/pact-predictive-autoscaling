"""U8 run manifest: git SHA, config hash, seeds, packages, host, τc."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from pact.config import load_cold_start_s

PACKAGE_NAMES: tuple[str, ...] = (
    "pact",
    "pyyaml",
    "numpy",
    "torch",
    "onnx",
    "onnxruntime",
    "statsmodels",
    "scipy",
    "pandas",
    "matplotlib",
)


def git_sha(repo: Path) -> str | None:
    override = os.environ.get("PACT_GIT_SHA", "").strip()
    if override:
        return override
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def sha256_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        digest.update(part)
        if index != len(parts) - 1:
            digest.update(b"\0")
    return digest.hexdigest()


def config_hash(config_path: Path, cold_start_path: Path) -> str:
    return sha256_bytes(config_path.read_bytes(), cold_start_path.read_bytes())


def package_versions(names: Sequence[str] = PACKAGE_NAMES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def host_specs() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def build_manifest(
    *,
    config_path: Path,
    cold_start_path: Path,
    seeds: Sequence[int],
    tau_c_s: float,
    repo: Path | None = None,
    methods: Sequence[str] | None = None,
    ticks: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = repo if repo is not None else Path.cwd()
    payload: dict[str, Any] = {
        "git_sha": git_sha(root),
        "config_hash": config_hash(config_path, cold_start_path),
        "config_path": str(config_path),
        "cold_start_path": str(cold_start_path),
        "seeds": [int(s) for s in seeds],
        "package_versions": package_versions(),
        "host": host_specs(),
        "tau_c_s": float(tau_c_s),
        "tau_c_s_from_file": load_cold_start_s(cold_start_path),
    }
    if methods is not None:
        payload["methods"] = list(methods)
    if ticks is not None:
        payload["ticks"] = int(ticks)
    if extra:
        payload.update(dict(extra))
    return payload


def write_run_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
