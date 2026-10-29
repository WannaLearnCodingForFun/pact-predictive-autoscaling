"""U8 manifest: config hash, τc, seeds, host specs, git SHA override."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pact.config import MissingColdStartError, load_cold_start_s
from pact.eval.manifest import build_manifest, config_hash, git_sha, write_run_manifest
from pact.eval.reproduce import min_series_length, require_cold_start

from tests.helpers import make_config


def _write_cold_start(path: Path, mean_s: float) -> None:
    path.write_text(
        json.dumps({"mean_s": mean_s, "std_s": 0.0, "n": 1, "samples_s": [mean_s]})
    )

def _delete_cold_start(path: Path) -> None:
    if path.exists():
        path.unlink():


def test_config_hash_is_sha256_of_config_and_cold_start(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cold = tmp_path / "cold_start.json"
    cfg.write_text("telemetry:\n  dt: 5.0\n")
    _write_cold_start(cold, 12.5)
    first = config_hash(cfg, cold)
    second = config_hash(cfg, cold)
    assert first == second
    assert len(first) == 64
    cold.write_text(
        json.dumps({"mean_s": 12.6, "std_s": 0.0, "n": 1, "samples_s": [12.6]})
    )
    assert config_hash(cfg, cold) != first


def test_manifest_records_sha_seeds_versions_host_and_tau(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cold = tmp_path / "cold_start.json"
    cfg.write_bytes(Path("configs/default.yaml").read_bytes())
    _write_cold_start(cold, 9.25)
    monkeypatch.setenv("PACT_GIT_SHA", "abc123def")
    payload = build_manifest(
        config_path=cfg,
        cold_start_path=cold,
        seeds=(0, 1, 2),
        tau_c_s=9.25,
        methods=("pact", "reactive"),
        ticks=40,
        repo=tmp_path,
    )
    assert payload["git_sha"] == "abc123def"
    assert payload["seeds"] == [0, 1, 2]
    assert payload["tau_c_s"] == pytest.approx(9.25)
    assert payload["tau_c_s_from_file"] == pytest.approx(load_cold_start_s(cold))
    assert payload["config_hash"] == config_hash(cfg, cold)
    assert "numpy" in payload["package_versions"]
    host = payload["host"]
    assert host["system"]
    assert host["python"]
    dest = tmp_path / "results" / "manifest.json"
    write_run_manifest(dest, payload)
    loaded = json.loads(dest.read_text())
    assert loaded["git_sha"] == "abc123def"
    assert loaded["ticks"] == 40


def test_git_sha_env_overrides_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PACT_GIT_SHA", "from-env")
    assert git_sha(tmp_path) == "from-env"


def test_require_cold_start_refuses_missing(tmp_path: Path) -> None:
    with pytest.raises(MissingColdStartError):
        require_cold_start(tmp_path / "missing.json")


def test_min_series_length_covers_val_split() -> None:
    # L=48, H=12 → need 60 samples in the 15% val split → ceil(60 / 0.15) = 400
    cfg = make_config()
    assert min_series_length(cfg) == 400
