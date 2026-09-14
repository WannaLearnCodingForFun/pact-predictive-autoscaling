from __future__ import annotations

from pathlib import Path

import pytest
from pact.config import (
    ControlConfig,
    MissingColdStartError,
    cold_start_ticks,
    load_config,
)


def test_load_config_injects_measured_tau_c(tmp_path: Path) -> None:
    cold = tmp_path / "cold_start.json"
    cold.write_text('{"mean_s": 12.5, "std_s": 0.4}\n')
    cfg = load_config(Path("configs/default.yaml"), cold_start_path=cold)
    assert cfg.control.tau_c_s == 12.5
    assert cfg.telemetry.dt == 5.0
    assert cfg.telemetry.alpha == 0.35
    assert cfg.telemetry.window == 48
    assert cfg.forecast.horizon == 12
    assert cfg.forecast.kappa == 2.5
    assert cfg.capacity.u_target == 0.65
    assert cfg.capacity.mu == 40.0
    assert cfg.control.q_up == 8.0
    assert cfg.control.delta_max == 4
    assert cfg.control.u_emergency == 0.92
    assert cfg.drift.kappa_gamma == 0.02
    assert cfg.simulator.mem_baseline == 0.20


def test_load_config_refuses_missing_cold_start(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("telemetry:\n  dt: 5.0\n")
    with pytest.raises(MissingColdStartError):
        load_config(yaml_path, cold_start_path=tmp_path / "missing.json")


def test_load_config_rejects_tau_c_in_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("control:\n  tau_c_s: 41.7\n")
    cold = tmp_path / "cold_start.json"
    cold.write_text('{"mean_s": 10.0}\n')
    with pytest.raises(ValueError, match="tau_c_s must not be set in YAML"):
        load_config(yaml_path, cold_start_path=cold)


def test_control_config_requires_tau_c_s() -> None:
    with pytest.raises(TypeError):
        ControlConfig()  # type: ignore[call-arg]


def test_cold_start_ticks_is_ceil_tau_over_dt() -> None:
    assert cold_start_ticks(10.0, 5.0) == 2
    assert cold_start_ticks(0.0, 5.0) == 0
    assert cold_start_ticks(41.7, 5.0) == 9
