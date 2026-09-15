"""Frozen configuration dataclasses and YAML loader.

Field names match the paper's symbols so the rest of the code can be read
against the equations. ``tau_c_s`` is never taken from YAML: it must come from
the measurement file written by ``scripts/measure_cold_start.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_COLD_START_PATH = Path("results/cold_start.json")


class MissingColdStartError(FileNotFoundError):
    """Raised when the measured cold-start file is absent.

    The control loop must refuse to start without a measured τc.
    """


@dataclass(frozen=True, kw_only=True)
class TelemetryConfig:
    dt: float = 5.0  # Δt, control interval (s)
    alpha: float = 0.35  # α, EWMA smoothing
    window: int = 48  # L, input window length


@dataclass(frozen=True, kw_only=True)
class ForecastConfig:
    horizon: int = 12  # H
    kernel_size: int = 3  # K
    depth: int = 4  # D
    channels: int = 64  # C
    dropout: float = 0.10
    lr: float = 3e-3
    batch_size: int = 128
    kappa: float = 2.5  # κ, under-prediction penalty
    beta: float = 0.15  # β, horizon decay
    huber_delta: float = 1.0  # δ


@dataclass(frozen=True, kw_only=True)
class CapacityConfig:
    u_target: float = 0.65  # u*
    r_target: float = 0.75  # r*
    mu: float = 40.0  # μ, per-replica service rate (req/s)
    slo_ms: float = 200.0  # ℓ_max
    ca2: float = 1.0  # c_a², arrival CV²
    cs2: float = 1.0  # c_s², service CV²


@dataclass(frozen=True, kw_only=True)
class ControlConfig:
    q_up: float = 8.0  # q⁺ (under-provision penalty)
    q_down: float = 1.0  # q⁻
    r_act: float = 0.15  # r, actuation weight
    s_churn: float = 0.60  # s, churn weight
    n_min: int = 1
    n_max: int = 16
    delta_max: int = 4  # Δ_max
    deadband: int = 1  # θ
    cooldown_s: float = 60.0  # T_cool
    u_emergency: float = 0.92  # utilisation that bypasses MPC and gates
    # Required: measured τc. Never defaulted — a guessed τc collapses the claim.
    tau_c_s: float


@dataclass(frozen=True, kw_only=True)
class DriftConfig:
    eta: float = 0.2  # η, error smoothing
    kappa_gamma: float = 0.02  # κ_γ, margin gain
    gamma_min: float = 0.0
    gamma_max: float = 0.50
    xi: float = 1.0  # ξ, drift threshold
    drift_window: int = 60  # W


@dataclass(frozen=True, kw_only=True)
class AblationConfig:
    """Flags that disable exactly one component. Defaults are the full system.

    ``tau_c_s`` is still never a YAML measurement. ``zero_tau_c`` overrides the
    loaded measurement to 0 after ``cold_start.json`` is read.
    """

    skip_mpc: bool = False
    freeze_gamma: bool = False
    include_rho: bool = True
    zero_tau_c: bool = False


@dataclass(frozen=True, kw_only=True)
class SimulatorConfig:
    """Analytical pool-simulator parameters (not paper control symbols)."""

    mem_baseline: float = 0.20  # idle per-replica memory fraction
    mem_load_coeff: float = 0.50  # additional memory at full utilisation


@dataclass(frozen=True, kw_only=True)
class PrometheusConfig:
    """cAdvisor / Prometheus scrape settings for the live collector."""

    base_url: str = "http://localhost:9090"
    timeout_s: float = 5.0
    cpu_counter_query: str = (
        'sum(container_cpu_usage_seconds_total{name=~"service.*"})'
    )
    cpu_counter_in_nanoseconds: bool = False
    memory_working_set_query: str = (
        'sum(container_memory_working_set_bytes{name=~"service.*"})'
    )
    memory_limit_query: str = (
        'sum(container_spec_memory_limit_bytes{name=~"service.*"})'
    )
    requests_total_query: str = "sum(nginx_http_requests_total)"
    latency_p95_query: str = (
        "histogram_quantile(0.95, "
        "sum by (le) (http_request_duration_seconds_bucket))"
    )
    replica_count_query: str = (
        'count(container_memory_working_set_bytes{name=~"service.*"})'
    )
    network_rx_bytes_query: str = (
        'sum(container_network_receive_bytes_total{name=~"service.*"})'
    )
    network_tx_bytes_query: str = (
        'sum(container_network_transmit_bytes_total{name=~"service.*"})'
    )
    fs_read_bytes_query: str = (
        'sum(container_fs_reads_bytes_total{name=~"service.*"})'
    )
    fs_write_bytes_query: str = (
        'sum(container_fs_writes_bytes_total{name=~"service.*"})'
    )


@dataclass(frozen=True, kw_only=True)
class PactConfig:
    telemetry: TelemetryConfig
    forecast: ForecastConfig
    capacity: CapacityConfig
    control: ControlConfig
    drift: DriftConfig
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


def cold_start_ticks(tau_c_s: float, dt: float) -> int:
    """Return τ = ceil(τc / Δt), the cold-start delay in control ticks."""

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if tau_c_s < 0.0:
        raise ValueError(f"tau_c_s must be non-negative, got {tau_c_s}")
    return math.ceil(tau_c_s / dt)


def load_cold_start_s(path: Path = DEFAULT_COLD_START_PATH) -> float:
    """Load measured mean cold-start time (seconds) from ``cold_start.json``."""

    if not path.is_file():
        raise MissingColdStartError(
            f"Measured cold-start file {path} is absent; the loop refuses to "
            "start without a measured τc"
        )
    raw: Any = json.loads(path.read_text())
    if not isinstance(raw, dict) or "mean_s" not in raw:
        raise ValueError(
            f"{path} must contain a 'mean_s' field written by "
            "scripts/measure_cold_start.py"
        )
    mean_s = float(raw["mean_s"])
    if mean_s < 0.0:
        raise ValueError(f"mean_s must be non-negative, got {mean_s}")
    return mean_s


def load_config(
    yaml_path: Path = DEFAULT_CONFIG_PATH,
    *,
    cold_start_path: Path = DEFAULT_COLD_START_PATH,
    tau_c_s: float | None = None,
) -> PactConfig:
    """Load a ``PactConfig`` from YAML, injecting measured ``tau_c_s``.

    Pass ``tau_c_s`` only in tests. Production loads it from ``cold_start_path``.
    """

    raw = _read_yaml_mapping(yaml_path)
    return config_from_mapping(
        raw, cold_start_path=cold_start_path, tau_c_s=tau_c_s
    )


def load_config_overlay(
    base_path: Path,
    overlay_path: Path,
    *,
    cold_start_path: Path = DEFAULT_COLD_START_PATH,
    tau_c_s: float | None = None,
) -> PactConfig:
    """Load ``base_path`` then apply ``overlay_path`` (ablation YAMLs)."""

    merged = deep_merge(
        _read_yaml_mapping(base_path), _read_yaml_mapping(overlay_path)
    )
    return config_from_mapping(
        merged, cold_start_path=cold_start_path, tau_c_s=tau_c_s
    )


def config_from_mapping(
    raw: Mapping[str, Any],
    *,
    cold_start_path: Path = DEFAULT_COLD_START_PATH,
    tau_c_s: float | None = None,
) -> PactConfig:
    """Build a ``PactConfig`` from an already-merged mapping."""

    if not isinstance(raw, Mapping):
        raise TypeError(f"Config root must be a mapping, got {type(raw).__name__}")

    control_raw = dict(_section(raw, "control"))
    if "tau_c_s" in control_raw:
        raise ValueError(
            "tau_c_s must not be set in YAML; load it from "
            "results/cold_start.json (written by scripts/measure_cold_start.py)"
        )
    if tau_c_s is None:
        tau_c_s = load_cold_start_s(cold_start_path)

    control = _from_mapping(ControlConfig, control_raw, extra={"tau_c_s": tau_c_s})
    ablation = _from_mapping(AblationConfig, _section(raw, "ablation"))
    if ablation.zero_tau_c:
        control = replace(control, tau_c_s=0.0)
    return PactConfig(
        telemetry=_from_mapping(TelemetryConfig, _section(raw, "telemetry")),
        forecast=_from_mapping(ForecastConfig, _section(raw, "forecast")),
        capacity=_from_mapping(CapacityConfig, _section(raw, "capacity")),
        control=control,
        drift=_from_mapping(DriftConfig, _section(raw, "drift")),
        simulator=_from_mapping(SimulatorConfig, _section(raw, "simulator")),
        prometheus=_from_mapping(PrometheusConfig, _section(raw, "prometheus")),
        ablation=ablation,
    )


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings. Overlay values replace base values."""

    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[str(key)] = deep_merge(current, value)
        else:
            out[str(key)] = value
    return out


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded: Any = yaml.safe_load(path.read_text())
    raw: dict[str, Any] = {} if loaded is None else loaded
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw).__name__}")
    return raw


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section {name!r} must be a mapping")
    return value


def _from_mapping(
    cls: type[T],
    data: Mapping[str, Any],
    *,
    extra: Mapping[str, Any] | None = None,
) -> T:
    allowed = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(allowed)
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {cls.__name__} field(s): {unknown_list}")
    kwargs: dict[str, Any] = {}
    for name, fld in allowed.items():
        if name not in data:
            continue
        kwargs[name] = _coerce(fld.type, data[name])
    if extra:
        kwargs.update(extra)
    return cls(**kwargs)


def _coerce(field_type: Any, value: Any) -> Any:
    origin = field_type
    if isinstance(origin, str):
        origin = {"float": float, "int": int, "str": str, "bool": bool}.get(
            origin, origin
        )
    if origin is float:
        return float(value)
    if origin is int:
        return int(value)
    if origin is bool:
        return bool(value)
    return value
