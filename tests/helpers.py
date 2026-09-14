from __future__ import annotations

from pact.config import (
    CapacityConfig,
    ControlConfig,
    DriftConfig,
    ForecastConfig,
    PactConfig,
    SimulatorConfig,
    TelemetryConfig,
)


def make_config(
    *,
    dt: float = 5.0,
    mu: float = 40.0,
    tau_c_s: float = 10.0,
    n_min: int = 1,
    n_max: int = 16,
    mem_baseline: float = 0.20,
    mem_load_coeff: float = 0.50,
    ca2: float = 1.0,
    cs2: float = 1.0,
) -> PactConfig:
    return PactConfig(
        telemetry=TelemetryConfig(dt=dt),
        forecast=ForecastConfig(),
        capacity=CapacityConfig(mu=mu, ca2=ca2, cs2=cs2),
        control=ControlConfig(tau_c_s=tau_c_s, n_min=n_min, n_max=n_max),
        drift=DriftConfig(),
        simulator=SimulatorConfig(
            mem_baseline=mem_baseline, mem_load_coeff=mem_load_coeff
        ),
    )
