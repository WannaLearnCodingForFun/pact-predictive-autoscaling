# PACT

Predictive Autoscaling with Control-Theoretic enforcement, for Docker container pools.

Implementation of the system described in *Deterministic Predictive Auto-Scaling of Docker Containers Using Temporal Convolutional Resource Forecasting and Model Predictive Control.*

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
ruff check .
mypy pact tests scripts
pytest
```

Implementation follows `CURSOR_SPEC.md` phase by phase. This tree is a skeleton only.
# pact-predictive-autoscaling
