"""Replay a finished results directory into the six live-panel payloads."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PANEL_FIELDS = (
    "tick",
    "timestamp",
    "observation",
    "forecast",
    "demand",
    "n_required",
    "applied_u",
    "n_desired",
    "drift",
    "safety",
    "gate",
    "mpc_u",
    "mpc_solve_time_s",
)

TICKS_JSONL = "ticks.jsonl"


def load_replay_ticks(results_dir: Path) -> list[dict[str, Any]]:
    """Prefer ``ticks.jsonl``; otherwise reconstruct from figure CSVs."""

    jsonl = results_dir / TICKS_JSONL
    if jsonl.is_file():
        return _load_jsonl(jsonl)
    return ticks_from_csvs(results_dir)


def write_ticks_jsonl(path: Path, ticks: Sequence[Mapping[str, Any]]) -> None:
    if not ticks:
        raise ValueError("refusing to write ticks.jsonl with no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in ticks:
            handle.write(json.dumps(dict(row)) + "\n")


def ticks_from_csvs(results_dir: Path) -> list[dict[str, Any]]:
    replicas = _read_csv(results_dir / "replicas.csv")
    latency = _read_csv(results_dir / "latency.csv")
    forecast = _read_csv(results_dir / "forecast.csv")
    actions = _read_csv(results_dir / "actions.csv")
    if not replicas:
        raise ValueError(f"no replica rows in {results_dir}")
    n = len(replicas)
    ticks: list[dict[str, Any]] = []
    for i in range(n):
        ts = float(replicas[i]["ts"])
        required = int(float(replicas[i]["required"]))
        n_pact = int(float(replicas[i]["pact"]))
        ell = float(latency[i]["pact"]) if i < len(latency) else 0.0
        actual_u = float(forecast[i]["actual"]) if i < len(forecast) else 0.0
        h1 = float(forecast[i]["pred_h1"]) if i < len(forecast) else actual_u
        h12 = float(forecast[i]["pred_h12"]) if i < len(forecast) else actual_u
        applied = int(float(actions[i]["pact"])) if i < len(actions) else 0
        ticks.append(
            {
                "tick": i,
                "timestamp": ts,
                "wall_time_s": ts,
                "observation": {
                    "u": actual_u,
                    "r": 0.0,
                    "lam": 0.0,
                    "ell_p95": ell,
                    "n": float(n_pact),
                    "t_s": ts,
                },
                "forecast": [[h1, 0.0], [h12, 0.0]],
                "demand": [required],
                "n_required": required,
                "mpc_u": None,
                "mpc_cost": None,
                "mpc_solve_time_s": None,
                "applied_u": applied,
                "gate": None,
                "safety": {"active": False, "u": 0},
                "drift": {
                    "tick": i,
                    "error": None,
                    "ebar": 0.0,
                    "gamma": 0.0,
                    "refit_requested": False,
                    "consecutive_high": 0,
                },
                "n_desired": n_pact,
                "deadline_slip_s": 0.0,
            }
        )
    return ticks


def replay_has_six_panels(ticks: Sequence[Mapping[str, Any]]) -> bool:
    if not ticks:
        return False
    first = ticks[0]
    return all(key in first for key in PANEL_FIELDS)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    if not rows:
        raise ValueError(f"{path} has no tick rows")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
