"""Launch, track, and cancel simulator experiment runs (Phase 11.4)."""

from __future__ import annotations

import csv
import logging
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pact.config import DEFAULT_COLD_START_PATH, load_config
from pact.dashboard.bus import BusObserver, TickBus, encode_tick
from pact.dashboard.docker_ctl import DockerCtl, DockerUnavailable
from pact.dashboard.replay import TICKS_JSONL, write_ticks_jsonl
from pact.dashboard.security import (
    REPO_ROOT,
    resolve_allowed_config,
    resolve_allowed_trace,
)
from pact.eval.datasets import ArrivalTrace, generate_burst
from pact.eval.export import export_run
from pact.eval.runner import (
    CONTROLLER_METHODS,
    FORECAST_METHODS,
    MethodTrace,
    metrics_from_trace,
    run_matrix,
)
from pact.loop import TickObserver, TickRecord

logger = logging.getLogger(__name__)

ALLOWED_METHODS = CONTROLLER_METHODS
ALLOWED_TARGETS = ("simulator", "docker")
DEFAULT_UI_SEEDS = 5
MAX_SEEDS = 20
MAX_TICKS = 400
MIN_TICKS = 8


@dataclass
class RunSpec:
    config: str
    trace: str
    methods: tuple[str, ...]
    n_seeds: int
    n_ticks: int
    target: str


@dataclass
class RunRecord:
    id: str
    status: str
    spec: RunSpec
    results_dir: str
    log: str = ""
    error: str | None = None
    metrics: list[dict[str, object]] = field(default_factory=list)
    started_s: float = 0.0
    finished_s: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "spec": asdict(self.spec),
            "results_dir": self.results_dir,
            "log": self.log,
            "error": self.error,
            "metrics": list(self.metrics),
            "started_s": self.started_s,
            "finished_s": self.finished_s,
        }


class JsonlTickWriter:
    """TickObserver that appends encoded ticks for replay."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def on_tick(self, record: TickRecord) -> None:
        self.rows.append(encode_tick(record))


class ExperimentManager:
    def __init__(
        self,
        *,
        bus: TickBus,
        results_root: Path,
        cold_start_path: Path | None = None,
        docker: DockerCtl | None = None,
        sync: bool = False,
    ) -> None:
        self._bus = bus
        self._results_root = results_root
        self._cold_start = cold_start_path or DEFAULT_COLD_START_PATH
        self._docker = docker
        self._sync = sync
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}
        self._cancel = threading.Event()

    def list_runs(self) -> list[dict[str, object]]:
        with self._lock:
            return [run.as_dict() for run in self._runs.values()]

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def start(self, body: dict[str, Any]) -> RunRecord:
        spec = parse_run_spec(body)
        run_id = uuid.uuid4().hex[:12]
        dest = self._results_root / "dashboard" / run_id
        dest.mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            id=run_id,
            status="queued",
            spec=spec,
            results_dir=str(dest.relative_to(self._results_root)),
            started_s=time.time(),
        )
        with self._lock:
            self._runs[run_id] = record
        if self._sync:
            self._execute(record, dest)
            return record
        thread = threading.Thread(
            target=self._execute, args=(record, dest), daemon=True
        )
        thread.start()
        return record

    def cancel(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise KeyError(run_id)
        self._cancel.set()
        with self._lock:
            if record.status in {"queued", "running"}:
                record.status = "cancelled"
                record.finished_s = time.time()
        return record

    def controller_running(self) -> bool:
        with self._lock:
            return any(r.status == "running" for r in self._runs.values())

    def _execute(self, record: RunRecord, dest: Path) -> None:
        record.status = "running"
        try:
            if record.spec.target == "docker":
                if self._docker is None:
                    raise DockerUnavailable("docker control is not configured")
                self._docker.up()
                record.log += "docker compose up (hardcoded project pact)\n"
            config_path = resolve_allowed_config(record.spec.config)
            config = load_config(config_path, cold_start_path=self._cold_start)
            trace = load_workload(
                record.spec.trace,
                n_ticks=record.spec.n_ticks,
                dt=config.telemetry.dt,
            )
            writer = JsonlTickWriter()
            observers: list[TickObserver] = [BusObserver(self._bus), writer]
            record.log += (
                f"run_matrix methods={record.spec.methods} "
                f"seeds={record.spec.n_seeds} ticks={record.spec.n_ticks}\n"
            )
            traces = run_matrix(
                config,
                trace,
                methods=record.spec.methods,
                seeds=tuple(range(record.spec.n_seeds)),
                evaluate_on="all",
                observers=observers,
            )
            try:
                export_run(dest, traces, config)
            except ValueError as exc:
                record.log += f"partial export ({exc})\n"
                from pact.eval.export import write_forecast_csv, write_table_csv

                metrics = [metrics_from_trace(item, config) for item in traces]
                if any(item.method == "pact" for item in traces):
                    write_forecast_csv(dest / "forecast.csv", traces)
                write_table_csv(
                    dest / "table6_baselines.csv",
                    metrics,
                    metric_name="sla_violation",
                )
            if writer.rows:
                write_ticks_jsonl(dest / TICKS_JSONL, writer.rows)
            record.metrics = _headline_metrics(traces, config)
            record.log += f"wrote {dest}\n"
            record.status = "completed"
        except Exception as exc:
            logger.exception("experiment %s failed", record.id)
            record.status = "failed"
            record.error = str(exc)
            record.log += f"error: {exc}\n"
        finally:
            record.finished_s = time.time()


def parse_run_spec(body: Mapping[str, Any]) -> RunSpec:
    config = str(body.get("config") or "configs/default.yaml")
    resolve_allowed_config(config)
    trace = str(body.get("trace") or "synthetic:step")
    resolve_allowed_trace(trace)
    raw_methods = body.get("methods", list(FORECAST_METHODS))
    if isinstance(raw_methods, str):
        methods = tuple(m.strip() for m in raw_methods.split(",") if m.strip())
    elif isinstance(raw_methods, Sequence):
        methods = tuple(str(m) for m in raw_methods)
    else:
        raise ValueError("methods must be a list")
    unknown = [m for m in methods if m not in ALLOWED_METHODS]
    if unknown:
        raise ValueError(f"methods not allowed: {unknown}")
    if not methods:
        raise ValueError("at least one method is required")
    target = str(body.get("target") or "simulator")
    if target not in ALLOWED_TARGETS:
        raise ValueError(f"target not allowed: {target!r}")
    n_seeds = int(body.get("seeds", DEFAULT_UI_SEEDS))
    n_ticks = int(body.get("ticks", 40))
    if n_seeds < 1 or n_seeds > MAX_SEEDS:
        raise ValueError(f"seeds must be 1..{MAX_SEEDS}")
    if n_ticks < MIN_TICKS or n_ticks > MAX_TICKS:
        raise ValueError(f"ticks must be {MIN_TICKS}..{MAX_TICKS}")
    return RunSpec(
        config=config,
        trace=trace,
        methods=methods,
        n_seeds=n_seeds,
        n_ticks=n_ticks,
        target=target,
    )


def load_workload(name: str, *, n_ticks: int, dt: float) -> ArrivalTrace:
    key = resolve_allowed_trace(name)
    if key.startswith("synthetic:"):
        kind = key.split(":", 1)[1]
        return generate_burst(
            kind,
            n_ticks=n_ticks,
            dt=dt,
            rise_ticks=4,
            base=20.0,
            peak=60.0,
            name="d3_step_cli" if kind == "step" else f"d3_{kind}",
        )
    path = REPO_ROOT / key
    return _load_processed_csv(path, dt=dt, name=key)


def _load_processed_csv(path: Path, *, dt: float, name: str) -> ArrivalTrace:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        orig = list(reader.fieldnames or [])
        fieldnames = [f.lower() for f in orig]
        rows = list(reader)
    if not rows or len(orig) < 2:
        raise ValueError(f"processed trace {path} is empty")
    t_key = _pick(fieldnames, ("timestamp", "ts", "t", "time"))
    r_key = _pick(fieldnames, ("arrival_rate", "rate", "lambda", "lam"))
    t_col = orig[fieldnames.index(t_key)] if t_key else orig[0]
    r_col = orig[fieldnames.index(r_key)] if r_key else orig[1]
    stamps = [float(row[t_col]) for row in rows]
    rates = [float(row[r_col]) for row in rows]
    return ArrivalTrace(
        name=name,
        timestamps=tuple(stamps),
        arrival_rates=tuple(rates),
        dt=dt,
        interpolated=False,
        notes=f"processed trace {name}",
    )


def _pick(fields: Sequence[str], names: Sequence[str]) -> str | None:
    for name in names:
        if name in fields:
            return name
    return None


def _headline_metrics(
    traces: Sequence[MethodTrace], config: Any
) -> list[dict[str, object]]:
    grouped: dict[str, list[Any]] = {}
    for trace in traces:
        grouped.setdefault(trace.method, []).append(metrics_from_trace(trace, config))
    out: list[dict[str, object]] = []
    for method, rows in grouped.items():
        sla = sum(r.sla_violation for r in rows) / len(rows)
        over = sum(r.over_provision for r in rows) / len(rows)
        out.append(
            {
                "method": method,
                "sla_violation": sla,
                "over_provision": over,
                "n_runs": len(rows),
            }
        )
    return out
