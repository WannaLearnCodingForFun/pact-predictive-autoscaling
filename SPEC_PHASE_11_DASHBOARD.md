# Phase 11 — Live Dashboard and Docker Control Plane

A local web UI at `http://127.0.0.1:8000` that watches the control loop in real
time, launches experiments, and drives the Docker testbed. Companion to
`CURSOR_SPEC.md`; everything there still applies.

This is an **operator and debugging tool**, not a paper artifact. Its job is to
make it obvious when the controller is misbehaving — oscillation, a stuck
margin, safety overrides firing constantly — which is very hard to see in a CSV
and immediately visible on a time-series plot.

---

## 11.0 Security constraints — non-negotiable

The dashboard talks to the Docker daemon. Docker socket access is equivalent to
root on the host. Therefore:

- Bind to `127.0.0.1` only. **Never** `0.0.0.0`.
- No shell passthrough. The Docker layer exposes a fixed set of operations
  (`up`, `down`, `scale`, `ps`, `logs`) against a **hardcoded compose project
  name**. There is no endpoint that accepts an arbitrary command, image name, or
  container ID from the client.
- Config selection is from an allow-list enumerated by scanning `configs/`, not
  from a client-supplied path. Reject anything resolving outside the repo.
- Read-only by default. Mutating endpoints (`/api/experiment/start`,
  `/api/docker/*`) require `PACT_DASHBOARD_ALLOW_CONTROL=1` in the environment.
- Log every mutating call with timestamp and parameters.

---

## 11.1 Stack

- **Backend**: FastAPI + uvicorn. WebSocket for live telemetry, REST for commands.
- **Docker**: `docker` Python SDK, scoped to the compose project.
- **Frontend**: single-page app, vanilla or lightweight (Preact/htm via CDN is
  fine — no build step). Charts via uPlot (fast, small, built for streaming
  time series) or Chart.js. **Do not pull in a React/Vite toolchain**; this is a
  local tool and a build step is friction you will regret.
- New optional extra in `pyproject.toml`: `dashboard = ["fastapi", "uvicorn[standard]", "docker", "websockets"]`

```
pact/dashboard/
  __init__.py
  app.py              # FastAPI app, routes, WS endpoint
  bus.py              # in-process pub/sub between loop and WS clients
  docker_ctl.py       # constrained Docker operations
  experiments.py      # launch/track/cancel experiment runs
  state.py            # ring buffers of recent ticks, run registry
  static/
    index.html
    app.js
    style.css
```

---

## 11.2 Wiring the loop to the dashboard

`pact/loop.py` already logs every tick. Do **not** make the loop depend on the
dashboard. Instead:

- Define a `TickObserver` protocol with one method, `on_tick(record: TickRecord)`.
- `loop.py` accepts an optional list of observers and calls them after each tick,
  inside a try/except that swallows observer errors. **A dashboard failure must
  never take down the controller.**
- The dashboard registers an observer that pushes onto `bus.py`.
- When the loop runs standalone with no dashboard, the observer list is empty
  and nothing changes.

`TickRecord` is a frozen dataclass with exactly the fields the spec's section 0
already requires in the log line: tick index, wall time, observed metrics,
forecast vector, demand trajectory, MPC solution, gate decisions (each stage),
applied action, current γ, smoothed error, safety-override flag, solve time.

Serialise with `dataclasses.asdict` + `orjson`.

---

## 11.3 Panels

Six panels. Each reads from the same WebSocket stream.

**1. Replicas vs demand** — stepped line. Three series: ground-truth required,
predicted demand `Ñ(t+1)`, actual `N(t)`. Shade the cold-start window after
each scale-up so the dead time is visible. This is the panel that shows whether
the controller leads or lags, which is the entire thesis.

**2. Latency** — p95 over time with the SLO as a horizontal rule. Shade regions
above the line red. Violation count as a running tile.

**3. Forecast overlay** — actual CPU utilisation with the `h=1` and `h=H`
predictions drawn forward from the current tick, so you can see the horizon
fan out ahead of the data and watch it be right or wrong.

**4. Control internals** — γ (safety margin), smoothed tracking error `ē(t)`,
and MPC solve time. A γ pinned at `gamma_max` for minutes means the forecaster
has drifted and the integral has saturated; that is invisible in any other view.

**5. Action log** — scrolling table, one row per tick where `u ≠ 0` or a gate
fired. Columns: tick, `u*` raw, after rate clip, after bounds clip, after
deadband, cooldown blocked, applied, reason. **Safety overrides highlighted.**
This panel is how you debug the gate ordering.

**6. Metric tiles** — SLA violation %, over-provisioning %, scaling actions,
mean utilisation, override count. Live, recomputed per tick.

### Visual direction

Neutral, data-first. One accent hue for PACT, one muted grey for ground truth,
one warm hue for violations/overrides. Avoid rainbow categorical palettes —
there are only ever three or four series per panel. Dark background suits a
long-running monitor. Grid lines at 10–15% opacity. No gradients, no shadows,
no animation on data updates beyond the scroll itself (animated transitions on
streaming data actively hide the oscillation you are trying to see).

---

## 11.4 Experiment launcher

Left rail panel:

- Dropdown of configs discovered by scanning `configs/*.yaml` and
  `configs/ablations/*.yaml`
- Dropdown of workload traces found under `data/processed/`
- Method selector (PACT, reactive, ARIMA, LSTM, static) with multi-select
- Seed count, defaulting to 5
- Target selector: **simulator** or **docker testbed**
- Start / Cancel

On start, spawn `pact.eval.runner` in a subprocess with the chosen arguments,
stream its stdout to a log pane, and register the run. On completion, parse the
written CSVs and add the run to the comparison view.

**Run comparison view**: table of completed runs with their headline metrics,
plus a scatter of SLA violation vs over-provisioning — the same trade-off
frontier as Fig. 7, but live and clickable. Clicking a point loads that run's
traces into the panels in replay mode.

---

## 11.5 Docker control

`docker_ctl.py`, constrained as described in 11.0:

- `GET /api/docker/status` — compose project state: service names, desired vs
  running replica count, health, uptime
- `POST /api/docker/up` / `POST /api/docker/down` — bring the testbed up or down
- `POST /api/docker/scale` — manual scale, for sanity-checking actuation by hand
- `GET /api/docker/logs?service=&tail=` — tail logs, bounded at 500 lines
- `POST /api/docker/measure_cold_start` — runs `scripts/measure_cold_start.py`,
  streams progress, writes `results/cold_start.json`

That last one is worth building first. It is the measurement the entire
cold-start argument rests on, and having it one click away means you will
actually re-measure it when the image changes rather than trusting a stale number.

Status strip at the top of the page: daemon reachable, project up/down, replica
count, controller running/stopped, τc currently loaded.

---

## 11.6 Replay mode

The dashboard must work without a live loop. Given a results directory, replay
the tick log at adjustable speed (1×, 10×, 100×, or scrub). You will spend far
more time looking at finished runs than watching live ones, and a dashboard that
only works live is a dashboard you use twice.

---

## 11.7 Acceptance criteria

- `python -m pact.dashboard` serves on `127.0.0.1:8000` and refuses to bind
  elsewhere; a test asserts the bind address.
- A test asserts that mutating endpoints return 403 when
  `PACT_DASHBOARD_ALLOW_CONTROL` is unset.
- A test asserts a config path outside the repo is rejected.
- A test asserts an observer raising an exception does not interrupt the loop.
- With the simulator target, starting a run from the UI produces live updates in
  all six panels and the same CSVs as the CLI path.
- Replay mode reconstructs identical panels from a finished run directory.
