"""Capture a per-tick decision trace around a burst onset for the inference table."""
from __future__ import annotations
import csv, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.harness import RESULTS, base_config, workload, _relaxed_rounded_u
from pact.capacity.mapper import map_to_demand
from pact.config import cold_start_ticks
from pact.control.gates import apply_gates
from pact.control.mpc import apply_plant
from pact.control.mpc_exact import solve_mpc_exact as solve_mpc
from pact.control.safety import evaluate_safety
from pact.sim.queue_model import PoolSimulator
from pact.telemetry.collector import Observation

cfg = base_config()
lam = workload("burst", 400, 1001)
tau = cold_start_ticks(cfg.control.tau_c_s, cfg.telemetry.dt)
mu, u_star, H = cfg.capacity.mu, cfg.capacity.u_target, cfg.forecast.horizon

sim = PoolSimulator(cfg, n_initial=4)
n, pending, last_u, last_up = 4, [0] * tau, 0, -1e9
rows = []
for t in range(len(lam)):
    st = sim.step(float(lam[t]), n)
    now = t * cfg.telemetry.dt
    # oracle forecast of CPU/memory at the current ready count
    fc = []
    for h in range(1, H + 1):
        j = min(t + h, len(lam) - 1)
        u = float(lam[j]) / (mu * max(st.n_ready, 1))
        r = cfg.simulator.mem_baseline + cfg.simulator.mem_load_coeff * min(u, 1.0)
        fc.append([u, r])
    dem = map_to_demand(fc, max(st.n_ready, 1), 0.0, cfg)
    sol = solve_mpc(n, dem, pending, last_u, cfg)
    r_u = _relaxed_rounded_u(n, dem, cfg, last_u)
    g = apply_gates(float(sol.u), n, cfg.control, now_s=now, last_scale_up_s=last_up)
    safe = evaluate_safety(st.utilisation, n, cfg)
    u = safe.u if safe.active else g.u
    req = max(1, int(math.ceil(float(lam[t]) / (mu * u_star))))
    rows.append({
        "tick": t, "time_s": round(now, 1),
        "arrival_rate": round(float(lam[t]), 1),
        "observed_util": round(st.utilisation, 3),
        "observed_p95_ms": round(min(st.p95_latency_ms, 5000.0), 1),
        "n_ready": st.n_ready, "n_desired": n,
        "forecast_u_h1": round(fc[0][0], 3),
        "forecast_u_h12": round(fc[-1][0], 3),
        "demand_h1": dem[0], "demand_h12": dem[-1],
        "mpc_u": sol.u, "rounded_u": r_u,
        "after_gates": g.u, "cooldown_blocked": g.cooldown_blocked,
        "safety_active": safe.active, "applied_u": u,
        "ground_truth_required": req,
        "error_vs_required": n - req,
    })
    if u > 0:
        last_up = now
    n += u
    last_u = u
    pending = (pending + [u])[-max(tau, 1):]

onset = int(np.argmax(np.diff(lam[:200]) > 0) )
burst = int(np.argmax(lam[:250]))
lo = max(0, burst - 26)
sel = rows[lo:lo + 16]
with (RESULTS / "e14_inference_trace.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(sel)
print(f"  wrote e14_inference_trace.csv ({len(sel)} ticks, burst peak at tick {burst})")
