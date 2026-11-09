"""Validate solve_mpc_exact against exhaustive enumeration, then quantify the
greedy-completion gap on the configuration used in the paper.

Part 1 brute-forces the full integer program on configurations small enough
to enumerate ((2*Delta_max+1)^H plans) and checks that solve_mpc_exact
attains the same optimal cost and an optimal first action. This is the test
of the two structural claims the exact solver rests on: dead-time truncation
and tail separability.

Part 2 runs the paper configuration and reports how often the greedy
completion returns a different first action, and by how much its objective
is off.

All three solvers are scored with the same reference objective,
``pact.control.mpc_exact.plan_cost``, which replicates the repository cost
term by term and adds the landing-admissibility test.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pact.config import (  # noqa: E402
    CapacityConfig,
    ControlConfig,
    DriftConfig,
    ForecastConfig,
    PactConfig,
    SimulatorConfig,
    TelemetryConfig,
    cold_start_ticks,
)
from pact.control.mpc import (  # noqa: E402
    _pending_of_length,
    _pipe_len,
    solve_mpc,
)
from pact.control.mpc_exact import (  # noqa: E402
    landing_lag,
    plan_cost,
    solve_mpc_exact,
    tracking_relevant_actions,
)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TOL = 1e-6


def make_config(*, dt, horizon, tau_c_s, delta_max, n_min, n_max,
                r=0.15, s=0.60, q_up=8.0, q_down=1.0):
    return PactConfig(
        telemetry=TelemetryConfig(dt=dt),
        forecast=ForecastConfig(horizon=horizon),
        capacity=CapacityConfig(),
        control=ControlConfig(
            tau_c_s=tau_c_s, delta_max=delta_max, n_min=n_min, n_max=n_max,
            r_act=r, s_churn=s, q_up=q_up, q_down=q_down,
        ),
        drift=DriftConfig(),
        simulator=SimulatorConfig(),
    )


def brute_force(n_current, demand, pending, last_u, config):
    """Exhaustive minimum of the reference objective over the integer box."""

    ctrl = config.control
    tau = cold_start_ticks(ctrl.tau_c_s, config.telemetry.dt)
    horizon = min(len(demand), config.forecast.horizon)
    pipe = _pending_of_length(pending, _pipe_len(tau))
    demand_h = [int(x) for x in demand[:horizon]]
    us = list(range(-ctrl.delta_max, ctrl.delta_max + 1))

    best_cost = float("inf")
    best_us: set[int] = set()
    for plan in itertools.product(us, repeat=horizon):
        cost = plan_cost(n_current, pipe, plan, demand_h, last_u, tau, ctrl)
        if cost is None:
            continue
        if cost < best_cost - TOL:
            best_cost, best_us = cost, {plan[0]}
        elif abs(cost - best_cost) <= TOL:
            best_us.add(plan[0])
    return best_cost, best_us


def part1_validate():
    rng = random.Random(20261030)
    cases = [
        # dt, H, tau_c_s, delta_max, n_min, n_max
        (5.0, 4, 10.0, 2, 1, 8),
        (5.0, 5, 10.0, 2, 1, 8),
        (5.0, 4, 20.0, 2, 1, 8),
        (5.0, 6, 15.0, 2, 1, 10),
        (5.0, 5, 5.0, 2, 1, 8),
        (5.0, 4, 0.0, 2, 1, 8),
        (5.0, 6, 30.0, 1, 1, 10),
        (5.0, 7, 25.0, 1, 1, 12),
        (5.0, 6, 45.0, 2, 1, 10),
        (5.0, 5, 15.0, 2, 2, 6),
    ]
    rows, failures = [], []
    for dt, H, tau_c, dmax, nmin, nmax in cases:
        cfg = make_config(dt=dt, horizon=H, tau_c_s=tau_c, delta_max=dmax,
                          n_min=nmin, n_max=nmax)
        ctrl = cfg.control
        tau = cold_start_ticks(tau_c, dt)
        pipe_len = _pipe_len(tau)
        m = tracking_relevant_actions(H, tau)
        for trial in range(40):
            n_cur = rng.randint(nmin, nmax)
            pipe = [rng.randint(-1, 1) for _ in range(pipe_len)]
            if not (nmin <= n_cur + sum(pipe) <= nmax):
                pipe = [0] * pipe_len
            demand = [rng.randint(nmin, nmax) for _ in range(H)]
            last_u = rng.randint(-dmax, dmax)

            bf_cost, bf_us = brute_force(n_cur, demand, pipe, last_u, cfg)
            if bf_cost == float("inf"):
                continue
            ex = solve_mpc_exact(n_cur, demand, pipe, last_u, cfg)
            gr = solve_mpc(n_cur, demand, pipe, last_u, cfg)
            gr_true = plan_cost(n_cur, pipe, gr.plan, demand, last_u, tau, ctrl)

            cost_ok = abs(ex.cost - bf_cost) <= TOL
            u_ok = ex.u in bf_us
            if not (cost_ok and u_ok):
                failures.append(dict(
                    case=(dt, H, tau_c, dmax, nmin, nmax), trial=trial,
                    n=n_cur, pipe=pipe, demand=demand, last_u=last_u,
                    bf_cost=bf_cost, bf_us=sorted(bf_us),
                    exact_cost=ex.cost, exact_u=ex.u,
                    cost_ok=cost_ok, u_ok=u_ok,
                ))
            rows.append(dict(
                H=H, tau=tau, delta_max=dmax, m=m,
                bf_cost=bf_cost, exact_cost=ex.cost,
                greedy_cost_true=gr_true,
                exact_ok=cost_ok and u_ok,
                greedy_u_optimal=gr.u in bf_us,
                greedy_excess=(None if gr_true is None else gr_true - bf_cost),
            ))
    return rows, failures


def part2_gap():
    """Paper configuration: how often does greedy completion pick differently?"""

    import harness as H  # noqa: N812

    cfg = H.base_config()
    ctrl = cfg.control
    tau = cold_start_ticks(ctrl.tau_c_s, cfg.telemetry.dt)
    horizon = cfg.forecast.horizon
    m = tracking_relevant_actions(horizon, tau)
    lag = landing_lag(tau)

    rng = random.Random(7)
    pipe_len = _pipe_len(tau)
    rows = []
    for _trial in range(1200):
        n_cur = rng.randint(ctrl.n_min, ctrl.n_max)
        pipe = [0] * pipe_len
        for i in range(pipe_len):
            if rng.random() < 0.35:
                pipe[i] = rng.randint(-2, 2)
        if not (ctrl.n_min <= n_cur + sum(pipe) <= ctrl.n_max):
            pipe = [0] * pipe_len
        base = rng.randint(ctrl.n_min, ctrl.n_max)
        slope = rng.choice([-2, -1, 0, 0, 1, 2])
        demand = []
        for k in range(horizon):
            v = base + slope * k + rng.randint(-1, 1)
            demand.append(max(ctrl.n_min, min(ctrl.n_max, v)))
        last_u = rng.randint(-ctrl.delta_max, ctrl.delta_max)

        ex = solve_mpc_exact(n_cur, demand, pipe, last_u, cfg)
        gr = solve_mpc(n_cur, demand, pipe, last_u, cfg)
        gr_true = plan_cost(n_cur, pipe, gr.plan, demand, last_u, tau, ctrl)
        rows.append(dict(
            exact_u=ex.u, greedy_u=gr.u,
            exact_cost=ex.cost,
            greedy_cost_true=gr_true,
            greedy_cost_reported=gr.cost,
            agree=ex.u == gr.u,
            exact_solve_s=ex.solve_time_s,
            greedy_solve_s=gr.solve_time_s,
        ))
    return rows, dict(tau=tau, horizon=horizon, m=m, lag=lag)


if __name__ == "__main__":
    SKIP1 = "--part2" in sys.argv
    if SKIP1:
        prev = json.loads((RESULTS / "e15_validation.json").read_text())
        n = prev["cases"]
        ex_ok = prev["exact_attains_optimum"]
        gr_ok = prev["greedy_first_action_optimal"]
        excess_mean = prev.get("greedy_excess_mean")
        excess_max = prev.get("greedy_excess_max")
        excess = []
        failures = []
        print("Part 1 - reusing results/e15_validation.json")
        print(f"  exact attains brute-force optimum: {ex_ok}/{n}")
    else:
        print("Part 1 - exhaustive validation of solve_mpc_exact")
        rows, failures = part1_validate()
        n = len(rows)
        excess_mean = excess_max = None
        ex_ok = sum(r["exact_ok"] for r in rows)
        gr_ok = sum(r["greedy_u_optimal"] for r in rows)
        excess = [r["greedy_excess"] for r in rows if r["greedy_excess"] is not None]
        print(f"  cases                            : {n}")
        print(f"  exact attains brute-force optimum: {ex_ok}/{n}")
        print(f"  greedy first action optimal      : {gr_ok}/{n} "
              f"({100.0*gr_ok/n:.1f}%)")
        if excess:
            excess_mean = sum(excess) / len(excess)
            excess_max = max(excess)
            print(f"  greedy objective excess mean/max : "
                  f"{excess_mean:.4f} / {excess_max:.4f}")
        (RESULTS / "e15_validation.json").write_text(json.dumps(dict(
            cases=n, exact_attains_optimum=ex_ok,
            greedy_first_action_optimal=gr_ok,
            greedy_excess_mean=excess_mean,
            greedy_excess_max=excess_max,
            failures=len(failures),
        ), indent=2))
    if failures:
        print(f"\n  FAILURES: {len(failures)}")
        print(json.dumps(failures[:2], indent=2, default=str))
        sys.exit(1)

    print()
    sys.stdout.flush()
    print("Part 2 - greedy-completion gap on the paper configuration")
    rows2, meta = part2_gap()
    n2 = len(rows2)
    agree = sum(r["agree"] for r in rows2)
    gaps = [r["greedy_cost_true"] - r["exact_cost"]
            for r in rows2 if r["greedy_cost_true"] is not None]
    worse = sum(1 for g in gaps if g > TOL)
    better = sum(1 for g in gaps if g < -TOL)
    rel = [g / max(abs(c), 1.0) for g, c in
           zip(gaps, [r["exact_cost"] for r in rows2])]
    ex_t = sorted(r["exact_solve_s"] for r in rows2)
    gr_t = sorted(r["greedy_solve_s"] for r in rows2)
    print(f"  tau={meta['tau']}  H={meta['horizon']}  "
          f"m={meta['m']}  lag={meta['lag']}")
    print(f"  ticks                        : {n2}")
    print(f"  first actions agree          : {agree}/{n2} = "
          f"{100.0*agree/n2:.2f}%")
    print(f"  greedy objective worse       : {worse} "
          f"({100.0*worse/len(gaps):.2f}%)")
    print(f"  greedy objective better      : {better}  (must be 0)")
    print(f"  mean / max absolute excess   : "
          f"{sum(gaps)/len(gaps):.4f} / {max(gaps):.4f}")
    print(f"  mean / max relative excess   : "
          f"{100.0*sum(rel)/len(rel):.3f}% / {100.0*max(rel):.3f}%")
    print(f"  exact  solve p50/p95/max ms  : "
          f"{1e3*ex_t[n2//2]:.3f} / {1e3*ex_t[int(0.95*n2)]:.3f} / "
          f"{1e3*ex_t[-1]:.3f}")
    print(f"  greedy solve p50/p95/max ms  : "
          f"{1e3*gr_t[n2//2]:.3f} / {1e3*gr_t[int(0.95*n2)]:.3f} / "
          f"{1e3*gr_t[-1]:.3f}")

    import csv as _csv
    tick_path = RESULTS / "e15_ticks.csv"
    with open(tick_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=[
            "exact_u", "greedy_u", "agree", "exact_cost", "greedy_cost_true",
            "abs_excess", "rel_excess_pct", "exact_ms", "greedy_ms",
        ])
        w.writeheader()
        for r in rows2:
            gt = r["greedy_cost_true"]
            ex = r["exact_cost"]
            w.writerow({
                "exact_u": r["exact_u"], "greedy_u": r["greedy_u"],
                "agree": int(bool(r["agree"])),
                "exact_cost": round(ex, 6),
                "greedy_cost_true": ("" if gt is None else round(gt, 6)),
                "abs_excess": ("" if gt is None else round(gt - ex, 6)),
                "rel_excess_pct": ("" if gt is None else
                                   round(100.0 * (gt - ex) / max(abs(ex), 1.0), 6)),
                "exact_ms": round(1e3 * r["exact_solve_s"], 6),
                "greedy_ms": round(1e3 * r["greedy_solve_s"], 6),
            })
    print(f"  wrote {tick_path} ({len(rows2)} rows)")

    out = RESULTS / "e15_exact_vs_greedy.json"
    out.write_text(json.dumps(dict(
        validation=dict(
            cases=n, exact_attains_optimum=ex_ok,
            greedy_first_action_optimal=gr_ok,
            greedy_excess_mean=excess_mean,
            greedy_excess_max=excess_max,
        ),
        paper_config=dict(
            **meta, ticks=n2, agree=agree, agree_pct=100.0*agree/n2,
            greedy_worse=worse, greedy_worse_pct=100.0*worse/len(gaps),
            greedy_better=better,
            cost_excess_mean=sum(gaps)/len(gaps), cost_excess_max=max(gaps),
            rel_excess_mean_pct=100.0*sum(rel)/len(rel),
            rel_excess_max_pct=100.0*max(rel),
            exact_p50_ms=1e3*ex_t[n2//2], exact_p95_ms=1e3*ex_t[int(0.95*n2)],
            exact_max_ms=1e3*ex_t[-1],
            greedy_p50_ms=1e3*gr_t[n2//2],
            greedy_p95_ms=1e3*gr_t[int(0.95*n2)],
            greedy_max_ms=1e3*gr_t[-1],
        ),
    ), indent=2))
    print(f"\n  wrote {out}")
