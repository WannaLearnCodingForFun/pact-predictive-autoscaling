"""Receding-horizon MPC with cold-start dead time (Eq. 20–24).

Approach (b): enumerate the first action over ``[-Δ_max, +Δ_max]``, fill the
rest of the horizon greedily (track the delayed landing), and pick the plan
with the exact piecewise-quadratic cost. Integrality is native — no QP.

Plant: ``u(t)`` first appears in ``N(t+τ)`` with ``τ = ceil(τc / Δt)``,
matching cold-start ready-at ``t+τ``. Discrete delay-line length is
``max(τ − 1, 0)`` so a step in demand ``τ`` ticks ahead is actionable now.

Only ``u(t)`` is applied (Eq. 24); the remainder is receding-horizon advice.
Gates and the safety ceiling are not applied here.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from pact.config import ControlConfig, PactConfig, cold_start_ticks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MpcSolution:
    u: int
    cost: float
    plan: tuple[int, ...]
    solve_time_s: float


def solve_mpc(
    n_current: int,
    demand: Sequence[int],
    pending: Sequence[int],
    last_u: int,
    config: PactConfig,
) -> MpcSolution:
    """Return the first action of the minimum-cost feasible plan.

    ``demand[k]`` is ``Ñ(t+1+k)`` for ``k = 0..H-1``.
    ``pending`` is the in-flight queue (length ``max(τ−1, 0)``); extra leading
    zeros are ignored. ``u(t)`` first appears in ``N(t+τ)``.
    """

    started = time.perf_counter()
    ctrl = config.control
    tau = cold_start_ticks(ctrl.tau_c_s, config.telemetry.dt)
    horizon = min(len(demand), config.forecast.horizon)
    if horizon < 1:
        raise ValueError("demand horizon must be >= 1")
    pipe = _pending_of_length(pending, _pipe_len(tau))
    demand_h = [int(x) for x in demand[:horizon]]

    best: MpcSolution | None = None
    for u0 in range(-ctrl.delta_max, ctrl.delta_max + 1):
        if not _landing_feasible(n_current, pipe, u0, tau, ctrl.n_min, ctrl.n_max):
            continue
        plan = _greedy_plan(n_current, pipe, demand_h, u0, tau, ctrl)
        cost = _piecewise_cost(
            n_current, pipe, plan, demand_h, last_u, tau, ctrl
        )
        candidate = MpcSolution(
            u=plan[0], cost=cost, plan=tuple(plan), solve_time_s=0.0
        )
        if best is None or _better(candidate, best):
            best = candidate

    if best is None:
        u0 = _clip_u(0, n_current, pipe, tau, ctrl.delta_max, ctrl.n_min, ctrl.n_max)
        plan = _greedy_plan(n_current, pipe, demand_h, u0, tau, ctrl)
        cost = _piecewise_cost(
            n_current, pipe, plan, demand_h, last_u, tau, ctrl
        )
        best = MpcSolution(
            u=plan[0], cost=cost, plan=tuple(plan), solve_time_s=0.0
        )

    elapsed = time.perf_counter() - started
    logger.debug("mpc solve_time_s=%.6f u=%s cost=%.4f", elapsed, best.u, best.cost)
    return MpcSolution(
        u=best.u, cost=best.cost, plan=best.plan, solve_time_s=elapsed
    )


def advance_pending(pending: Sequence[int], u: int, tau: int) -> list[int]:
    """Shift the delay line after applying ``u``."""

    _, pipe = apply_plant(0, pending, u, tau, n_min=-(10**9), n_max=10**9)
    return pipe


def apply_plant(
    n_current: int,
    pending: Sequence[int],
    u: int,
    tau: int,
    n_min: int,
    n_max: int,
) -> tuple[int, list[int]]:
    """One plant step. ``u(t)`` appears in ``N(t+τ)`` (cold-start delay)."""

    if tau <= 1:
        n_next = _clip(n_current + u, n_min, n_max)
        return n_next, []
    pipe = _pending_of_length(pending, _pipe_len(tau))
    pipe.append(u)
    landed = pipe.pop(0)
    n_next = _clip(n_current + landed, n_min, n_max)
    return n_next, pipe


def _greedy_plan(
    n_current: int,
    pending: list[int],
    demand: list[int],
    u0: int,
    tau: int,
    ctrl: ControlConfig,
) -> list[int]:
    plan = [u0]
    n = n_current
    pipe = list(pending)
    n, pipe = _step(n, pipe, u0, tau, ctrl.n_min, ctrl.n_max)
    for k in range(1, len(demand)):
        u_k = _greedy_u(
            n, pipe, demand, k, tau, ctrl.delta_max, ctrl.n_min, ctrl.n_max
        )
        plan.append(u_k)
        n, pipe = _step(n, pipe, u_k, tau, ctrl.n_min, ctrl.n_max)
    return plan


def _greedy_u(
    n: int,
    pending: list[int],
    demand: list[int],
    k: int,
    tau: int,
    delta_max: int,
    n_min: int,
    n_max: int,
) -> int:
    # u_k first appears in N(t+k+τ) → demand index k+τ-1 (Ñ(t+1+...)).
    land_idx = k if tau <= 1 else k + tau - 1
    if land_idx >= len(demand):
        return _clip_u(0, n, pending, tau, delta_max, n_min, n_max)
    committed = _committed(n, pending, tau)
    raw = int(demand[land_idx] - committed)
    return _clip_u(raw, n, pending, tau, delta_max, n_min, n_max)


def _piecewise_cost(
    n_current: int,
    pending: list[int],
    plan: Sequence[int],
    demand: list[int],
    last_u: int,
    tau: int,
    ctrl: ControlConfig,
) -> float:
    n = n_current
    pipe = list(pending)
    prev = last_u
    cost = 0.0
    for k, u in enumerate(plan):
        cost += ctrl.r_act * float(u * u)
        du = u - prev
        cost += ctrl.s_churn * float(du * du)
        n, pipe = _step(n, pipe, u, tau, ctrl.n_min, ctrl.n_max)
        err = float(n - demand[k])
        q = ctrl.q_up if n < demand[k] else ctrl.q_down
        cost += q * err * err
        prev = u
    return cost


def _step(
    n: int,
    pending: list[int],
    u: int,
    tau: int,
    n_min: int,
    n_max: int,
) -> tuple[int, list[int]]:
    n_next, pipe = apply_plant(n, pending, u, tau, n_min, n_max)
    return n_next, pipe


def _landing_feasible(
    n: int,
    pending: Sequence[int],
    u: int,
    tau: int,
    n_min: int,
    n_max: int,
) -> bool:
    committed = _committed(n, pending, tau)
    land = committed + u
    return n_min <= land <= n_max


def _clip_u(
    u: int,
    n: int,
    pending: Sequence[int],
    tau: int,
    delta_max: int,
    n_min: int,
    n_max: int,
) -> int:
    u = _clip(u, -delta_max, delta_max)
    committed = _committed(n, pending, tau)
    land = committed + u
    if land > n_max:
        u -= land - n_max
    elif land < n_min:
        u += n_min - land
    return _clip(u, -delta_max, delta_max)


def _committed(n: int, pending: Sequence[int], tau: int) -> int:
    if tau <= 1:
        return n
    return n + sum(pending)


def _pipe_len(tau: int) -> int:
    return max(tau - 1, 0)


def _pending_of_length(pending: Sequence[int], length: int) -> list[int]:
    if length <= 0:
        return []
    pipe = [int(x) for x in pending]
    if len(pipe) < length:
        pipe = [0] * (length - len(pipe)) + pipe
    return pipe[-length:]


def _clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _better(candidate: MpcSolution, incumbent: MpcSolution) -> bool:
    if candidate.cost < incumbent.cost:
        return True
    if candidate.cost > incumbent.cost:
        return False
    return abs(candidate.u) < abs(incumbent.u)
