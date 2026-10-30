"""Globally optimal integer MPC for the dead-time plant (Eq. 20-24).

``solve_mpc`` in :mod:`pact.control.mpc` enumerates the admissible first
actions but completes each horizon greedily, so the action it returns is
optimal only within the family of greedily-completed plans. This module
solves the same integer program exactly and is the reference against which
the greedy-completion solver is measured.

The program
-----------
Minimise, over integer sequences ``u = (u_0, ..., u_{H-1})`` with
``|u_k| <= Delta_max``,

    J(u) = sum_k [ q(e_k) e_k^2 + r u_k^2 + s (u_k - u_{k-1})^2 ],
    e_k  = N(t+k+1) - Ntilde(t+1+k),

subject to the plant of :func:`pact.control.mpc.apply_plant` -- which
saturates ``N`` at ``[n_min, n_max]`` on every tick -- and to landing
admissibility

    n_min <= committed_k + u_k <= n_max,
    committed_k = N(t+k) + sum(pending_k).

Exactness
---------
Two structural facts make the exact optimum cheap.

(1) *Dead-time truncation.* ``u(t+j)`` first appears in ``N(t+j+lag)`` with
    ``lag = tau`` (or 1 when ``tau <= 1``), so it enters a tracking term
    ``e_k`` only when ``j + lag <= k + 1``. Over ``k = 0..H-1`` that leaves

        m = max(1, min(H, H - lag + 1))

    tracking-relevant actions. The remaining ``u_m, ..., u_{H-1}`` land
    strictly after the horizon and therefore cannot move any ``e_k`` -- nor
    the saturation pattern, which depends only on what has landed.

(2) *Tail separability.* Those late actions enter ``J`` only through the
    decoupled chain ``sum_k [ r u_k^2 + s (u_k - u_{k-1})^2 ]`` and their own
    admissibility. Because they never land in-horizon, the saturation
    corrections to ``committed`` over the tail are fixed by the prefix, so
    the chain is a scalar shortest-path problem in ``(committed, u_{k-1})``
    and is solved exactly by backward recursion.

This tail minimum is *not* the all-zero tail whenever ``s > 0``: with
r = 0.15, s = 0.60 and ``u_{m-1} = 4``, the all-zero tail costs 9.60 while a
ramp-down costs 4.35. Zeroing the tail is exactly what the greedy completion
does, which is why its objective is a biased overestimate rather than a
harmless approximation.

Cost: ``O((2 Delta_max + 1)^m H)`` prefix work plus a memoised tail
recursion. For the configuration reported in the paper (H = 12, tau = 9,
Delta_max = 4) that is at most 9^4 = 6561 prefixes rather than the 9^12 of
naive full enumeration.

Tie-breaking matches :mod:`pact.control.mpc`: lower cost wins, then smaller
``|u(t)|``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

from pact.config import ControlConfig, PactConfig, cold_start_ticks
from pact.control.mpc import (
    MpcSolution,
    _pending_of_length,
    _pipe_len,
    apply_plant,
)

logger = logging.getLogger(__name__)

_INF = float("inf")


def _fast_step(
    n: int, pipe: tuple[int, ...], u: int, tau: int, lo: int, hi: int
) -> tuple[int, tuple[int, ...]]:
    """Allocation-light equivalent of :func:`pact.control.mpc.apply_plant`.

    Requires ``pipe`` to already have length ``max(tau-1, 0)``. Part 1 of
    ``experiments/verify_exact.py`` checks the solver that uses this against
    a brute force built on ``apply_plant`` itself, so any divergence between
    the two would surface as a validation failure.
    """

    if tau <= 1:
        v = n + u
        return (lo if v < lo else hi if v > hi else v), ()
    v = n + pipe[0]
    if v < lo:
        v = lo
    elif v > hi:
        v = hi
    return v, pipe[1:] + (u,)


def landing_lag(tau: int) -> int:
    """Ticks between issuing ``u(t)`` and its appearance in ``N``.

    ``apply_plant`` applies ``u`` immediately when ``tau <= 1``, so the lag
    is one tick there and ``tau`` ticks otherwise.
    """

    return 1 if tau <= 1 else tau


def tracking_relevant_actions(horizon: int, tau: int) -> int:
    """Number of leading actions that can move an in-horizon tracking term."""

    lag = landing_lag(tau)
    return max(1, min(horizon, horizon - lag + 1))


# Largest number of recursion nodes we are willing to spend on one tick.
# The solve is abandoned rather than allowed to stall the control loop.
EXACT_NODE_BUDGET = 400_000


class ExactBudgetExceeded(RuntimeError):
    """The exact solve would cost more than ``EXACT_NODE_BUDGET`` nodes.

    Raised instead of returning a partial answer, so a caller can never
    mistake a truncated search for the global optimum. The caller is
    expected to fall back to an approximate solver and record that it did.
    """


def exact_is_tractable(
    horizon: int, tau: int, delta_max: int,
    n_min: int = 1, n_max: int = 16,
    budget: int = EXACT_NODE_BUDGET,
) -> bool:
    """Cheap a-priori estimate of whether the exact solve is affordable.

    Two reductions can make it cheap and they are complementary. Plain
    enumeration of the ``m`` tracking-relevant actions costs
    ``(2*delta_max+1)^m`` leaves, which is small when the dead time is long.
    Memoisation on ``(k, N, pending, u_prev)`` costs at most
    ``m * |N| * (2*delta_max+1)^(min(m, tau-1) + 1)`` states, which is small
    when the dead time is short, because then the pending queue is short and
    distinct prefixes collapse onto the same state.

    Note what the memo bound is *not*: the memo still indexes the pending
    queue, so it does not collapse anything while the queue is longer than
    the search is deep. An earlier version of this function assumed it did
    and declared intractable configurations tractable.

    This is only an estimate -- the reachable state count is smaller than
    either bound, because landing admissibility constrains the queue. The
    binding guarantee is the runtime node cap in :func:`solve_mpc_exact`.
    """

    m = tracking_relevant_actions(horizon, tau)
    pipe_len = max(tau - 1, 0)
    a = 2 * delta_max + 1
    span = max(n_max - n_min + 1, 1)

    def powc(base: int, exp: int) -> float:
        """``base**exp``, saturating instead of building a huge integer."""
        if exp * math.log10(max(base, 2)) > 12:
            return float("inf")
        return float(base ** exp)

    tree = powc(a, m)
    memo = m * span * powc(a, min(m, pipe_len) + 1)
    return min(tree, memo) <= budget


def committed_of(n: int, pending: Sequence[int], tau: int) -> int:
    """Eventual replica count if no further action is taken."""

    return n if tau <= 1 else n + sum(pending)


def plan_cost(
    n_current: int,
    pending: Sequence[int],
    plan: Sequence[int],
    demand: Sequence[int],
    last_u: int,
    tau: int,
    ctrl: ControlConfig,
) -> float | None:
    """Reference objective for one full plan, or ``None`` if inadmissible.

    Replicates :func:`pact.control.mpc._piecewise_cost` term by term and adds
    the landing-admissibility test, so this is the ground truth both the
    exact solver and the brute-force checker are measured against.
    """

    n = n_current
    pipe = list(pending)
    prev = last_u
    cost = 0.0
    for k, u in enumerate(plan):
        land = committed_of(n, pipe, tau) + u
        if land < ctrl.n_min or land > ctrl.n_max:
            return None
        cost += ctrl.r_act * float(u * u)
        du = u - prev
        cost += ctrl.s_churn * float(du * du)
        n, pipe = apply_plant(n, pipe, u, tau, ctrl.n_min, ctrl.n_max)
        err = float(n - demand[k])
        q = ctrl.q_up if n < demand[k] else ctrl.q_down
        cost += q * err * err
        prev = u
    return cost


def solve_mpc_exact(
    n_current: int,
    demand: Sequence[int],
    pending: Sequence[int],
    last_u: int,
    config: PactConfig,
    node_budget: int = EXACT_NODE_BUDGET,
) -> MpcSolution:
    """Return the globally optimal first action of the integer program.

    Arguments and conventions are identical to
    :func:`pact.control.mpc.solve_mpc`: ``demand[k]`` is ``Ntilde(t+1+k)``
    and ``pending`` is the in-flight queue of length ``max(tau-1, 0)``.
    """

    started = time.perf_counter()
    ctrl = config.control
    tau = cold_start_ticks(ctrl.tau_c_s, config.telemetry.dt)
    horizon = min(len(demand), config.forecast.horizon)
    if horizon < 1:
        raise ValueError("demand horizon must be >= 1")

    pipe0 = _pending_of_length(pending, _pipe_len(tau))
    demand_h = [int(x) for x in demand[:horizon]]
    m = tracking_relevant_actions(horizon, tau)
    tail_cache: dict[tuple[int, ...], dict[tuple[int, int, int], float]] = {}

    lo, hi = ctrl.n_min, ctrl.n_max
    r, s, q_up, q_dn = ctrl.r_act, ctrl.s_churn, ctrl.q_up, ctrl.q_down
    us = tuple(range(-ctrl.delta_max, ctrl.delta_max + 1))
    memo: dict[tuple[int, int, tuple[int, ...], int], float] = {}
    nodes = 0

    def cost_to_go(
        depth: int, n: int, pipe: tuple[int, ...], prev_u: int
    ) -> float:
        """Exact minimum of the remaining objective from this plant state.

        Memoised on the full controllable state ``(depth, N, pipe, u_prev)``.
        When the dead time is short the pipe is short, so distinct action
        sequences collapse onto the same state and the recursion is
        polynomial; when the dead time is long the pipe is long but ``m`` is
        small, so the tree is shallow. The two regimes are complementary,
        which is why one procedure covers both.
        """

        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            # Abandon rather than return a truncated search as if it were
            # the optimum. The caller falls back and records that it did.
            raise ExactBudgetExceeded(
                f"exact solve exceeded {node_budget} nodes "
                f"(H={horizon}, tau={tau}, m={m}, delta_max={ctrl.delta_max})"
            )

        if depth == m:
            tail_cost, tracking = _complete(
                n, pipe, demand_h, prev_u, tau, ctrl, m, horizon, tail_cache
            )
            return _INF if tail_cost is None else tracking + tail_cost

        key = (depth, n, pipe, prev_u)
        hit = memo.get(key)
        if hit is not None:
            return hit

        dk = demand_h[depth]
        committed = committed_of(n, pipe, tau)
        best_v = _INF
        for u in us:
            land = committed + u
            if land < lo or land > hi:
                continue
            n_next, pipe_next = _fast_step(n, pipe, u, tau, lo, hi)
            err = float(n_next - dk)
            q = q_up if n_next < dk else q_dn
            du = u - prev_u
            step = r * float(u * u) + s * float(du * du) + q * err * err
            rest = cost_to_go(depth + 1, n_next, pipe_next, u)
            if rest == _INF:
                continue
            total = step + rest
            if total < best_v:
                best_v = total
        memo[key] = best_v
        return best_v

    pipe_t = tuple(pipe0)
    committed0 = committed_of(n_current, pipe_t, tau)
    dk0 = demand_h[0]
    best_cost = _INF
    best_u0: int | None = None
    best_next: tuple[int, tuple[int, ...]] | None = None
    for u in us:
        land = committed0 + u
        if land < lo or land > hi:
            continue
        n_next, pipe_next = _fast_step(n_current, pipe_t, u, tau, lo, hi)
        err = float(n_next - dk0)
        q = q_up if n_next < dk0 else q_dn
        du = u - last_u
        step = r * float(u * u) + s * float(du * du) + q * err * err
        rest = cost_to_go(1, n_next, pipe_next, u)
        if rest == _INF:
            continue
        total = step + rest
        if total < best_cost - 1e-12 or (
            abs(total - best_cost) <= 1e-12
            and best_u0 is not None
            and abs(u) < abs(best_u0)
        ):
            best_cost, best_u0, best_next = total, u, (n_next, pipe_next)

    if best_u0 is not None:
        # Walk the argmin forward for reporting; the cost is already final.
        mid: list[int] = []
        n, pipe, prev_u = (*best_next, best_u0)  # type: ignore[misc]
        for depth in range(1, m):
            dk = demand_h[depth]
            committed = committed_of(n, pipe, tau)
            pick, pick_v, pick_next = None, _INF, None
            for u in us:
                land = committed + u
                if land < lo or land > hi:
                    continue
                n_next, pipe_next = _fast_step(n, pipe, u, tau, lo, hi)
                err = float(n_next - dk)
                q = q_up if n_next < dk else q_dn
                du = u - prev_u
                step = r * float(u * u) + s * float(du * du) + q * err * err
                rest = cost_to_go(depth + 1, n_next, pipe_next, u)
                if rest == _INF:
                    continue
                if step + rest < pick_v:
                    pick, pick_v, pick_next = u, step + rest, (n_next, pipe_next)
            if pick is None:
                break
            mid.append(pick)
            n, pipe = pick_next  # type: ignore[misc]
            prev_u = pick
        tail_plan = _recover_tail(
            n, pipe, prev_u, tau, ctrl, m, horizon, tail_cache
        )
        best = MpcSolution(
            u=best_u0, cost=best_cost,
            plan=(best_u0,) + tuple(mid) + tail_plan, solve_time_s=0.0,
        )
    else:
        best = None

    if best is None:
        # No admissible sequence: hold, mirroring the fallback in
        # pact.control.mpc so the two solvers degrade identically.
        plan = (0,) * horizon
        cost = plan_cost(
            n_current, pipe0, plan, demand_h, last_u, tau, ctrl
        )
        best = MpcSolution(
            u=0, cost=_INF if cost is None else cost, plan=plan, solve_time_s=0.0
        )

    elapsed = time.perf_counter() - started
    logger.debug(
        "mpc_exact solve_time_s=%.6f u=%s cost=%.4f m=%d",
        elapsed, best.u, best.cost, m,
    )
    return MpcSolution(
        u=best.u, cost=best.cost, plan=best.plan, solve_time_s=elapsed
    )


_ZERO_DELTAS: dict[int, tuple[int, ...]] = {}


def _zeros(k: int) -> tuple[int, ...]:
    z = _ZERO_DELTAS.get(k)
    if z is None:
        z = (0,) * k
        _ZERO_DELTAS[k] = z
    return z


def _zero_tail_rollout(
    n: int,
    pipe: tuple[int, ...],
    demand: Sequence[int],
    tau: int,
    ctrl: ControlConfig,
    m: int,
    horizon: int,
) -> tuple[float, int, tuple[int, ...]]:
    """Tracking cost of stages ``m..H-1``, the committed level entering the
    tail, and the saturation corrections to committed across it.

    Tail actions land strictly after the horizon, so a zero-tail rollout
    gives the exact in-horizon trajectory -- and hence the exact tracking
    cost and saturation corrections -- for every tail choice.

    With every tail action zero the queue simply drains, so the item landing
    at stage ``m+j`` is ``pipe[j]`` and the queue behind it is ``pipe[j+1:]``.
    Walking that directly avoids rebuilding a tuple and re-summing the queue
    at every step. Saturation is rare, so the all-zero correction vector is
    shared rather than rebuilt: that also lets every unsaturated prefix share
    one tail memo instead of allocating its own.
    """

    lo, hi = ctrl.n_min, ctrl.n_max
    q_up, q_dn = ctrl.q_up, ctrl.q_down
    deep = tau > 1
    tracking = 0.0
    nn = n
    rem = sum(pipe) if deep else 0
    length = len(pipe)
    committed0 = nn + rem if deep else nn
    prev_c = committed0
    deltas: list[int] | None = None
    for k in range(m, horizon):
        j = k - m
        if j < length:
            landed = pipe[j]
            rem -= landed
        else:
            landed = 0
        v = nn + landed
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        nn = v
        c = nn + rem if deep else nn
        d = c - prev_c
        if d:
            if deltas is None:
                deltas = [0] * (horizon - m)
            deltas[j] = d
        prev_c = c
        dk = demand[k]
        err = float(nn - dk)
        tracking += (q_up if nn < dk else q_dn) * err * err
    return (tracking, committed0,
            _zeros(horizon - m) if deltas is None else tuple(deltas))


def _complete(
    n: int,
    pipe: tuple[int, ...],
    demand: Sequence[int],
    prev_u: int,
    tau: int,
    ctrl: ControlConfig,
    m: int,
    horizon: int,
    cache: dict[tuple[int, ...], dict[tuple[int, int, int], float]],
) -> tuple[float | None, float]:
    """Exact tail cost and the residual tracking cost of stages ``m..H-1``."""

    tracking, committed0, deltas = _zero_tail_rollout(
        n, pipe, demand, tau, ctrl, m, horizon
    )
    if m >= horizon:
        return 0.0, tracking
    memo = cache.get(deltas)
    if memo is None:
        memo = cache[deltas] = {}
    value = _tail_value(m, committed0, prev_u, deltas, m, horizon, ctrl, memo)
    return (None if value == _INF else value), tracking


def _recover_tail(
    n: int,
    pipe: tuple[int, ...],
    prev_u: int,
    tau: int,
    ctrl: ControlConfig,
    m: int,
    horizon: int,
    cache: dict[tuple[int, ...], dict[tuple[int, int, int], float]],
) -> tuple[int, ...]:
    """One optimal tail for the winning prefix. Reporting only."""

    if m >= horizon:
        return ()
    _, committed0, deltas = _zero_tail_rollout(
        n, pipe, [0] * horizon, tau, ctrl, m, horizon
    )
    memo = cache.setdefault(deltas, {})
    return _tail_argmin(m, committed0, prev_u, deltas, m, horizon, ctrl, memo)


def _tail_value(
    k: int,
    committed: int,
    prev_u: int,
    deltas: tuple[int, ...],
    m: int,
    horizon: int,
    ctrl: ControlConfig,
    memo: dict[tuple[int, int, int], float],
) -> float:
    """Minimum of the decoupled tail chain from stage ``k``."""

    if k >= horizon:
        return 0.0
    key = (k, committed, prev_u)
    hit = memo.get(key)
    if hit is not None:
        return hit
    best = _INF
    for u in range(-ctrl.delta_max, ctrl.delta_max + 1):
        land = committed + u
        if land < ctrl.n_min or land > ctrl.n_max:
            continue
        nxt = _tail_value(
            k + 1, land + deltas[k - m], u, deltas, m, horizon, ctrl, memo
        )
        if nxt == _INF:
            continue
        du = u - prev_u
        total = ctrl.r_act * float(u * u) + ctrl.s_churn * float(du * du) + nxt
        if total < best:
            best = total
    memo[key] = best
    return best


def _tail_argmin(
    k: int,
    committed: int,
    prev_u: int,
    deltas: tuple[int, ...],
    m: int,
    horizon: int,
    ctrl: ControlConfig,
    memo: dict[tuple[int, int, int], float],
) -> tuple[int, ...]:
    """Recover one optimal tail, for reporting. Not used in the cost."""

    plan: list[int] = []
    c, p = committed, prev_u
    for stage in range(k, horizon):
        best_u, best_total = None, _INF
        for u in range(-ctrl.delta_max, ctrl.delta_max + 1):
            land = c + u
            if land < ctrl.n_min or land > ctrl.n_max:
                continue
            nxt = _tail_value(
                stage + 1, land + deltas[stage - m], u, deltas, m, horizon,
                ctrl, memo,
            )
            if nxt == _INF:
                continue
            du = u - p
            total = ctrl.r_act * float(u * u) + ctrl.s_churn * float(du * du) + nxt
            if total < best_total:
                best_total, best_u = total, u
        if best_u is None:
            break
        plan.append(best_u)
        c, p = c + best_u + deltas[stage - m], best_u
    return tuple(plan)
