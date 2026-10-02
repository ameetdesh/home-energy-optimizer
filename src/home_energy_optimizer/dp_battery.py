"""Battery dynamic program.

Backward DP over (time, state-of-energy) with a discretised power action grid.
Maximises reward = -import_cost + export_revenue, so V[t, s] is the value-to-go
in currency from holding s kWh at step t.

Ported from the POC's `battery_dp`, with three changes:

1. No module globals - config is passed in.
2. V and POL are always returned (the POC dropped them on the compiled path).
3. `dp_load` semantics documented: it is every OTHER power flow on the meter,
   so the battery internalises its own marginal effect on the grid exchange.
   This is also why the resulting dV/ds is a marginal value CONDITIONAL on the
   other devices' plans, not a true system-wide shadow price. See
   docs/NOTES.md - measuring that gap is an open question.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from ._kernels import kernel_battery
from .interp import interp_grid
from .meter import Limits, limit_cost
from .types import BatteryConfig, DeviceSolution, Horizon, SocGate


def _gate_thresholds(
    gates: tuple[SocGate, ...], horizon: Horizon, capacity_kwh: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step minimum stored energy (or -1 for no gate) and its penalty."""
    n = horizon.steps
    min_soe = np.full(n + 1, -1.0)
    penalty = np.zeros(n + 1)
    for gate in gates:
        step = int(round(gate.hour / horizon.dt))
        step = max(0, min(n, step))
        threshold = gate.soc_frac * capacity_kwh
        if threshold > min_soe[step]:
            min_soe[step] = threshold
            penalty[step] = gate.penalty
    return min_soe, penalty


def terminal_price(cfg: BatteryConfig, buy: np.ndarray) -> float:
    """Value of a kWh left in the battery at the horizon edge.

    Defaults to the cheapest import price on the horizon - the same choice
    evcc's optimizer documents as "end of forecast commercial value". It is a
    lower bound on what that energy can save you, so it never encourages
    hoarding beyond what the prices justify.
    """
    if cfg.terminal_price is not None:
        return float(cfg.terminal_price)
    return float(np.min(buy))


def terminal_value(cfg: BatteryConfig, S: np.ndarray, buy: np.ndarray) -> np.ndarray:
    """V at the horizon edge, as a function of stored energy."""
    if cfg.terminal_mode == "linear":
        return terminal_price(cfg, buy) * S
    target = cfg.capacity_kwh * cfg.terminal_target_frac
    return -cfg.terminal_weight * (S - target) ** 2 / cfg.capacity_kwh


def _per_slot(values: npt.ArrayLike | None, n: int, name: str) -> np.ndarray | None:
    """Normalise an optional per-slot array, padding or trimming to n."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return None
    if arr.size < n:  # pad by holding the last value, never silently zero
        arr = np.concatenate([arr, np.full(n - arr.size, arr[-1])])
    return arr[:n]


def _action_grid(cfg: BatteryConfig) -> np.ndarray:
    """Action grid, with the semi-continuous charge band removed.

    Deleting infeasible actions is all a DP needs to honour a charger's minimum
    power; an LP would need a binary per timestep for the same thing.
    """
    A = np.linspace(-cfg.p_discharge_max_kw, cfg.p_charge_max_kw, cfg.n_actions)
    if cfg.charge_deadband_kw > 0.0:
        keep = (A <= 0.0) | (A >= cfg.charge_deadband_kw)
        A = A[keep]
        if cfg.charge_deadband_kw <= cfg.p_charge_max_kw and not np.any(
            np.isclose(A, cfg.charge_deadband_kw)
        ):
            A = np.sort(np.append(A, cfg.charge_deadband_kw))
    return A


def _feasible_actions(
    A_row: np.ndarray,
    S_col: np.ndarray,
    dt: float,
    cap: float,
    min_charge: float = 0.0,
    deadband: float = 0.0,
    eta_c: float = 1.0,
    eta_d: float = 1.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Clip the action grid to what the state physically allows.

    `floor` is the reserve (BatteryConfig.soe_floor_kwh): discharge stops there
    instead of at empty.

    The bounds carry the efficiencies, and that is not cosmetic. Storing energy
    at meter power `a` adds `eta_c * a * dt`, so filling the remaining headroom
    needs `a <= (cap - s) / (eta_c * dt)`. Delivering `|a|` to the meter drains
    `|a| * dt / eta_d`, so emptying the store allows only
    `|a| <= s * eta_d / dt`.

    Using the un-adjusted `(cap - s)/dt` and `(0 - s)/dt` instead lets a nearly
    empty battery be discharged past zero, where the projection onto [0, cap]
    silently absorbs the deficit - delivering energy that never existed (0.111
    kWh in one slot at s = 1 kWh, eta = 0.9). It also under-charges near full,
    since the store can never quite reach capacity. With these bounds
    `s + eff * dt` lands inside [0, cap] by construction and the projection is
    a no-op, which `test_soe_bounds_need_no_projection` asserts.

    With a p_demand floor the lower bound is raised to `min_charge`, but never
    above what the remaining headroom permits: a full battery cannot absorb its
    demand, and returning an infeasible action would be worse than charging as
    much as fits.

    `min_charge = 0` must mean "no floor", NOT "never discharge" - taking a
    max against zero there would silently forbid every discharge action, which
    is a whole-battery outage that looks like a plan.
    """
    headroom = (cap - S_col) / (eta_c * dt)
    lo = -np.maximum(S_col - floor, 0.0) * eta_d / dt
    if min_charge > 0.0:
        lo = np.minimum(np.maximum(lo, min_charge), headroom)
    out = np.clip(A_row, lo, headroom)
    if deadband > 0.0:
        # Clipping can land a value inside the forbidden band - a full battery
        # has headroom below the charger's floor. Snap those to off, which is
        # always achievable, rather than emitting an unexecutable setpoint.
        out = np.where((out > 0.0) & (out < deadband), 0.0, out)
    return out


def solve_battery(
    cfg: BatteryConfig,
    horizon: Horizon,
    buy: np.ndarray,
    sell: np.ndarray,
    dp_load: np.ndarray | None = None,
    admm_target: np.ndarray | None = None,
    admm_rho: float = 0.0,
    soc_gates: tuple[SocGate, ...] = (),
    limits: Limits | None = None,
) -> DeviceSolution:
    """Solve the battery DP and roll out the optimal trajectory.

    `limits`: price the grid limits into the reward (home_energy_optimizer.meter).

    Sign convention: positive power = CHARGING (drawing from the meter),
    negative = discharging. This matches the POC and is the opposite of
    EMHASS's p_sto_pos convention - see docs/NOTES.md.
    """
    t_start = time.perf_counter()
    n = horizon.steps
    dt = horizon.dt
    cap = cfg.capacity_kwh
    floor = cfg.soe_floor_kwh
    eta_c, eta_d = cfg.eta_c, cfg.eta_d

    if dp_load is None:
        dp_load = np.zeros(n)
    use_admm = admm_target is not None and admm_rho > 0
    if admm_target is None:
        admm_target = np.zeros(n)

    S = np.linspace(floor, cap, cfg.n_states)
    A = _action_grid(cfg)
    V = np.zeros((n + 1, cfg.n_states))
    POL = np.zeros((n, cfg.n_states))

    min_charge = _per_slot(cfg.min_charge_kw, n, "min_charge_kw")
    soc_goal = _per_slot(cfg.soc_goal_kwh, n, "soc_goal_kwh")
    gate_min_soe, gate_penalty = _gate_thresholds(soc_gates, horizon, cap)

    V[n] = terminal_value(cfg, S, buy)
    if gate_min_soe[n] >= 0:
        shortfall = np.maximum(gate_min_soe[n] - S, 0.0)
        V[n] -= gate_penalty[n] * (shortfall / cap)

    S_col = S[:, None]
    A_row = A[None, :]

    # The compiled kernel covers the DEFAULT configuration only. Anything that
    # adds a term to the recursion - SoC gates, a charge floor, a soc goal, a
    # charger deadband - stays on the Python path rather than being duplicated
    # in C where the two would have to be kept in step by hand.
    plain = (
        min_charge is None
        and soc_goal is None
        and cfg.charge_deadband_kw <= 0.0
        and not np.any(gate_min_soe >= 0)
        and floor == 0.0
        and limits is None
    )
    compiled = plain and kernel_battery(
        np.ascontiguousarray(buy, dtype=np.float64),
        np.ascontiguousarray(sell, dtype=np.float64),
        np.ascontiguousarray(dp_load, dtype=np.float64),
        np.ascontiguousarray(admm_target, dtype=np.float64),
        np.ascontiguousarray(S, dtype=np.float64),
        np.ascontiguousarray(A, dtype=np.float64),
        V, POL, n, cfg.n_states, len(A),
        dt, cap, eta_c, eta_d, float(admm_rho), 1 if use_admm else 0,
    )

    for t in (() if compiled else range(n - 1, -1, -1)):
        # Clip actions to what the current state physically allows, then apply
        # efficiency: charging stores eta * a, discharging drains a / eta.
        Ac = _feasible_actions(
            A_row, S_col, dt, cap,
            0.0 if min_charge is None else float(min_charge[t]),
            cfg.charge_deadband_kw, eta_c, eta_d, floor,
        )
        effA = np.where(Ac > 0, Ac * eta_c, Ac / eta_d)
        S_next = np.clip(S_col + effA * dt, floor, cap)
        V_next = interp_grid(S_next, S, V[t + 1])

        # Load-aware reward: the battery sees its marginal effect on net import.
        imp = Ac + dp_load[t]
        reward = (
            -buy[t] * np.maximum(imp, 0.0) + sell[t] * np.maximum(-imp, 0.0)
            - limit_cost(imp, sell[t], limits)
        ) * dt
        if use_admm:
            reward = reward - (admm_rho / 2.0) * (Ac - admm_target[t]) ** 2 * dt

        Q = reward + V_next
        best = np.argmax(Q, axis=1)
        rows = np.arange(cfg.n_states)
        V[t] = Q[rows, best]
        POL[t] = Ac[rows, best]

        if gate_min_soe[t] >= 0:
            shortfall = np.maximum(gate_min_soe[t] - S, 0.0)
            V[t] -= gate_penalty[t] * (shortfall / cap)

        # Soft per-slot SoC goal (evcc s_goal). Priced rather than enforced so
        # an unreachable goal degrades to "as close as possible".
        if soc_goal is not None:
            V[t] -= cfg.soc_goal_penalty * np.maximum(soc_goal[t] - S, 0.0)

    soc, power = rollout_battery(
        cfg, horizon, V, S, A, buy, sell, dp_load, admm_target if use_admm else None,
        admm_rho if use_admm else 0.0, start_step=0, start_soe=cap * cfg.soc_initial_frac,
        limits=limits,
    )

    return DeviceSolution(
        trajectory=soc,
        power=power,
        value=V,
        policy=POL,
        states=S,
        actions=A,
        solve_ms=(time.perf_counter() - t_start) * 1000.0,
    )


def rollout_battery(
    cfg: BatteryConfig,
    horizon: Horizon,
    V: np.ndarray,
    S: np.ndarray,
    A: np.ndarray,
    buy: np.ndarray,
    sell: np.ndarray,
    dp_load: np.ndarray,
    admm_target: np.ndarray | None,
    admm_rho: float,
    start_step: int,
    start_soe: float,
    limits: Limits | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy forward replay of a stored value function from ANY (t, soe).

    This is the counterfactual engine. Re-deriving the action from V rather
    than reading POL means it works for states between grid points, which is
    what makes "what if I were at 3.7 kWh right now" a microsecond question
    instead of a re-solve.

    Returns (soe trajectory of length steps - start_step + 1, power).
    """
    n = horizon.steps
    dt = horizon.dt
    cap = cfg.capacity_kwh
    floor = cfg.soe_floor_kwh
    eta_c, eta_d = cfg.eta_c, cfg.eta_d
    use_admm = admm_target is not None and admm_rho > 0

    start_step = max(0, min(n, start_step))
    horizon_len = n - start_step
    soe = np.zeros(horizon_len + 1)
    power = np.zeros(horizon_len)
    s = float(np.clip(start_soe, floor, cap))
    soe[0] = s

    min_charge = _per_slot(cfg.min_charge_kw, n, "min_charge_kw")

    for i in range(horizon_len):
        t = start_step + i
        Ac = _feasible_actions(
            A, np.array(s), dt, cap,
            0.0 if min_charge is None else float(min_charge[t]),
            cfg.charge_deadband_kw, eta_c, eta_d, floor,
        )
        effA = np.where(Ac > 0, Ac * eta_c, Ac / eta_d)
        S_next = np.clip(s + effA * dt, floor, cap)
        V_next = interp_grid(S_next.reshape(1, -1), S, V[t + 1]).ravel()

        imp = Ac + dp_load[t]
        reward = (
            -buy[t] * np.maximum(imp, 0.0) + sell[t] * np.maximum(-imp, 0.0)
            - limit_cost(imp, sell[t], limits)
        ) * dt
        if use_admm:
            reward = reward - (admm_rho / 2.0) * (Ac - admm_target[t]) ** 2 * dt

        a = float(Ac[np.argmax(reward + V_next)])
        power[i] = a
        eff = a * eta_c if a > 0 else a / eta_d
        s = float(np.clip(s + eff * dt, floor, cap))
        soe[i + 1] = s

    return soe, power
