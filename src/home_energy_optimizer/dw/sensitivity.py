"""A device's price sensitivity, read from its DP.

The idea: instead of (or as well as) proposing one plan per price, a device
says "this is my plan, and this is how I would respond if you asked me to
consume a little more or a little less in any hour". The DP already holds the
answer. Force a one-step change in the action at hour t, then let the stored
optimal policy re-plan every later hour from wherever that leaves the device.
The result is a runnable plan - the physics are respected by construction -
that includes the knock-on shifts ("more now, so less at 18:00"), and its
cost at the current price is the device's marginal cost of that deviation.

No re-solve is involved: every variant is a forward replay of the policy the
pricing DP just computed, batched so that all 2n of them run in one pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from home_energy_optimizer.dp_battery import _feasible_actions
from home_energy_optimizer.dp_thermal import (
    HVAC_ACTIONS,
    _hvac_duty_cap,
    _hvac_heat_flow,
    _max_duty,
    _usable_outflow,
)
from home_energy_optimizer.types import DeviceSolution

if TYPE_CHECKING:  # dw.coordinator imports this module, so only for checkers
    from home_energy_optimizer.dw.coordinator import Device, DWCoordinator


def _nearest(values: npt.ArrayLike, lo: float, hi: float, n: int) -> np.ndarray:
    idx = np.rint((np.asarray(values) - lo) / (hi - lo) * (n - 1)).astype(int)
    return np.clip(idx, 0, n - 1)


def _replay(n: int, starts: np.ndarray, forced: np.ndarray, base_power: np.ndarray,
            base_traj: np.ndarray,
            policy_action: Callable[[int, np.ndarray], np.ndarray],
            step: Callable[[int, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
            ) -> tuple[np.ndarray, np.ndarray]:
    """Run every variant forward at once.

    Variant k follows the base plan before starts[k], takes forced[k] at
    starts[k], then follows `policy_action(t, state)`. `step(t, state, action)`
    returns (realised power, next state), vectorised over variants.
    """
    K = len(starts)
    power = np.tile(base_power, (K, 1))
    traj = np.tile(base_traj, (K, 1))
    x = np.full(K, base_traj[0], dtype=float)
    for t in range(n):
        x = np.where(starts >= t, base_traj[t], x)      # on the base plan until the change
        live = starts <= t
        if not live.any():
            continue
        act = np.where(starts == t, forced, policy_action(t, x))
        p, x_next = step(t, x, act)
        power[live, t] = p[live]
        traj[live, t + 1] = x_next[live]
        x = np.where(live, x_next, x)
    return power, traj


def variants(co: DWCoordinator, dev: Device, sol: DeviceSolution
             ) -> tuple[list[tuple[int, int, np.ndarray, np.ndarray]], np.ndarray]:
    """All one-step deviations of `sol`'s plan, each followed by the policy.

    Returns ([(t, direction, power, trajectory), ...], base_power), with
    direction +1 for "consume more at t" and -1 for "consume less". Only
    admissible, genuinely different changes are included.
    """
    n, dt, fc = co.n, co.dt, co.fc
    base_p, base_x = sol.power.copy(), sol.trajectory.copy()
    starts, forced, dirs = [], [], []

    if dev.kind == "water_heater":
        wh = dev.tank_cfg
        levels = wh.duty_actions
        C, P = wh.heat_capacity_kwh_per_k, wh.power_kw
        base_level = _nearest(base_p / P, 0.0, 1.0, len(levels))
        for t in range(n):
            for d in (+1, -1):
                j = base_level[t] + d
                if 0 <= j < len(levels):
                    starts.append(t); forced.append(levels[j]); dirs.append(d)

        def policy_action(t: int, x: np.ndarray) -> np.ndarray:
            return levels[sol.policy[t, _nearest(x, wh.t_min, wh.t_max, wh.n_states)]]

        def step(t: int, x: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q_out = _usable_outflow(x, wh, fc.hot_water_demand[t], dt)
            duty = np.minimum(a, _max_duty(x, wh, q_out, dt))
            return P * duty, x + (P * duty - q_out) / C * dt

    elif dev.kind == "hvac":
        hv = dev.hvac_cfg
        base_a = np.zeros(n)
        # recover the base action from the policy along the base trajectory
        for t in range(n):
            base_a[t] = hv.duty_actions[sol.policy[t, _nearest(base_x[t], hv.t_min, hv.t_max, hv.n_states)]]
        for t in range(n):
            for a in HVAC_ACTIONS:                 # off, full cool, full heat
                if a != base_a[t]:
                    # "more" = a mode that draws power where the base did not
                    d = +1 if (a != 0 and base_a[t] == 0) else (-1 if a == 0 else 0)
                    starts.append(t); forced.append(a); dirs.append(d)

        def policy_action(t: int, x: np.ndarray) -> np.ndarray:
            return hv.duty_actions[sol.policy[t, _nearest(x, hv.t_min, hv.t_max, hv.n_states)]]

        def step(t: int, x: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q_wall = (fc.outdoor_temp[t] - x) / hv.r_wall_k_per_kw
            duty = np.ones_like(x)
            q_ac = np.zeros_like(x)
            for mode in (-1.0, 1.0):
                m = a == mode
                if m.any():
                    duty[m] = _hvac_duty_cap(x[m], hv, q_wall[m], mode, dt)
                    q_ac[m] = _hvac_heat_flow(mode, hv) * duty[m]
            p = hv.power_kw * np.abs(a) * np.where(a == 0, 0.0, duty)
            return p, x + (q_wall + q_ac) / hv.c_room_kwh_per_k * dt

    else:  # battery proposed as columns
        b = dev.battery_cfg
        cap, ec, ed = b.capacity_kwh, b.eta_c, b.eta_d
        delta = max(b.p_charge_max_kw, b.p_discharge_max_kw) / 4.0
        for t in range(n):
            for d in (+1, -1):
                a = float(_feasible_actions(np.array([base_p[t] + d * delta]), np.array([base_x[t]]),
                                            dt, cap, 0.0, b.charge_deadband_kw, ec, ed,
                                            b.soe_floor_kwh)[0])
                if abs(a - base_p[t]) > 1e-6:
                    starts.append(t); forced.append(a); dirs.append(d)

        def policy_action(t: int, x: np.ndarray) -> np.ndarray:
            return sol.policy[t, _nearest(x, b.soe_floor_kwh, cap, b.n_states)]

        def step(t: int, x: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a = _feasible_actions(a, x, dt, cap, 0.0, b.charge_deadband_kw, ec, ed, b.soe_floor_kwh)
            eff = np.where(a > 0, a * ec, a / ed)
            return a, np.clip(x + eff * dt, b.soe_floor_kwh, cap)

    if not starts:
        return [], base_p
    start_at, forced_at = np.array(starts), np.array(forced, dtype=float)
    power, traj = _replay(n, start_at, forced_at, base_p, base_x, policy_action, step)
    out = []
    for k in range(len(starts)):
        if np.max(np.abs(power[k] - base_p)) > 1e-6:
            out.append((int(starts[k]), int(dirs[k]), power[k], traj[k]))
    return out, base_p


def flexibility(co: DWCoordinator, dev: Device, sol: DeviceSolution, price_kwh: np.ndarray,
                base_cost: float,
                found: list[tuple[int, int, np.ndarray, np.ndarray]]) -> dict:
    """Per hour, the cheapest 'more' and 'less' deviation, as cost at `price_kwh`.

    Cost = change in (private cost + energy at the price) against the base
    plan: how much it costs the device, all effects included, to be asked to
    move. Near zero means the device is flexible at that hour; a DP that is
    optimal at the price makes it non-negative up to discretisation.
    """
    n, dt = co.n, co.dt
    base_val = base_cost + float(price_kwh @ sol.power) * dt
    up: list[float | None] = [None] * n
    down: list[float | None] = [None] * n
    for t, d, p, x in found:
        val = co.private_cost(dev, x) + float(price_kwh @ p) * dt - base_val
        if d > 0 and ((cur := up[t]) is None or val < cur):
            up[t] = val
        if d < 0 and ((cur := down[t]) is None or val < cur):
            down[t] = val
    return {"up": up, "down": down}
