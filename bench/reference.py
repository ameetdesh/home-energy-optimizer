"""Exact reference solvers, for measuring how much the fast path gives up.

Two DIFFERENT error sources need isolating, and conflating them is the easiest
way to get a misleading number:

1. **Discretisation gap** - the battery DP uses a 50-point state grid and a
   21-point action grid where the true problem is continuous. Measured by
   `lp_battery`, a continuous LP over the identical model, solved exactly by
   HiGHS.

2. **Decomposition gap** - per-device 1-D DPs coordinated by ADMM/price, where
   the true problem is a joint optimisation over the product state space.
   Measured by `joint_dp_battery_water_heater`, an exact 2-D DP over the
   identical model.

Comparing against EMHASS's MILP would conflate both with *model* differences
(different devices, different constraints), which is why these references
reimplement this package's own model rather than reaching for another project.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from home_energy_optimizer.dp_battery import terminal_price
from home_energy_optimizer.dp_thermal import (
    WATER_HEATER_ACTIONS,
    _max_duty,
    _usable_outflow,
    wh_discomfort,
)
from home_energy_optimizer.interp import interp_uniform
from home_energy_optimizer.types import BatteryConfig, Horizon, WaterHeaterConfig


# --------------------------------------------------------------------------
# 1. Continuous LP reference for the battery  (isolates discretisation)
# --------------------------------------------------------------------------


def lp_battery(
    cfg: BatteryConfig,
    horizon: Horizon,
    buy: np.ndarray,
    sell: np.ndarray,
    dp_load: np.ndarray,
    soe_initial: float | None = None,
) -> dict:
    """Exact continuous optimum for the battery model, via HiGHS.

    Variables per step: charge power c[t] >= 0, discharge power d[t] >= 0, and
    the grid split g+[t], g-[t] >= 0. Energy uses the same one-way efficiency
    convention as the DP (store eta*c, drain d/eta), so the only difference
    from `solve_battery` is that this is continuous in both power and energy.

    Requires buy >= sell elementwise: otherwise the grid split needs a binary
    to stop the LP inflating import and export together.
    """
    n = horizon.steps
    dt = horizon.dt
    cap = cfg.capacity_kwh
    eta_c, eta_d = cfg.eta_c, cfg.eta_d
    if np.any(sell > buy + 1e-12):
        raise ValueError("lp_battery requires sell <= buy (else the split needs binaries)")

    soe0 = cap * cfg.soc_initial_frac if soe_initial is None else soe_initial

    # Layout: [c(0..n-1), d(0..n-1), gp(0..n-1), gn(0..n-1)]
    C, D, GP, GN = 0, n, 2 * n, 3 * n
    nvar = 4 * n

    # minimise  sum(buy*gp - sell*gn)*dt  -  terminal_price * SoE[n]
    obj = np.zeros(nvar)
    obj[GP : GP + n] = buy * dt
    obj[GN : GN + n] = -sell * dt

    # SoE[n] = soe0 + sum(eta*c - d/eta)*dt, so the terminal term contributes
    # -tp*eta*dt to each c and +tp*dt/eta to each d (constant part dropped).
    tp = terminal_price(cfg, buy)
    obj[C : C + n] += -tp * eta_c * dt
    obj[D : D + n] += tp * dt / eta_d

    rows_eq, rhs_eq = [], []
    rows_ub, rhs_ub = [], []

    for t in range(n):
        # Meter identity: gp - gn = c - d + dp_load
        r = np.zeros(nvar)
        r[GP + t], r[GN + t] = 1.0, -1.0
        r[C + t], r[D + t] = -1.0, 1.0
        rows_eq.append(r)
        rhs_eq.append(dp_load[t])

        # 0 <= SoE[t+1] <= cap, as cumulative sums
        r = np.zeros(nvar)
        r[C : C + t + 1] = eta_c * dt
        r[D : D + t + 1] = -dt / eta_d
        rows_ub.append(r.copy())
        rhs_ub.append(cap - soe0)
        rows_ub.append(-r)
        rhs_ub.append(soe0)

    bounds = (
        [(0.0, cfg.p_charge_max_kw)] * n
        + [(0.0, cfg.p_discharge_max_kw)] * n
        + [(0.0, None)] * n
        + [(0.0, None)] * n
    )

    res = linprog(
        obj,
        A_ub=np.array(rows_ub),
        b_ub=np.array(rhs_ub),
        A_eq=np.array(rows_eq),
        b_eq=np.array(rhs_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    c = res.x[C : C + n]
    d = res.x[D : D + n]
    power = c - d
    soe = np.concatenate([[soe0], soe0 + np.cumsum((c * eta_c - d / eta_d) * dt)])
    net = power + dp_load
    bill = float(np.sum(np.maximum(net, 0) * buy * dt) - np.sum(np.maximum(-net, 0) * sell * dt))

    return {
        "power": power,
        "soe": soe,
        "bill": bill,
        # Objective net of terminal value, so it is comparable to the DP's.
        "objective": bill - tp * (soe[-1] - soe0),
        "status": res.message,
    }


# --------------------------------------------------------------------------
# 2. Exact joint DP  (isolates decomposition)
# --------------------------------------------------------------------------


def joint_dp_battery_water_heater(
    batt: BatteryConfig,
    wh: WaterHeaterConfig,
    horizon: Horizon,
    buy: np.ndarray,
    sell: np.ndarray,
    hot_water_demand: np.ndarray,
    fixed_load: np.ndarray,
) -> dict:
    """Exact optimum over the JOINT (SoE, tank temperature) state space.

    Same physics, same costs, same grids as the decomposed solvers - the only
    difference is that the two devices are optimised together instead of being
    coordinated by a price signal. The gap between this and `coordinate()` is
    therefore purely the decomposition heuristic's cost.

    Size: n_batt_states * n_wh_states * n_steps * n_batt_actions * 2.
    At the defaults that is 50*40*96*21*2 = 8.1M transitions - slow in numpy
    but exact, which is the point.
    """
    n = horizon.steps
    dt = horizon.dt
    cap = batt.capacity_kwh
    eta_c, eta_d = batt.eta_c, batt.eta_d
    C = wh.heat_capacity_kwh_per_k

    S = np.linspace(0.0, cap, batt.n_states)
    T = np.linspace(wh.t_min, wh.t_max, wh.n_states)
    A = np.linspace(-batt.p_discharge_max_kw, batt.p_charge_max_kw, batt.n_actions)
    ns, nt, na = batt.n_states, wh.n_states, batt.n_actions

    tp = terminal_price(batt, buy)
    ref_price = float(np.mean(buy))
    # V[s, t_idx]; terminal = battery linear value + tank comfort terminal.
    # MUST use the same discomfort function as the decomposed solver, or the
    # two are optimising different objectives and the "gap" is meaningless.
    if wh.comfort_mode == "linear":
        wh_terminal = wh.heat_capacity_kwh_per_k * ref_price * np.maximum(0.0, wh.t_comfort - T)
    else:
        wh_terminal = wh.terminal_weight * np.maximum(0.0, wh.t_comfort - T) ** 2
    V = tp * S[:, None] - wh_terminal[None, :]
    POL_A = np.zeros((n, ns, nt))
    POL_W = np.zeros((n, ns, nt), dtype=np.int64)
    # Keep every V[t] so the joint costate dV/ds can be read off afterwards.
    # 96 * 50 * 40 * 8 B ~ 1.5 MB, so there is no reason to be clever.
    V_ALL = np.zeros((n + 1, ns, nt))
    V_ALL[n] = V



    for step in range(n - 1, -1, -1):
        best = np.full((ns, nt), -np.inf)
        q_out = _usable_outflow(T, wh, hot_water_demand[step], dt)

        # Battery transition (depends on s only)
        # Bounds carry the efficiencies, matching home_energy_optimizer.dp_battery: without
        # them a nearly empty store discharges past zero and the projection
        # silently creates energy, which would make this "exact" reference
        # optimistic by the same amount as the solver it is checking.
        Ac = np.clip(
            A[None, :], -S[:, None] * eta_d / dt, (cap - S[:, None]) / (eta_c * dt)
        )  # (ns, na)
        effA = np.where(Ac > 0, Ac * eta_c, Ac / eta_d)
        s_next = np.clip(S[:, None] + effA * dt, 0.0, cap)
        s_idx, s_w = interp_uniform(S, 0.0, cap, ns, s_next)

        for wi, wa in enumerate(WATER_HEATER_ACTIONS):
            # Tank transition (depends on t_idx only). Uses the SAME bounded
            # outflow and cut-out-limited duty as home_energy_optimizer.dp_thermal - a
            # reference solving different physics is not a reference.
            duty = np.minimum(wa, _max_duty(T, wh, q_out, dt))
            dT = (wh.power_kw * duty - q_out) / C * dt
            t_next = T + dT
            t_idx, t_w = interp_uniform(T, wh.t_min, wh.t_max, nt, t_next)
            comfort = -wh_discomfort(wh, t_next, ref_price, dt)

            # Bilinear interpolation of V over the product grid.
            # v_next[s, a, t] via (s_idx, s_w) on axis 0 and (t_idx, t_w) on axis 1.
            v00 = V[s_idx[:, :, None], t_idx[None, None, :]]
            v01 = V[s_idx[:, :, None], np.minimum(t_idx + 1, nt - 1)[None, None, :]]
            v10 = V[np.minimum(s_idx + 1, ns - 1)[:, :, None], t_idx[None, None, :]]
            v11 = V[
                np.minimum(s_idx + 1, ns - 1)[:, :, None],
                np.minimum(t_idx + 1, nt - 1)[None, None, :],
            ]
            sw = s_w[:, :, None]
            tw = t_w[None, None, :]
            v_next = (
                v00 * (1 - sw) * (1 - tw)
                + v01 * (1 - sw) * tw
                + v10 * sw * (1 - tw)
                + v11 * sw * tw
            )  # (ns, na, nt)

            imp = Ac[:, :, None] + wh.power_kw * wa + fixed_load[step]
            reward = (
                -buy[step] * np.maximum(imp, 0.0) + sell[step] * np.maximum(-imp, 0.0)
            ) * dt

            Q = reward + comfort[None, None, :] + v_next  # (ns, na, nt)
            best_a = np.argmax(Q, axis=1)  # (ns, nt)
            q_best = np.take_along_axis(Q, best_a[:, None, :], axis=1)[:, 0, :]

            improve = q_best > best
            best = np.where(improve, q_best, best)
            POL_W[step] = np.where(improve, wi, POL_W[step])
            POL_A[step] = np.where(
                improve, np.take_along_axis(Ac[:, :, None], best_a[:, None, :], axis=1)[:, 0, :], POL_A[step]
            )
        V = best
        V_ALL[step] = V

    # Forward rollout
    soe = np.zeros(n + 1)
    temp = np.zeros(n + 1)
    p_batt = np.zeros(n)
    p_wh = np.zeros(n)
    soe[0] = cap * batt.soc_initial_frac
    temp[0] = wh.t_comfort

    for step in range(n):
        si = int(np.clip(round(soe[step] / cap * (ns - 1)), 0, ns - 1))
        ti = int(
            np.clip(round((temp[step] - wh.t_min) / (wh.t_max - wh.t_min) * (nt - 1)), 0, nt - 1)
        )
        a = float(POL_A[step, si, ti])
        wa = float(WATER_HEATER_ACTIONS[POL_W[step, si, ti]])
        p_batt[step] = a
        p_wh[step] = wh.power_kw * wa

        eff = a * eta_c if a > 0 else a / eta_d
        soe[step + 1] = float(np.clip(soe[step] + eff * dt, 0.0, cap))
        q_out_i = float(_usable_outflow(temp[step], wh, hot_water_demand[step], dt))
        duty_i = float(np.minimum(wa, _max_duty(temp[step], wh, q_out_i, dt)))
        p_wh[step] = wh.power_kw * duty_i
        temp[step + 1] = float(temp[step] + (wh.power_kw * duty_i - q_out_i) / C * dt)

    net = p_batt + p_wh + fixed_load
    bill = float(np.sum(np.maximum(net, 0) * buy * dt) - np.sum(np.maximum(-net, 0) * sell * dt))
    comfort_pen = float(np.sum(wh_discomfort(wh, temp[1:], ref_price, dt)))
    return {
        "value": V_ALL,
        "states": S,
        "temps": T,
        "p_batt": p_batt,
        "p_wh": p_wh,
        "soe": soe,
        "temp": temp,
        "bill": bill,
        "comfort_penalty": comfort_pen,
        "objective": bill + comfort_pen + tp * (soe[0] - soe[-1]),
    }
