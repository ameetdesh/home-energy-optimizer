"""A HiGHS MILP over the same model - the reference an EMHASS user already has.

`reference.py` answers "how close is the fast path to the exact optimum of ITS
own model". This answers a different and more practical question: **how does it
compare to what a mixed-integer solver would give you today?** That is the bar
EMHASS sets in Home Assistant, where CVXPY defaults hard to `cp.HIGHS`
(`optimization.py`: `selected_solver = cp.HIGHS`), and evcc's cloud optimizer is
MILP too (`core/optimizer.md`).

Deliberately built here rather than by calling EMHASS: EMHASS models different
devices with different constraints, so comparing against it would conflate
*model* differences with *algorithm* differences. This is the same model, solved
exactly.

**This only became expressible when comfort became linear.** A
`weight * degC^2` penalty is not MILP-representable; the currency-per-kelvin-hour
price (docs/NOTES.md 3a) is.

Formulation, battery + hot-water tank:

  vars/step   c, d >= 0 (kW)   gp, gn >= 0 (kW)   sh >= 0 (K)
              on in {0,1} (command)   u in [0,1] (realised duty)
  state       soe[0..n] (kWh)  T[0..n] (degC)
  soe[t+1] =  soe[t] + (eta_c*c - d/eta_d)*dt
  T[t+1]   =  T[t] - rate[t]*dt*(T[t] - t_inf[t]) + P*u*dt/C
  duty      :  u = min(on, alpha_bar(T[t])),
               alpha_bar(T) = ((t_max - T)*C/dt + q_out[t](T)) / P
  meter    :  gp - gn = c - d + P*u + fixed
  comfort  :  sh >= t_comfort - T[t+1]
  min         sum(buy*gp - sell*gn)*dt + price_K*sum(sh)*dt
              + C*ref*sh_term - p_terminal*soe[n]

The tank's *physics* is entirely linear here. Both heat-loss terms - the
standing loss to `t_ambient` and the enthalpy carried out by the draw to
`t_inlet` - are linear in T, so their sum is `C*rate[t]*(T - t_inf[t])` with
`rate` and `t_inf` exogenous (they depend only on the demand forecast), and
`alpha_bar` is linear in T as well. See docs/theory.tex, Appendix A. The one
nonlinearity left is the `min` that realises the command against the cut-out,
and it is modelled exactly, with one binary per slot on each side.

This matches the DP's model exactly: the element is commanded on or off and
realises `min(a, alpha_bar(T))`. Writing `u <= alpha_bar` alone would be a
relaxation - it would let the solver run the element for an arbitrary fraction
of a slot, which the DP cannot - and the benchmark would then be measuring the
relaxation as much as the decomposition.

Requires `sell <= buy` elementwise so the grid split stays exact without a
direction binary.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from hemspolicy.dp_battery import terminal_price
from hemspolicy.dp_thermal import _relaxation
from hemspolicy.types import BatteryConfig, Horizon, WaterHeaterConfig


def milp_battery_water_heater(
    batt: BatteryConfig,
    wh: WaterHeaterConfig,
    horizon: Horizon,
    buy: np.ndarray,
    sell: np.ndarray,
    hot_water_demand: np.ndarray,
    fixed_load: np.ndarray,
    time_limit: float = 120.0,
    mip_rel_gap: float = 1e-4,
) -> dict:
    """Exact MILP optimum for the coupled battery + tank problem, via HiGHS."""
    t_start = time.perf_counter()
    n = horizon.steps
    dt = horizon.dt
    cap = batt.capacity_kwh
    eta_c, eta_d = batt.eta_c, batt.eta_d
    C = wh.heat_capacity_kwh_per_k
    ref_price = float(np.mean(buy))
    price_k = wh.discomfort_price_per_kelvin_hour(ref_price)
    tp = terminal_price(batt, buy)

    if np.any(sell > buy + 1e-12):
        raise ValueError("milp_battery_water_heater requires sell <= buy elementwise")
    if wh.comfort_mode != "linear":
        raise ValueError("MILP reference requires linear comfort pricing")

    # Passive relaxation rate and mixing temperature, per slot. Both are
    # exogenous: they depend on the demand forecast, not on the decision.
    rate = np.zeros(n)
    t_inf = np.zeros(n)
    for t in range(n):
        rate[t], t_inf[t] = _relaxation(wh, float(hot_water_demand[t]))
    if np.any(rate * dt > 1.0):
        raise ValueError(
            "the tank's Euler rate cap is active on this forecast; the DP applies "
            "it and this linear model does not, so the two would not be comparable"
        )

    # Variable layout. Wv is the REALISED duty; ON is the binary command.
    Cv, Dv, GP, GN, Wv, SH = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
    ON, BSEL = 6 * n, 7 * n  # command, and the selector for the min
    SOE, TT = 8 * n, 8 * n + (n + 1)
    SHT = TT + (n + 1)  # terminal tank shortfall
    nvar = SHT + 1

    obj = np.zeros(nvar)
    obj[GP : GP + n] = buy * dt
    obj[GN : GN + n] = -sell * dt
    obj[SH : SH + n] = price_k * dt
    obj[SOE + n] = -tp
    obj[SHT] = C * ref_price

    integrality = np.zeros(nvar)
    integrality[ON : ON + n] = 1
    integrality[BSEL : BSEL + n] = 1

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[Cv : Cv + n] = batt.p_charge_max_kw
    ub[Dv : Dv + n] = batt.p_discharge_max_kw
    ub[Wv : Wv + n] = 1.0
    lb[SOE : SOE + n + 1], ub[SOE : SOE + n + 1] = 0.0, cap
    ub[ON : ON + n] = 1.0
    ub[BSEL : BSEL + n] = 1.0
    lb[TT : TT + n + 1], ub[TT : TT + n + 1] = wh.t_min, wh.t_max

    # Big-M for the duty min. alpha_bar is largest at the coldest state under
    # the heaviest draw; one unit of slack above that is ample, since the other
    # side of the min (`on`) never exceeds 1.
    alpha_max = float(
        np.max(
            ((wh.t_max - wh.t_min) * C / dt + C * rate * (wh.t_max - t_inf)) / wh.power_kw
        )
    )
    big_m = alpha_max + 1.0

    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    def add(row: np.ndarray, low: float, high: float) -> None:
        rows.append(row)
        lo.append(low)
        hi.append(high)

    # Initial states
    r = np.zeros(nvar); r[SOE] = 1.0
    add(r, cap * batt.soc_initial_frac, cap * batt.soc_initial_frac)
    r = np.zeros(nvar); r[TT] = 1.0
    add(r, wh.t_comfort, wh.t_comfort)

    for t in range(n):
        # Battery SoE recursion
        r = np.zeros(nvar)
        r[SOE + t + 1] = 1.0
        r[SOE + t] = -1.0
        r[Cv + t] = -eta_c * dt
        r[Dv + t] = dt / eta_d
        add(r, 0.0, 0.0)

        # Meter balance
        r = np.zeros(nvar)
        r[GP + t], r[GN + t] = 1.0, -1.0
        r[Cv + t], r[Dv + t] = -1.0, 1.0
        r[Wv + t] = -wh.power_kw
        add(r, fixed_load[t], fixed_load[t])

        # Tank dynamics.
        #   T[t+1] = T[t] - rate*dt*(T[t] - t_inf) + P*u*dt/C
        r = np.zeros(nvar)
        r[TT + t + 1] = 1.0
        r[TT + t] = -1.0 + rate[t] * dt
        r[Wv + t] = -wh.power_kw * dt / C
        const = rate[t] * dt * t_inf[t]
        add(r, const, const)

        # u = min(on, alpha_bar(T[t])), exactly.
        #   alpha_bar(T)*P = (t_max - T)*C/dt + C*rate*(T - t_inf)
        #   ->  alpha_bar*P + T*(C/dt - C*rate) = t_max*C/dt - C*rate*t_inf
        acoef = C / dt - C * rate[t]  # coefficient of T[t] moved to the left
        arhs = wh.t_max * C / dt - C * rate[t] * t_inf[t]  # = alpha_bar*P + acoef*T

        # u <= on
        r = np.zeros(nvar); r[Wv + t] = 1.0; r[ON + t] = -1.0
        add(r, -np.inf, 0.0)
        # u <= alpha_bar
        r = np.zeros(nvar); r[Wv + t] = wh.power_kw; r[TT + t] = acoef
        add(r, -np.inf, arhs)
        # u >= on - M*b
        r = np.zeros(nvar); r[Wv + t] = 1.0; r[ON + t] = -1.0; r[BSEL + t] = big_m
        add(r, 0.0, np.inf)
        # u >= alpha_bar - M*(1-b)
        r = np.zeros(nvar)
        r[Wv + t] = wh.power_kw; r[TT + t] = acoef; r[BSEL + t] = -wh.power_kw * big_m
        add(r, arhs - wh.power_kw * big_m, np.inf)

        # Comfort shortfall: sh >= t_comfort - T[t+1]
        r = np.zeros(nvar)
        r[SH + t] = 1.0
        r[TT + t + 1] = 1.0
        add(r, wh.t_comfort, np.inf)

    # Terminal tank shortfall
    r = np.zeros(nvar); r[SHT] = 1.0; r[TT + n] = 1.0
    add(r, wh.t_comfort, np.inf)

    A = np.array(rows)
    res = milp(
        c=obj,
        constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap},
    )
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")

    x = res.x
    p_batt = x[Cv : Cv + n] - x[Dv : Dv + n]
    p_wh = x[Wv : Wv + n] * wh.power_kw
    soe = x[SOE : SOE + n + 1]
    temp = x[TT : TT + n + 1]

    net = p_batt + p_wh + fixed_load
    bill = float(np.sum(np.maximum(net, 0) * buy * dt) - np.sum(np.maximum(-net, 0) * sell * dt))
    comfort = float(np.sum(price_k * np.maximum(0.0, wh.t_comfort - temp[1:]) * dt))

    return {
        "p_batt": p_batt,
        "p_wh": p_wh,
        "soe": soe,
        "temp": temp,
        "bill": bill,
        "comfort_penalty": comfort,
        # Same convention as coordinate.total_objective: bill + comfort +
        # battery depletion priced at the terminal rate.
        "objective": bill + comfort + tp * (soe[0] - soe[-1]),
        "status": res.message,
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        "solve_ms": (time.perf_counter() - t_start) * 1000.0,
    }
