"""Thermal device dynamic programs: hot-water tank and HVAC.

Both are binary/ternary devices over a 1-D temperature state. Comfort is a
soft quadratic penalty, so these devices trade money against discomfort rather
than treating the comfort band as a hard constraint.

**Change from the POC:** these now return V and POL. The POC's compiled WASM
path returned only ``(temperature, on)``, silently discarding two-thirds of the
value functions at the FFI boundary - which meant the counterfactual/marginal
value machinery only ever worked for the battery.
"""

from __future__ import annotations

import time

import numpy as np

from ._kernels import kernel_hvac, kernel_water_heater
from .interp import interp_uniform
from .meter import Limits, limit_cost
from .types import DeviceSolution, Horizon, HvacConfig, WaterHeaterConfig

# Action grids. Index into these with the integer stored in POL. The tank's is
# per config (WaterHeaterConfig.duty_actions); this is the on/off default.
WATER_HEATER_ACTIONS = np.array([0.0, 1.0])  # off, on
HVAC_ACTIONS = np.array([0.0, -1.0, 1.0])  # off, cool, heat


def wh_discomfort(cfg: WaterHeaterConfig, temp, ref_price: float, dt: float):
    """Cost of the tank sitting below its setpoint, in currency.

    Linear mode prices a kelvin-hour of shortfall at the cost of restoring it
    (see WaterHeaterConfig). Quadratic mode reproduces the POC's arbitrary
    degC^2 weight for parity.
    """
    short = np.maximum(0.0, cfg.t_comfort - temp)
    if cfg.comfort_mode == "linear":
        return cfg.discomfort_price_per_kelvin_hour(ref_price) * short * dt
    return cfg.comfort_weight * short**2 * dt


def hvac_discomfort(cfg: HvacConfig, temp, ref_price: float, dt: float):
    """Cost of the room sitting outside its comfort band, in currency."""
    too_hot = np.maximum(0.0, temp - cfg.t_comfort_high)
    too_cold = np.maximum(0.0, cfg.t_comfort_low - temp)
    if cfg.comfort_mode == "linear":
        return cfg.discomfort_price_per_kelvin_hour(ref_price) * (too_hot + too_cold) * dt
    return cfg.comfort_weight * (too_hot**2 + too_cold**2) * dt


def _draw_factor(temp: np.ndarray | float, cfg: WaterHeaterConfig) -> np.ndarray:
    """Enthalpy carried away per unit of demand, relative to demand at setpoint.

    Drawing water at temperature T and replacing it with mains water at
    t_inlet removes rho*cp*W*(T - t_inlet) from the tank, so the loss is
    linear in (T - t_inlet) - NOT in (T - t_ambient). The ambient air around
    the tank is what the insulation leaks to; the mains is what the tap
    replaces. They are different temperatures and they belong to different
    terms.

    `hot_water_demand` is carried in kW-at-setpoint, i.e. w = rho*cp*W*
    (t_comfort - t_inlet), so this factor is exactly (T - t_inlet) /
    (t_comfort - t_inlet) and equals 1 at the setpoint by construction. There
    is no cap: a hotter tank really does give up more per litre, and the
    energy actually available is bounded in `_usable_outflow` instead.
    """
    return (np.asarray(temp, dtype=float) - cfg.t_inlet) / (cfg.t_comfort - cfg.t_inlet)


def _relaxation(cfg: WaterHeaterConfig, demand_kw: float):
    """Rate constant and equilibrium of the tank's passive dynamics.

    With the element off the energy balance is

        C dT/dt = -(T - t_ambient)/R - w * (T - t_inlet)/(t_comfort - t_inlet)
                = -C (beta + gamma) (T - t_inf)

    so the tank relaxes exponentially at rate beta + gamma toward t_inf, the
    flow-weighted mixture of the air it leaks to and the water that refills
    it. With no draw t_inf is t_ambient; under heavy draw it tends to
    t_inlet, which is what a tank run flat actually settles at.

    Returns (rate [1/h], t_inf [degC]).
    """
    C = cfg.heat_capacity_kwh_per_k
    beta = 1.0 / (cfg.r_thermal * C)
    gamma = float(demand_kw) / (C * (cfg.t_comfort - cfg.t_inlet))
    rate = beta + gamma
    t_inf = (beta * cfg.t_ambient + gamma * cfg.t_inlet) / rate
    return rate, t_inf


def _usable_outflow(temp, cfg: WaterHeaterConfig, demand_kw: float, dt: float):
    """Heat the tank gives up this slot, in kW: standing loss AND draw.

    The demanded outflow is C * rate * (T - t_inf). An explicit Euler step at
    that rate overshoots t_inf whenever rate * dt > 1, so the rate is capped
    at 1/dt: the tank may relax all the way to the mixing temperature in one
    slot, never past it. That cap is what makes t_inf an exact floor without
    any projection on the state - the same move equation (2) makes for the
    battery, bounding the flow by what the state can supply rather than
    letting the flow run and clamping the state afterwards.

    The cap is inactive at any realistic tank and slot length (rate * dt is
    ~0.16 at the shipped fixtures); it is here so the bound is a property of
    the model rather than of the parameters.
    """
    rate, t_inf = _relaxation(cfg, demand_kw)
    return cfg.heat_capacity_kwh_per_k * min(rate, 1.0 / dt) * (
        np.asarray(temp, dtype=float) - t_inf
    )


def _max_duty(temp, cfg: WaterHeaterConfig, q_out, dt: float):
    """Largest duty fraction in [0, 1] that does not carry the tank past t_max.

    This is the thermostat cut-out, expressed where it belongs: in the ADMISSIBLE
    ACTION SET rather than as a clamp applied to the state afterwards. Over a
    fifteen-minute slot a thermostatic element cycles many times, so a duty
    fraction is the more faithful reading of "on" anyway.
    """
    headroom_kw = (cfg.t_max - temp) * cfg.heat_capacity_kwh_per_k / dt + q_out
    return np.clip(headroom_kw / cfg.power_kw, 0.0, 1.0)


def solve_water_heater(
    cfg: WaterHeaterConfig,
    horizon: Horizon,
    buy: np.ndarray,
    sell: np.ndarray,
    hot_water_demand: np.ndarray,
    dp_load: np.ndarray | None = None,
    limits: Limits | None = None,
    admm_target: np.ndarray | None = None,
    admm_rho: float = 0.0,
    ref_price: float | None = None,
) -> DeviceSolution:
    """Solve the hot-water tank DP. `limits`: see hemspolicy.meter.

    `admm_target`/`admm_rho`: an optional tether (admm_rho/2)(p - target)^2 dt
    on the element's power, as the battery DP has. The legacy ADMM loop leaves
    it off - the on/off element is non-convex, so it coordinates through
    `dp_load` alone - while textbook ADMM (hemspolicy.exchange) tethers every
    device, running this on a relaxed (fractional) element.
    `ref_price`: the price discomfort is valued at (default: the mean of `buy`),
    for callers whose `buy` is not the tariff.
    """
    t_start = time.perf_counter()
    n = horizon.steps
    dt = horizon.dt
    C = cfg.heat_capacity_kwh_per_k

    if dp_load is None:
        dp_load = np.zeros(n)

    ref_price = float(np.mean(buy)) if ref_price is None else float(ref_price)
    T = np.linspace(cfg.t_min, cfg.t_max, cfg.n_states)
    V = np.zeros((n + 1, cfg.n_states))
    POL = np.zeros((n, cfg.n_states), dtype=np.int64)

    # Terminal: a cold tank at the horizon edge still has to be reheated, so
    # price the shortfall at the restoration cost rather than an arbitrary
    # weight. (In quadratic mode, keep the POC's shape.)
    if cfg.comfort_mode == "linear":
        V[n] = -cfg.heat_capacity_kwh_per_k * ref_price * np.maximum(0.0, cfg.t_comfort - T)
    else:
        V[n] = -cfg.terminal_weight * np.maximum(0.0, cfg.t_comfort - T) ** 2

    # Per-slot relaxation constants, computed once here so the mixing-
    # temperature physics lives in _relaxation and not also in C.
    rate = np.empty(n)
    t_inf = np.empty(n)
    for _t in range(n):
        rate[_t], t_inf[_t] = _relaxation(cfg, float(hot_water_demand[_t]))
    rate = np.minimum(rate, 1.0 / dt)      # the Euler cap, applied once

    actions = cfg.duty_actions
    POL_f = np.zeros(POL.shape)
    # The compiled kernel is the on/off element only.
    tether = admm_target is not None and admm_rho > 0
    compiled = cfg.n_duty_levels == 2 and limits is None and not tether and kernel_water_heater(
        np.ascontiguousarray(buy, dtype=np.float64),
        np.ascontiguousarray(sell, dtype=np.float64),
        np.ascontiguousarray(dp_load, dtype=np.float64),
        rate, t_inf, np.ascontiguousarray(T, dtype=np.float64),
        V, POL_f, n, cfg.n_states,
        dt, C, cfg.power_kw, cfg.t_min, cfg.t_max, cfg.t_comfort,
        cfg.discomfort_price_per_kelvin_hour(ref_price)
        if cfg.comfort_mode == "linear" else 0.0,
    ) if cfg.comfort_mode == "linear" else False
    if compiled:
        # POL holds action INDICES; the kernel writes float64 because `long`
        # is 32-bit on wasm32 and 64-bit natively (see dp_kernels.pyx).
        POL[:] = POL_f.astype(np.int64)

    for t in (() if compiled else range(n - 1, -1, -1)):
        best = np.full(cfg.n_states, -np.inf)
        q_out = _usable_outflow(T, cfg, hot_water_demand[t], dt)
        duty_cap = _max_duty(T, cfg, q_out, dt)

        for ai, a in enumerate(actions):
            # The action set is state-dependent: the element may run only as
            # far as the cut-out allows from HERE. No projection follows.
            duty = np.minimum(a, duty_cap)
            q_heat = cfg.power_kw * duty
            dT = (q_heat - q_out) / C * dt
            T_next = T + dT

            # NOT extrapolated: with the flows bounded, T_next cannot leave
            # the grid, so extrapolation is unreachable here and would only
            # add an unvalidated linear guess on a path nothing takes.
            idx, w = interp_uniform(V[t + 1], cfg.t_min, cfg.t_max, cfg.n_states, T_next)
            V_next = V[t + 1, idx] * (1 - w) + V[t + 1, np.minimum(idx + 1, cfg.n_states - 1)] * w

            imp = q_heat + dp_load[t]
            cost = (
                -buy[t] * np.maximum(imp, 0.0) + sell[t] * np.maximum(-imp, 0.0)
                - limit_cost(imp, sell[t], limits)
            ) * dt
            # Priced on the TRUE next temperature. Under the old projection this
            # read the clamped value, so the shortfall saturated at
            # (t_comfort - t_min) however cold the tank actually got.
            comfort = -wh_discomfort(cfg, T_next, ref_price, dt)
            if tether:
                cost = cost - (admm_rho / 2.0) * (q_heat - admm_target[t]) ** 2 * dt

            Q = cost + comfort + V_next
            improve = Q > best
            best = np.where(improve, Q, best)
            POL[t] = np.where(improve, ai, POL[t])
        V[t] = best

    temp, on = rollout_water_heater(
        cfg, horizon, POL, hot_water_demand, start_step=0, start_temp=cfg.t_comfort
    )

    return DeviceSolution(
        trajectory=temp,
        power=on * cfg.power_kw,
        value=V,
        policy=POL,
        states=T,
        actions=actions,
        solve_ms=(time.perf_counter() - t_start) * 1000.0,
    )


def rollout_water_heater(
    cfg: WaterHeaterConfig,
    horizon: Horizon,
    POL: np.ndarray,
    hot_water_demand: np.ndarray,
    start_step: int,
    start_temp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward replay of a stored tank policy from any (t, temperature)."""
    n = horizon.steps
    dt = horizon.dt
    C = cfg.heat_capacity_kwh_per_k
    start_step = max(0, min(n, start_step))
    length = n - start_step

    temp = np.zeros(length + 1)
    on = np.zeros(length)
    temp[0] = float(start_temp)

    for i in range(length):
        t = start_step + i
        si = int(
            np.clip(
                round((temp[i] - cfg.t_min) / (cfg.t_max - cfg.t_min) * (cfg.n_states - 1)),
                0,
                cfg.n_states - 1,
            )
        )
        a = float(cfg.duty_actions[POL[t, si]])
        q_out_i = float(_usable_outflow(temp[i], cfg, hot_water_demand[t], dt))
        duty = float(np.minimum(a, _max_duty(temp[i], cfg, q_out_i, dt)))
        on[i] = duty
        q = cfg.power_kw * duty - q_out_i
        # No projection: the trajectory reports the temperature the tank has.
        temp[i + 1] = float(temp[i] + q / C * dt)

    return temp, on


def _hvac_heat_flow(a: float, cfg: HvacConfig) -> float:
    """Thermal power delivered to the room for action `a`.

    Cooling moves COP times the electrical power; heating moves COP+1 (the
    compressor work also ends up in the room). Faithful to the POC.
    """
    if a < 0:
        return cfg.power_kw * cfg.cop * a
    if a > 0:
        return cfg.power_kw * (cfg.cop + 1.0) * a
    return 0.0



def _hvac_duty_cap(temp, cfg: HvacConfig, q_wall, a: float, dt: float):
    """Fraction of the HVAC action admissible from `temp` without leaving the grid.

    The wall coupling drives the room toward outdoor and is exogenous; the
    ACTION is what could push the room past the modelled range, so the action
    is what gets bounded. Same move as the tank's duty cap and equation (2)'s
    storage action set: constrain the control, do not clamp the state.
    """
    if a == 0.0:
        return np.ones_like(np.asarray(temp, dtype=float))
    q_ac = _hvac_heat_flow(a, cfg)
    limit = cfg.t_max if a > 0 else cfg.t_min
    room_kw = (limit - temp) * cfg.c_room_kwh_per_k / dt - q_wall
    return np.clip(room_kw / q_ac, 0.0, 1.0)


def solve_hvac(
    cfg: HvacConfig,
    horizon: Horizon,
    buy: np.ndarray,
    sell: np.ndarray,
    outdoor_temp: np.ndarray,
    dp_load: np.ndarray | None = None,
    limits: Limits | None = None,
    admm_target: np.ndarray | None = None,
    admm_rho: float = 0.0,
    ref_price: float | None = None,
) -> DeviceSolution:
    """Solve the HVAC DP. Ternary action, soft two-sided comfort band.
    `limits`: see hemspolicy.meter. `admm_target`/`admm_rho`: an optional
    tether on its power, as for the tank. `ref_price`: as for the tank."""
    t_start = time.perf_counter()
    n = horizon.steps
    dt = horizon.dt

    if dp_load is None:
        dp_load = np.zeros(n)

    ref_price = float(np.mean(buy)) if ref_price is None else float(ref_price)
    T = np.linspace(cfg.t_min, cfg.t_max, cfg.n_states)
    V = np.zeros((n + 1, cfg.n_states))
    POL = np.zeros((n, cfg.n_states), dtype=np.int64)

    if cfg.comfort_mode == "linear":
        V[n] = -hvac_discomfort(cfg, T, ref_price, 1.0)
    else:
        V[n] = -cfg.terminal_weight * (T - cfg.t_comfort_mid) ** 2

    actions = cfg.duty_actions
    POL_f = np.zeros(POL.shape)
    # The compiled kernel is the three-way (off / full cool / full heat) unit only.
    tether = admm_target is not None and admm_rho > 0
    compiled = limits is None and cfg.n_duty_levels == 2 and not tether and kernel_hvac(
        np.ascontiguousarray(buy, dtype=np.float64),
        np.ascontiguousarray(sell, dtype=np.float64),
        np.ascontiguousarray(dp_load, dtype=np.float64),
        np.ascontiguousarray(outdoor_temp, dtype=np.float64),
        np.ascontiguousarray(T, dtype=np.float64),
        V, POL_f, n, cfg.n_states,
        dt, cfg.c_room_kwh_per_k, cfg.r_wall_k_per_kw, cfg.power_kw, cfg.cop,
        cfg.t_min, cfg.t_max, cfg.t_comfort_low, cfg.t_comfort_high,
        cfg.discomfort_price_per_kelvin_hour(ref_price)
        if cfg.comfort_mode == "linear" else 0.0,
    ) if cfg.comfort_mode == "linear" else False
    if compiled:
        # POL holds action INDICES; the kernel writes float64 because `long`
        # is 32-bit on wasm32 and 64-bit natively (see dp_kernels.pyx).
        POL[:] = POL_f.astype(np.int64)

    for t in (() if compiled else range(n - 1, -1, -1)):
        best = np.full(cfg.n_states, -np.inf)
        q_wall = (outdoor_temp[t] - T) / cfg.r_wall_k_per_kw

        for ai, a in enumerate(actions):
            duty = _hvac_duty_cap(T, cfg, q_wall, float(a), dt)
            q_ac = _hvac_heat_flow(float(a), cfg) * duty
            dT = (q_wall + q_ac) / cfg.c_room_kwh_per_k * dt
            T_next = T + dT

            # NOT extrapolated: with the flows bounded, T_next cannot leave
            # the grid, so extrapolation is unreachable here and would only
            # add an unvalidated linear guess on a path nothing takes.
            idx, w = interp_uniform(V[t + 1], cfg.t_min, cfg.t_max, cfg.n_states, T_next)
            V_next = V[t + 1, idx] * (1 - w) + V[t + 1, np.minimum(idx + 1, cfg.n_states - 1)] * w

            imp = cfg.power_kw * abs(a) * duty + dp_load[t]
            elec = (
                -buy[t] * np.maximum(imp, 0.0) + sell[t] * np.maximum(-imp, 0.0)
                - limit_cost(imp, sell[t], limits)
            ) * dt
            comfort = -hvac_discomfort(cfg, T_next, ref_price, dt)
            if tether:
                elec = elec - (admm_rho / 2.0) * (cfg.power_kw * abs(a) * duty - admm_target[t]) ** 2 * dt

            Q = elec + comfort + V_next
            improve = Q > best
            best = np.where(improve, Q, best)
            POL[t] = np.where(improve, ai, POL[t])
        V[t] = best

    temp, power = rollout_hvac(
        cfg, horizon, POL, outdoor_temp, start_step=0, start_temp=cfg.t_comfort_mid
    )

    return DeviceSolution(
        trajectory=temp,
        power=power,
        value=V,
        policy=POL,
        states=T,
        actions=actions,
        solve_ms=(time.perf_counter() - t_start) * 1000.0,
    )


def rollout_hvac(
    cfg: HvacConfig,
    horizon: Horizon,
    POL: np.ndarray,
    outdoor_temp: np.ndarray,
    start_step: int,
    start_temp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward replay of a stored HVAC policy from any (t, room temperature)."""
    n = horizon.steps
    dt = horizon.dt
    start_step = max(0, min(n, start_step))
    length = n - start_step

    temp = np.zeros(length + 1)
    power = np.zeros(length)
    temp[0] = float(start_temp)

    for i in range(length):
        t = start_step + i
        si = int(
            np.clip(
                round((temp[i] - cfg.t_min) / (cfg.t_max - cfg.t_min) * (cfg.n_states - 1)),
                0,
                cfg.n_states - 1,
            )
        )
        a = float(cfg.duty_actions[POL[t, si]])
        q_wall = (outdoor_temp[t] - temp[i]) / cfg.r_wall_k_per_kw
        duty = float(_hvac_duty_cap(temp[i], cfg, q_wall, a, dt))
        power[i] = cfg.power_kw * abs(a) * duty
        q_ac = _hvac_heat_flow(a, cfg) * duty
        # No projection: the trajectory reports the temperature the room has.
        temp[i + 1] = float(temp[i] + (q_wall + q_ac) / cfg.c_room_kwh_per_k * dt)

    return temp, power


# --------------------------------------------------------------------------
# Thermostat baselines (what these devices do with no optimization at all)
# --------------------------------------------------------------------------


def baseline_water_heater(
    cfg: WaterHeaterConfig, horizon: Horizon, hot_water_demand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bang-bang thermostat: heat whenever below the comfort setpoint."""
    n = horizon.steps
    dt = horizon.dt
    C = cfg.heat_capacity_kwh_per_k
    temp = np.zeros(n + 1)
    power = np.zeros(n)
    temp[0] = cfg.t_comfort

    for t in range(n):
        on = 1.0 if temp[t] < cfg.t_comfort else 0.0
        q_out = float(_usable_outflow(temp[t], cfg, hot_water_demand[t], dt))
        # The thermostat's own cut-out is the only ceiling the baseline has,
        # and it is the same action bound the DP uses. No state clamp.
        duty = float(np.minimum(on, _max_duty(temp[t], cfg, q_out, dt)))
        power[t] = cfg.power_kw * duty
        q = cfg.power_kw * duty - q_out
        temp[t + 1] = float(temp[t] + q / C * dt)

    return temp, power


def baseline_hvac(
    cfg: HvacConfig, horizon: Horizon, outdoor_temp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Deadband thermostat: cool above the band, heat below it, else off."""
    n = horizon.steps
    dt = horizon.dt
    temp = np.zeros(n + 1)
    power = np.zeros(n)
    temp[0] = cfg.t_comfort_mid

    for t in range(n):
        if temp[t] > cfg.t_comfort_high:
            mode = -1.0
        elif temp[t] < cfg.t_comfort_low:
            mode = 1.0
        else:
            mode = 0.0
        power[t] = cfg.power_kw * abs(mode)
        q_wall = (outdoor_temp[t] - temp[t]) / cfg.r_wall_k_per_kw
        q_ac = _hvac_heat_flow(mode, cfg)
        temp[t + 1] = float(
            np.clip(temp[t] + (q_wall + q_ac) / cfg.c_room_kwh_per_k * dt, cfg.t_min, cfg.t_max)
        )

    return temp, power
