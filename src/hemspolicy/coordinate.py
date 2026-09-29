"""ADMM coordination of the devices, and the pieces both coordinators share.

`coordinate()` plans a site by ADMM - proximal message passing, in the
`admm` package beside this one (admm/coordinator.py). This module also holds what every planner uses to score
and finish a plan: the meter's curtailment rule and cost, the comfort, breach
and battery end-value terms of the objective, the thermostat baseline, the
baseline fallback, the polish (Gauss-Seidel best responses, kept only if the
objective falls) and the battery's pricing re-solve for the execution tier.

ADMM carries no optimality certificate; the Dantzig-Wolfe coordinator's bound
certifies its plans (docs/theory.tex).
"""

from __future__ import annotations

import numpy as np

from .dp_battery import solve_battery, terminal_price
from .meter import Limits
from .dp_thermal import (
    baseline_hvac,
    baseline_water_heater,
    hvac_discomfort,
    solve_hvac,
    solve_water_heater,
    wh_discomfort,
)
from .types import (
    CoordinationResult,
    DeviceSolution,
    Forecasts,
    SiteConfig,
)


def apply_curtailment(
    net_raw: np.ndarray, solar: np.ndarray, sell: np.ndarray, grid
) -> tuple[np.ndarray, np.ndarray]:
    """Throw away the PV that cannot usefully leave the site.

    Returns (net_grid_after, curtailment). Curtailment reduces export 1:1 and
    is bounded by the PV actually available, so this is exact rather than an
    approximation - there is no interaction with the devices to iterate over.

    Two reasons to curtail, and only two:

    * **an export cap** that storage cannot absorb. Once the battery is full
      the surplus has nowhere to go, and no price can make it vanish - this is
      the only lever left.
    * **a negative export price**, where being paid to stop generating beats
      paying someone to take the energy.

    Deliberately applied AFTER the device solves rather than inside them: it
    only ever discards energy that was already going to be exported, so it
    cannot change what a device should have done. That keeps it out of the
    coordination loop entirely.
    """
    curtail = np.zeros_like(net_raw)
    if grid is None or not grid.allow_curtailment:
        return net_raw, curtail

    export = np.maximum(-net_raw, 0.0)
    if grid.max_export_kw is not None:
        curtail = np.minimum(solar, np.maximum(export - grid.max_export_kw, 0.0))
    # Never pay to export.
    negative_price = sell < 0
    if negative_price.any():
        curtail = np.where(
            negative_price, np.minimum(solar, export), curtail
        )
    return net_raw + curtail, curtail


def device_sell_price(sell: np.ndarray, grid) -> np.ndarray:
    """The export price a DEVICE should be optimising against.

    Curtailment is free and always available up to the PV on the roof, so a
    negative export price is never actually paid on PV-driven surplus - you
    stop generating instead. A device priced at the raw negative tariff sees a
    phantom benefit for absorbing that surplus, and will spend round-trip
    losses to capture something curtailment delivers for nothing.

    Measured on a day/night fixture with a -0.10 midday export price: pricing
    devices at the raw tariff costs 0.31 currency/day and 7.3 kWh of extra
    battery throughput against pricing them at max(sell, 0). The gap scales
    linearly with how negative the price is, and is exactly zero for sell >= 0,
    so this is a no-op on every non-negative tariff.

    The BILL is still computed on the true tariff - this changes what devices
    optimise, not what the site is charged.
    """
    if grid is None or not grid.allow_curtailment:
        return sell
    return np.maximum(sell, 0.0)


def net_cost(net_grid: np.ndarray, buy: np.ndarray, sell: np.ndarray, dt: float) -> float:
    """Bill for a net grid exchange series. Positive net_grid = import."""
    imp = float(np.sum(np.maximum(net_grid, 0.0) * buy * dt))
    exp = float(np.sum(np.maximum(-net_grid, 0.0) * sell * dt))
    return imp - exp


def comfort_penalty(
    cfg: SiteConfig,
    wh_temp: np.ndarray | None,
    hvac_temp: np.ndarray | None,
    reference_price: float = 0.0,
) -> float:
    """Total discomfort cost. In linear mode this is genuinely in currency, so
    adding it to a bill is meaningful rather than an arbitrary exchange rate.

    Must stay in lockstep with the penalties inside the DPs - they are the same
    two functions, deliberately shared rather than reimplemented.
    """
    dt = cfg.horizon.dt
    total = 0.0
    if cfg.water_heater is not None and wh_temp is not None:
        total += float(np.sum(wh_discomfort(cfg.water_heater, wh_temp[1:], reference_price, dt)))
    if cfg.hvac is not None and hvac_temp is not None:
        total += float(np.sum(hvac_discomfort(cfg.hvac, hvac_temp[1:], reference_price, dt)))
    return total


def breach_energy(cfg: SiteConfig, net_grid: np.ndarray) -> np.ndarray:
    """Energy drawn or pushed beyond the grid limits, per slot, in kWh.

    Zero everywhere when no limit is configured, which is why every benchmark
    objective is untouched by the penalty built on it.
    """
    over = np.zeros_like(net_grid)
    if cfg.grid.max_import_kw is not None:
        over = over + np.maximum(net_grid - cfg.grid.max_import_kw, 0.0)
    if cfg.grid.max_export_kw is not None:
        over = over + np.maximum(-net_grid - cfg.grid.max_export_kw, 0.0)
    return over * cfg.horizon.dt


def breach_price(grid, buy: np.ndarray, sell: np.ndarray | None = None) -> float:
    """Currency per kWh beyond a grid limit: one constant for the horizon.

    `grid.breach_price` if set, else `breach_price_multiplier` times the
    largest |price| on the horizon (1.0 if every price is zero).
    """
    if grid.breach_price is not None:
        return float(grid.breach_price)
    prices = np.abs(np.asarray(buy, dtype=float))
    if sell is not None:
        prices = np.concatenate([prices, np.abs(np.asarray(sell, dtype=float))])
    scale = float(prices.max()) if prices.size and prices.max() > 0 else 1.0
    return grid.breach_price_multiplier * scale


def grid_penalty(cfg: SiteConfig, net_grid: np.ndarray, buy: np.ndarray,
                 sell: np.ndarray | None = None) -> float:
    """Cost of breaching the grid limits, in currency.

    Over-limit energy is priced at `breach_price` - a constant, not the slot's
    tariff. A grid limit is usually a fuse or a regulatory cap; it does not
    matter less at a cheap hour, and a tariff-scaled penalty vanishes at a zero
    import price and turns into a reward at a negative one.

    Putting the breach in the objective is what lets round selection be a
    single comparison on money. The alternative - rank on violation first and
    cost second - never lets the two trade at all, which sounds safer but in
    practice decides on the eighth decimal place of a breach that is
    infeasible either way.

    This does NOT enforce anything. The coordinators steer by pricing the
    limit the same way (Dantzig-Wolfe's master, ADMM's grid connection), and
    anything safety-critical is clamped outside the solver entirely
    (policy.clamp). This term scores a finished plan; it does not produce one.
    """
    if not cfg.grid.active:
        return 0.0
    price = breach_price(cfg.grid, buy, sell)
    if price <= 0:
        return 0.0
    return float(np.sum(breach_energy(cfg, net_grid)) * price)


def battery_terminal_penalty(
    cfg: SiteConfig,
    soe: np.ndarray | dict[str, np.ndarray] | None,
    buy: np.ndarray | None = None,
) -> float:
    """Cost of ending the horizon with less energy than you started with.

    In linear mode this bills the net depletion at the terminal price, which is
    what stops a plan from looking cheap simply by emptying the battery. In
    quadratic mode it reproduces the POC's pull toward a target SoC.

    Accepts either a single trajectory (battery 0, the common case) or a
    key -> trajectory mapping, in which case every battery is charged.
    """
    if soe is None or not cfg.battery_list:
        return 0.0

    if not isinstance(soe, dict):
        soe = {SiteConfig.battery_key(0): soe}

    total = 0.0
    for i, b in enumerate(cfg.battery_list):
        traj = soe.get(SiteConfig.battery_key(i))
        if traj is None:
            continue
        if b.terminal_mode == "linear":
            if buy is None:
                continue
            total += float(traj[0] - traj[-1]) * terminal_price(b, buy)
        else:
            target = b.capacity_kwh * b.terminal_target_frac
            total += float(
                b.terminal_weight * (traj[-1] - target) ** 2 / max(b.capacity_kwh, 0.01)
            )
    return total


def total_objective(
    cfg: SiteConfig,
    net_grid: np.ndarray,
    fc: Forecasts,
    wh_temp: np.ndarray | None,
    hvac_temp: np.ndarray | None,
    soe: np.ndarray | dict[str, np.ndarray] | None,
) -> float:
    """Bill + comfort + grid-breach penalties + battery terminal penalty.

    Lower is better, and every term is in currency, so the sum is a money
    number that can be compared against another solver's objective. Used for
    round selection, which is therefore a single comparison: the coordinator
    cannot trade a comfort violation or a grid breach for a smaller bill,
    because both already cost money here.

    The grid term is identically zero with no limit configured, so the
    benchmark objectives are exactly the bill-plus-comfort they always were.
    """
    return (
        net_cost(net_grid, fc.buy, fc.sell, cfg.horizon.dt)
        + comfort_penalty(cfg, wh_temp, hvac_temp, float(np.mean(fc.buy)))
        + grid_penalty(cfg, net_grid, fc.buy, fc.sell)
        + battery_terminal_penalty(cfg, soe, fc.buy)
    )


def baseline_solution(cfg: SiteConfig, fc: Forecasts) -> tuple[np.ndarray, float]:
    """What the house does with no optimization: thermostats, idle battery.

    Curtailed on the same terms as the optimised plan. A baseline that keeps
    paying to export when curtailment is available is not the right comparison -
    it would flatter the optimiser by an amount that has nothing to do with
    scheduling.
    """
    net = fc.net_fixed_demand.copy()
    if cfg.water_heater is not None:
        net = net + baseline_water_heater(cfg.water_heater, cfg.horizon, fc.hot_water_demand)[1]
    if cfg.hvac is not None:
        net = net + baseline_hvac(cfg.hvac, cfg.horizon, fc.outdoor_temp)[1]
    net, _ = apply_curtailment(net, fc.solar, fc.sell, cfg.grid)
    return net, net_cost(net, fc.buy, fc.sell, cfg.horizon.dt)


def coordinate(cfg: SiteConfig, fc: Forecasts, progress=None, warm=None) -> CoordinationResult:
    """Plan a site by ADMM (admm.coordinator) and return its best plan.

    `progress`, if given, is called as progress(round, max_rounds) when an
    iteration starts, progress(round, max_rounds, objective, None, best) when it
    has been scored (None: ADMM has no bound), and progress(-1, max_rounds)
    before the fallback and polish - the same shape the DW app reads. `warm`:
    an earlier result's `warm_start` (see admm.coordinator.WarmStart).
    """
    from admm.coordinator import coordinate_exchange      # it imports this module
    return coordinate_exchange(cfg, fc, progress, warm)


def _polish(cfg: SiteConfig, fc: Forecasts, res: CoordinationResult, dev_keys: list[str],
            buy: np.ndarray, sell: np.ndarray, sweeps: int, limits: Limits | None = None) -> bool:
    """Gauss-Seidel best response from the returned plan; keep only improvements.

    The devices re-plan one at a time, each against the others' CURRENT plans
    (at `buy`/`sell`, with the grid limits priced), and a re-plan is kept only
    if `total_objective` falls. So it is monotone: it cannot undo what the
    iterations found. Updates `res` in place; returns whether anything changed.
    """
    h, dt = cfg.horizon, cfg.horizon.dt
    batt = {SiteConfig.battery_key(i): b for i, b in enumerate(cfg.battery_list)}

    def objective(devices) -> tuple[float, np.ndarray, np.ndarray]:
        net = fc.net_fixed_demand + sum(devices[k].power for k in dev_keys)
        net, curtail = apply_curtailment(net, fc.solar, fc.sell, cfg.grid)
        soe = {k: devices[k].trajectory for k in dev_keys if k in batt} or None
        obj = total_objective(cfg, net, fc,
                              devices["water_heater"].trajectory if "water_heater" in devices else None,
                              devices["hvac"].trajectory if "hvac" in devices else None, soe)
        return obj, net, curtail

    cur, net, curtail = objective(res.devices)
    changed = False
    for _ in range(max(sweeps, 0)):
        improved = False
        for k in dev_keys:
            others = fc.net_fixed_demand + sum(res.devices[j].power for j in dev_keys if j != k)
            if k in batt:
                sol = solve_battery(batt[k], h, buy, sell, dp_load=others, admm_rho=0.0,
                                    soc_gates=cfg.soc_gates if k == "battery" else (), limits=limits)
            elif k == "water_heater":
                sol = solve_water_heater(cfg.water_heater, h, buy, sell, fc.hot_water_demand,
                                         dp_load=others, limits=limits)
            else:
                sol = solve_hvac(cfg.hvac, h, buy, sell, fc.outdoor_temp, dp_load=others,
                                 limits=limits)
            trial = dict(res.devices)
            trial[k] = sol
            obj, t_net, t_curtail = objective(trial)
            if obj < cur - 1e-9:
                res.devices, cur, net, curtail = trial, obj, t_net, t_curtail
                improved = changed = True
        if not improved:
            break
    if not changed:
        return False
    res.net_grid, res.curtailment = net, curtail
    res.curtailed_kwh = float(curtail.sum() * dt)
    res.import_cost = float(np.sum(np.maximum(net, 0.0) * fc.buy * dt))
    res.export_revenue = float(np.sum(np.maximum(-net, 0.0) * fc.sell * dt))
    res.net_cost = res.import_cost - res.export_revenue
    res.total_objective = cur
    if cfg.grid.max_import_kw is not None:
        res.grid_import_excess = float(np.maximum(net - cfg.grid.max_import_kw, 0.0).max())
    if cfg.grid.max_export_kw is not None:
        res.grid_export_excess = float(np.maximum(-net - cfg.grid.max_export_kw, 0.0).max())
    if "battery" in res.devices:
        # the pricing re-solve is conditioned on the others' FINAL plans
        res.battery_dp_load = fc.net_fixed_demand + sum(
            res.devices[j].power for j in dev_keys if j != "battery")
    return True


def _pricing_resolve(cfg: SiteConfig, fc: Forecasts, res: CoordinationResult,
                     limits: Limits | None = None) -> DeviceSolution:
    """Re-solve the battery on pure economics, holding the other devices fixed.

    A coordinator's own device solves carry its machinery - ADMM's proximal
    term -(rho/2)(a - target)^2, or Dantzig-Wolfe's master prices instead of
    the tariff - and differentiating such a value function gives a "price"
    polluted by it. So the pricing/execution tier gets its own solve: the
    battery at the tariff, against the other devices' planned powers, with no
    tether. dV/ds is then an honest marginal value of stored energy,
    CONDITIONAL on those plans - which is exactly what the price signal claims
    to be, and no more.
    """
    dp_load = (
        res.battery_dp_load
        if res.battery_dp_load is not None
        else fc.net_fixed_demand
    )
    return solve_battery(
        cfg.battery,
        cfg.horizon,
        fc.buy,
        device_sell_price(fc.sell, cfg.grid),
        dp_load=dp_load,
        admm_rho=0.0,
        soc_gates=cfg.soc_gates,
        limits=limits,
    )


def _apply_baseline_fallback(
    cfg: SiteConfig, fc: Forecasts, res: CoordinationResult, dev_keys: list[str]
) -> bool:
    """Safety net: if a device's thermostat baseline beats its optimised plan
    on TOTAL objective, keep the baseline.

    Compares the objective rather than the bill so we never buy a smaller bill
    with a comfort violation. Iterates because swapping one device changes the
    net grid the others are scored against.

    Returns whether any device was actually swapped, so the caller can tell a
    retained plan that survived the check from one that was rewritten by it.
    """
    n = cfg.horizon.steps
    dt = cfg.horizon.dt

    batt_keys = [k for k in dev_keys if k.startswith("battery")]

    def rebuild(overrides: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Net grid for a candidate swap, curtailed the same way a round is.

        Skipping curtailment here silently produced a result whose net_grid
        breached the export cap while grid_export_excess read 0.000, because the
        two were computed from different arrays.
        """
        net = fc.net_fixed_demand.copy()
        for k in dev_keys:
            net = net + overrides.get(k, res.devices[k].power)
        return apply_curtailment(net, fc.solar, fc.sell, cfg.grid)

    swapped = False
    changed = True
    while changed:
        changed = False
        soes = {k: res.devices[k].trajectory for k in batt_keys if k in res.devices}
        cur_obj = total_objective(
            cfg,
            res.net_grid,
            fc,
            res.devices["water_heater"].trajectory if "water_heater" in res.devices else None,
            res.devices["hvac"].trajectory if "hvac" in res.devices else None,
            soes or None,
        )

        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        for idx, key in enumerate(batt_keys):
            b = cfg.battery_list[idx]
            # An idle battery is only a valid fallback when nothing obliges it
            # to move: SoC gates and p_demand are enforced inside the DP, and
            # doing nothing would violate them.
            if (key == "battery" and cfg.soc_gates) or b.min_charge_kw or b.soc_goal_kwh:
                continue
            idle_soe = np.full(n + 1, b.capacity_kwh * b.soc_initial_frac)
            candidates.append((key, np.zeros(n), idle_soe))
        if "water_heater" in dev_keys:
            temp, pw = baseline_water_heater(cfg.water_heater, cfg.horizon, fc.hot_water_demand)
            candidates.append(("water_heater", pw, temp))
        if "hvac" in dev_keys:
            temp, pw = baseline_hvac(cfg.hvac, cfg.horizon, fc.outdoor_temp)
            candidates.append(("hvac", pw, temp))

        for key, alt_power, alt_traj in candidates:
            alt_net, alt_curtail = rebuild({key: alt_power})
            trajs = {k: res.devices[k].trajectory for k in res.devices}
            trajs[key] = alt_traj
            alt_obj = total_objective(
                cfg,
                alt_net,
                fc,
                trajs.get("water_heater"),
                trajs.get("hvac"),
                {k: trajs[k] for k in batt_keys if k in trajs} or None,
            )
            if alt_obj < cur_obj - 1e-12:
                res.devices[key].power = alt_power
                res.devices[key].trajectory = alt_traj
                res.net_grid = alt_net
                res.curtailment = alt_curtail
                res.curtailed_kwh = float(alt_curtail.sum() * dt)
                if cfg.grid.max_import_kw is not None:
                    res.grid_import_excess = float(
                        np.maximum(alt_net - cfg.grid.max_import_kw, 0.0).max()
                    )
                if cfg.grid.max_export_kw is not None:
                    res.grid_export_excess = float(
                        np.maximum(-alt_net - cfg.grid.max_export_kw, 0.0).max()
                    )
                res.import_cost = float(np.sum(np.maximum(alt_net, 0.0) * fc.buy * dt))
                res.export_revenue = float(np.sum(np.maximum(-alt_net, 0.0) * fc.sell * dt))
                res.net_cost = res.import_cost - res.export_revenue
                res.total_objective = alt_obj
                changed = swapped = True
                break

    return swapped
