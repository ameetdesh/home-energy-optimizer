"""Dantzig-Wolfe as a drop-in planner for Home Assistant and evcc.

Both integrations consume a `CoordinationResult`: evcc reads the plan, Home
Assistant also builds a `PolicySnapshot` from it (a battery value function, so
the fast tier can act from whatever state the battery is really in). This
module runs the DW coordinator and returns exactly that object, so neither
integration has to know which coordinator planned the day.

Two things differ from the ADMM path, and both are additions:

* the result carries a certificate - `lower_bound`, `plan_objective` and `gap`;
* it carries the master's meter price `meter_price` (pi): what one more kWh at
  the meter costs the whole house in each slot, grid limits included.

The value function for the fast tier comes from the same place as in ADMM:
one battery DP at the tariff, with every other device held at its planned
power. DW with the battery in the master has no value function of its own (the
battery is LP variables there), and the DP re-solve is what lets the execution
tier answer "what now?" from a state the plan did not predict.
"""

from __future__ import annotations

import numpy as np

from home_energy_optimizer.dw.coordinator import DWCoordinator, DWResult
from home_energy_optimizer.coordinate import (
    apply_curtailment,
    baseline_solution,
    device_sell_price,
    net_cost,
    total_objective,
)
from home_energy_optimizer.dp_battery import solve_battery
from home_energy_optimizer.types import CoordinationResult, DeviceSolution, Forecasts, SiteConfig

_EMPTY2 = np.empty((0, 0))
_EMPTY1 = np.empty(0)


def dw_plan(cfg: SiteConfig, fc: Forecasts, tank_in_master: bool | str = "auto",
            ev_duty_cycle: bool = False, **run_kw) -> CoordinationResult:
    """Plan with the recommended DW configuration; return a CoordinationResult.

    `tank_in_master="auto"` puts the tank in the master LP when its element
    can modulate (`n_duty_levels > 2`) and its comfort is priced linearly;
    an on/off element bids plans instead. `ev_duty_cycle=True` lets an EV
    whose charger has a minimum power run a blend of its plans, as on/off
    cycling within a slot (only if the charger can be switched that often;
    otherwise each EV runs one of its plans). `run_kw` goes to
    `DWCoordinator.run` (its defaults are the recommended ones).
    """
    for b in cfg.battery_list:
        if b.terminal_mode != "linear":
            # The master values stored energy linearly; a quadratic pull
            # toward a target SoC has no place in an LP.
            raise ValueError("the DW planner needs terminal_mode='linear' on every battery")
    wh = cfg.water_heater
    if tank_in_master == "auto":
        tank_in_master = bool(wh is not None and wh.n_duty_levels > 2 and wh.comfort_mode == "linear")
    co = DWCoordinator(cfg, fc, tank_in_master=bool(tank_in_master and wh is not None),
                       ev_duty_cycle=ev_duty_cycle)
    return to_coordination_result(co, co.run(**run_kw))


def to_coordination_result(co: DWCoordinator, r: DWResult) -> CoordinationResult:
    cfg, fc, dt = co.cfg, co.fc, co.dt
    devices = {
        k: DeviceSolution(trajectory=c.trajectory.copy(), power=c.power.copy(),
                          value=_EMPTY2, policy=_EMPTY2, states=_EMPTY1, actions=_EMPTY1)
        for k, c in r.plan.items()
    }
    raw = co.d + sum(c.power for c in r.plan.values())
    net, curtail = apply_curtailment(raw, fc.solar, fc.sell, cfg.grid)
    trajs = {k: c.trajectory for k, c in r.plan.items()}
    soe = {k: v for k, v in trajs.items() if k.startswith("battery")} or None

    # The fast tier's value function: battery 0 at the tariff, everything else
    # held at the DW plan - the same conditioning ADMM's pricing re-solve uses.
    pricing, dp_load = None, None
    key0 = SiteConfig.battery_key(0)
    if cfg.battery is not None and key0 in r.plan:
        dp_load = co.d + sum(c.power for k, c in r.plan.items() if k != key0)
        pricing = solve_battery(cfg.battery, cfg.horizon, fc.buy,
                                device_sell_price(fc.sell, cfg.grid),
                                dp_load=dp_load, soc_gates=cfg.soc_gates)

    g = cfg.grid
    return CoordinationResult(
        devices=devices,
        net_grid=net,
        import_cost=float(np.sum(np.maximum(net, 0.0) * fc.buy) * dt),
        export_revenue=float(np.sum(np.maximum(-net, 0.0) * fc.sell) * dt),
        net_cost=net_cost(net, fc.buy, fc.sell, dt),
        total_objective=total_objective(cfg, net, fc, trajs.get("water_heater"),
                                        trajs.get("hvac"), soe),
        rounds_run=r.iterations,
        battery_pricing=pricing,
        grid_import_excess=(float(np.maximum(net - g.max_import_kw, 0.0).max())
                            if g.max_import_kw is not None else 0.0),
        grid_export_excess=(float(np.maximum(-net - g.max_export_kw, 0.0).max())
                            if g.max_export_kw is not None else 0.0),
        curtailment=curtail,
        curtailed_kwh=float(curtail.sum() * dt),
        battery_dp_load=dp_load,
        baseline_cost=baseline_solution(cfg, fc)[1],
        method="dw",
        lower_bound=float(r.lower),
        plan_objective=float(r.upper),
        meter_price=np.asarray(r.prices, dtype=float).copy(),
    )
