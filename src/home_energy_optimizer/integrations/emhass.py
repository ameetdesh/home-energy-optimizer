"""EMHASS adapter: coordinate EMHASS's devices as separate participants.

EMHASS calls `optimize` from `Optimization.perform_optimization` when its
`optimization_backend` is not "cvxpy". Each participant - by default one per
device - is answered by EMHASS's own model with only that participant's
devices enabled (`EmhassParticipant`): the price-response query is EMHASS's own
problem with no PV and no load, priced at the coordinator's price. Devices
assigned to "home_energy_optimizer" are planned by this package's solvers. The
plan comes back as EMHASS's own `opt_res` columns, plus `fed_*` columns.

`optimize` returns None, after logging why, whenever the configuration uses
something it cannot split per device yet; EMHASS then runs its default solver.

    "optimization_backend": "dantzig_wolfe",
    "participants": [
      {"devices": ["battery"], "solver": "emhass"},
      {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},
      {"devices": ["water_heater"], "solver": "home_energy_optimizer", "config": {...}}
    ]

EMHASS works in W, this package in kW; EMHASS's P_batt is + when discharging,
the interface's plans are + when drawn from the meter.
"""

from __future__ import annotations

import copy
import time
from dataclasses import fields, replace

import numpy as np

from home_energy_optimizer.coordinate import apply_curtailment
from home_energy_optimizer.dw.coordinator import DWCoordinator
from home_energy_optimizer.interface import Answer, Query
from home_energy_optimizer.types import (
    BatteryConfig,
    Forecasts,
    GridLimits,
    Horizon,
    HvacConfig,
    SiteConfig,
    WaterHeaterConfig,
)

PACKAGE = "home_energy_optimizer"
# DWCoordinator.run settings. A participant answered by a black-box model gives
# a weak lower bound, so the gap test alone would run to max_iter: stop once the
# master's value has stopped falling, then fix the on/off participants to one
# plan each and let the rest re-plan around them (the dive).
RUN_DEFAULTS = {"stall": 2, "smoothing": 0.0, "dive": 6}
OK_STATUSES = ("Optimal", "Optimal (Relaxed)")
# Runtime per-load lists perform_optimization takes; cut to a participant's loads.
RUNTIME_LOAD_LISTS = ("def_total_hours", "def_total_timestep", "def_start_timestep",
                      "def_end_timestep", "def_init_temp", "min_power_of_deferrable_loads")
# Inputs a participant cannot split per device yet (perform_optimization arguments).
RUNTIME_UNSUPPORTED = ("soc_target", "soc_target_timestep", "current_period_peak",
                       "capacity_charge_window", "capacity_charge_consideration",
                       "capacity_charge_current_interval_history")
# Arguments that change nothing here. Any other argument that is set means a
# feature this adapter does not know, so EMHASS's own solver runs.
RUNTIME_IGNORED = ("debug", "stage_times")


def _per_load_keys():
    from emhass.utils import DEF_LOAD_ARRAY_PARAMS
    return set(DEF_LOAD_ARRAY_PARAMS) | {"minimum_power_of_deferrable_loads", "cost_forecast_per_deferrable_load",
                                         "is_electric_load", "def_load_config"}


def unsupported(optim_conf: dict, plant_conf: dict, costfun: str, runtime: dict) -> str | None:
    """Why this configuration cannot be split per device yet, or None."""
    oc, pc = optim_conf, plant_conf
    if costfun not in ("profit", "cost"):
        return f"costfun {costfun!r} (supported: profit, cost)"
    if oc.get("set_total_pv_sell"):
        return "set_total_pv_sell"
    if oc.get("set_use_battery") and oc.get("set_nocharge_from_grid"):
        return "set_nocharge_from_grid (ties the battery to PV)"
    if oc.get("set_battery_first_priority"):
        return "set_battery_first_priority"
    if pc.get("inverter_is_hybrid"):
        return "a hybrid inverter (ties the battery to PV)"
    if int(pc.get("number_of_batteries", 1)) > 1:
        return "more than one battery"
    if oc.get("heat_topology") or oc.get("shared_thermal_tanks") or oc.get("deferrable_load_groups"):
        return "heat_topology, shared thermal tanks or deferrable load groups"
    if any(v is not None for v in (oc.get("cost_forecast_per_deferrable_load") or [])):
        return "cost_forecast_per_deferrable_load"
    if any(float(v or 0) > 0 for v in (oc.get("set_deferrable_startup_penalty") or [])):
        return "set_deferrable_startup_penalty (priced at the import tariff)"
    if any(float(v or 0) > 0 for v in (oc.get("deferrable_load_max_cost") or [])):
        return "deferrable_load_max_cost"
    cap = oc.get("capacity_cost_per_kw") or 0.0
    if (any(float(c or 0) > 0 for c in cap) if isinstance(cap, list) else float(cap) > 0):
        return "capacity charges"
    for name in RUNTIME_UNSUPPORTED:
        if runtime.get(name) is not None:
            return name
    known = set(RUNTIME_LOAD_LISTS) | set(RUNTIME_UNSUPPORTED) | set(RUNTIME_IGNORED)
    for name, value in runtime.items():
        if name not in known and value is not None:
            return f"the argument {name!r}"
    return None


# ------------------------------------------------------------------ participants
class EmhassParticipant:
    """One participant answered by EMHASS's own model, restricted to its devices.

    The model is built once and re-solved at each query's prices: EMHASS holds
    prices, PV and load in CVXPY parameters, so a re-solve does not rebuild it.
    """

    def __init__(self, opt, key: str, battery: bool, loads: list[int], data_opt, soc_init, soc_final, runtime: dict,
                 buy: np.ndarray, reach_w: float = 1e5):
        from emhass.optimization import Optimization

        oc, pc = copy.deepcopy(opt.optim_conf), copy.deepcopy(opt.plant_conf)
        oc["optimization_backend"] = "cvxpy"         # its own solves run the default solver
        oc["set_use_battery"] = bool(battery)
        # PV stays enabled: a best response meters the participant under the rest
        # of the house, given as load and PV (zero in a price response)
        oc["set_use_pv"] = True
        oc["set_nodischarge_to_grid"] = False        # the coordinator holds the meter
        oc["number_of_deferrable_loads"] = len(loads)
        for name in _per_load_keys():
            if isinstance(oc.get(name), list):
                oc[name] = [oc[name][k] for k in loads if k < len(oc[name])]
        # The coordinator holds the meter, so the view's own limits must never
        # bind: `reach_w` exceeds every flow the house can make. Not 'infinite':
        # EMHASS uses these as big-M bounds, and a huge one breaks HiGHS.
        pc["maximum_power_from_grid"] = float(reach_w)
        pc["maximum_power_to_grid"] = float(reach_w)
        pc["compute_curtailment"] = False
        self.key, self.battery, self.loads = key, bool(battery), list(loads)
        self.opt = Optimization(opt.retrieve_hass_conf, oc, pc, opt.var_load_cost, opt.var_prod_price,
                                "profit", opt.emhass_conf, opt.logger, num_timesteps=len(data_opt))
        self.data = data_opt
        self.n, self.dt = len(data_opt), float(opt.time_step)
        self.soc_init, self.soc_final = soc_init, soc_final
        self.runtime = {k: (None if v is None else [v[i] for i in loads if i < len(v)])
                        for k, v in runtime.items() if k in RUNTIME_LOAD_LISTS}
        p = 0.0
        if battery:
            p += max(float(pc["battery_charge_power_max"]), float(pc["battery_discharge_power_max"])) / 1000.0
        for k in range(len(loads)):
            nominal = oc["nominal_power_of_deferrable_loads"][k]
            p += (max(nominal) if isinstance(nominal, list) else float(nominal)) / 1000.0
        self.max_power_kw = p
        self.cap_wh = float(pc.get("battery_nominal_energy_capacity", 0.0))
        self.buy = np.asarray(buy, dtype=float)
        # Convex: loads that run at any power, with no minimum on/off time, start
        # count or single block, and no battery (EMHASS's battery has a charge or
        # discharge binary per slot). A mix of such plans runs as it is, so the
        # coordinator may keep one (see blend).
        self.modulating = not battery and all(
            not oc["treat_deferrable_load_as_semi_cont"][k] and not oc["set_deferrable_load_single_constant"][k]
            and not oc["def_minimum_on_time"][k] and not oc["def_minimum_off_time"][k]
            and not oc["set_deferrable_max_startups"][k]
            and not ((oc.get("def_load_config") or [{}] * len(loads))[k] or {}).get("thermal_config", {}).get("overshoot_temperature")
            for k in range(len(loads)))
        # On/off scheduling (a load at fixed power, a single block, minimum on/off
        # times, a start limit): the coordinator fixes such a participant to one
        # plan before the others re-plan around it (DWCoordinator.run dive).
        self.onoff = any(
            oc["treat_deferrable_load_as_semi_cont"][k] or oc["set_deferrable_load_single_constant"][k]
            or oc["def_minimum_on_time"][k] or oc["def_minimum_off_time"][k] or oc["set_deferrable_max_startups"][k]
            for k in range(len(loads)))
        self._good: Answer | None = None   # the last answer that solved, a stand-in if one fails
        self.solves, self.solve_s = 0, 0.0
        self._last: tuple | None = None   # (query key, answer): a repeated query is not re-solved

    def _run(self, pv_w, load_w, buy, sell) -> Answer:
        t0 = time.perf_counter()
        res = self.opt.perform_optimization(self.data, pv_w, load_w, np.asarray(buy, dtype=float),
                                            np.asarray(sell, dtype=float), soc_init=self.soc_init,
                                            soc_final=self.soc_final, **self.runtime)
        self.solves += 1
        self.solve_s += time.perf_counter() - t0
        if self.opt.optim_status not in OK_STATUSES or (self.battery and "P_batt" not in res):
            if self._good is None:
                raise RuntimeError(f"EMHASS could not plan participant {self.key}: {self.opt.optim_status}")
            g = self._good
            return Answer(g.plan_kw, g.private_cost, g.trajectory, status="fallback", detail=g.detail)
        plan_w = np.zeros(self.n)
        for k in range(len(self.loads)):
            plan_w += res[f"P_deferrable{k}"].values
        if self.battery:
            plan_w -= res["P_batt"].values
        plan_kw = plan_w / 1000.0
        # EMHASS maximises -(bill + private); the bill is on the metered flow
        scale = 0.001 * self.dt
        bill = scale * float(np.sum(np.asarray(buy) * res["P_grid_pos"].values + np.asarray(sell) * res["P_grid_neg"].values))
        private = -float(self.opt.prob.value) - bill
        if self.battery:
            s0 = (self.soc_init if self.soc_init is not None else self.opt.plant_conf["battery_target_state_of_charge"])
            traj = np.concatenate([[float(s0)], res["SOC_opt"].values]) * self.cap_wh / 1000.0
        else:
            traj = np.concatenate([[0.0], np.cumsum(plan_kw) * self.dt])
        self._good = Answer(plan_kw, private, traj, status="ok", detail=res)
        return self._good

    def respond(self, q: Query) -> Answer:
        key = (q.kind,) + tuple(None if a is None else np.asarray(a, dtype=float).tobytes()
                                for a in (q.price_draw, q.price_supply, q.residual_kw))
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        a = self._respond(q)
        self._last = (key, a)
        return a

    def _respond(self, q: Query) -> Answer:
        if q.kind == "price_response":
            zero = np.zeros(self.n)
            return self._run(zero, zero, q.price_draw, q.price_supply)
        if q.kind == "best_response":
            r = np.asarray(q.residual_kw, dtype=float)
            return self._run(np.maximum(-r, 0.0) * 1000.0, np.maximum(r, 0.0) * 1000.0, q.price_draw, q.price_supply)
        raise NotImplementedError("a proximal step needs a quadratic term EMHASS's model does not have yet")

    def blend(self, weights, details):
        """EMHASS's result for a mix of this participant's plans: every number
        mixed with the same weights, the rest taken from the heaviest plan."""
        out = details[int(np.argmax(weights))].copy()
        for c in out.columns:
            if np.issubdtype(out[c].dtype, np.number):
                out[c] = sum(w * d[c].values for w, d in zip(weights, details))
        return out

    def baseline(self) -> Answer:
        """A feasible plan to start from: its plan at the import tariff, metered alone."""
        return self.respond(Query("price_response", self.buy, self.buy))


# ------------------------------------------------------------------ the call
def _groups(optim_conf: dict, devices: list[str]) -> list[dict]:
    spec = optim_conf.get("participants") or []
    if not spec:
        return [{"devices": [d], "solver": "emhass"} for d in devices]
    seen, out = set(), []
    for g in spec:
        devs = list(g["devices"])
        seen.update(devs)
        out.append({"devices": devs, "solver": g.get("solver", "emhass"), "config": g.get("config", {})})
    for d in devices:                      # a device the spec leaves out stays EMHASS's
        if d not in seen:
            out.append({"devices": [d], "solver": "emhass"})
    return out


def _config(cls, overrides: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in names})


def _battery(plant_conf: dict, soc_init: float, buy: np.ndarray) -> BatteryConfig:
    """EMHASS's battery as this package's: its SoC window becomes the store, and
    energy left at the end is valued at the horizon's average import price (as
    the package's own planner does), not pinned to soc_final."""
    pc = plant_conf
    cap = float(pc["battery_nominal_energy_capacity"]) / 1000.0
    lo, hi = float(pc["battery_minimum_state_of_charge"]), float(pc["battery_maximum_state_of_charge"])
    return BatteryConfig(capacity_kwh=cap * hi, p_charge_max_kw=float(pc["battery_charge_power_max"]) / 1000.0,
                         p_discharge_max_kw=float(pc["battery_discharge_power_max"]) / 1000.0,
                         eta_charge=float(pc["battery_charge_efficiency"]),
                         eta_discharge=float(pc["battery_discharge_efficiency"]),
                         soc_initial_frac=min(max(soc_init / hi, 0.0), 1.0), soe_min_frac=lo / hi,
                         terminal_mode="linear", terminal_price=float(np.mean(buy)))


def _linear_battery(oc: dict, pc: dict) -> bool:
    """Whether EMHASS's battery is exactly its linear constraints - the state
    of charge stepped by both efficiencies, power limits, the SoC window and
    the end-of-day target - so the coordinator can hold it in its master LP.
    (The charge-or-discharge binary is relaxed; it matters only at negative
    prices, where the plan is run as net power.)"""
    zero = lambda v: all(float(x or 0) == 0 for x in (v if isinstance(v, list) else [v]))   # noqa: E731
    return (not oc.get("set_battery_dynamic") and zero(oc.get("weight_battery_discharge", 0))
            and zero(oc.get("weight_battery_charge", 0)) and zero(pc.get("battery_stress_cost", 0))
            and zero(pc.get("battery_soc_deficit_cost", 0)) and zero(pc.get("battery_soc_surplus_cost", 0))
            and not pc.get("battery_charge_power_derating"))


def optimize(opt, data_opt, p_pv, p_load, unit_load_cost, unit_prod_price, soc_init=None, soc_final=None,
             **runtime):
    """Plan with the coordinator `opt.optim_conf["optimization_backend"]` names;
    return EMHASS's opt_res, or None to let EMHASS run its default solver."""
    log = opt.logger
    oc, pc = opt.optim_conf, opt.plant_conf
    backend = oc.get("optimization_backend", "cvxpy")
    reason = unsupported(oc, pc, opt.costfun, runtime)
    if backend != "dantzig_wolfe":
        reason = reason or f"backend {backend!r} (this version runs dantzig_wolfe)"
    if reason:
        log.warning(f"optimization_backend={backend}: not available for {reason}; using the default solver")
        return None

    t0 = time.perf_counter()
    n, dt = len(data_opt), float(opt.time_step)
    buy = np.asarray(unit_load_cost, dtype=float)
    sell_in = np.asarray(unit_prod_price, dtype=float)
    sell = sell_in if opt.costfun == "profit" else np.zeros(n)
    if np.any(sell > buy + 1e-12):
        log.warning("optimization_backend: an export price above the import price; using the default solver")
        return None
    pv_w, load_w = np.asarray(p_pv, dtype=float), np.asarray(p_load, dtype=float)
    if soc_init is None and oc.get("set_use_battery"):
        soc_init = pc["battery_target_state_of_charge"]
    if soc_final is None and oc.get("set_use_battery"):
        soc_final = pc["battery_target_state_of_charge"]

    devices = (["battery"] if oc.get("set_use_battery") else []) + [
        f"deferrable{k}" for k in range(int(oc.get("number_of_deferrable_loads", 0)))]
    nominal = oc.get("nominal_power_of_deferrable_loads") or []
    device_w = sum((max(v) if isinstance(v, list) else float(v)) for v in nominal[: len(devices)])
    if oc.get("set_use_battery"):
        device_w += max(float(pc["battery_charge_power_max"]), float(pc["battery_discharge_power_max"]))
    reach_w = 2.0 * (float(np.max(np.abs(load_w))) + float(np.max(np.abs(pv_w))) + device_w) + 1000.0
    participants, site_kw, targets = [], {}, {}
    for g in _groups(oc, devices):
        if g["solver"] == "emhass" and g["devices"] == ["battery"] and _linear_battery(oc, pc):
            # EMHASS's battery, as its linear model in the master: the same
            # constraints from the same parameters, held exactly
            b = replace(_battery(pc, float(soc_init), buy), terminal_price=0.0)
            cap = float(pc["battery_nominal_energy_capacity"]) / 1000.0
            s0 = b.capacity_kwh * b.soc_initial_frac
            reach_up = s0 + n * dt * b.eta_c * b.p_charge_max_kw
            reach_dn = s0 - n * dt * b.p_discharge_max_kw / b.eta_d
            target = float(soc_final) * cap
            site_kw["battery"] = b
            targets["battery"] = min(max(target, b.soe_floor_kwh, reach_dn), b.capacity_kwh, reach_up)
        elif g["solver"] == "emhass":
            loads = sorted(int(d[len("deferrable"):]) for d in g["devices"] if d.startswith("deferrable"))
            unknown = [d for d in g["devices"] if d != "battery" and not d.startswith("deferrable")]
            if unknown:
                log.warning(f"optimization_backend: EMHASS has no device {unknown}; using the default solver")
                return None
            key = "+".join(g["devices"])
            participants.append(EmhassParticipant(opt, key, "battery" in g["devices"], loads, data_opt,
                                                  soc_init, soc_final, runtime, buy, reach_w))
        elif g["solver"] == PACKAGE:
            for d in g["devices"]:
                if d == "battery":
                    site_kw["battery"] = _battery(pc, float(soc_init), buy)
                elif d == "water_heater":
                    site_kw["water_heater"] = _config(WaterHeaterConfig, g.get("config", {}))
                elif d == "hvac":
                    site_kw["hvac"] = _config(HvacConfig, g.get("config", {}))
                else:
                    log.warning(f"optimization_backend: {PACKAGE} has no solver for {d!r}; using the default solver")
                    return None
        else:
            log.warning(f"optimization_backend: unknown solver {g['solver']!r}; using the default solver")
            return None

    horizon = Horizon(dt=dt, hours=n * dt)
    col = lambda name, default: (np.asarray(data_opt[name].values, dtype=float)       # noqa: E731
                                 if name in data_opt else np.full(n, default))
    fc = Forecasts(buy=buy, sell=sell, load=load_w / 1000.0, solar=pv_w / 1000.0,
                   outdoor_temp=col("outdoor_temperature_forecast", 20.0),
                   hot_water_demand=col("hot_water_demand_kw", 0.0))
    grid = GridLimits(max_import_kw=float(pc.get("maximum_power_from_grid", 9000)) / 1000.0,
                      max_export_kw=float(pc.get("maximum_power_to_grid", 9000)) / 1000.0,
                      allow_curtailment=bool(pc.get("compute_curtailment", False)))
    site = SiteConfig(horizon=horizon, battery=site_kw.get("battery"), water_heater=site_kw.get("water_heater"),
                      hvac=site_kw.get("hvac"), grid=grid)
    ceiling = np.maximum(pv_w - load_w, 0.0) / 1000.0 if oc.get("set_nodischarge_to_grid") else None
    wh = site.water_heater
    co = DWCoordinator(site, fc, tank_in_master=bool(wh is not None and wh.n_duty_levels > 2),
                       participants=participants, export_ceiling=ceiling, soe_targets=targets)
    try:
        # with no EMHASS participant, exactly the package's own planner (its defaults)
        r = co.run(**(RUN_DEFAULTS if participants else {}))
    except Exception as exc:                            # a plan is always published
        log.warning(f"optimization_backend: the coordinator failed ({exc}); using the default solver")
        return None

    res = _results(opt, co, r, data_opt, pv_w, load_w, buy, sell_in, soc_init, devices, participants, site)
    opt.optim_status = "Optimal"
    solves = sum(p.solves for p in participants)
    log.info(f"optimization_backend=dantzig_wolfe: {r.iterations} iterations, gap {r.upper - r.lower:.4f}, "
             f"{solves} participant solves, {time.perf_counter() - t0:.2f} s")
    return res


def _results(opt, co, r, data_opt, pv_w, load_w, buy, sell_in, soc_init, devices, participants, site):
    """The plan as EMHASS's opt_res: the same columns and units, plus fed_*."""
    import pandas as pd

    dt, n = float(opt.time_step), len(data_opt)
    plan = r.plan
    net = co.d + sum(c.power for c in plan.values())
    net, curtail = apply_curtailment(net, co.fc.solar, co.fc.sell, co.cfg.grid)
    if co.export_ceiling is not None:
        over = np.maximum(-net - co.export_ceiling, 0.0)
        net, curtail = net + over, curtail + over

    out = pd.DataFrame(index=data_opt.index)
    out["P_PV"] = pv_w
    out["P_Load"] = load_w
    if opt.plant_conf.get("compute_curtailment"):
        out["P_PV_curtailment"] = curtail * 1000.0
    out["P_grid_pos"] = np.maximum(net, 0.0) * 1000.0
    out["P_grid_neg"] = np.minimum(net, 0.0) * 1000.0
    out["P_grid"] = out["P_grid_pos"] + out["P_grid_neg"]

    by_load, batt_detail, extra = {}, None, {}
    for p in participants:
        res = plan[p.key].detail
        for local, k in enumerate(p.loads):
            by_load[k] = res[f"P_deferrable{local}"].values
            for c in res.columns:                              # thermal detail, renumbered
                if c.endswith(f"heater{local}"):
                    extra[c[: -len(str(local))] + str(k)] = res[c].values
        if p.battery:
            batt_detail = res
    for d in devices:
        if d.startswith("deferrable"):
            k = int(d[len("deferrable"):])
            out[f"P_deferrable{k}"] = by_load.get(k, np.zeros(n))
    if opt.optim_conf.get("set_use_battery"):
        if batt_detail is not None:
            out["P_batt"] = batt_detail["P_batt"].values
            out["SOC_opt"] = batt_detail["SOC_opt"].values
            for c in ("soc_deficit_cost", "soc_surplus_cost", "batt_stress_cost"):
                if c in batt_detail:
                    out[c] = batt_detail[c].values
        elif "battery" in plan:
            col = plan["battery"]
            out["P_batt"] = -col.power * 1000.0
            out["SOC_opt"] = col.trajectory[1:] / (float(opt.plant_conf["battery_nominal_energy_capacity"]) / 1000.0)
            # EMHASS writes these for every battery; held linearly, both costs are zero
            out["soc_deficit_cost"] = 0.0
            out["soc_surplus_cost"] = 0.0
    for key, label in (("water_heater", "water_heater"), ("hvac", "hvac")):
        if key in plan:
            out[f"P_{label}"] = plan[key].power * 1000.0
            out[f"temp_{label}"] = plan[key].trajectory[1:]
    out["unit_load_cost"] = buy
    out["unit_prod_price"] = sell_in
    out["maximum_power_from_grid"] = opt._prepare_power_limit_array(
        opt.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n)
    out["maximum_power_to_grid"] = opt._prepare_power_limit_array(
        opt.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n)
    scale = -0.001 * dt
    cost_profit = scale * (buy * out["P_grid_pos"].values + sell_in * out["P_grid_neg"].values)
    out["cost_profit"] = cost_profit
    if opt.costfun == "profit":
        out["cost_fun_profit"] = cost_profit
    else:
        out["cost_fun_cost"] = scale * buy * out["P_grid_pos"].values
    out["optim_status"] = "Optimal"
    for c, v in extra.items():
        out[c] = v
    out["fed_meter_price"] = r.prices
    out["fed_lower_bound"] = float(r.lower)
    out["fed_gap"] = float(r.upper - r.lower)
    return out
