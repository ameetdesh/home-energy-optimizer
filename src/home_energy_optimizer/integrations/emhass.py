"""EMHASS adapter: coordinate EMHASS's devices as separate participants.

EMHASS calls `optimize` from `Optimization.perform_optimization` when its
`optimization_backend` is not "cvxpy". Each participant - by default one per
device - is answered by EMHASS's own model with only that participant's
devices enabled (`EmhassParticipant`): the price-response query is EMHASS's own
problem with no PV and no load, priced at the coordinator's price. Devices
assigned to "home_energy_optimizer" are planned by this package's solvers. The
plan comes back as EMHASS's own `opt_res` columns, plus `fed_*` columns,
including each player's share of the saving (`fed_share_*`).

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
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from home_energy_optimizer.coordinate import apply_curtailment
from home_energy_optimizer.dw.attribution import ledger
from home_energy_optimizer.dw.coordinator import Column, DWCoordinator, DWResult, RunOptions
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

if TYPE_CHECKING:  # pandas arrives with EMHASS; this package does not require it
    import pandas as pd

_CfgT = TypeVar("_CfgT", WaterHeaterConfig, HvacConfig)


class EmhassOptimization(Protocol):
    """What this adapter uses of EMHASS's `Optimization` (emhass.optimization):
    the one `optimize` is handed, and the one each participant builds for its
    own devices. EMHASS is not a dependency, so its surface is written down here
    rather than imported - one list of what an EMHASS upgrade must keep."""

    optim_conf: dict[str, Any]
    plant_conf: dict[str, Any]
    retrieve_hass_conf: dict[str, Any]
    emhass_conf: dict[str, Any]
    var_load_cost: str
    var_prod_price: str
    costfun: str                   # "profit", "cost" or "self-consumption"
    time_step: float               # hours
    logger: logging.Logger
    optim_status: str
    prob: Any                      # the solved problem; its .value is the objective
    _persist_q_input: Callable[..., None]

    def perform_optimization(self, data_opt: pd.DataFrame, p_pv: np.ndarray, p_load: np.ndarray,
                             unit_load_cost: np.ndarray, unit_prod_price: np.ndarray,
                             **kwargs: Any) -> pd.DataFrame: ...

    def _prepare_power_limit_array(self, value: Any, name: str, n: int) -> np.ndarray: ...

PACKAGE = "home_energy_optimizer"
# DWCoordinator.run settings. A participant answered by a black-box model gives
# a weak lower bound, so the gap test alone would run to max_iter: stop once the
# master's value has stopped falling, then fix the on/off participants to one
# plan each and let the rest re-plan around them (the dive).
RUN_DEFAULTS: RunOptions = {"stall": 2, "smoothing": 0.0, "dive": 6}
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


def _per_load_keys() -> set[str]:
    """The optim_conf keys that hold one entry per deferrable load, so are cut
    to a participant's loads (EMHASS's own list, plus a few it handles apart)."""
    from emhass.utils import DEF_LOAD_ARRAY_PARAMS
    return set(DEF_LOAD_ARRAY_PARAMS) | {"minimum_power_of_deferrable_loads", "cost_forecast_per_deferrable_load",
                                         "is_electric_load", "def_load_config"}


def unsupported(optim_conf: dict[str, Any], plant_conf: dict[str, Any], costfun: str,
                runtime: dict[str, Any]) -> str | None:
    """Why this configuration cannot be split per device yet, or None if it can.

    `optim_conf`, `plant_conf`: EMHASS's configuration dicts. `costfun`: EMHASS's
    cost function ("profit", "cost" or "self-consumption"). `runtime`: the
    remaining keyword arguments of perform_optimization. Returns a short reason
    for the log (the option that blocks it), or None.
    """
    oc, pc = optim_conf, plant_conf
    if costfun not in ("profit", "cost"):
        return f"costfun {costfun!r} (supported: profit, cost)"
    if oc.get("set_total_pv_sell"):
        return "set_total_pv_sell"
    named = {d for g in oc.get("participants") or [] for d in g.get("devices", [])}
    if "battery" in named and not oc.get("set_use_battery"):
        # EMHASS has no battery, so no state of charge or plant_conf to plan one from
        return "a participant names the battery, but set_use_battery is off"
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


def _hold_q_input(k: int, params: dict[str, Any], hc: dict[str, Any]) -> None:
    """Stand-in for Optimization._persist_q_input in a participant's model: keep
    the heat input a thermal battery starts from (its configured
    `q_input_initial`, else the value it was built with) on every re-solve.
    `k`: the load's index; `params`: its thermal parameters; `hc`: its
    thermal_battery config."""
    if "q_input_initial" in hc:
        params["q_input_start"].value = float(hc.get("q_input_initial", 0.0) or 0.0)


# ------------------------------------------------------------------ participants
class EmhassParticipant:
    """One participant answered by EMHASS's own model, restricted to its devices.

    The model is built once and re-solved at each query's prices: EMHASS holds
    prices, PV and load in CVXPY parameters, so a re-solve does not rebuild it.

    `opt`: the calling EMHASS Optimization (its configuration is copied, then
    restricted). `key`: the participant's name in the coordinator. `battery`:
    whether the participant includes EMHASS's battery. `loads`: indices of its
    deferrable loads in EMHASS's configuration. `data_opt`: EMHASS's input
    DataFrame (its index sets the horizon). `soc_init`, `soc_final`: the
    battery's start and end-of-day state of charge (fractions, 0-1). `runtime`:
    perform_optimization's other keyword arguments; the per-load lists are cut
    to `loads`. `buy`: the import tariff per slot (currency/kWh), for its
    baseline. `reach_w`: grid limits (W) for its own model, sized above every
    flow the house can make.
    """

    def __init__(self, opt: EmhassOptimization, key: str, battery: bool, loads: list[int],
                 data_opt: pd.DataFrame, soc_init: float | None, soc_final: float | None,
                 runtime: dict[str, Any], buy: np.ndarray, reach_w: float = 1e5) -> None:
        """Build the participant's own EMHASS model: a copy of `opt`'s
        configuration with only this participant's devices enabled (the
        parameters are described on the class). Solves nothing yet; sets
        `max_power_kw`, `modulating` and `onoff` for the coordinator."""
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
        self.opt: EmhassOptimization = Optimization(
            opt.retrieve_hass_conf, oc, pc, opt.var_load_cost, opt.var_prod_price,
            "profit", opt.emhass_conf, opt.logger, num_timesteps=len(data_opt))
        # Every query starts from the same state. On a re-solve EMHASS carries a
        # heat pump's heat input (thermal inertia) over from the last solve, as
        # MPC needs when the horizon moves on; here every solve is the same
        # horizon at other prices, so only a configured q_input_initial applies.
        self.opt._persist_q_input = _hold_q_input
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
        self._last: tuple[tuple[Any, ...], Answer] | None = None   # (query key, answer): a
                                                                   # repeated query is not re-solved

    def _run(self, pv_w: np.ndarray, load_w: np.ndarray, buy: np.ndarray,
             sell: np.ndarray) -> Answer:
        """Solve EMHASS's model of this participant once and read off its answer.

        `pv_w`, `load_w`: the PV and load (W per slot) it is metered with - zero
        in a price response, the rest of the house in a best response. `buy`,
        `sell`: the import and export prices (currency/kWh per slot). Returns the
        plan (kW, + = drawn), its private cost (EMHASS's objective less the
        bill) and its state: stored energy (kWh) with a battery, else energy used
        so far. If the solve fails, returns the last good answer with status
        "fallback"; raises RuntimeError if there has been none.
        """
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
        """Answer a coordinator's query (interface.Participant). A query equal
        to the last one returns the same answer without solving again."""
        key = (q.kind,) + tuple(None if a is None else np.asarray(a, dtype=float).tobytes()
                                for a in (q.price_draw, q.price_supply, q.residual_kw))
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        a = self._respond(q)
        self._last = (key, a)
        return a

    def _respond(self, q: Query) -> Answer:
        """Map the query to an EMHASS solve: a price response meters the
        participant alone; a best response meters it under `q.residual_kw`,
        split into PV (exported part) and load (imported part). A proximal
        query raises NotImplementedError."""
        if q.kind in ("price_response", "best_response") and (q.price_draw is None or q.price_supply is None):
            raise ValueError(f"a {q.kind} query needs price_draw and price_supply")
        if q.kind == "price_response":
            assert q.price_draw is not None and q.price_supply is not None
            zero = np.zeros(self.n)
            return self._run(zero, zero, q.price_draw, q.price_supply)
        if q.kind == "best_response":
            assert q.price_draw is not None and q.price_supply is not None
            r = np.asarray(q.residual_kw, dtype=float)
            return self._run(np.maximum(-r, 0.0) * 1000.0, np.maximum(r, 0.0) * 1000.0, q.price_draw, q.price_supply)
        raise NotImplementedError("a proximal step needs a quadratic term EMHASS's model does not have yet")

    def blend(self, weights: Sequence[float] | np.ndarray,
              details: list[pd.DataFrame]) -> pd.DataFrame:
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
def _groups(optim_conf: dict[str, Any], devices: list[str]) -> list[dict[str, Any]]:
    """The participant groups: `optim_conf["participants"]`, with every device
    in `devices` it leaves out as its own group, solved by EMHASS. With no
    `participants`, one EMHASS group per device. Each group is a dict with
    "devices" (names), "solver" and, if given, "config"."""
    spec = optim_conf.get("participants") or []
    if not spec:
        return [{"devices": [d], "solver": "emhass"} for d in devices]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for g in spec:
        devs = list(g["devices"])
        seen.update(devs)
        out.append({"devices": devs, "solver": g.get("solver", "emhass"), "config": g.get("config", {})})
    for d in devices:                      # a device the spec leaves out stays EMHASS's
        if d not in seen:
            out.append({"devices": [d], "solver": "emhass"})
    return out


def _config(cls: type[_CfgT], overrides: dict[str, Any]) -> _CfgT:
    """An instance of the config dataclass `cls` (e.g. WaterHeaterConfig) from
    `overrides`, ignoring keys it has no field for; the rest keep defaults."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in names})


def _battery(plant_conf: dict[str, Any], soc_init: float, buy: np.ndarray) -> BatteryConfig:
    """EMHASS's battery as this package's: its SoC window becomes the store, and
    energy left at the end is valued at the horizon's average import price (as
    the package's own planner does), not pinned to soc_final.

    `plant_conf`: EMHASS's plant configuration (W, Wh, fractions). `soc_init`:
    the starting state of charge (fraction of nominal capacity). `buy`: the
    import tariff per slot (currency/kWh). Returns a BatteryConfig in kW/kWh.

    Power limits are at the meter here. EMHASS bounds the meter-side power by
    each limit twice - directly, and through the efficiency (charging at most
    limit / charge efficiency, discharging at most limit * discharge
    efficiency) - so the tighter of the two is used: with efficiencies below 1,
    the charge limit as given and the discharge limit times its efficiency.
    """
    pc = plant_conf
    cap = float(pc["battery_nominal_energy_capacity"]) / 1000.0
    lo, hi = float(pc["battery_minimum_state_of_charge"]), float(pc["battery_maximum_state_of_charge"])
    eta_c, eta_d = float(pc["battery_charge_efficiency"]), float(pc["battery_discharge_efficiency"])
    chg, dis = float(pc["battery_charge_power_max"]) / 1000.0, float(pc["battery_discharge_power_max"]) / 1000.0
    return BatteryConfig(capacity_kwh=cap * hi, p_charge_max_kw=min(chg, chg / eta_c),
                         p_discharge_max_kw=min(dis, dis * eta_d),
                         eta_charge=eta_c, eta_discharge=eta_d,
                         soc_initial_frac=min(max(soc_init / hi, 0.0), 1.0), soe_min_frac=lo / hi,
                         terminal_mode="linear", terminal_price=float(np.mean(buy)))


def _linear_battery(oc: dict[str, Any], pc: dict[str, Any]) -> bool:
    """Whether EMHASS's battery is exactly its linear constraints - the state
    of charge stepped by both efficiencies, power limits, the SoC window and
    the end-of-day target - so the coordinator can hold it in its master LP.
    (The charge-or-discharge binary is relaxed; it matters only at negative
    prices, where the plan is run as net power.)"""
    def zero(v: Any) -> bool:
        return all(float(x or 0) == 0 for x in (v if isinstance(v, list) else [v]))

    return (not oc.get("set_battery_dynamic") and zero(oc.get("weight_battery_discharge", 0))
            and zero(oc.get("weight_battery_charge", 0)) and zero(pc.get("battery_stress_cost", 0))
            and zero(pc.get("battery_soc_deficit_cost", 0)) and zero(pc.get("battery_soc_surplus_cost", 0))
            and not pc.get("battery_charge_power_derating"))


def optimize(opt: EmhassOptimization, data_opt: pd.DataFrame, p_pv: npt.ArrayLike, p_load: npt.ArrayLike,
             unit_load_cost: npt.ArrayLike, unit_prod_price: npt.ArrayLike,
             soc_init: float | None = None, soc_final: float | None = None,
             **runtime: Any) -> pd.DataFrame | None:
    """Plan with the coordinator `opt.optim_conf["optimization_backend"]` names.

    Called by EMHASS's Optimization.perform_optimization with its own
    arguments. `opt`: that Optimization (configuration, logger, time step).
    `data_opt`: its input DataFrame, one row per slot. `p_pv`, `p_load`: PV and
    house load forecasts (W per slot). `unit_load_cost`, `unit_prod_price`:
    import and export prices (currency/kWh per slot). `soc_init`, `soc_final`:
    the battery's start and end-of-day state of charge (fractions); EMHASS's
    battery_target_state_of_charge when None. `runtime`: perform_optimization's
    other keyword arguments.

    Returns a DataFrame with the columns perform_optimization returns (the same
    names and units) plus `fed_meter_price`, `fed_lower_bound`, `fed_gap` and
    one `fed_share_<player>` per player (solar, each device or participant):
    its share of the saving over the horizon, in currency, the same in every row.
    Returns None, after logging why, when it cannot plan this configuration or
    the coordinator fails; EMHASS then runs its default solver.
    """
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
    participants: list[EmhassParticipant] = []
    targets: dict[str, float] = {}
    battery_cfg: BatteryConfig | None = None
    tank_cfg: WaterHeaterConfig | None = None
    hvac_cfg: HvacConfig | None = None
    for g in _groups(oc, devices):
        if g["solver"] == "emhass" and g["devices"] == ["battery"] and _linear_battery(oc, pc):
            # EMHASS's battery, as its linear model in the master: the same
            # constraints from the same parameters, held exactly
            # unsupported() declined a battery without set_use_battery, which sets both
            assert soc_init is not None and soc_final is not None
            b = replace(_battery(pc, float(soc_init), buy), terminal_price=0.0)
            cap = float(pc["battery_nominal_energy_capacity"]) / 1000.0
            s0 = b.capacity_kwh * b.soc_initial_frac
            reach_up = s0 + n * dt * b.eta_c * b.p_charge_max_kw
            reach_dn = s0 - n * dt * b.p_discharge_max_kw / b.eta_d
            target = float(soc_final) * cap
            battery_cfg = b
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
                    assert soc_init is not None      # as above: set_use_battery is on
                    battery_cfg = _battery(pc, float(soc_init), buy)
                elif d == "water_heater":
                    tank_cfg = _config(WaterHeaterConfig, g.get("config", {}))
                elif d == "hvac":
                    hvac_cfg = _config(HvacConfig, g.get("config", {}))
                else:
                    log.warning(f"optimization_backend: {PACKAGE} has no solver for {d!r}; using the default solver")
                    return None
        else:
            log.warning(f"optimization_backend: unknown solver {g['solver']!r}; using the default solver")
            return None

    horizon = Horizon(dt=dt, hours=n * dt)
    def col(name: str, default: float) -> np.ndarray:
        return np.asarray(data_opt[name].values, dtype=float) if name in data_opt else np.full(n, default)

    fc = Forecasts(buy=buy, sell=sell, load=load_w / 1000.0, solar=pv_w / 1000.0,
                   outdoor_temp=col("outdoor_temperature_forecast", 20.0),
                   hot_water_demand=col("hot_water_demand_kw", 0.0))
    grid = GridLimits(max_import_kw=float(pc.get("maximum_power_from_grid", 9000)) / 1000.0,
                      max_export_kw=float(pc.get("maximum_power_to_grid", 9000)) / 1000.0,
                      allow_curtailment=bool(pc.get("compute_curtailment", False)))
    site = SiteConfig(horizon=horizon, battery=battery_cfg, water_heater=tank_cfg, hvac=hvac_cfg, grid=grid)
    ceiling = np.maximum(pv_w - load_w, 0.0) / 1000.0 if oc.get("set_nodischarge_to_grid") else None
    wh = site.water_heater
    tank_in_master = bool(wh is not None and wh.n_duty_levels > 2)
    run_kw: RunOptions = RUN_DEFAULTS if participants else {}   # with no EMHASS participant, the package's own defaults
    co = DWCoordinator(site, fc, tank_in_master=tank_in_master,
                       participants=participants, export_ceiling=ceiling, soe_targets=targets)
    try:
        r = co.run(**run_kw)
    except Exception as exc:                            # a plan is always published
        log.warning(f"optimization_backend: the coordinator failed ({exc}); using the default solver")
        return None

    res = _results(opt, co, r, data_opt, pv_w, load_w, buy, sell_in, soc_init, devices, participants, site)
    try:
        # The saving split needs one more plan, the same devices with no PV
        # (the export ceiling, PV surplus, is then zero).
        dark = replace(fc, solar=np.zeros(n))
        co_dark = DWCoordinator(site, dark, tank_in_master=tank_in_master, participants=participants,
                                export_ceiling=None if ceiling is None else np.zeros(n), soe_targets=targets)
        for player, share in shares(co, r.plan, co_dark, co_dark.run(**run_kw).plan).items():
            res[f"fed_share_{player}"] = share
    except Exception as exc:                            # the plan stands without its split
        log.warning(f"optimization_backend: could not split the saving ({exc})")
    opt.optim_status = "Optimal"
    solves = sum(p.solves for p in participants)
    log.info(f"optimization_backend=dantzig_wolfe: {r.iterations} iterations, gap {r.upper - r.lower:.4f}, "
             f"{solves} participant solves, {time.perf_counter() - t0:.2f} s")
    return res


def shares(co: DWCoordinator, plan: Mapping[str, Column], co_dark: DWCoordinator,
           plan_dark: Mapping[str, Column]) -> dict[str, float]:
    """Each player's share of the saving over the horizon (currency): solar and
    every device or participant (dw.attribution: reimbursed its private-cost
    change, then an Owen value between solar and the devices, Aumann-Shapley
    among the devices). `plan` / `plan_dark`: the coordinator's plans with and
    without PV. The shares, with the reimbursements, add up to the saving over
    the baseline with no PV and no coordination."""
    rows = ledger(co, plan, co_dark, plan_dark)["rows"]
    return {row["player"]: float(row["net_gain"]) for row in rows if row["player"] != "household load"}


def _results(opt: Any, co: DWCoordinator, r: DWResult, data_opt: pd.DataFrame,
             pv_w: np.ndarray, load_w: np.ndarray, buy: np.ndarray, sell_in: np.ndarray,
             soc_init: float | None, devices: list[str], participants: list[EmhassParticipant],
             site: SiteConfig) -> pd.DataFrame:
    """The coordinator's result `r` as EMHASS's opt_res: the same columns and
    units as perform_optimization returns, plus fed_* columns.

    `co`: the coordinator that produced `r`. The other arguments are those
    `optimize` worked with: EMHASS's inputs (W, currency/kWh), the device names,
    the EMHASS participants (whose own results fill their devices' columns) and
    the package's SiteConfig. Returns a DataFrame indexed like `data_opt`.
    """
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
