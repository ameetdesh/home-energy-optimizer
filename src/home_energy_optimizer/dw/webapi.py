"""Backend for the Dantzig-Wolfe testbed UI (dw/gui/), independent of transport.

Mirrors admm.webapi in shape - so the charts read the same fields - but
runs dw.coordinator instead of the ADMM loop, and adds what only DW has: a
lower bound, the master's meter price, battery costates from the master's
duals, and a per-iteration view of the master's convex mix.

Everything returned is JSON-safe (lists and floats, no numpy).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from home_energy_optimizer.dw.attribution import dark, ledger
from home_energy_optimizer.dw.coordinator import Column, DWCoordinator, baseline_objective
from home_energy_optimizer.coordinate import baseline_solution, net_cost
from home_energy_optimizer.profiles import demo_forecasts
from home_energy_optimizer.submeter import site_meter
from home_energy_optimizer.types import Forecasts, Horizon, SiteConfig, hybrid_inverter
from home_energy_optimizer.admm.webapi import build_site


def _dark(fc: Forecasts) -> Forecasts:
    """The same forecasts with no PV: the devices' plan before solar arrives,
    which the ledger needs (src/home_energy_optimizer/dw/attribution.py)."""
    return dark(fc)


# Who saves what for the last solve, computed when the page asks for it: it
# needs a second plan (the devices with no PV), which costs about as much as a
# quarter of the solve, so it is not worth doing for every solve.
_LEDGER: dict = {}


def _defer_ledger(co: DWCoordinator, plan: dict[str, tuple[np.ndarray, np.ndarray]],
                  dark: Callable[[], tuple[DWCoordinator, dict[str, tuple[np.ndarray, np.ndarray]]]]) -> None:
    """Remember how to build the last solve's ledger. `dark()` returns the
    no-PV coordinator and its plan."""
    _LEDGER.clear()
    _LEDGER["make"] = lambda: _ledger_json(co, plan, *dark())


def ledger_route(p: dict) -> dict:
    """The last solve's ledger (built on first request, then cached)."""
    if "make" not in _LEDGER:
        return {"error": "solve first"}
    if "result" not in _LEDGER:
        _LEDGER["result"] = _LEDGER["make"]()
    return {"ledger": _LEDGER["result"]}


def _ledger_json(co: DWCoordinator, plan: dict[str, tuple[np.ndarray, np.ndarray]], co_dark: DWCoordinator,
                 plan_dark: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Who saves what (src/home_energy_optimizer/dw/attribution.py), rounded for the page."""
    L = ledger(co, plan, co_dark, plan_dark)
    return {"rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in L["rows"]],
            "totals": {k: round(v, 4) for k, v in L["totals"].items()}}


def _series(x: npt.ArrayLike) -> list[float]:
    return [round(float(v), 5) for v in np.asarray(x).ravel()]


def _view(co: DWCoordinator, fc: Forecasts, snap: dict, batt_keys: list[str]) -> dict:
    """One plottable plan: a master snapshot, or the recovered plan."""
    flows = site_meter(co.cfg, fc, snap["powers"])
    net, curtail = flows.net, flows.curtail
    view: dict[str, Any] = {
        # per device, per hour: cost of being asked to consume one step more /
        # less, re-planning after (src/home_energy_optimizer/dw/sensitivity.py); None where inadmissible
        "flex": {k: {d: [None if v is None else round(v, 5) for v in arr] for d, arr in f.items()}
                 for k, f in snap.get("flex", {}).items()},
        "net_grid": _series(net),
        "curtailment": _series(curtail),
        "devices": {k: {"power": _series(snap["powers"][k]),
                        "trajectory": _series(snap["trajectories"][k])}
                    for k in snap["powers"]},
        "price": _series(snap["price"]) if snap.get("price") is not None else None,
        # behind a sub-meter (the hybrid inverter): its AC power (+ = to the
        # house) and its bus's own price, where the master has one
        "inverter_ac": _series(flows.ac["inverter"]) if "inverter" in flows.ac else None,
        "local_price": (_series(snap["local_prices"]["inverter"])
                        if "inverter" in (snap.get("local_prices") or {}) else None),
        "lambdas": {k: _series(v) for k, v in snap.get("lambda", {}).items()},
        "active": {k: [[round(a[0], 4), a[1], (a[2] if len(a) > 2 else None)] for a in v]
                   for k, v in snap.get("active", {}).items()},
        "costs": {
            "net_cost": round(net_cost(net, fc.buy, fc.sell, co.dt), 4),
            "import_cost": round(float(np.sum(np.maximum(net, 0) * fc.buy * co.dt)), 4),
            "export_revenue": round(float(np.sum(np.maximum(-net, 0) * fc.sell * co.dt)), 4),
        },
        "curtailed_kwh": round(float(curtail.sum() * co.dt), 3),
        "import_excess": round(float(np.maximum(net - co.cfg.grid.max_import_kw, 0).max()), 4)
        if co.cfg.grid.max_import_kw is not None else 0.0,
        "export_excess": round(float(np.maximum(-net - co.cfg.grid.max_export_kw, 0).max()), 4)
        if co.cfg.grid.max_export_kw is not None else 0.0,
    }
    # Reservation band for battery 0, from the MASTER's costate rather than a
    # conditional re-solve: import below lambda*eta_c, export above lambda/eta_d.
    if batt_keys and batt_keys[0] in view["lambdas"]:
        b = co.cfg.battery_list[0]
        lam = np.asarray(snap["lambda"][batt_keys[0]])
        view["lambda"] = _series(lam)
        view["import_below"] = _series(lam * b.eta_c)
        view["export_above"] = _series(lam / b.eta_d)
    else:
        view["lambda"] = view["import_below"] = view["export_above"] = None
    return view


def apply_edits(fc: Forecasts, horizon: Horizon, p: dict) -> Forecasts:
    """Replace the preset prices / load with the ones dragged in the UI.

    The UI edits one point per hour (`buy_h`, `sell_h`, `load_h`, `draw_h`, at hours
    0, 1, ..., H); they are interpolated onto the slots, as the single-battery
    app does. Export is held at or below import - the master's meter cost is
    only convex then - and load at or above zero.
    """
    xs = np.arange(horizon.steps) * horizon.dt
    def expand(key: str) -> np.ndarray | None:
        v = p.get(key)
        if not v:
            return None
        v = np.asarray(v, dtype=float)
        return np.interp(xs, np.arange(v.size), v)
    buy, sell, load = expand("buy_h"), expand("sell_h"), expand("load_h")
    draw = expand("draw_h")                 # hot-water draw, kW of heat (Forecasts.hot_water_demand)
    if buy is None and sell is None and load is None and draw is None:
        return fc
    buy = fc.buy if buy is None else buy
    sell = np.minimum(fc.sell if sell is None else sell, buy)
    load = fc.load if load is None else np.maximum(load, 0.0)
    draw = fc.hot_water_demand if draw is None else np.maximum(draw, 0.0)
    return replace(fc, buy=buy, sell=sell, load=load, hot_water_demand=draw)


def apply_band(site: SiteConfig, p: dict) -> SiteConfig:
    """Replace the room's flat comfort band with the one dragged in the UI.

    As for the prices: one low and one high point per hour (`room_low_h`,
    `room_high_h`, at hours 0, 1, ..., H), interpolated here onto the n + 1
    points of the room trajectory (HvacConfig.comfort_band). Each point's pair
    is sorted, as build_site sorts the flat band's.
    """
    hv = site.hvac
    if hv is None or not (p.get("room_low_h") or p.get("room_high_h")):
        return site
    xs = np.arange(site.horizon.steps + 1) * site.horizon.dt
    def expand(key: str, flat: float) -> np.ndarray:
        v = p.get(key)
        return np.full(xs.size, flat) if not v else np.interp(xs, np.arange(len(v)), np.asarray(v, dtype=float))
    low, high = expand("room_low_h", hv.t_comfort_low), expand("room_high_h", hv.t_comfort_high)
    low, high = np.minimum(low, high), np.maximum(low, high)
    return replace(site, hvac=replace(hv, comfort_low_profile=tuple(low.tolist()),
                                      comfort_high_profile=tuple(high.tolist())))


def _inverter(site: SiteConfig) -> dict | None:
    """The hybrid inverter the page drew, or None: its AC ratings (None: no limit)."""
    sm = site.pv_submeter
    return None if sm is None else {"max_output_kw": sm.max_export_kw, "max_input_kw": sm.max_import_kw,
                                    "eta": sm.eta_export, "batteries": list(sm.members)}


def _comfort(site: SiteConfig) -> dict:
    """The thresholds the Temperatures chart draws: the tank's setpoint, and
    the room's band at each point of its trajectory."""
    low, high = site.hvac.comfort_band(site.horizon.steps) if site.hvac else (None, None)
    return {"tank": site.water_heater.t_comfort if site.water_heater else None,
            "room_low": None if low is None else _series(low),
            "room_high": None if high is None else _series(high)}


# Solve progress for the served UI, which polls it while a solve runs:
# request id -> [iteration, max_iter]; iteration -1 means "finishing".
# (The in-browser page is pushed the same numbers by its worker instead.)
_PROGRESS: dict[str, list[int]] = {}
# Paused (or mid-way) ADMM runs, by the page's solve id. One at a time.
_RUNS: dict = {}
# Where the last finished ADMM solve stood at its best plan, for a warm start.
_LAST_WARM: dict = {}


def progress(p: dict) -> dict:
    return {"progress": _PROGRESS.get(str(p.get("id", "")))}


def _site_fc(p: dict) -> tuple[SiteConfig, Forecasts, bool]:
    """The site and forecasts both methods solve, from the UI's settings."""
    # The master values stored energy linearly; the quadratic POC terminal has
    # no place in an LP, so it is not offered here.
    site = build_site({**p, "terminal_mode": "linear"})
    # Defaults are the recommended configuration: batteries and a modulating
    # tank in the master LP, used+newest pool, MILP recovery, numpy solver.
    tank_lp = str(p.get("tank_levels", "lp")) == "lp"
    levels = 2 if tank_lp else int(p.get("tank_levels", 2))
    if site.water_heater is not None and levels != 2:
        # A fractional element needs a temperature grid that resolves one
        # duty step, or its DP plans moves its rollout cannot follow.
        wh = replace(site.water_heater, n_duty_levels=levels)
        wh = replace(wh, n_states=wh.states_for_duty_levels(site.horizon.dt))
        site = replace(site, water_heater=wh)
    hvac_levels = int(p.get("hvac_levels", 2))
    if site.hvac is not None and hvac_levels != 2:
        hv = replace(site.hvac, n_duty_levels=hvac_levels)
        site = replace(site, hvac=replace(hv, n_states=hv.states_for_duty_levels(site.horizon.dt)))
    site = apply_band(site, p)
    fc = demo_forecasts(site.horizon, tariff=p.get("tariff", "dynamic"),
                        solar_peak_kw=float(p.get("solar_peak", 5.0)))
    fc = apply_edits(fc, site.horizon, p)
    return value_stored_energy(apply_inverter(site, p), fc), fc, tank_lp


def apply_inverter(site: SiteConfig, p: dict) -> SiteConfig:
    """With `hybrid`, the PV and every battery behind one hybrid inverter
    (types.hybrid_inverter): `inverter_kw` its AC rating each way (0 or
    absent: no limit), `inverter_eta` its efficiency each way."""
    if not p.get("hybrid"):
        return site
    kw = float(p.get("inverter_kw") or 0.0)
    eta = float(p.get("inverter_eta", 0.97))
    keys = tuple(SiteConfig.battery_key(i) for i, b in enumerate(site.battery_list) if b.capacity_kwh > 0)
    return replace(site, submeters=(hybrid_inverter(keys, kw or None, kw or None, eta, eta),))


def value_stored_energy(site: SiteConfig, fc: Forecasts) -> SiteConfig:
    """Value a kWh left in a battery at the end at the horizon's AVERAGE import
    price, in the plan and in the ledger alike. The library default, the
    cheapest import price, is a floor on that energy's worth; on this page it
    let the batteries empty themselves by the end of the day."""
    if site.battery is None:
        return site
    price = float(np.mean(fc.buy))
    return replace(site, battery=replace(site.battery, terminal_price=price),
                   batteries=tuple(replace(b, terminal_price=price) for b in site.batteries))


def solve(p: dict, progress: Callable[..., None] | None = None) -> dict:
    if p.get("method") == "admm":
        return solve_admm(p, progress)
    site, fc, tank_lp = _site_fc(p)

    def coordinator(fc: Forecasts) -> DWCoordinator:
        return DWCoordinator(site, fc, battery_in_master=bool(p.get("battery_in_master", True)),
                             tank_in_master=tank_lp and site.water_heater is not None,
                             solver=p.get("solver", "numpy"))
    co = coordinator(fc)
    settings = dict(
        max_iter=int(p.get("max_iter", 100)),
        # "auto": stronger smoothing, and every plan kept, when batteries bid plans
        smoothing="auto" if p.get("smoothing", True) else 0.0,
        heuristic_columns=bool(p.get("heuristic_columns", True)),
        integer=p.get("recovery", "milp"),
        polish=bool(p.get("polish", True)),
        pool=p.get("pool", "auto"),
        seed_admm=bool(p.get("seed_admm", False)),
        response=p.get("response", "proposals"),
    )
    r = co.run(**settings, record=True, anytime=True, progress=_reporter(p, progress))

    def dark() -> tuple[DWCoordinator, dict[str, tuple[np.ndarray, np.ndarray]]]:
        co_dark = coordinator(_dark(fc))
        r_dark = co_dark.run(**settings, anytime=True)
        return co_dark, {k: (c.power, c.trajectory) for k, c in r_dark.plan.items()}
    _defer_ledger(co, {k: (c.power, c.trajectory) for k, c in r.plan.items()}, dark)
    n = site.horizon.steps
    batt_keys = [SiteConfig.battery_key(i) for i in range(len(site.battery_list))]

    # The recovered plan carries the FINAL master's prices: it is the plan
    # those prices were computed for, minus the fractional mixing.
    final = r.relaxed_view
    assert final is not None and r.plan_parts is not None    # both kept by run(record=True)
    plan_snap = {
        "powers": {k: c.power for k, c in r.plan.items()},
        "trajectories": {k: c.trajectory for k, c in r.plan.items()},
        "price": final["price"],
        "local_prices": final.get("local_prices"),
        "lambda": final["lambda"],
        # A modulating tank's plan is its blend: show the weights it kept.
        "active": {k: (c.mix if c.source == "blend" else [(1.0, c.source, c.born)])
                   for k, c in r.plan.items() if k in final["active"]},
    }
    plan = _view(co, fc, plan_snap, batt_keys)

    base_obj = baseline_objective(site, fc)
    _, base_bill = baseline_solution(site, fc)
    available = base_obj - r.lower

    iterations = []
    for h in r.history:
        v = _view(co, fc, h["snapshot"], batt_keys)
        v.update(index=h["iter"], rmp=round(h["rmp"], 4),
                 lb=round(h["lb"], 4) if np.isfinite(h["lb"]) else None,
                 gap=round(h["gap"], 4) if np.isfinite(h["gap"]) else None,
                 columns=h["columns"], added=h["added"], ms=round(h["ms"], 1),
                 plan_value=round(h["plan_value"], 4), best_plan=round(h["best_plan"], 4),
                 plan_parts={k: round(v, 4) for k, v in h["plan_parts"].items()})
        iterations.append(v)
    relaxed = _view(co, fc, final, batt_keys)

    pool = {}
    for k, cols in (r.columns or {}).items():
        by_src: dict[str, int] = {}
        for c in cols:
            by_src[c.source] = by_src.get(c.source, 0) + 1
        pool[k] = by_src

    return {
        "dt": site.horizon.dt,
        "steps": n,
        "hours": site.horizon.hours,
        "buy": _series(fc.buy),
        "sell": _series(fc.sell),
        "solar": _series(fc.solar),
        "load": _series(fc.load),
        "hot_water": _series(fc.hot_water_demand),
        "outdoor_temp": _series(fc.outdoor_temp),
        "plan": plan,
        "relaxed": relaxed,
        "iterations": iterations,
        "battery_keys": [k for k in batt_keys if k in plan["devices"]],
        "battery_capacities": [b.capacity_kwh for b in site.battery_list],
        "battery_floors": [b.soe_floor_kwh for b in site.battery_list],
        "grid": {"max_import_kw": site.grid.max_import_kw,
                 "max_export_kw": site.grid.max_export_kw},
        "inverter": _inverter(site),
        "comfort": _comfort(site),
        "pool": pool,
        "ledger": ledger_route(p)["ledger"] if p.get("ledger") else None,
        "summary": {
            "upper": round(r.upper, 4),
            "lower": round(r.lower, 4),
            "relaxed": round(r.relaxed, 4),
            "gap": round(r.upper - r.lower, 4),
            "baseline_objective": round(base_obj, 4),
            "baseline_bill": round(base_bill, 4),
            "capture": round((base_obj - r.upper) / available, 4) if available > 1e-9 else None,
            "pricing_error": round(r.pricing_error, 4),
            "stop_reason": r.stop_reason,
            "incumbent_used": r.incumbent_used,
            "max_iter": int(p.get("max_iter", 100)),
            "gap_tol": round(1e-3 * max(1.0, abs(r.relaxed)), 5),
            "plan_parts": {k: round(v, 4) for k, v in r.plan_parts.items()},
            "fractional_devices": r.fractional_devices,
            "n_columns": r.n_columns,
            "iterations": r.iterations,
            "solve_ms": round(r.ms, 1),
            "method": "dw",
            "battery_in_master": bool(p.get("battery_in_master", True)),
            "grid": f"{site.battery.n_states}x{site.battery.n_actions}" if site.battery else "-",
            "tank": ("continuous duty, in the master LP" if tank_lp and site.water_heater
                     else f"{site.water_heater.n_duty_levels} duty levels, {site.water_heater.n_states} states"
                     if site.water_heater else "-"),
            "pool": co.pool_used,
            "smoothing": co.smoothing_used,
            "solver": p.get("solver", "numpy"),
            "response": p.get("response", "proposals"),
            "peak_pool": max((h["columns"] for h in r.history), default=r.n_columns),
            "admm_value": round(r.admm_value, 4) if r.admm_value is not None else None,
        },
    }


def _reporter(p: dict, progress: Callable[..., None] | None
              ) -> Callable[..., None] | None:
    """One callback for both transports: the caller's (the browser worker's)
    and the polled table (the server's), whichever are present."""
    pid = p.get("_progress_id")
    if progress is None and pid is None:
        return None

    last: list = []

    def report(it: int, n: int, *metrics: float | None) -> None:
        # metrics come when an iteration ends (blend, bound, best runnable);
        # the call as the next one starts carries none, and must not wipe the
        # last ones from what a poll between the two sees
        if metrics:
            last[:] = metrics
        if pid is not None:
            _PROGRESS[str(pid)] = [it, n, *last]
        if progress is not None:
            progress(it, n, *metrics)
    return report


def solve_admm(p: dict, progress: Callable[..., None] | None = None) -> dict:
    """The same view of the same site, planned by ADMM instead (proximal
    message passing, admm.coordinator).

    Returns the fields the page reads for DW, with what ADMM has: one runnable
    plan per device per iteration (no blending), no lower bound, and each
    iteration's objective in place of the blend. `xrho` is the starting rho,
    `rho_adapt` lets it adapt. `battery_step`: "dp", each battery's DP on its
    grid, or "lp", its LP solved exactly (admm.battery_qp); on a site of
    batteries alone the bill's kink is then left unrounded, so the loop
    converges to the optimum. The tank is its DP as configured - "continuous
    (LP)" is a DW option and runs here as the on/off element. `warm_start`:
    start from where the last finished solve stood at its best plan (if the
    devices and horizon are the same) and warm-start each battery's LP step;
    off, every solve starts from zero and does not depend on the one before.

    The solve is a resumable object, so the page can run it a few iterations
    at a time (live figures; the in-browser worker cannot be interrupted
    mid-call), pause it to look at the best plan so far, and carry on.
    `chunk`: iterations per call; `resume`: continue the run with this
    `_progress_id`; `pause`: return the best plan so far and keep it.
    """
    from home_energy_optimizer.coordinate import coordinate
    from home_energy_optimizer.admm.coordinator import ExchangeRun
    from home_energy_optimizer.policy import PolicySnapshot, marginal_value
    from home_energy_optimizer.types import CoordinationConfig

    site, fc, _ = _site_fc(p)
    rounds = int(p.get("max_iter", 100))
    lp = p.get("battery_step", "dp") == "lp"
    # Batteries alone with LP steps: every step exact and convex, so the loop
    # converges to the optimum - and rounding the kink would only move it.
    # With the on/off tank or HVAC the rounding helps runs converge.
    exact = lp and site.water_heater is None and site.hvac is None
    exact_only: dict[str, Any] = {"kink_smoothing": 0.0} if exact else {}
    cc = CoordinationConfig(exchange_rho=float(p.get("xrho", 0.1)),
                            exchange_rho_gain=0.01 if p.get("rho_adapt", True) else 0.0,
                            exchange_rounds=rounds, polish=bool(p.get("polish", True)),
                            exchange_battery_step="lp" if lp else "dp",
                            exchange_warm_battery=bool(p.get("warm_start", False)),
                            **exact_only)
    site = replace(site, coordination=cc)
    pid = str(p.get("_progress_id", ""))
    run = _RUNS.get(pid) if p.get("resume") else None
    if run is None:
        _RUNS.clear()
        warm = _LAST_WARM.get("state") if p.get("warm_start") else None
        run = _RUNS[pid] = ExchangeRun(site, fc, warm=warm)
    run.progress = _reporter(p, progress)
    if not p.get("pause"):
        chunk = p.get("chunk")
        if not run.step(int(chunk) if chunk else None) and chunk:
            last = run.records[-1].total_objective if run.records else None
            best = run.best_obj if np.isfinite(run.best_obj) else None
            return {"running": True, "progress": [run.k, run.max_rounds, last, None, best]}
    res = run.result()
    if run.done:
        _RUNS.pop(pid, None)
        _LAST_WARM["state"] = run.warm_state()          # for the next solve's warm start
    solve_ms = run.seconds * 1000
    co = DWCoordinator(site, fc)           # for the shared plan view and scoring only

    def dark() -> tuple[DWCoordinator, dict[str, tuple[np.ndarray, np.ndarray]]]:
        res_dark = coordinate(site, _dark(fc))
        return (DWCoordinator(site, _dark(fc)),
                {k: (d.power, d.trajectory) for k, d in res_dark.devices.items()})
    _defer_ledger(co, {k: (d.power, d.trajectory) for k, d in res.devices.items()}, dark)
    batt_keys = [SiteConfig.battery_key(i) for i in range(len(site.battery_list))]

    def snap_of(powers: dict[str, np.ndarray], trajs: dict[str, np.ndarray],
                lam: dict | None = None) -> dict:
        return {"powers": powers, "trajectories": trajs, "price": None,
                "lambda": lam or {}, "active": {}}

    def parts(powers: dict[str, np.ndarray], trajs: dict[str, np.ndarray]) -> dict:
        return co.parts({k: Column(powers[k], trajs[k], 0.0, "admm") for k in powers})

    powers = {k: d.power for k, d in res.devices.items()}
    trajs = {k: d.trajectory for k, d in res.devices.items()}
    lam = {}
    if "battery" in res.devices and res.battery_pricing is not None:
        # value of a stored kWh along the plan, from the fast tier's value function
        ps = PolicySnapshot.from_result(site, fc, res)
        soe = trajs["battery"]
        lam["battery"] = np.array([marginal_value(ps, t, float(soe[t])) for t in range(site.horizon.steps)])
    plan = _view(co, fc, snap_of(powers, trajs, lam), batt_keys)
    plan_parts = parts(powers, trajs)

    # Every iteration's numbers; full plans only for about ten evenly spaced
    # iterations, the best and the last - the rest would be most of the payload.
    n_rounds = len(res.rounds)
    kept = {int(i) for i in np.linspace(0, n_rounds - 1, min(n_rounds, 10)).round()} | {res.selected_round, n_rounds - 1}
    iterations, best = [], np.inf
    for r in res.rounds:
        pp = parts(r.powers, r.trajectories)
        best = min(best, pp["total"])
        v = _view(co, fc, snap_of(r.powers, r.trajectories), batt_keys) if r.index in kept else {}
        v.update(index=r.index + 1, rmp=None, lb=None, gap=None, columns=None, added=None,
                 ms=round(r.round_ms, 1), plan_value=round(pp["total"], 4), best_plan=round(best, 4),
                 rho=round(float(r.rho), 4), snapshot=r.index in kept,
                 relaxed_value=None if r.relaxed_objective is None else round(r.relaxed_objective, 4))
        if r.index in kept:
            v["plan_parts"] = {k: round(x, 4) for k, x in pp.items()}
        iterations.append(v)

    base_obj = baseline_objective(site, fc)
    upper = plan_parts["total"]
    return {
        "dt": site.horizon.dt, "steps": site.horizon.steps, "hours": site.horizon.hours,
        "buy": _series(fc.buy), "sell": _series(fc.sell), "solar": _series(fc.solar),
        "load": _series(fc.load), "hot_water": _series(fc.hot_water_demand),
        "outdoor_temp": _series(fc.outdoor_temp),
        "plan": plan, "relaxed": plan, "iterations": iterations,
        "battery_keys": [k for k in batt_keys if k in plan["devices"]],
        "battery_capacities": [b.capacity_kwh for b in site.battery_list],
        "battery_floors": [b.soe_floor_kwh for b in site.battery_list],
        "grid": {"max_import_kw": site.grid.max_import_kw, "max_export_kw": site.grid.max_export_kw},
        "inverter": _inverter(site),
        "comfort": _comfort(site),
        "pool": {},
        "ledger": ledger_route(p)["ledger"] if p.get("ledger") else None,
        "summary": {
            "method": "admm", "upper": round(upper, 4), "lower": None, "relaxed": None, "gap": None,
            "baseline_objective": round(base_obj, 4), "capture": None,
            "stop_reason": res.stop_reason or ("iteration cap" if res.rounds_run >= rounds else "converged"),
            "patience": site.coordination.exchange_patience,
            "battery_step": site.coordination.exchange_battery_step,
            "warm_started": run.warm_used,
            # the iterations plan with a fractional tank / HVAC; the returned
            # plan is recovered for the real devices from the best iteration
            "relaxed_rounds": (site.water_heater is not None or site.hvac is not None)
                              and site.coordination.relax_levels > 2,
            "incumbent_used": res.selected_round != len(res.rounds) - 1,
            "selected_round": res.selected_round + 1,
            "plan_parts": {k: round(x, 4) for k, x in plan_parts.items()},
            "iterations": res.rounds_run, "solve_ms": round(solve_ms, 1), "admm_value": None,
            "rho": float(p.get("xrho", 0.1)),
            "rho_adapt": bool(p.get("rho_adapt", True)),
        },
    }


ROUTES: dict[str, Callable[..., dict]] = {"solve": solve, "progress": progress, "ledger": ledger_route}


def call(name: str, payload: dict,
         progress: Callable[..., None] | None = None) -> dict:
    fn = ROUTES.get(name)
    if fn is None:
        return {"error": f"unknown route {name!r}"}
    try:
        if name == "solve":
            try:
                return fn(payload, progress)
            finally:
                _PROGRESS.pop(str(payload.get("_progress_id", "")), None)
        return fn(payload)
    except Exception as exc:  # surfaced in the UI rather than swallowed
        return {"error": f"{type(exc).__name__}: {exc}"}
