"""Backend for the testbed UI, independent of how the call arrives.

`admm/gui/server.py` wraps this in HTTP; the standalone page in `admm/wasm/` calls the
same functions directly inside Pyodide. Keeping the logic here rather than in
the server is what lets the two share an implementation instead of drifting -
the same reason the solver core is a package rather than something the GUI
reimplements.

Everything in and out is plain JSON-safe Python: dicts, lists, floats. No
numpy arrays escape, because the browser side has to hand them to
`json.dumps`.
"""

from __future__ import annotations

import threading

import numpy as np

from home_energy_optimizer.coordinate import (
    _pricing_resolve,
    baseline_solution,
    coordinate,
    grid_penalty,
)
from home_energy_optimizer.policy import (
    PolicySnapshot,
    HardLimits,
    action,
    clamp,
    evaluate,
    marginal_value,
    reservation_prices,
    rollout,
)
from home_energy_optimizer.profiles import demo_forecasts
from home_energy_optimizer.types import (
    BatteryConfig,
    CoordinationConfig,
    CoordinationResult,
    Forecasts,
    GridLimits,
    Horizon,
    HvacConfig,
    SiteConfig,
    WaterHeaterConfig,
)

# Last solve. A single-user tool in both transports - one browser tab, or one
# local server - so a plain dict behind a lock is enough.
_state: dict = {}
_lock = threading.Lock()


def build_site(p: dict) -> SiteConfig:
    horizon = Horizon(dt=0.25, hours=float(p.get("hours", 48)))
    grid = int(p.get("grid", 200))
    # Capacity 0 means "no battery", not an invalid battery: the slider bottoms
    # out at 0 and that should disable the device, not raise.
    capacity = float(p.get("batt_capacity", 10.0))

    def make_batt(cap: float) -> BatteryConfig:
        return BatteryConfig(
            capacity_kwh=cap,
            p_charge_max_kw=float(p.get("batt_charge", 5.0)),
            p_discharge_max_kw=float(p.get("batt_discharge", 5.0)),
            eta=float(p.get("eta", 0.90)),
            n_states=grid,
            n_actions=max(9, grid // 2 + 1),
            terminal_mode=p.get("terminal_mode", "linear"),
            soe_min_frac=float(p.get("soe_min", 0.0)) / 100.0,
        )

    n_batt = int(p.get("n_batteries", 1))
    on = p.get("enable_battery", True) and capacity > 0
    batt = make_batt(capacity) if on else None
    # Extra units are deliberately DIFFERENT sizes - a fleet of identical
    # batteries is the easy case and hides whether the coordination shares work
    # sensibly. Sizes fan out around the base scale rather than all matching,
    # so a 20-unit fleet is heterogeneous the way a real one would be.
    scale = float(p.get("batt2_scale", 2.0))
    extra = (
        tuple(
            make_batt(capacity * scale * (0.6 + 0.8 * i / max(n_batt - 2, 1)))
            for i in range(n_batt - 1)
        )
        if on and n_batt > 1 else ()
    )
    wh = (WaterHeaterConfig(t_comfort=float(p.get("tank_comfort", 55.0)))
          if p.get("enable_wh", True) else None)
    # the room's flat band, whichever order the two values are in (the DW page
    # drags an hourly one instead: dw.webapi.apply_band)
    lo, hi = sorted((float(p.get("room_low", 22.0)), float(p.get("room_high", 26.0))))
    hvac = HvacConfig(t_comfort_low=lo, t_comfort_high=hi) if p.get("enable_hvac", True) else None
    return SiteConfig(
        horizon=horizon,
        battery=batt,
        batteries=extra,
        water_heater=wh,
        hvac=hvac,
        grid=GridLimits(
            max_import_kw=(float(p["max_import_kw"]) if p.get("max_import_kw") else None),
            max_export_kw=(float(p["max_export_kw"]) if p.get("max_export_kw") else None),
            allow_curtailment=bool(p.get("allow_curtailment", True)),
        ),
        coordination=CoordinationConfig(exchange_rounds=int(p.get("max_rounds", 100))),
    )


def _series(x) -> list[float]:
    return [round(float(v), 5) for v in np.asarray(x).ravel()]


def _price_curve(site: SiteConfig, fc: Forecasts, res, soe) -> dict:
    """lambda and the reservation band along a state-of-energy trajectory.

    Split out because a coordination round needs exactly the same thing,
    computed against that round's residual load rather than the retained one's.
    """
    n = site.horizon.steps
    snap = PolicySnapshot.from_result(site, fc, res)
    lam = [float(marginal_value(snap, t, float(soe[t]))) for t in range(n)]
    rp = [reservation_prices(snap, t, float(soe[t])) for t in range(n)]
    return {
        "snapshot": snap,
        "lambda": [round(v, 5) for v in lam],
        "import_below": [round(float(r["import_below"]), 5) for r in rp],
        "export_above": [round(float(r["export_above"]), 5) for r in rp],
    }


def _round_result(res, rec):
    """A CoordinationResult standing in for one round, for pricing purposes.

    Only the fields PolicySnapshot.from_result reads are populated. The value
    function is re-solved from the round's own dp_load - rounds do not carry
    one, because storing V for every round is two orders of magnitude more
    memory than storing the plans.
    """
    return CoordinationResult(
        devices=res.devices,
        net_grid=rec.net_grid,
        import_cost=rec.import_cost,
        export_revenue=rec.export_revenue,
        net_cost=rec.net_cost,
        total_objective=rec.total_objective,
        rounds_run=res.rounds_run,
        battery_dp_load=rec.battery_dp_load,
    )


def _rounds_payload(res, dt: float, site, fc) -> list[dict]:
    """Every round's plan, in the same shape the top-level payload uses.

    The charts read these directly, so a round swap is a redraw rather than a
    re-solve or a second call.
    """
    return [
        {
            "index": r.index,
            "selected": r.selected,
            "net_grid": _series(r.net_grid),
            "curtailment": _series(r.curtailment),
            "devices": {
                k: {"power": _series(v), "trajectory": _series(r.trajectories[k])}
                for k, v in r.powers.items()
            },
            "costs": {
                "net_cost": round(r.net_cost, 4),
                "import_cost": round(r.import_cost, 4),
                "export_revenue": round(r.export_revenue, 4),
                "total_objective": round(r.total_objective, 4),
            },
            "violation": round(r.violation, 4),
            "breach_cost": round(grid_penalty(site, np.asarray(r.net_grid), fc.buy, fc.sell), 4),
            "score_objective": round(r.score_objective, 4),
            "fallback_applied": r.fallback_applied,
            "import_excess": round(r.import_excess, 4),
            "export_excess": round(r.export_excess, 4),
            "primal_res": round(r.primal_res, 4),
            "dual_res": round(r.dual_res, 4),
            "rho": round(r.rho, 3),
            "round_ms": round(r.round_ms, 1),
            "curtailed_kwh": round(float(np.sum(r.curtailment)) * dt, 3),
        }
        for r in res.rounds
    ]


def solve(p: dict) -> dict:
    site = build_site(p)
    fc = demo_forecasts(
        site.horizon,
        tariff=p.get("tariff", "dynamic"),
        solar_peak_kw=float(p.get("solar_peak", 5.0)),
    )
    res = coordinate(site, fc)
    n = site.horizon.steps

    snap = None
    lam: list = [None] * n
    bid = ask = None
    if "battery" in res.devices:
        pc = _price_curve(site, fc, res, res.devices["battery"].trajectory)
        snap, lam, bid, ask = pc["snapshot"], pc["lambda"], pc["import_below"], pc["export_above"]

    with _lock:
        _state.clear()
        # `prices` caches the per-round lambda curves: each one costs a battery
        # re-solve, so it is computed when a round is first opened rather than
        # for all of them up front.
        _state.update(site=site, forecasts=fc, result=res, snapshot=snap, prices={})

    _, base_bill = baseline_solution(site, fc)

    devices = {
        name: {
            "power": _series(sol.power),
            "trajectory": _series(sol.trajectory),
            "solve_ms": round(sol.solve_ms, 2),
        }
        for name, sol in res.devices.items()
    }

    return {
        "dt": site.horizon.dt,
        "steps": n,
        "hours": site.horizon.hours,
        "buy": _series(fc.buy),
        "sell": _series(fc.sell),
        "solar": _series(fc.solar),
        "load": _series(fc.load),
        "outdoor_temp": _series(fc.outdoor_temp),
        "net_grid": _series(res.net_grid),
        "curtailment": _series(res.curtailment if res.curtailment is not None else np.zeros(n)),
        "lambda": [None if v is None else round(v, 5) for v in lam],
        "import_below": bid,
        "export_above": ask,
        "devices": devices,
        "rounds": _rounds_payload(res, site.horizon.dt, site, fc),
        "selected_round": res.selected_round,
        "battery_capacity": site.battery.capacity_kwh if site.battery else 0.0,
        "battery_capacities": [b.capacity_kwh for b in site.battery_list],
        "battery_keys": [SiteConfig.battery_key(i) for i in range(len(site.battery_list))],
        "grid": {
            "max_import_kw": site.grid.max_import_kw,
            "max_export_kw": site.grid.max_export_kw,
            "import_excess": round(res.grid_import_excess, 4),
            "export_excess": round(res.grid_export_excess, 4),
            "curtailed_kwh": round(res.curtailed_kwh, 3),
        },
        "costs": {
            "net_cost": round(res.net_cost, 4),
            "import_cost": round(res.import_cost, 4),
            "export_revenue": round(res.export_revenue, 4),
            "baseline_cost": round(base_bill, 4),
            "savings": round(base_bill - res.net_cost, 4),
            "total_objective": round(res.total_objective, 4),
        },
        "meta": {
            "rounds_run": res.rounds_run,
            "solve_ms": round(sum(t.get("round_ms", 0.0) for t in res.timings), 1),
            "grid": f"{site.battery.n_states}x{site.battery.n_actions}" if site.battery else "-",
        },
    }


def _snap_or_error() -> PolicySnapshot:
    with _lock:
        snap = _state.get("snapshot")
    if snap is None:
        raise ValueError("no battery snapshot; solve with the battery enabled first")
    return snap


def round_prices(p: dict) -> dict:
    """lambda for a coordination round other than the retained one.

    Computed on demand and cached: each answer is a battery re-solve at
    rho = 0, against THAT round's residual load.
    """
    with _lock:
        site, fc = _state.get("site"), _state.get("forecasts")
        res, cache = _state.get("result"), _state.get("prices")
    if res is None:
        raise ValueError("no solve yet")
    r = int(p["r"])
    if not 0 <= r < len(res.rounds):
        raise ValueError(f"round {r} out of range (0..{len(res.rounds) - 1})")
    if r not in cache:
        rec = res.rounds[r]
        if "battery" not in res.devices or rec.battery_dp_load is None:
            cache[r] = {"lambda": None, "import_below": None, "export_above": None}
        else:
            stand_in = _round_result(res, rec)
            stand_in.battery_pricing = _pricing_resolve(site, fc, stand_in)
            pc = _price_curve(site, fc, stand_in, rec.trajectories["battery"])
            cache[r] = {k: pc[k] for k in ("lambda", "import_below", "export_above")}
    return {"round": r, **cache[r]}


def policy_at(p: dict) -> dict:
    snap = _snap_or_error()
    t, soe = int(p["t"]), float(p["soe"])
    return {
        "action_kw": round(action(snap, t, soe), 4),
        "lambda_per_kwh": round(marginal_value(snap, t, soe), 5),
    }


def lambda_at(p: dict) -> dict:
    snap = _snap_or_error()
    return {"lambda_per_kwh": round(marginal_value(snap, int(p["t"]), float(p["soe"])), 5)}


def rollout_at(p: dict) -> dict:
    snap = _snap_or_error()
    t = int(p["t"])
    s, pw = rollout(snap, t, float(p["soe"]))
    return {
        "start_step": t,
        "soe": [round(float(v), 4) for v in s],
        "power": [round(float(v), 4) for v in pw],
    }


def evaluate_at(p: dict) -> dict:
    snap = _snap_or_error()
    cf = evaluate(snap, int(p["t"]), float(p["soe"]), float(p["action"]))
    return {
        "forced_action": round(cf.forced_action, 4),
        "optimal_action": round(cf.optimal_action, 4),
        "opportunity_cost": round(cf.opportunity_cost, 5),
        "elapsed_ms": round(cf.elapsed_ms, 4),
    }


def setpoint_at(p: dict) -> dict:
    snap = _snap_or_error()
    t, soe = int(p["t"]), float(p["soe"])
    raw = action(snap, t, soe)
    safe, bound_by = clamp(
        raw,
        other_load_kw=float(p.get("other_load_kw", 0.0)),
        limits=HardLimits(
            max_import_kw=p.get("max_import_kw"),
            max_export_kw=p.get("max_export_kw"),
            max_charge_kw=p.get("max_charge_kw"),
            max_discharge_kw=p.get("max_discharge_kw"),
            max_ramp_kw_per_step=p.get("max_ramp_kw_per_step"),
        ),
        previous_kw=p.get("previous_kw"),
    )
    return {
        "setpoint_kw": round(safe, 4),
        "unclamped_kw": round(raw, 4),
        "bound_by": bound_by,
        "lambda_per_kwh": round(marginal_value(snap, t, soe), 5),
    }


# The single entry point the UI uses, whatever the transport. Names match the
# `/api/<name>` paths the HTTP server exposes, so one table serves both.
ROUTES = {
    "solve": solve,
    "round": round_prices,
    "policy": policy_at,
    "lambda": lambda_at,
    "rollout": rollout_at,
    "evaluate": evaluate_at,
    "setpoint": setpoint_at,
}


def call(name: str, payload: dict) -> dict:
    fn = ROUTES.get(name)
    if fn is None:
        raise ValueError(f"no such route: {name}")
    return fn(payload or {})
