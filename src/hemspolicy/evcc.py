"""evcc optimizer wire contract, so a stock evcc binary can use this solver.

evcc reaches its optimizer over HTTP at `OPTIMIZER_URI`, which is an ordinary
environment variable (`core/site_optimizer.go:341`):

    OPTIMIZER_URI=http://127.0.0.1:8765 evcc

so implementing this contract makes `hems-policy` a drop-in replacement for
`https://optimizer.evcc.io` with **zero changes to evcc**. No fork, no patch.

The contract is transcribed from the generated client in
`github.com/evcc-io/optimizer/client` (pinned by commit in evcc's go.mod), not
guessed from traffic:

    POST /optimize/charge-schedule   OptimizationInput -> OptimizationResult
    GET  /optimize/health            -> {status, message}

Three things to get right, and all three are easy to get wrong:

* **Units.** evcc speaks W and Wh; this package speaks kW and kWh. Prices are
  per **Wh** on the wire (`p_N`, `p_E`, `p_a`), per kWh here.
* **Energy, not power.** `ft`, `gt` and the battery result arrays are energies
  per slot in Wh, despite `charging_power`'s name. Slot length comes from
  `dt[]`, in seconds, and evcc's FIRST slot is usually short - it is the
  remainder of the current quarter-hour.
* **Sign.** evcc splits charge and discharge into separate non-negative arrays;
  this package uses one signed series with positive = charging.

**Which coordinator.** Requests are planned by Dantzig-Wolfe by default
(`hemspolicy.plan`). A home battery then sits in the master LP exactly, and an
EV loadpoint - which carries p_demand, s_goal and a c_min floor - bids charging
plans from its DP, which honours all three. Grid limits become rows of the
master: met, or - only when physically unavoidable - breached at a fixed
breach price (10x the highest price on the horizon) and then reported in the
overshoot arrays. Set `HEMS_METHOD=admm` for the previous behaviour (ADMM, limits reported only).

Non-uniform `dt` is the one structural mismatch: `Horizon` assumes a fixed
step. We solve on the modal step and report which slots were resampled, rather
than silently pretending the grid matched - a misaligned plan that reports
`Optimal` is exactly the failure mode EMHASS's AGENTS.md warns about.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import replace

import numpy as np

from .planner import METHODS, plan
from .types import (
    BatteryConfig,
    Forecasts,
    GridLimits,
    Horizon,
    SiteConfig,
)

W_PER_KW = 1000.0
SECONDS_PER_HOUR = 3600.0


class ContractError(ValueError):
    """The request does not satisfy the evcc optimizer contract."""


# --------------------------------------------------------------------------
# Request -> internal
# --------------------------------------------------------------------------


def _require(d: dict, key: str):
    if key not in d:
        raise ContractError(f"missing required field {key!r}")
    return d[key]


def horizon_from_dt(dt_seconds: list[int]) -> tuple[Horizon, float, bool]:
    """Pick a uniform horizon from evcc's per-slot durations.

    Returns (horizon, modal_dt_hours, was_uniform). evcc's first slot is
    typically a partial quarter-hour, so `was_uniform` is usually False and the
    caller should say so rather than hide it.
    """
    if not dt_seconds:
        raise ContractError("time_series.dt is empty")
    if any(d <= 0 for d in dt_seconds):
        raise ContractError("time_series.dt contains a non-positive duration")

    modal = Counter(dt_seconds).most_common(1)[0][0]
    dt_h = modal / SECONDS_PER_HOUR
    n = len(dt_seconds)
    return Horizon(dt=dt_h, hours=dt_h * n), dt_h, len(set(dt_seconds)) == 1


def request_to_site(payload: dict) -> tuple[SiteConfig, Forecasts, dict]:
    """Translate an `OptimizationInput` into a SiteConfig + Forecasts.

    Returns the pair plus a `meta` dict describing what had to be approximated,
    so the caller can surface it instead of pretending the mapping was exact.
    """
    ts = _require(payload, "time_series")
    dt_seconds = [int(v) for v in _require(ts, "dt")]
    horizon, dt_h, uniform = horizon_from_dt(dt_seconds)
    n = horizon.steps

    def energy_wh_to_kw(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size != n:
            raise ContractError(f"time_series.{name} has {arr.size} entries, expected {n}")
        # Wh per slot -> average kW over the slot, using each slot's own length
        # so a short first slot is not read as a dip in demand.
        secs = np.array(dt_seconds, dtype=float)
        return arr / W_PER_KW / (secs / SECONDS_PER_HOUR)

    def price_per_wh_to_kwh(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size != n:
            raise ContractError(f"time_series.{name} has {arr.size} entries, expected {n}")
        return arr * W_PER_KW

    load = energy_wh_to_kw(_require(ts, "gt"), "gt")
    solar = energy_wh_to_kw(ts.get("ft", [0.0] * n), "ft")
    buy = price_per_wh_to_kwh(_require(ts, "p_N"), "p_N")
    sell = price_per_wh_to_kwh(_require(ts, "p_E"), "p_E")

    batteries = _require(payload, "batteries")
    if not batteries:
        raise ContractError("batteries is empty; nothing to optimise")

    eta_c = float(payload.get("eta_c") or 0.95)
    eta_d = float(payload.get("eta_d") or 0.95)
    secs = np.array(dt_seconds, dtype=float)
    slot_h = secs / SECONDS_PER_HOUR

    configs: list[BatteryConfig] = []
    offsets: list[float] = []
    notes: list[str] = []

    for idx, b in enumerate(batteries):
        s_max = float(_require(b, "s_max")) / W_PER_KW
        s_min = float(b.get("s_min", 0.0)) / W_PER_KW
        s_cap = float(b.get("s_capacity") or 0.0) / W_PER_KW or s_max
        s_initial = float(_require(b, "s_initial")) / W_PER_KW

        if s_initial > s_cap + 1e-9:
            raise ContractError(
                f"batteries.{idx}: s_initial ({s_initial} kWh) exceeds s_capacity ({s_cap} kWh)"
            )

        # The usable window is [s_min, s_max]; model a battery of that size and
        # carry the offset through on the way out.
        usable = max(s_max - s_min, 1e-6)

        # p_demand is a minimum charge ENERGY per slot (Wh). It is how evcc
        # states that a car needs charging - a loadpoint arrives as a battery
        # with d_max = 0 and its requirement entirely in p_demand, so ignoring
        # it means never charging the EV.
        min_charge = None
        if b.get("p_demand"):
            arr = np.asarray(b["p_demand"], dtype=float).ravel()
            if arr.size != n:
                raise ContractError(
                    f"batteries.{idx}.p_demand has {arr.size} entries, expected {n}"
                )
            if np.any(arr > 0):
                min_charge = tuple(arr / W_PER_KW / slot_h)

        # s_goal is a per-slot target stored energy (Wh), absolute, so shift it
        # into the usable window's coordinates.
        soc_goal = None
        if b.get("s_goal"):
            arr = np.asarray(b["s_goal"], dtype=float).ravel()
            if arr.size != n:
                raise ContractError(
                    f"batteries.{idx}.s_goal has {arr.size} entries, expected {n}"
                )
            if np.any(arr > 0):
                soc_goal = tuple(np.clip(arr / W_PER_KW - s_min, 0.0, usable))

        configs.append(
            BatteryConfig(
                capacity_kwh=usable,
                p_charge_max_kw=float(_require(b, "c_max")) / W_PER_KW,
                # A loadpoint sends d_max = 0 (no V2G). A battery that cannot
                # discharge is legitimate; guard the zero so the action grid
                # stays well formed.
                p_discharge_max_kw=max(float(_require(b, "d_max")) / W_PER_KW, 1e-9),
                eta_charge=eta_c,
                eta_discharge=eta_d,
                soc_initial_frac=float(np.clip((s_initial - s_min) / usable, 0.0, 1.0)),
                terminal_mode="linear",
                # p_a is "monetary value of stored energy per Wh at end of
                # horizon" - exactly this package's linear terminal price.
                terminal_price=float(b.get("p_a", 0.0)) * W_PER_KW or None,
                min_charge_kw=min_charge,
                soc_goal_kwh=soc_goal,
                # c_min is a charger's physical floor: off, or at least this.
                # Ignoring it plans setpoints the hardware cannot execute.
                charge_deadband_kw=float(b.get("c_min", 0.0)) / W_PER_KW,
            )
        )
        offsets.append(s_min)
        if b.get("c_priority"):
            notes.append(f"batteries.{idx}: c_priority ignored")

    battery = configs[0]
    grid = payload.get("grid") or {}
    site = SiteConfig(
        horizon=horizon,
        battery=battery,
        batteries=tuple(configs[1:]),
        water_heater=None,  # evcc's contract carries batteries only
        hvac=None,
    )

    forecasts = Forecasts(
        buy=buy,
        sell=sell,
        load=load,
        solar=solar,
        outdoor_temp=np.zeros(n),  # unused without thermal devices
        hot_water_demand=np.zeros(n),
    )

    meta = {
        "slots": n,
        "dt_seconds_modal": int(dt_h * SECONDS_PER_HOUR),
        "uniform_dt": uniform,
        "dt_seconds": dt_seconds,
        "n_batteries": len(configs),
        "s_min_kwh": offsets[0],
        "offsets_kwh": offsets,
        "s_capacity_kwh": float(batteries[0].get("s_capacity") or batteries[0]["s_max"]) / W_PER_KW,
        "eta_c": eta_c,
        "eta_d": eta_d,
        "max_import_kw": float(grid.get("p_max_imp") or 0.0) / W_PER_KW or None,
        "max_export_kw": float(grid.get("p_max_exp") or 0.0) / W_PER_KW or None,
        "charge_from_grid": bool(b.get("charge_from_grid", False)),
        "discharge_to_grid": bool(b.get("discharge_to_grid", False)),
        "strategy": payload.get("strategy") or {},
        "notes": notes,
    }
    if not uniform:
        meta["notes"].append(
            f"non-uniform dt ({sorted(set(dt_seconds))}s); solved on the modal "
            f"{meta['dt_seconds_modal']}s step and reported per slot"
        )
    # Grid limits are honoured as hard bounds on the plan, not as penalties.
    if meta["max_import_kw"] or meta["max_export_kw"]:
        meta["notes"].append("grid limits applied post-hoc via clamping, not inside the LP")

    return site, forecasts, meta


# --------------------------------------------------------------------------
# Internal -> response
# --------------------------------------------------------------------------


def result_to_response(site: SiteConfig, fc: Forecasts, res, meta: dict) -> dict:
    """Translate a CoordinationResult into evcc's `OptimizationResult`."""
    dt_seconds = np.array(meta["dt_seconds"], dtype=float)
    slot_h = dt_seconds / SECONDS_PER_HOUR
    n = site.horizon.steps

    def series(a) -> list[float]:
        return [round(float(v), 4) for v in np.asarray(a).ravel()]

    # One BatteryResult per requested battery, in the order they were sent -
    # evcc matches them back positionally to its own loadpoints and meters
    # (site_optimizer.go: applyOptimizerResult walks details index-for-index),
    # so the ordering is load-bearing, not cosmetic.
    battery_results = []
    for i, offset in enumerate(meta["offsets_kwh"]):
        sol = res.devices.get(SiteConfig.battery_key(i))
        if sol is None:
            power_kw = np.zeros(n)
            soe_kwh = np.zeros(n + 1) + offset
        else:
            power_kw = sol.power
            soe_kwh = sol.trajectory
        battery_results.append(
            {
                "charging_power": series(np.maximum(power_kw, 0.0) * slot_h * W_PER_KW),
                "discharging_power": series(np.maximum(-power_kw, 0.0) * slot_h * W_PER_KW),
                # SoC back to absolute Wh, undoing the s_min offset.
                "state_of_charge": series((soe_kwh[1:] + offset) * W_PER_KW),
            }
        )

    net_kw = res.net_grid
    grid_import_wh = np.maximum(net_kw, 0.0) * slot_h * W_PER_KW
    grid_export_wh = np.maximum(-net_kw, 0.0) * slot_h * W_PER_KW
    flow = [(0 if v >= 0 else 1) for v in net_kw]

    # Grid limits: report where the plan would breach them rather than
    # silently clipping, which is what evcc's overshoot arrays are for.
    import_overshoot = np.zeros(n)
    export_overshoot = np.zeros(n)
    if meta.get("max_import_kw"):
        over = np.maximum(net_kw - meta["max_import_kw"], 0.0)
        import_overshoot = over * slot_h * W_PER_KW
    if meta.get("max_export_kw"):
        over = np.maximum(-net_kw - meta["max_export_kw"], 0.0)
        export_overshoot = over * slot_h * W_PER_KW

    response = {
        "status": "Optimal",
        # evcc's objective is an economic BENEFIT; ours is a cost.
        "objective_value": round(-float(res.net_cost), 6),
        "batteries": battery_results,
        "flow_direction": flow,
        "grid_import": series(grid_import_wh),
        "grid_export": series(grid_export_wh),
        "grid_import_overshoot": series(import_overshoot),
        "grid_export_overshoot": series(export_overshoot),
        "limit_violations": {
            "grid_import_limit_exceeded": bool(import_overshoot.sum() > 1e-6),
            "grid_export_limit_hit": bool(export_overshoot.sum() > 1e-6),
        },
    }
    if res.lower_bound is not None:
        # Not part of evcc's schema. Costs, not benefits: no plan costs less
        # than `lower_bound`, so this one is within `gap` of the best possible
        # (up to the device DPs' grids).
        response["_hems_policy_certificate"] = {
            "method": res.method,
            "plan_cost": round(float(res.plan_objective), 6),
            "lower_bound": round(float(res.lower_bound), 6),
            "gap": round(float(res.gap), 6),
        }
    return response


def optimize_charge_schedule(payload: dict, method: str | None = None) -> dict:
    """Full `POST /optimize/charge-schedule` handler.

    Returns evcc's `OptimizationResult`. Contract violations raise
    `ContractError`, which the caller should surface as a 400 with evcc's
    `Error` shape. `method` is "dw" or "admm";
    unset, it is read from the `HEMS_METHOD` environment variable, and defaults
    to "dw".
    """
    method = method or os.environ.get("HEMS_METHOD", "dw")
    if method not in METHODS:
        raise ContractError(f"HEMS_METHOD must be one of {METHODS}, got {method!r}")
    site, fc, meta = request_to_site(payload)
    if method == "dw" and (meta["max_import_kw"] or meta["max_export_kw"]):
        # DW prices the limits inside its master, so hand them over rather
        # than only checking the plan against them afterwards.
        site = replace(site, grid=GridLimits(max_import_kw=meta["max_import_kw"],
                                             max_export_kw=meta["max_export_kw"]))
    res = plan(site, fc, method=method)
    if res.method == "dw" and site.grid.active:
        meta["notes"] = [n for n in meta["notes"] if not n.startswith("grid limits applied post-hoc")]
        meta["notes"].append("grid limits enforced in the plan (DW master); any overshoot "
                             "reported is a breach bought at the breach price")
    if res.note:
        meta["notes"].append(res.note)
    response = result_to_response(site, fc, res, meta)
    if meta["notes"]:
        # Not part of evcc's schema; harmless extra field, and it keeps the
        # approximations visible to a human reading the raw response.
        response["_hems_policy_notes"] = meta["notes"]
    return response


def health() -> dict:
    """`GET /optimize/health`."""
    from . import __version__

    return {"status": "ok", "message": f"hems-policy {__version__}"}
