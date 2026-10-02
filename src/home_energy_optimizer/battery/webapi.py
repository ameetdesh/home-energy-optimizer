"""Backend for the single-battery page (battery/gui/), independent of transport.

One battery, its dynamic programme, and the two tiers the whole package is
built around, made visible:

* slow - `solve`: the backward recursion over every slot and every state of
  charge (`dp_battery.solve_battery`). It leaves a value function V[t, s]
  behind, and a policy over the whole state space, not one trajectory.
* fast - `rollout`: from ANY time and state of charge, what the battery would
  do from there to the end of the horizon (`policy.rollout`) - one look-ahead
  step per slot against the stored V, no re-solve. A single decision is
  `policy.action`, timed on its own.

The page draws the policy table as a flow field: at each (hour, state of
charge), which way the battery is pushed.

This is the page the package grew out of (ORIGIN.md), rebuilt on the
package's own solver. The proof of concept carried a private copy of the DP;
now there is one.

Routes, all JSON in and out:

    solve    settings + hourly curves -> plan, flow field, savings, timing
    rollout  {hour, soe}              -> the policy from there, its timing, lambda
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from home_energy_optimizer.dp_battery import solve_battery
from home_energy_optimizer.policy import PolicySnapshot, action, marginal_value, rollout
from home_energy_optimizer.types import BatteryConfig, Horizon, SocGate

DT_HOURS = 0.25
# The proof of concept's defaults: a 48 h horizon, one point per hour.
DEFAULT_HOURS = 48
# State x action grids the page offers. "coarse" is the proof of concept's
# resolution class; docs/NOTES.md measures it giving up 3-6% of the savings
# that the package default (200 x 81) captures. The slow tier's cost grows with
# both; one fast decision only with the number of actions.
GRIDS = {"coarse": (50, 21), "medium": (100, 41), "fine": (200, 81)}
# The flow field is a picture of the policy table; it does not need every row.
FIELD_STATES = 60

_lock = threading.Lock()
_state: dict[str, Any] = {}


def default_curves(hours: int = DEFAULT_HOURS) -> dict[str, list[float]]:
    """The proof of concept's starting inputs, one point per hour: an evening
    peak in the import price, a flat export price, a flat 3 kW load."""
    xs = range(hours + 1)
    return {
        "buy_h": [0.25 if 18 <= (h % 24) < 21 else 0.15 for h in xs],
        "sell_h": [0.05 for _ in xs],
        "load_h": [3.0 for _ in xs],
    }


def _slots(hourly: list[float], horizon: Horizon) -> np.ndarray:
    """An hourly curve (points at 0, 1, ..., H) interpolated onto the slots."""
    v = np.asarray(hourly, dtype=float)
    if v.size < 2:
        raise ValueError("a curve needs at least two hourly points")
    return np.interp(np.arange(horizon.steps) * horizon.dt, np.arange(v.size), v)


def _bill(net_kw: np.ndarray, buy: np.ndarray, sell: np.ndarray, dt: float) -> float:
    """Import paid at `buy`, export earned at `sell`, over the horizon."""
    return float(np.sum(buy * np.maximum(net_kw, 0.0) - sell * np.maximum(-net_kw, 0.0)) * dt)


def _rounded(a: np.ndarray, places: int = 4) -> list[float]:
    return [round(float(v), places) for v in np.asarray(a).ravel()]


def solve(p: dict[str, Any]) -> dict[str, Any]:
    """The slow tier: build the battery, run its DP, keep the value function.

    `p` (every key optional): hours, capacity_kwh, p_charge_kw, p_discharge_kw,
    reserve_pct, eta (one-way), grid ("coarse" | "medium" | "fine"), buy_h,
    sell_h, load_h (hourly points, currency/kWh and kW; a negative load is PV
    surplus), gates ([{hour, soc_frac}], soft minimum state of charge).
    """
    hours = int(p.get("hours", DEFAULT_HOURS))
    horizon = Horizon(dt=DT_HOURS, hours=float(hours))
    horizon.validate()
    curves = default_curves(hours)
    buy = _slots(p.get("buy_h") or curves["buy_h"], horizon)
    sell = _slots(p.get("sell_h") or curves["sell_h"], horizon)
    load = _slots(p.get("load_h") or curves["load_h"], horizon)

    grid = str(p.get("grid", "fine"))
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {sorted(GRIDS)}, got {grid!r}")
    n_states, n_actions = GRIDS[grid]
    reserve = min(max(float(p.get("reserve_pct", 10.0)), 0.0), 99.0) / 100.0
    # Energy left at the end is worth what it cost to buy on average, so the
    # DP and the savings figure below value it the same way.
    price_left = float(np.mean(buy))
    cfg = BatteryConfig(
        capacity_kwh=float(p.get("capacity_kwh", 40.0)),
        p_charge_max_kw=max(float(p.get("p_charge_kw", 5.0)), 0.0),
        p_discharge_max_kw=max(float(p.get("p_discharge_kw", 5.0)), 0.0),
        eta=float(p.get("eta", 0.90)),
        n_states=n_states,
        n_actions=n_actions,
        soc_initial_frac=(reserve + 1.0) / 2.0,     # halfway between the reserve and full
        soe_min_frac=reserve,
        terminal_mode="linear",
        terminal_price=price_left,
    )
    cfg.validate()
    gates = tuple(SocGate(hour=float(g["hour"]), soc_frac=float(g["soc_frac"]))
                  for g in p.get("gates") or [])

    t0 = time.perf_counter()
    sol = solve_battery(cfg, horizon, buy, sell, dp_load=load, soc_gates=gates)
    slow_ms = (time.perf_counter() - t0) * 1000.0

    snap = PolicySnapshot(horizon=horizon, battery=cfg, value=sol.value, policy=sol.policy,
                          states=sol.states, actions=sol.actions, buy=buy, sell=sell,
                          dp_load=load, generated_at=time.time())
    with _lock:
        _state.update(snapshot=snap, load=load)

    dt = horizon.dt
    soe, power = sol.trajectory, sol.power
    bill_without = _bill(load, buy, sell, dt)
    bill_with = _bill(load + power, buy, sell, dt)
    saving = bill_without - bill_with + (soe[-1] - soe[0]) * price_left

    rows = np.unique(np.linspace(0, n_states - 1, min(FIELD_STATES, n_states)).round().astype(int))
    return {
        "hours": hours,
        "dt": dt,
        "steps": horizon.steps,
        "capacity_kwh": cfg.capacity_kwh,
        "eta_charge": cfg.eta_c,
        "eta_discharge": cfg.eta_d,
        "soe_floor_kwh": cfg.soe_floor_kwh,
        "soe_start_kwh": float(soe[0]),
        "n_states": n_states,
        "n_actions": n_actions,
        "slow_ms": round(slow_ms, 2),
        "plan": {"soe": _rounded(soe), "power": _rounded(power), "import": _rounded(load + power)},
        "field": {"states": _rounded(sol.states[rows]),
                  "policy": [_rounded(row, 3) for row in sol.policy[:, rows]]},
        "bill_without": round(bill_without, 4),
        "bill_with": round(bill_with, 4),
        "price_left": round(price_left, 5),
        "savings_per_day": round(saving / (hours / 24.0), 4),
    }


def rollout_from(p: dict[str, Any]) -> dict[str, Any]:
    """The fast tier: the policy replayed from {hour, soe} to the horizon.

    No solve happens here. Each slot is one look-ahead against the stored V,
    so the path is exact for a state between grid points - which is where a
    real battery always is.
    """
    with _lock:
        snap, load = _state.get("snapshot"), _state.get("load")
    if snap is None or load is None:
        raise ValueError("solve first: there is no value function to read yet")
    dt, steps = snap.horizon.dt, snap.horizon.steps
    t = int(np.clip(np.floor(float(p["hour"]) / dt), 0, steps - 1))
    soe = float(np.clip(float(p["soe"]), snap.battery.soe_floor_kwh, snap.battery.capacity_kwh))

    t0 = time.perf_counter()
    first = action(snap, t, soe)
    decision_us = (time.perf_counter() - t0) * 1e6
    t0 = time.perf_counter()
    path, power = rollout(snap, t, soe)
    replay_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "start_step": t,
        "start_hour": t * dt,
        "soe": _rounded(path),
        "power": _rounded(power),
        "import": _rounded(load[t:] + power),
        "action_kw": round(first, 4),
        "lambda": round(marginal_value(snap, t, soe), 5),
        "decision_us": round(decision_us, 1),
        "replay_ms": round(replay_ms, 3),
        "replay_steps": int(power.size),
    }


ROUTES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "solve": solve,
    "rollout": rollout_from,
}


def call(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a named route (the server's /api/<name>, and the browser
    worker's single entry point)."""
    fn = ROUTES.get(name)
    if fn is None:
        raise ValueError(f"no such route: {name}")
    return fn(payload or {})
