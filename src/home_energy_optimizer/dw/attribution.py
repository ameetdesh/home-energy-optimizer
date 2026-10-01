"""Who saves what: each player's net bill before and after, after sharing.

The house pays one bill, and it does not split by player: whether a kWh is
imported or exported depends on everything else on the meter. This module
splits it anyway, against a baseline with no PV and nothing coordinated
(thermostats on, batteries idle). The total saving over that baseline is
shared among solar and the devices, and each device is first reimbursed its
private-cost change Delta f_k (comfort, stored energy, a missed EV goal) -
negative when the plan leaves it better off.

The split is an Owen value (Owen 1977) with two groups, solar and the devices:
average two orders of arrival.

* PV first, then the devices. Solar earns what PV saves on its own (the
  thermostat baseline with PV); the devices share the rest - the saving from
  coordination - by Aumann-Shapley: every kWh a device shifts is priced at
  the average slope of the bill along the straight path from the baseline to
  the plan, all devices moving together. Per slot that is an exact secant.
* Devices first (a plan made with no PV), then PV. The devices share that
  plan's saving the same way; solar earns the rest.

Needs one extra plan, the devices' plan with no PV. The shares add up to the
total exactly. Why not Aumann-Shapley with solar as one more player: pricing
each slot on its own misses that storage turns midday PV into evening
energy, so it credits nearly everything to solar - on the demo site, with the
batteries negative - where the Shapley value (2^(m+1) re-solves,
bench/der_attribution.py) and this rule give them a fair share
(docs/theory.tex, "Who saves what").

Each player's bill before is its share of the baseline bill, its energy priced
along the path from nothing to the baseline; its bill after is that, less what
it is paid (net gain plus reimbursement). The household load does not move,
and its bill does not change.

`coordination_saving` is the part of the total that coordination alone makes:
the saving over the thermostat baseline WITH PV.

Participants (interface.Participant, e.g. an EMHASS device) take part like the
site's own devices: their baseline is their own `baseline()` answer (their plan
without coordination), and their private cost is what their answers report.
"""

from __future__ import annotations

import numpy as np

from home_energy_optimizer.dw.coordinator import Device, DWCoordinator
from home_energy_optimizer.coordinate import apply_curtailment, breach_price
from home_energy_optimizer.dp_thermal import baseline_hvac, baseline_water_heater
from home_energy_optimizer.types import SiteConfig


def _tariff_cost(co: DWCoordinator, z: np.ndarray) -> np.ndarray:
    """Grid cost per slot at a meter flow z that is already final (no
    curtailment left to apply): the tariff plus any priced breach."""
    cfg, fc, dt = co.cfg, co.fc, co.dt
    c = (np.maximum(z, 0.0) * fc.buy - np.maximum(-z, 0.0) * fc.sell) * dt
    g = cfg.grid
    if g.active:
        b = breach_price(g, fc.buy, fc.sell)
        if g.max_import_kw is not None:
            c = c + np.maximum(z - g.max_import_kw, 0.0) * b * dt
        if g.max_export_kw is not None:
            c = c + np.maximum(-z - g.max_export_kw, 0.0) * b * dt
    return c


def _meter_cost(co: DWCoordinator, z: np.ndarray) -> np.ndarray:
    """Grid cost per slot at raw meter flow z, exactly as the objective scores
    it: the curtailment rule, then the tariff and any priced breach."""
    zc, _ = apply_curtailment(z, co.fc.solar, co.fc.sell, co.cfg.grid)
    return _tariff_cost(co, zc)


def _path_prices(co, start: np.ndarray, end: np.ndarray, cost) -> np.ndarray:
    """Average slope of the grid cost `cost(co, z)` along the straight path start -> end,
    per slot (currency per kW-slot).

    Exact, with no integration: the grid cost is separable by slot and the path
    is a straight line in each slot's meter flow, so the average slope is the
    secant (cost at the end - cost at the start) / (flow at the end - flow at
    the start). Where the flow does not move, it is the slope there.
    """
    dz = end - start
    dm = cost(co, end) - cost(co, start)
    eps = 1e-6
    local = (cost(co, start + eps) - cost(co, start - eps)) / (2 * eps)
    moving = np.abs(dz) > 1e-9
    return np.where(moving, dm / np.where(moving, dz, 1.0), local)


def _participant(co: DWCoordinator, key: str) -> Device | None:
    """The coordinator's participant called `key`, or None for a site device."""
    return next((d for d in co.devices if d.key == key and d.kind == "participant"), None)


def _entry(e) -> tuple[np.ndarray, np.ndarray, float | None]:
    """A plan entry as (power kW, state trajectory, private cost or None).
    Accepts a coordinator Column or a (power, trajectory[, cost]) tuple."""
    if hasattr(e, "power"):
        return e.power, e.trajectory, float(e.cost)
    cost = e[2] if len(e) > 2 else None
    return e[0], e[1], None if cost is None else float(cost)


def _private_cost(co: DWCoordinator, key: str, traj: np.ndarray, cost: float | None) -> float:
    """A plan's private cost (currency): the cost a participant reported for
    it, or the coordinator's own pricing of a site device's trajectory."""
    if _participant(co, key) is not None:
        if cost is None:
            raise ValueError(f"participant {key!r}: its plan entry must carry its private cost")
        return cost
    return co.private_cost(_device(co, key), traj)


def _device(co: DWCoordinator, key: str) -> Device:
    for d in co.devices:
        if d.key == key:
            return d
    for k, b in co.lp_batts:
        if k == key:
            return Device(key, "battery", b)
    if key == "water_heater":
        return Device(key, "water_heater", co.cfg.water_heater)
    return Device(key, "hvac", co.cfg.hvac)


def baseline(co: DWCoordinator, key: str) -> tuple[np.ndarray, np.ndarray]:
    """The uncoordinated plan of device `key` as (power kW, trajectory): idle
    battery, thermostat tank and HVAC, or a participant's own `baseline()`."""
    power, traj, _ = _baseline(co, key)
    return power, traj


def _baseline(co: DWCoordinator, key: str) -> tuple[np.ndarray, np.ndarray, float | None]:
    """`baseline`, plus the private cost a participant reported for it (None
    for a site device). A participant is asked once per coordinator."""
    p = _participant(co, key)
    if p is not None:
        memo = co.__dict__.setdefault("_baseline_answers", {})
        if key not in memo:
            memo[key] = p.cfg.baseline()
        a = memo[key]
        return np.asarray(a.plan_kw, dtype=float), np.asarray(a.trajectory, dtype=float), float(a.private_cost)
    cfg, fc, h = co.cfg, co.fc, co.cfg.horizon
    if key.startswith("battery"):
        b = _device(co, key).cfg
        return np.zeros(h.steps), np.full(h.steps + 1, b.capacity_kwh * b.soc_initial_frac), None
    if key == "water_heater":
        t, p = baseline_water_heater(cfg.water_heater, h, fc.hot_water_demand)
        return p, t, None
    t, p = baseline_hvac(cfg.hvac, h, fc.outdoor_temp)
    return p, t, None


def _device_shares(co: DWCoordinator, plan: dict, keys: list[str]) -> tuple[dict, dict]:
    """Aumann-Shapley shares of the devices' saving over their baseline, with
    the PV (whatever co.fc holds) fixed: each device's net gain, and its
    private-cost change."""
    fc = co.fc
    base = {k: _baseline(co, k) for k in keys}
    plan = {k: _entry(plan[k]) for k in keys}
    z0 = fc.load - fc.solar + sum(base[k][0] for k in keys)
    z1 = fc.load - fc.solar + sum(plan[k][0] for k in keys)
    q = _path_prices(co, z0, z1, _meter_cost)
    df = {k: _private_cost(co, k, plan[k][1], plan[k][2]) - _private_cost(co, k, base[k][1], base[k][2])
          for k in keys}
    return {k: -float(q @ (plan[k][0] - base[k][0])) - df[k] for k in keys}, df


def device_keys(co: DWCoordinator) -> list[str]:
    """Every device that takes a share: the site's batteries, tank and HVAC,
    then the participants, in the coordinator's order."""
    keys = [SiteConfig.battery_key(i) for i, b in enumerate(co.cfg.battery_list) if b.capacity_kwh > 0]
    keys += [k for k in ("water_heater", "hvac") if getattr(co.cfg, k) is not None]
    return keys + [d.key for d in co.devices if d.kind == "participant"]


def ledger(co: DWCoordinator, plan: dict[str, tuple[np.ndarray, np.ndarray]],
           co_dark: DWCoordinator, plan_dark: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Each player's bill before and after, the private-cost change and the gain.

    `plan` maps device key -> its plan, for every device on the site and every
    participant: a coordinator Column, or (power kW, state trajectory), plus
    the private cost as a third item for a participant. `co_dark` /
    `plan_dark` are the same site and method with the PV set to zero.

    Returns {"rows": [...], "totals": {...}}: one row per player (household
    load, solar, then each device) with its bill before and after, its
    private-cost change and its net gain (currency), and the house totals.
    """
    fc = co.fc
    keys = device_keys(co)
    base = {k: _baseline(co, k) for k in keys}
    plan = {k: _entry(plan[k]) for k in keys}

    # meter flows: baseline with no PV, thermostats with PV, the plan
    z_dark = fc.load + sum(base[k][0] for k in keys)
    z_thermo = z_dark - fc.solar
    z_plan = z_thermo + sum(plan[k][0] - base[k][0] for k in keys)
    bill_dark = float(_tariff_cost(co, z_dark).sum())
    bill_thermo = float(_meter_cost(co, z_thermo).sum())
    bill_plan = float(_meter_cost(co, z_plan).sum())

    # the two orders of arrival
    first_pv, df = _device_shares(co, plan, keys)             # devices after PV
    first_dev, _ = _device_shares(co_dark, plan_dark, keys)   # devices before PV
    total = bill_dark - bill_plan - sum(df.values())
    v_pv, v_dev = bill_dark - bill_thermo, sum(first_dev.values())
    gain = {"household load": 0.0, "solar": 0.5 * (v_pv + (total - v_dev))}
    gain.update({k: 0.5 * (first_pv[k] + first_dev[k]) for k in keys})
    df = {"household load": 0.0, "solar": 0.0, **df}

    # bills before: the baseline bill, split along 0 -> baseline
    x_base = {"household load": fc.load, "solar": np.zeros_like(fc.solar)}
    x_base.update({k: base[k][0] for k in keys})
    q0 = _path_prices(co, np.zeros_like(z_dark), z_dark, _tariff_cost)

    rows = []
    for j, x in x_base.items():
        before = float(q0 @ x)
        after = before - (gain[j] + df[j])       # paid: its gain plus its reimbursement
        rows.append({"player": j, "bill_before": before, "bill_after": after,
                     "bill_change": after - before, "private_cost_change": df[j],
                     "net_gain": gain[j]})
    tot = {c: sum(r[c] for r in rows) for c in ("bill_before", "bill_after", "bill_change",
                                                  "private_cost_change", "net_gain")}
    _, curtailed = apply_curtailment(z_plan, fc.solar, fc.sell, co.cfg.grid)
    tot.update(house_bill_before=bill_dark, house_bill_after=bill_plan, house_bill_with_pv=bill_thermo,
               coordination_saving=bill_thermo - bill_plan - tot["private_cost_change"],
               pv_alone_saving=v_pv, devices_alone_saving=v_dev,
               curtailed_kwh=float(curtailed.sum() * co.dt))
    return {"rows": rows, "totals": tot}
