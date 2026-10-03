"""The meter, given every device's power: the one place that knows how a
sub-meter (a hybrid inverter, a shared breaker; types.SubMeter) connects its
devices and the PV to the house.

Every scorer (the coordinators' objectives, the saving split, the results)
goes through `site_meter`, so they all agree with each other and with the
Dantzig-Wolfe master, which models the same connection as LP rows. With no
sub-meter it is exactly what it replaces: the inflexible load, minus the PV,
plus every device, then `apply_curtailment`.

At a sub-meter's bus, each slot:

* the members' draw minus the PV on the bus (if any) is what the bus needs
  from the house (positive) or can send to it (negative);
* sending is capped at the connection's rating; PV beyond it is clipped, at
  no cost (an inverter does this itself). Anything the bus still cannot send
  - a member discharging into a full connection - is sent anyway and charged
  as a breach, at the grid's breach price, like a grid-limit breach;
* taking from the house beyond the rating is likewise planned, and charged;
* the house sees what is sent times eta_export, or what is taken divided by
  eta_import.

Then curtailment for the meter's own reasons (an export cap, a negative
export price), from the PV on the house's AC bus and from what the PV's bus
is sending; each kWh not sent from the bus is 1/eta_export kWh of PV.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from .coordinate import apply_curtailment, breach_price
from .meter import Bus, bus_flow
from .types import Forecasts, SiteConfig, SubMeter


class MeterFlows(NamedTuple):
    net: np.ndarray                 # kW at the meter, after curtailment (+ = import)
    curtail: np.ndarray             # PV not used, kW (clipped at a bus and curtailed)
    breach_kwh: np.ndarray          # energy beyond a sub-meter's ratings, per slot
    ac: dict[str, np.ndarray]       # sub-meter name -> its power to the house, kW (+ = to the house)
    bus: dict[str, np.ndarray]      # sub-meter name -> what its bus sends, bus-side kW (+ = to the house)
    raw: np.ndarray                 # the meter before the meter's own curtailment (an export cap,
                                    # a negative export price): what a device DP takes as dp_load
    ac_raw: dict[str, np.ndarray]   # `ac` before that curtailment


def ac_solar(cfg: SiteConfig, fc: Forecasts) -> np.ndarray:
    """The PV on the house's AC bus: all of it, unless a sub-meter has it."""
    return np.zeros_like(fc.solar) if cfg.pv_submeter is not None else fc.solar


def clip_cap(cfg: SiteConfig, fc: Forecasts, sm: SubMeter) -> np.ndarray:
    """PV sub-meter `sm` may clip, per slot (kW): free, but beyond what its
    rating forces only if the grid allows curtailment (as the master's clip
    variable)."""
    if not sm.pv:
        return np.zeros_like(fc.solar)
    return fc.solar if cfg.grid.allow_curtailment else np.maximum(fc.solar - sm.export_cap_dc, 0.0)


def device_bus(cfg: SiteConfig, fc: Forecasts, key: str,
               powers: Mapping[str, np.ndarray]) -> tuple[Bus | None, np.ndarray]:
    """What device `key` meets, holding every other device at `powers`: its
    Bus (None on the house's AC bus) and the rest of the meter (kW, the
    dp_load a device DP takes): everything on the meter but its own bus."""
    s = cfg.submeter_of(key)
    others = {k: p for k, p in powers.items() if k != key}
    if s is None:
        return None, site_meter(cfg, fc, others).raw
    sm = cfg.submeters[s]
    n = len(fc.load)
    rest = {k: p for k, p in others.items() if k not in sm.members}
    flows = site_meter(cfg, fc, rest)          # this bus idle: its PV still flows
    raw = flows.raw + flows.ac_raw[sm.name]    # the meter without this bus at all
    bus_others = sum((others[k] for k in sm.members if k in others), np.zeros(n))
    if sm.pv:
        bus_others = bus_others - fc.solar
    bus = Bus(bus_others, sm.eta_export, sm.eta_import, sm.export_cap_dc, sm.import_cap_dc,
              clip_cap(cfg, fc, sm), breach_price(cfg.grid, fc.buy, fc.sell))
    return bus, raw


def fixed_demand(cfg: SiteConfig, fc: Forecasts) -> np.ndarray:
    """The meter's inflexible part: the load minus the PV on the AC bus."""
    return fc.load - ac_solar(cfg, fc)


def site_meter(cfg: SiteConfig, fc: Forecasts, powers: Mapping[str, np.ndarray]) -> MeterFlows:
    """The meter, curtailment and sub-meter flows for device powers `powers`
    (device key -> kW, + = drawing). See the module docstring."""
    n = len(fc.load)
    behind = {k for sm in cfg.submeters for k in sm.members}
    # the devices summed first, then added to the load: the order every
    # scorer used before sub-meters, so a site without one is bit-for-bit
    # what it was
    ac = fixed_demand(cfg, fc) + sum((p for k, p in powers.items() if k not in behind), np.zeros(n))
    bus = {sm.name: sum((powers[k] for k in sm.members if k in powers), np.zeros(n)) for sm in cfg.submeters}
    return meter_from_draws(cfg, fc, ac, bus)


def meter_from_draws(cfg: SiteConfig, fc: Forecasts, ac_draw: np.ndarray,
                     bus_draw: Mapping[str, np.ndarray]) -> MeterFlows:
    """`site_meter` from totals: `ac_draw`, everything on the house's AC bus
    (the load, minus the PV there, plus the devices there), and `bus_draw`,
    sub-meter name -> its members' total draw (kW). The PV on a sub-meter's
    bus comes from `fc`."""
    dt = cfg.horizon.dt
    n = len(fc.load)
    net = np.asarray(ac_draw, dtype=float).copy()
    breach = np.zeros(n)
    clipped = np.zeros(n)                     # PV clipped at its bus, kW
    curtailable = ac_solar(cfg, fc).astype(float).copy()   # AC-side kW the meter could curtail
    ac: dict[str, np.ndarray] = {}
    ac_raw: dict[str, np.ndarray] = {}
    bus_out: dict[str, np.ndarray] = {}
    pv_bus: tuple[str, float, np.ndarray] | None = None
    for sm in cfg.submeters:
        pv = fc.solar if sm.pv else np.zeros(n)
        draw = np.asarray(bus_draw.get(sm.name, np.zeros(n)), dtype=float) - pv
        flow, clip, over, send = bus_flow(draw, clip_cap(cfg, fc, sm), sm.eta_export, sm.eta_import,
                                          sm.export_cap_dc, sm.import_cap_dc)
        breach += over * dt
        net = net - flow
        ac[sm.name], bus_out[sm.name] = flow, send - np.maximum(draw, 0.0)
        ac_raw[sm.name] = flow
        if sm.pv:
            clipped = clipped + clip
            spare = np.minimum(send, pv - clip)       # PV still being sent, bus-side
            curtailable = curtailable + sm.eta_export * spare
            pv_bus = (sm.name, sm.eta_export, spare)
    raw = net
    net, cut = apply_curtailment(net, curtailable, fc.sell, cfg.grid)
    # Curtail the AC-side PV first, then what the PV's bus sends.
    from_ac = np.minimum(cut, ac_solar(cfg, fc))
    curtail = clipped + from_ac
    if pv_bus is not None:
        name, eta, spare = pv_bus
        from_bus = np.minimum(cut - from_ac, eta * spare)
        ac[name] = ac[name] - from_bus
        bus_out[name] = bus_out[name] - from_bus / eta
        curtail = curtail + from_bus / eta
    return MeterFlows(net, curtail, breach, ac, bus_out, raw, ac_raw)


def submeter_penalty(cfg: SiteConfig, fc: Forecasts, flows: MeterFlows) -> float:
    """What the energy beyond the sub-meters' ratings costs (currency): the
    grid's breach price per kWh, the price a grid-limit breach pays."""
    total = float(np.sum(flows.breach_kwh))
    if total <= 0.0:
        return 0.0
    return total * breach_price(cfg.grid, fc.buy, fc.sell)
