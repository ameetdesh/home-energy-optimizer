"""The grid limits as a device solver sees them: a priced breach.

A device DP prices its own draw on the meter bill given the others' load
(`dp_load`). With `limits` it also pays for any energy beyond a grid limit,
exactly as `coordinate.total_objective` scores it, so a device re-planning
against the others takes the headroom that is left and no more - instead of
learning about the limit only through a price that climbs round by round.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class BusLevel(NamedTuple):
    """One node on a device's way to its grid connection (types.SubMeter):
    per slot, the node's draw is what comes up from below plus `others` (its
    other devices' draw, minus its PV, minus what its other child nodes send
    up); its parent sees what `bus_flow` says. Ratings are bus-side kW."""
    others: np.ndarray
    eta_export: float
    eta_import: float
    export_cap: float            # inf: no limit
    import_cap: float
    clip_cap: np.ndarray         # PV the node may clip, per slot (kW)


class Bus(NamedTuple):
    """A device's chain of nodes up to its grid connection, nearest first
    (a battery on a hybrid inverter's DC side: one level; behind a panel
    behind a breaker: three). submeter.device_bus builds it."""
    levels: tuple[BusLevel, ...]
    breach_price: float          # currency per kWh beyond a rating


def bus_flow(draw: np.ndarray | float, clip_cap: np.ndarray | float, eta_export: float, eta_import: float,
             export_cap: float, import_cap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One bus's connection to the house, at bus draw `draw` (kW, + = the bus
    needs power; PV on it already subtracted): returns (flow to the house in
    kW, PV clipped, kW beyond a rating, what the bus sends bus-side).

    Sending is capped at `export_cap`; PV beyond it is clipped (free) up to
    `clip_cap`, and anything still over is sent anyway, as a breach. Taking
    beyond `import_cap` is a breach too. The house sees eta_export x sent, or
    taken / eta_import. submeter.meter_from_draws and the battery DP both use
    this, so they cannot disagree.
    """
    draw = np.asarray(draw, dtype=float)
    surplus = np.maximum(-draw, 0.0)
    forced = np.maximum(surplus - export_cap, 0.0)
    clip = np.minimum(forced, clip_cap)
    send = surplus - clip
    take = np.maximum(draw, 0.0)
    over = (forced - clip) + np.maximum(take - import_cap, 0.0)
    return eta_export * send - take / eta_import, clip, over, send


def bus_cost(a: np.ndarray, t: int, bus: Bus) -> tuple[np.ndarray, np.ndarray]:
    """For device actions `a` (kW, + = drawing) at slot t on `bus`: (what the
    grid connection's meter draws through the chain, kW; the breach cost per
    kW-slot, every level's)."""
    x = np.asarray(a, dtype=float)
    over_all = np.zeros_like(x)
    for lv in bus.levels:
        flow, _, over, _ = bus_flow(x + lv.others[t], lv.clip_cap[t], lv.eta_export, lv.eta_import,
                                    lv.export_cap, lv.import_cap)
        over_all = over_all + over
        x = -flow                    # what this node draws from its parent
    return x, bus.breach_price * over_all


class Limits(NamedTuple):
    max_import_kw: float | None
    max_export_kw: float | None
    breach_price: float          # currency per kWh beyond a limit
    curtail_exports: bool        # export beyond the cap is curtailed PV, not a breach


def limit_cost(flow: np.ndarray, sell_t: float, limits: Limits | None) -> np.ndarray | float:
    """Cost per kW-slot (multiply by dt) of meter flow `flow` beyond the limits.

    Beyond the export cap, with curtailment allowed, the extra export is
    curtailed: it simply earns nothing, so this cancels its revenue. (That
    assumes enough PV to curtail, which holds wherever the site exports.)
    """
    if limits is None:
        return 0.0
    cost: np.ndarray | float = 0.0
    if limits.max_import_kw is not None:
        cost = cost + limits.breach_price * np.maximum(flow - limits.max_import_kw, 0.0)
    if limits.max_export_kw is not None:
        over = np.maximum(-flow - limits.max_export_kw, 0.0)
        cost = cost + (sell_t if limits.curtail_exports else limits.breach_price) * over
    return cost
