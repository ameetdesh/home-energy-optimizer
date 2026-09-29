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
    cost = 0.0
    if limits.max_import_kw is not None:
        cost = cost + limits.breach_price * np.maximum(flow - limits.max_import_kw, 0.0)
    if limits.max_export_kw is not None:
        over = np.maximum(-flow - limits.max_export_kw, 0.0)
        cost = cost + (sell_t if limits.curtail_exports else limits.breach_price) * over
    return cost
