"""One entry point for both coordinators.

    plan(site, fc)                          # Dantzig-Wolfe (the default)
    plan(site, fc, method="admm")           # ADMM, src/home_energy_optimizer/admm/coordinator.py

Both return a `CoordinationResult`, so everything downstream - the policy
snapshot, Home Assistant publishing, the evcc contract - is the same. The DW
result additionally carries a certificate (`lower_bound`, `gap`) and the
master's meter price (`meter_price`).

The DW coordinator lives in the top-level `dw` package (installed alongside
this one) and is imported only when asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .coordinate import coordinate
from .types import CoordinationResult, Forecasts, SiteConfig

if TYPE_CHECKING:  # a runtime import would be circular, or load DW for a type
    from typing import Unpack

    from .admm.coordinator import WarmStart
    from .dw.integrate import PlanOptions

METHODS = ("dw", "admm")


def _admm(cfg: SiteConfig, fc: Forecasts, warm: WarmStart | None = None) -> CoordinationResult:
    return coordinate(cfg, fc, warm=warm)


def plan(cfg: SiteConfig, fc: Forecasts, method: str = "dw", fallback: bool = True,
         warm: WarmStart | None = None, **kw: Unpack[PlanOptions]) -> CoordinationResult:
    """Plan a horizon with the chosen coordinator.

    With `fallback` (the default), a DW run that cannot start or finish -
    an export price above the import price in some slot, which makes the
    master's meter cost non-convex, or a master LP the solver gives up on -
    returns ADMM's plan instead, with the reason in `result.note`. A
    home automation loop should get a plan, not an exception.

    `warm` (ADMM only): the previous ADMM result's `warm_start`, shifted by
    the slots that have passed (`WarmStart.shift`), to start from there.
    """
    if method == "admm":
        return _admm(cfg, fc, warm)
    if method != "dw":
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    from home_energy_optimizer.dw.integrate import dw_plan

    try:
        return dw_plan(cfg, fc, **kw)
    except (ValueError, RuntimeError) as exc:
        if not fallback:
            raise
        res = _admm(cfg, fc)
        res.note = f"DW unavailable ({exc}); planned with ADMM"
        return res
