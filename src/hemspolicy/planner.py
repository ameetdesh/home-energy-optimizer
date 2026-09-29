"""One entry point for both coordinators.

    plan(site, fc)                          # Dantzig-Wolfe (the default)
    plan(site, fc, method="admm")           # textbook ADMM, hemspolicy.exchange
    plan(site, fc, method="admm_legacy")    # the older ADMM loop, hemspolicy.coordinate

Both return a `CoordinationResult`, so everything downstream - the policy
snapshot, Home Assistant publishing, the evcc contract - is the same. The DW
result additionally carries a certificate (`lower_bound`, `gap`) and the
master's meter price (`meter_price`).

The DW coordinator lives in the top-level `dw` package (installed alongside
this one) and is imported only when asked for.
"""

from __future__ import annotations

from dataclasses import replace

from .coordinate import coordinate
from .types import CoordinationResult, Forecasts, SiteConfig

METHODS = ("dw", "admm", "admm_legacy")


def _admm(cfg: SiteConfig, fc: Forecasts, algorithm: str, warm=None) -> CoordinationResult:
    return coordinate(replace(cfg, coordination=replace(cfg.coordination, algorithm=algorithm)), fc,
                      warm=warm)


def plan(cfg: SiteConfig, fc: Forecasts, method: str = "dw", fallback: bool = True,
         warm=None, **kw) -> CoordinationResult:
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
        return _admm(cfg, fc, "exchange", warm)
    if method == "admm_legacy":
        return _admm(cfg, fc, "legacy")
    if method != "dw":
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    from dw.integrate import dw_plan

    try:
        return dw_plan(cfg, fc, **kw)
    except (ValueError, RuntimeError) as exc:
        if not fallback:
            raise
        res = _admm(cfg, fc, "exchange")
        res.note = f"DW unavailable ({exc}); planned with textbook ADMM"
        return res
