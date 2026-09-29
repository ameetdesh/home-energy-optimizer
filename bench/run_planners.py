"""ADMM against Dantzig-Wolfe on the two integration paths.

    .venv/bin/python bench/run_planners.py

Prints the tables quoted in docs/theory.tex, section "Two coordinators, side
by side":

1. the Home Assistant demo site (tools/ha-lambda-demo/run.py's defaults:
   battery on a 100-state grid, on/off tank, HVAC, 24 h) on three tariffs;
2. an evcc request with a home battery and an EV (4.14 kW charger minimum,
   40 kWh by 07:00) under three import limits.

Both plans are scored by the same function, `DWCoordinator.parts`, on the
basis the DW lower bound is stated on. The bound holds for every plan, so
"above bound" certifies ADMM's plan too: no plan is better than it by more than
that. "Capture" (Home Assistant site only) is the share of the savings
available over the thermostat baseline that a plan realises, measured against
the bound: (baseline - plan) / (baseline - bound). It is left out for the EV
rows, where the idle baseline misses the EV's goal and its penalty dwarfs
everything else.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dw.coordinator import Column, DWCoordinator, baseline_objective  # noqa: E402
from hemspolicy import (  # noqa: E402
    BatteryConfig, CoordinationConfig, GridLimits, Horizon, HvacConfig, SiteConfig,
    WaterHeaterConfig, demo_forecasts, plan,
)
from hemspolicy.evcc import request_to_site  # noqa: E402


def score(site, fc, res) -> float:
    co = DWCoordinator(site, fc)
    return co.parts({k: Column(d.power, d.trajectory, 0.0, "x") for k, d in res.devices.items()})["total"]


def row(label, site, fc, capture: bool = True) -> None:
    base = baseline_objective(site, fc)
    out = {}
    for m in ("admm", "dw"):
        t0 = time.perf_counter()
        res = plan(site, fc, method=m, fallback=False) if m == "dw" else plan(site, fc, method=m)
        out[m] = (score(site, fc, res), (time.perf_counter() - t0) * 1000, res)
    lb = out["dw"][2].lower_bound
    avail = base - lb
    cells = [f"{out[m][0]:8.4f} {out[m][0] - lb:7.4f} "
             + (f"{100 * (base - out[m][0]) / avail:5.1f}%" if capture else "     -")
             + f" {out[m][1]:5.0f} ms" for m in ("admm", "dw")]
    print(f"{label:<22} {cells[0]}   {cells[1]}   bound {lb:8.4f}")


def ha_site() -> SiteConfig:
    return SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0, n_states=100, n_actions=51),
        water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
        coordination=CoordinationConfig(max_rounds=6),
    )


def ev_site(limit_kw: float):
    from test_dw_integration import ev_request

    site, fc, _ = request_to_site(ev_request(limit_kw * 1000.0))
    return replace(site, grid=GridLimits(max_import_kw=limit_kw)), fc


def main() -> None:
    head = "objective  above-bound capture time"
    print(f"{'':<22} ADMM: {head}   DW: {head}")
    print("Home Assistant demo site (battery + on/off tank + HVAC, 24 h)")
    for tariff in ("flat", "day_night", "dynamic"):
        site = ha_site()
        row(f"  {tariff}", site, demo_forecasts(site.horizon, tariff=tariff))
    print("evcc: home battery + EV (c_min 4.14 kW, 40 kWh by 07:00), 24 h")
    for lim in (17.25, 7.0, 5.0):
        site, fc = ev_site(lim)
        row(f"  import limit {lim:g} kW", site, fc, capture=False)


if __name__ == "__main__":
    main()
