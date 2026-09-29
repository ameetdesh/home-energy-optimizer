"""One battery alone: Dantzig-Wolfe against the battery's own DP at the tariff.

    .venv/bin/python bench/one_battery.py

With a single device, the house problem is exactly the one the battery's DP
solves when handed the real tariff and the house load, so DW should land on
the same plan or a better one (the LP battery is continuous; the DP is on a
grid). Prints that check as a table (a sanity check; docs/theory.tex does not
quote it).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dw.coordinator import Column, DWCoordinator  # noqa: E402
from hemspolicy import BatteryConfig, GridLimits, Horizon, SiteConfig, demo_forecasts  # noqa: E402
from hemspolicy.coordinate import device_sell_price  # noqa: E402
from hemspolicy.dp_battery import solve_battery  # noqa: E402


def main() -> None:
    print(f"{'tariff':10s} {'battery DP':>11s} {'DW, in LP':>22s} {'DW, bidding plans':>24s} {'no load-aware':>20s}")
    for tariff in ("flat", "day_night", "dynamic"):
        site = SiteConfig(horizon=Horizon(dt=0.25, hours=24.0),
                          battery=BatteryConfig(capacity_kwh=10.0, n_states=200, n_actions=81),
                          water_heater=None, hvac=None, grid=GridLimits())
        fc = demo_forecasts(site.horizon, tariff=tariff, solar_peak_kw=5.0)
        co = DWCoordinator(site, fc)
        dp = solve_battery(site.battery, site.horizon, fc.buy, device_sell_price(fc.sell, site.grid),
                           dp_load=fc.net_fixed_demand)
        direct = co.parts({"battery": Column(dp.power, dp.trajectory, 0.0, "dp")})["total"]
        lp = DWCoordinator(site, fc).run(max_iter=40)
        bid = DWCoordinator(site, fc, battery_in_master=False).run(max_iter=40)
        bid0 = DWCoordinator(site, fc, battery_in_master=False).run(max_iter=40, heuristic_columns=False)
        cell = lambda r: f"{r.upper:8.4f} ({r.iterations:2d} it)"  # noqa: E731
        print(f"{tariff:10s} {direct:11.4f} {cell(lp):>22s} {cell(bid):>24s} {cell(bid0):>20s}")


if __name__ == "__main__":
    main()
