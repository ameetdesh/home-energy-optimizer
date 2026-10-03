"""A hybrid inverter, swept from no limit down to a tight one.

    .venv/bin/python bench/hybrid_inverter.py [--limits none,5,4,3,2.5] [--eta 0.95]

The demo site (dynamic tariff, 9 kW of PV, a 10 kWh battery and an on/off
water heater), with the PV and the battery on one hybrid inverter's DC bus
(types.hybrid_inverter). For each AC rating, prints what Dantzig-Wolfe plans
(its cost and bound, the PV clipped, the inverter's largest delivery to the
house), how far its bus's price falls below the meter's while PV is clipped,
ADMM's plan for comparison, and who saves what (dw.attribution: the battery
priced at its bus's price). docs/theory.tex, "Sub-meters and local prices",
quotes this.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from home_energy_optimizer import (
    BatteryConfig,
    Horizon,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    hybrid_inverter,
)
from home_energy_optimizer.dw.attribution import ledger
from home_energy_optimizer.dw.coordinator import DWCoordinator
from home_energy_optimizer.profiles import demo_forecasts
from home_energy_optimizer.submeter import site_meter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limits", default="none,5,4,3,2.5")
    ap.add_argument("--eta", type=float, default=0.95)
    a = ap.parse_args()
    h = Horizon(dt=0.5, hours=24)
    fc = demo_forecasts(h, solar_peak_kw=9.0)
    dark = replace(fc, solar=np.zeros(h.steps))
    print(f"{'limit':>6} {'DW cost':>8} {'bound':>8} {'ADMM':>8} {'clipped':>8} {'max AC':>7} "
          f"{'mu clip':>8} {'solar':>7} {'battery':>8} {'tank':>7}")
    for tok in a.limits.split(","):
        lim = None if tok == "none" else float(tok)
        site = SiteConfig(horizon=h, battery=BatteryConfig(), water_heater=WaterHeaterConfig(), hvac=None,
                          submeters=(hybrid_inverter(("battery",), lim, lim, a.eta, a.eta),))
        co, co_dark = DWCoordinator(site, fc), DWCoordinator(site, dark)
        r = co.run()
        flows = site_meter(site, fc, {k: c.power for k, c in r.plan.items()})
        clipped = flows.curtail > 1e-6
        mu = r.local_prices["inverter"]
        mu_clip = f"{float(mu[clipped].max()):8.3f}" if clipped.any() else f"{'-':>8}"
        admm = coordinate(site, fc).total_objective
        gains = {row["player"]: row["net_gain"] for row in ledger(co, r.plan, co_dark, co_dark.run().plan)["rows"]}
        print(f"{tok:>6} {r.upper:8.3f} {r.lower:8.3f} {admm:8.3f} {flows.curtail.sum() * h.dt:7.2f}k "
              f"{flows.ac['inverter'].max():7.2f} {mu_clip} {gains['solar']:7.3f} {gains['battery']:8.3f} "
              f"{gains['water_heater']:7.3f}")


if __name__ == "__main__":
    main()
