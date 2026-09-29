"""Who saves what: the page's ledger, and how its sharing rule compares.

    .venv/bin/python bench/der_attribution.py [--tariffs day_night,dynamic,flat] [--export-limit KW]

On the app's two-battery site (on/off tank, HVAC, 5 kW PV, 7 kW import limit,
stored energy valued at the average import price, as on the page) planned by
Dantzig-Wolfe, prints:

1. the ledger src/home_energy_optimizer/dw/attribution.py computes against a baseline with no PV and
   nothing coordinated - each player's bill before and after, its private-cost
   change and its net gain (docs/theory.tex, "Who saves what");
2. five ways to split the same total saving among solar and the devices:
   - surplus at the final meter price pi, and that shared in proportion;
   - Aumann-Shapley with solar as one more player (per-slot secant prices);
   - the ledger's rule: an Owen value - solar and the devices as two groups,
     two orders of arrival averaged, Aumann-Shapley within the devices;
   - Shapley, from 2^(m+1) re-solves, as the reference - devices left out of
     a coalition run their baseline (idle, thermostat), and without solar
     the PV is zero.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from home_energy_optimizer.dw.attribution import _path_prices, _tariff_cost, baseline, ledger  # noqa: E402
from home_energy_optimizer.dw.coordinator import DWCoordinator  # noqa: E402
from home_energy_optimizer.dw.webapi import build_site, value_stored_energy  # noqa: E402
from home_energy_optimizer.coordinate import apply_curtailment  # noqa: E402
from home_energy_optimizer import SiteConfig, demo_forecasts  # noqa: E402

APP = {"n_batteries": 2, "batt_capacity": 10, "solar_peak": 5, "max_import_kw": 7,
       "hours": 24, "grid": 100, "terminal_mode": "linear"}


def study(tariff: str, export_limit: float | None) -> None:
    site = build_site({**APP, "max_export_kw": export_limit})
    h, dt = site.horizon, site.horizon.dt
    fc = demo_forecasts(h, tariff=tariff, solar_peak_kw=APP["solar_peak"])
    site = value_stored_energy(site, fc)
    co = DWCoordinator(site, fc, tank_in_master=False)
    r = co.run(max_iter=40)
    plan = {k: (c.power, c.trajectory) for k, c in r.plan.items()}
    co_dark = DWCoordinator(site, replace(fc, solar=np.zeros_like(fc.solar)), tank_in_master=False)
    plan_dark = {k: (c.power, c.trajectory) for k, c in co_dark.run(max_iter=40).plan.items()}
    L = ledger(co, plan, co_dark, plan_dark)
    devs = [row["player"] for row in L["rows"] if row["player"] not in ("household load", "solar")]
    keys = ["solar"] + devs

    print(f"\n== {tariff}, export limit {export_limit}: DW plan gap {r.upper - r.lower:.3f}")
    print(f"   {'player':15s} {'bill before':>11s} {'bill after':>10s} {'change':>8s} {'private d':>9s} {'net gain':>9s}")
    for row in L["rows"]:
        print(f"   {row['player']:15s} {row['bill_before']:11.3f} {row['bill_after']:10.3f} {row['bill_change']:8.3f}"
              f" {row['private_cost_change']:9.3f} {row['net_gain']:9.3f}")
    t = L["totals"]
    print(f"   {'house':15s} {t['bill_before']:11.3f} {t['bill_after']:10.3f} {t['bill_change']:8.3f}"
          f" {t['private_cost_change']:9.3f} {t['net_gain']:9.3f}")
    print(f"   of which coordination {t['coordination_saving']:.3f}; PV curtailed {t['curtailed_kwh']:.2f} kWh")

    # the same total, split four ways
    total = t["net_gain"]
    pi = np.asarray(r.prices)
    df = {row["player"]: row["private_cost_change"] for row in L["rows"]}
    base = {k: baseline(co, k) for k in devs}
    z_raw = fc.load - fc.solar + sum(plan[k][0] for k in devs)
    _, curtailed = apply_curtailment(z_raw, fc.solar, fc.sell, site.grid)
    surplus = {k: dt * float(pi @ (base[k][0] - plan[k][0])) - df[k] for k in devs}
    surplus["solar"] = dt * float(pi @ (fc.solar - curtailed))
    prop = {k: total * surplus[k] / sum(surplus.values()) for k in keys}
    owen = {row["player"]: row["net_gain"] for row in L["rows"] if row["player"] in keys}

    # Aumann-Shapley with solar as a player: its delivered PV and each device's
    # shift priced at one secant per slot, from the no-PV baseline to the plan
    x0 = {"solar": np.zeros_like(fc.solar), **{k: base[k][0] for k in devs}}
    x1 = {"solar": -(fc.solar - curtailed), **{k: plan[k][0] for k in devs}}
    z0 = fc.load + sum(x0.values())
    q = _path_prices(co, z0, fc.load + sum(x1.values()), _tariff_cost)
    asv = {k: -float(q @ (x1[k] - x0[k])) - (df[k] if k in devs else 0.0) for k in keys}

    fbase = {k: co.private_cost(_dev(co, k), base[k][1]) for k in devs}

    def cost(S):
        """Best cost with only the players in S free; devices not in S at
        their baseline, and no PV unless solar is in S."""
        bl = [b for i, b in enumerate(site.battery_list) if SiteConfig.battery_key(i) in S]
        s2 = replace(site, battery=bl[0] if bl else None, batteries=tuple(bl[1:]))
        load, extra = fc.load.copy(), 0.0
        for k in ("water_heater", "hvac"):
            if k not in S:
                s2 = replace(s2, **{k: None})
                load = load + base[k][0]
                extra += fbase[k]
        solar = fc.solar if "solar" in S else np.zeros_like(fc.solar)
        c2 = DWCoordinator(s2, replace(fc, load=load, solar=solar), tank_in_master=False)
        if not c2.devices and not c2.lp_batts:
            return c2.parts({})["total"] + extra
        return c2.run(max_iter=40).upper + extra

    C = {S: cost(S) for m in range(len(keys) + 1) for S in itertools.combinations(keys, m)}
    v = {S: C[()] - C[S] for S in C}
    n = len(keys)
    shap = {}
    for k in keys:
        others = [j for j in keys if j != k]
        shap[k] = sum(math.factorial(m) * math.factorial(n - m - 1) / math.factorial(n)
                      * (v[tuple(j for j in keys if j in S or j == k)] - v[S])
                      for m in range(n) for S in itertools.combinations(others, m))

    print(f"   split of the total saving {total:.3f}:")
    print(f"   {'player':15s} {'surplus @ final pi':>18s} {'proportional':>12s} {'A-S, solar a player':>19s}"
          f" {'Owen (ledger)':>13s} {'Shapley':>8s}")
    for k in keys:
        print(f"   {k:15s} {surplus[k]:18.3f} {prop[k]:12.3f} {asv[k]:19.3f} {owen[k]:13.3f} {shap[k]:8.3f}")
    print(f"   {'sum':15s} {sum(surplus.values()):18.3f} {sum(prop.values()):12.3f}"
          f" {sum(asv.values()):19.3f} {sum(owen.values()):13.3f} {sum(shap.values()):8.3f}")


def _dev(co, key):
    from home_energy_optimizer.dw.attribution import _device
    return _device(co, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tariffs", default="day_night,dynamic,flat")
    ap.add_argument("--export-limit", type=float, default=None)
    a = ap.parse_args()
    for tariff in a.tariffs.split(","):
        study(tariff, a.export_limit)


if __name__ == "__main__":
    main()
