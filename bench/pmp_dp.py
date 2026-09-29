"""Textbook ADMM with the testbed's own device models, on relaxed devices.

    .venv/bin/python bench/pmp_dp.py [--levels 13] [--iters 300]

bench/relaxed_devices.py asks whether ADMM's gap is the on/off devices. This
asks the other half: with the devices relaxed (the tank and HVAC allowed
`levels` duty steps each way), does a textbook ADMM - proximal message passing,
bench/prior_art.py's loop, exchange form, every device tethered, the grid as a
device - close the gap, using the SAME dynamic programmes the ADMM loop uses as
each device's prox (the battery, tank and HVAC DPs with a tether term)?

If it does, what keeps the ADMM loop from the optimum is the loop's structure,
not the devices. Plans are scored by DWCoordinator.parts against the DW bound
for the same (relaxed) devices; the best iterate is reported alongside the
last, and the best iterate after the testbed's polish.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "bench")]

from home_energy_optimizer.dw.coordinator import Column, DWCoordinator, _PriceVector  # noqa: E402
from home_energy_optimizer.dw.webapi import _site_fc  # noqa: E402
from home_energy_optimizer.coordinate import breach_price, device_sell_price  # noqa: E402
from home_energy_optimizer.dp_battery import solve_battery  # noqa: E402
from home_energy_optimizer.dp_thermal import solve_hvac, solve_water_heater  # noqa: E402
from home_energy_optimizer.meter import Limits  # noqa: E402
from home_energy_optimizer.types import SiteConfig  # noqa: E402
from prior_art import PV, Fixed, Tie  # noqa: E402

SITES = list(itertools.product(("dynamic", "day_night", "flat"), (1, 2, 3), (7, None), (5, 8), (10, 20)))


class DPDevice:
    """A device whose prox is its own DP with a tether:
    argmin f(p) + (rho/2)|p - v|^2  ==  the DP with admm_target = v and
    admm_rho = rho / dt (the DPs weigh the tether by dt)."""

    def __init__(self, key, solve, dt):
        self.key, self.solve, self.dt, self.sol = key, solve, dt, None

    def prox(self, v, rho):
        self.sol = self.solve(np.asarray(v, float), rho / self.dt)
        return self.sol.power


def devices(site, fc, co):
    h, n, dt = site.horizon, site.horizon.steps, site.horizon.dt
    zero = np.zeros(n)
    free = _PriceVector(zero, co.ref)       # no bill in the prox (the grid device has it); comfort at ref
    g = site.grid
    c_br = breach_price(g, fc.buy, fc.sell) if g.active else 0.0
    devs = [Fixed(fc.load), PV(fc.solar, g.allow_curtailment),
            Tie(fc.buy, fc.sell, dt, g.max_import_kw, g.max_export_kw, c_br)]
    for i, b in enumerate(site.battery_list):
        if b.capacity_kwh > 0:
            devs.append(DPDevice(SiteConfig.battery_key(i),
                                 lambda v, r, b=b: solve_battery(b, h, zero, zero, dp_load=zero,
                                                                 admm_target=v, admm_rho=r), dt))
    if site.water_heater is not None:
        devs.append(DPDevice("water_heater",
                             lambda v, r: solve_water_heater(site.water_heater, h, free, free, fc.hot_water_demand,
                                                             dp_load=zero, admm_target=v, admm_rho=r), dt))
    if site.hvac is not None:
        devs.append(DPDevice("hvac",
                             lambda v, r: solve_hvac(site.hvac, h, free, free, fc.outdoor_temp,
                                                     dp_load=zero, admm_target=v, admm_rho=r), dt))
    return devs


def plan_of(devs):
    return {d.key: Column(d.sol.power, d.sol.trajectory, 0.0, "tb") for d in devs if isinstance(d, DPDevice)}


def polish(site, fc, co, plan, sweeps=5):
    """The testbed's polish: each device in turn re-plans against the others,
    limits priced, kept only if the objective falls."""
    h, g = site.horizon, site.grid
    limits = (Limits(g.max_import_kw, g.max_export_kw, breach_price(g, fc.buy, fc.sell), g.allow_curtailment)
              if g.active else None)
    sell = device_sell_price(fc.sell, g)
    batt = {SiteConfig.battery_key(i): b for i, b in enumerate(site.battery_list)}
    cur = co.parts(plan)["total"]
    for _ in range(sweeps):
        improved = False
        for k in list(plan):
            others = fc.net_fixed_demand + sum(plan[j].power for j in plan if j != k)
            if k in batt:
                sol = solve_battery(batt[k], h, fc.buy, sell, dp_load=others, limits=limits)
            elif k == "water_heater":
                sol = solve_water_heater(site.water_heater, h, fc.buy, sell, fc.hot_water_demand,
                                         dp_load=others, limits=limits)
            else:
                sol = solve_hvac(site.hvac, h, fc.buy, sell, fc.outdoor_temp, dp_load=others, limits=limits)
            trial = dict(plan)
            trial[k] = Column(sol.power, sol.trajectory, 0.0, "polish")
            val = co.parts(trial)["total"]
            if val < cur - 1e-9:
                plan, cur, improved = trial, val, True
        if not improved:
            break
    return cur


def run(args):
    (tariff, nb, limit, solar, cap), levels, iters, *rest = args
    rho0, adapt = rest if rest else (0.1, True)        # starting rho; the paper's rho controller on/off
    site, fc, _ = _site_fc({"tariff": tariff, "n_batteries": nb, "grid": 50, "max_import_kw": limit,
                            "solar_peak": solar, "batt_capacity": cap,
                            "tank_levels": levels, "hvac_levels": levels})
    co = DWCoordinator(site, fc, tank_in_master=False)
    r = co.run(max_iter=40)
    t = time.perf_counter()
    devs = devices(site, fc, co)
    # proximal message passing (bench/prior_art.py's pmp), scoring every iterate
    N, n = len(devs), site.horizon.steps
    p, u, rho, w_prev = np.zeros((N, n)), np.zeros(n), rho0, 0.0
    pbar, prev_dev = p.mean(axis=0), p - p.mean(axis=0)
    best, best_plan, last = np.inf, None, np.inf
    eps = 1e-3 * np.sqrt(N * n)
    k = 0
    for k in range(iters):
        v = p - pbar - u
        p = np.array([d.prox(v[i], rho) for i, d in enumerate(devs)])
        pbar = p.mean(axis=0)
        u = u + pbar
        dev = p - pbar
        res_r, res_s = np.sqrt(N) * np.linalg.norm(pbar), rho * np.linalg.norm(dev - prev_dev)
        prev_dev = dev
        plan = plan_of(devs)
        last = co.parts(plan)["total"]
        if last < best:
            best, best_plan = last, plan
        if res_r <= eps and res_s <= eps:
            break
        if adapt and k < 200 and res_s > 0:
            w = rho * res_r / res_s - 1.0
            new = rho * np.exp(0.01 * w + 0.01 * (w - w_prev))
            w_prev = w
            u, rho = u * (rho / new), new
    polished = polish(site, fc, co, best_plan)
    return dict(key=f"{tariff}-{nb}-{limit}-{solar}-{cap}", dw=r.upper - r.lower, last=last - r.lower,
                best=best - r.lower, polished=polished - r.lower, iters=k + 1, s=time.perf_counter() - t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=13)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for row in pool.map(run, [(s, a.levels, a.iters) for s in SITES]):
            rows.append(row)
            print(f"{row['key']:24s} DW {row['dw']:.3f} | textbook ADMM (DP proxes): last {row['last']:.3f}"
                  f" best {row['best']:.3f} polished {row['polished']:.3f} | {row['iters']} it {row['s']:.0f}s",
                  flush=True)
    m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    print(f"mean above the bound ({a.levels} duty levels): DW {m('dw'):.3f}  textbook ADMM last {m('last'):.3f}"
          f"  best {m('best'):.3f}  best + polish {m('polished'):.3f}; iterations {m('iters'):.0f}")


if __name__ == "__main__":
    main()
