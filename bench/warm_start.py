"""Warm starts for ADMM: how much do they save?

    .venv/bin/python bench/warm_start.py

1. Within a solve: each LP battery step started from that battery's previous
   solution (CoordinationConfig.exchange_warm_battery) against from scratch -
   time in the battery steps, and in the whole solve.
2. Across solves, as the page re-solves after a setting changes: solve a
   site, change one thing (PV 5 -> 6 kW, a battery 10 -> 12 kWh, the import
   limit 7 -> 6 kW), and solve again cold or from the first solve's best
   state (ExchangeRun(warm=...)).
3. The next planning cycle: the same site with its forecast one slot (15 min)
   later, cold or from the previous solve's state shifted by that slot
   (WarmStart.shift).

For 2 and 3: iterations run, whether the run converged, the returned plan
above the Dantzig-Wolfe bound, the first iteration whose runnable plan came
within 0.05 of the run's best, and time. The sites are the testbed's
(src/home_energy_optimizer/dw/webapi.py _site_fc): dynamic tariff, 24 hours, a 7 kW import limit, the
tank and HVAC, one or two batteries, with DP or LP battery steps.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from home_energy_optimizer.dw.coordinator import Column, DWCoordinator  # noqa: E402
from home_energy_optimizer.dw.webapi import _site_fc  # noqa: E402
from home_energy_optimizer.admm.coordinator import ExchangeRun  # noqa: E402
from home_energy_optimizer.types import CoordinationConfig, Forecasts  # noqa: E402

BASE = {"tariff": "dynamic", "hours": 24, "max_import_kw": 7, "batt_capacity": 10, "solar_peak": 5,
        "grid": 50, "tank_levels": 2}
CHANGES = {"PV 5 -> 6 kW": {"solar_peak": 6}, "battery 10 -> 12 kWh": {"batt_capacity": 12},
           "import limit 7 -> 6 kW": {"max_import_kw": 6}}


def config(step: str, warm_battery: bool = True) -> CoordinationConfig:
    return CoordinationConfig(exchange_battery_step=step, exchange_warm_battery=warm_battery)


def solve(site, fc, cc, warm=None):
    run = ExchangeRun(replace(site, coordination=cc), fc, warm=warm)
    t = time.perf_counter()
    run.step()
    res = run.result()
    return run, res, time.perf_counter() - t


def score(site, fc, run, res, seconds) -> dict:
    co = DWCoordinator(site, fc, tank_in_master=False)
    lower = co.run(max_iter=40).lower
    val = co.parts({k: Column(d.power, d.trajectory, 0.0, "x") for k, d in res.devices.items()})["total"]
    objs = [r.total_objective for r in run.records]
    near = next(i for i, o in enumerate(objs) if o <= run.best_obj + 0.05) + 1
    return dict(iters=run.k, stop=run.stop_reason, above=val - lower, near=near, s=seconds, warm=run.warm_used)


def within(args) -> str:
    nb, thermal = args
    site, fc, _ = _site_fc({**BASE, "n_batteries": nb, "enable_wh": thermal, "enable_hvac": thermal})
    out = []
    for warm_battery in (False, True):
        run, res, s = solve(site, fc, config("lp", warm_battery))
        batt = sum(v for r in run.records for k, v in r.device_ms.items() if k.startswith("battery")) / 1000
        out.append((batt, s, run.k, res.total_objective))
    (bc, sc, kc, oc), (bw, sw, kw, ow) = out
    return (f"{nb} batteries{' + tank + HVAC' if thermal else ', alone'}, LP steps: battery steps {bc:.2f} -> {bw:.2f} s,"
            f" solve {sc:.1f} -> {sw:.1f} s ({kc} and {kw} iterations; plans {oc:.4f} and {ow:.4f})")


def across(args) -> list[str]:
    nb, step, name = args
    site_a, fc_a, _ = _site_fc({**BASE, "n_batteries": nb})
    site_b, fc_b, _ = _site_fc({**BASE, "n_batteries": nb, **CHANGES[name]})
    run_a, _, _ = solve(site_a, fc_a, config(step))
    rows = []
    for label, warm in (("cold", None), ("warm", run_a.warm_state())):
        run, res, s = solve(site_b, fc_b, config(step, warm is not None), warm)
        rows.append((label, score(site_b, fc_b, run, res, s)))
    return [f"{nb} batt, {step} steps, {name:22s} {label}: " + fmt(m) for label, m in rows]


def rolling(args) -> list[str]:
    nb, step = args
    site, _, _ = _site_fc({**BASE, "n_batteries": nb})
    _, long, _ = _site_fc({**BASE, "n_batteries": nb, "hours": 24.25})        # one slot longer
    n = site.horizon.steps
    cut = lambda a, b: Forecasts(**{f: getattr(long, f)[a:b] for f in  # noqa: E731
                                    ("buy", "sell", "load", "solar", "outdoor_temp", "hot_water_demand")})
    fc_a, fc_b = cut(0, n), cut(1, n + 1)
    run_a, _, _ = solve(site, fc_a, config(step))
    rows = []
    for label, warm in (("cold", None), ("warm", run_a.warm_state().shift(1))):
        run, res, s = solve(site, fc_b, config(step, warm is not None), warm)
        rows.append((label, score(site, fc_b, run, res, s)))
    return [f"{nb} batt, {step} steps, one slot later       {label}: " + fmt(m) for label, m in rows]


def fmt(m: dict) -> str:
    return (f"{m['iters']:3d} it ({m['stop']:13s}) | above bound {m['above']:6.3f} | within 0.05 of its best by it"
            f" {m['near']:3d} | {m['s']:5.1f} s{'' if m['warm'] or True else ''}")


def main() -> None:
    with ProcessPoolExecutor(6) as pool:
        w = list(pool.map(within, [(2, True), (8, True), (3, False)]))
        a = list(pool.map(across, [(nb, st, nm) for nb in (1, 2) for st in ("dp", "lp") for nm in CHANGES]))
        r = list(pool.map(rolling, [(nb, st) for nb in (1, 2) for st in ("dp", "lp")]))
    print("1. within a solve (LP battery steps, cold -> warm):")
    for line in w:
        print("  ", line)
    print("2. across solves, one setting changed:")
    for rows in a:
        for line in rows:
            print("  ", line)
    print("3. the next planning cycle (forecast one slot later):")
    for rows in r:
        for line in rows:
            print("  ", line)


if __name__ == "__main__":
    main()
