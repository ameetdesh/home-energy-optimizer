"""Tune ADMM (admm.coordinator) for speed without losing quality.

    .venv/bin/python bench/exchange_tune.py ['{"relax_levels": 5}' ...]

On 12 of the 72 sites (each tariff x 1 or 3 batteries x a 7 kW limit or none;
5 kW of PV, 10 kWh per battery, 48 hours), each variant - a set of
CoordinationConfig overrides on top of the defaults - is scored by
DWCoordinator.parts against the DW bound of the site's (on/off) problem, with
its iterations, why it stopped, and its time. No argument runs a default set.
"""

from __future__ import annotations

import itertools
import json
import os
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
from home_energy_optimizer.coordinate import coordinate  # noqa: E402
from home_energy_optimizer.types import CoordinationConfig  # noqa: E402

SITES = [(t, nb, lim, 5, 10) for t, nb, lim in itertools.product(("dynamic", "day_night", "flat"), (1, 3), (7, None))]
DEFAULT = [{}, {"relax_levels": 5}, {"kink_smoothing": 0.25}, {"exchange_momentum": True},
           {"exchange_rho": 0.03}, {"relax_levels": 5, "kink_smoothing": 0.25, "exchange_momentum": True},
           {"exchange_rounds": 300}]


def load(s):
    tariff, nb, limit, solar, cap = s
    return _site_fc({"tariff": tariff, "n_batteries": nb, "grid": 50, "max_import_kw": limit,
                     "solar_peak": solar, "batt_capacity": cap})


def bound(s):
    site, fc, _ = load(s)
    return DWCoordinator(site, fc, tank_in_master=False).run(max_iter=40).lower


def task(args):
    s, over, b = args
    site, fc, _ = load(s)
    cc = CoordinationConfig(**over)
    t = time.perf_counter()
    res = coordinate(replace(site, coordination=cc), fc)
    sec = time.perf_counter() - t
    co = DWCoordinator(site, fc, tank_in_master=False)
    val = co.parts({k: Column(d.power, d.trajectory, 0.0, "x") for k, d in res.devices.items()})["total"]
    return dict(site=s, over=json.dumps(over), dist=val - b, rounds=res.rounds_run, stop=res.stop_reason, s=sec)


def main() -> None:
    variants = [json.loads(a) for a in sys.argv[1:]] or DEFAULT
    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers) as pool:
        bounds = dict(zip(SITES, pool.map(bound, SITES)))
        rows = list(pool.map(task, [(s, v, bounds[s]) for v in variants for s in SITES]))
    for v in variants:
        sel = [r for r in rows if r["over"] == json.dumps(v)]
        stops = {k: sum(r["stop"] == k for r in sel) for k in ("converged", "no improvement", "iteration cap")}
        print(f"{json.dumps(v) or '{}':70s} above bound {np.mean([r['dist'] for r in sel]):.3f}"
              f" (worst {max(r['dist'] for r in sel):.3f}) | iterations median {np.median([r['rounds'] for r in sel]):.0f}"
              f" | stops {stops} | time mean {np.mean([r['s'] for r in sel]):.0f}s", flush=True)


if __name__ == "__main__":
    main()
