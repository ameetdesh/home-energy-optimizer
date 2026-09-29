"""ADMM variants on 72 sites, scored against the Dantzig-Wolfe lower bound.

    .venv/bin/python bench/admm_stability.py                   # ADMM, the defaults
    .venv/bin/python bench/admm_stability.py '{"kink_smoothing": 0}'

Each argument is a set of CoordinationConfig overrides (JSON); no argument runs
ADMM with its defaults. The sites are the
testbed's (dw/webapi.py _site_fc): the three tariffs x 1-3 batteries x a 7 kW
import limit or none x 5 or 8 kW of PV x 10 or 20 kWh per battery, a 50-state
battery grid, 48 hours (the site builder's default).

Per variant it prints the mean and worst distance above the DW lower bound -
each plan scored by DWCoordinator.parts, the basis the bound is stated on, so
each number is a guaranteed distance from the best possible plan - split
by whether a limit applies; how many runs stopped within 3 iterations or ran
into the iteration cap; the mean time; and how many plans exceed the import
limit, and by how much. The 72-site numbers in docs/theory.tex (Appendix
"Numerical evidence", ADMM on the test sites) come from this script.
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

from dw.coordinator import Column, DWCoordinator  # noqa: E402
from dw.webapi import _site_fc  # noqa: E402
from hemspolicy.coordinate import coordinate  # noqa: E402
from hemspolicy.types import CoordinationConfig  # noqa: E402

SITES = list(itertools.product(("dynamic", "day_night", "flat"), (1, 2, 3), (7, None), (5, 8), (10, 20)))


def _load(key):
    tariff, nb, limit, solar, cap = key
    site, fc, _ = _site_fc({"tariff": tariff, "n_batteries": nb, "grid": 50, "max_import_kw": limit,
                            "solar_peak": solar, "batt_capacity": cap})
    return site, fc


def _bound(key):
    site, fc = _load(key)
    return DWCoordinator(site, fc, tank_in_master=False).run(max_iter=40).lower


def _run(args):
    key, over, bound = args
    site, fc = _load(key)
    cc = CoordinationConfig(**over)
    t = time.perf_counter()
    res = coordinate(replace(site, coordination=cc), fc)
    ms = (time.perf_counter() - t) * 1000
    co = DWCoordinator(site, fc, tank_in_master=False)
    plan = {k: Column(d.power, d.trajectory, 0.0, "admm") for k, d in res.devices.items()}
    return co.parts(plan)["total"] - bound, res.rounds_run, ms, cc, res.grid_import_excess


def main() -> None:
    variants = [json.loads(a) for a in sys.argv[1:]] or [{}]
    workers = max(1, (os.cpu_count() or 2) - 2)        # the sites are independent
    with ProcessPoolExecutor(workers) as pool:
        bounds = list(pool.map(_bound, SITES))
        for over in variants:
            out = list(pool.map(_run, [(k, over, b) for k, b in zip(SITES, bounds)]))
            d = np.array([o[0] for o in out])
            r = np.array([o[1] for o in out])
            ms = [o[2] for o in out]
            brk = np.array([o[4] for o in out])
            cap = out[0][3].exchange_rounds
            lim = np.array([k[2] is not None for k in SITES])
            print(f"{json.dumps(over) or '{}':40s} mean {d.mean():.3f} worst {d.max():.3f} | "
                  f"limit {d[lim].mean():.3f} (worst {d[lim].max():.3f}) no limit {d[~lim].mean():.3f} | "
                  f"<=3 rounds {(r <= 3).sum()}, at the cap {(r >= cap).sum()} | {np.mean(ms) / 1000:.2f} s | "
                  f"over the limit: {(brk > 1e-3).sum()} sites, at most {brk.max():.3f} kW",
                  flush=True)


if __name__ == "__main__":
    main()
