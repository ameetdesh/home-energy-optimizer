"""Is ADMM's gap the on/off devices? Relax them and see.

    .venv/bin/python bench/relaxed_devices.py [--levels 13]

On bench/admm_stability.py's 72 sites (48 hours each), twice: with the tank
on/off and the HVAC off / full cool / full heat, and with both allowed to run
a fraction of each slot (`levels` duty steps each way; the DP grids refined to
match). Each time it plans with Dantzig-Wolfe (bound and runnable plan), with
the ADMM loop, and with the ADMM loop stopped after its first round (the seed:
each battery's own best plan) and polished. Every plan is scored by
DWCoordinator.parts against the DW bound of the same problem, so each number is
a guaranteed distance from the best possible plan with those devices.

If the on/off devices were what kept ADMM from the optimum, relaxing them would
close its gap. It prints, per setting, the mean distance above the bound for
DW, ADMM and seed + polish, and how often ADMM's rounds beat the seed.
"""

from __future__ import annotations

import argparse
import itertools
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


def run(p: dict) -> dict:
    site, fc, _ = _site_fc(p)
    co = DWCoordinator(site, fc, tank_in_master=False)
    t = time.perf_counter()
    r = co.run(max_iter=40)
    dw_s = time.perf_counter() - t

    def above(res) -> float:
        return co.parts({k: Column(d.power, d.trajectory, 0.0, "admm") for k, d in res.devices.items()})["total"] - r.lower

    t = time.perf_counter()
    full = coordinate(replace(site, coordination=CoordinationConfig(rho=6.0, max_rounds=100)), fc)
    admm_s = time.perf_counter() - t
    seed = coordinate(replace(site, coordination=CoordinationConfig(rho=6.0, max_rounds=1)), fc)
    o = np.asarray(full.round_objectives)
    return dict(bound=r.lower, dw=r.upper - r.lower, admm=above(full), seed=above(seed),
                beat_seed=bool(o.min() < o[0] - 1e-3), rounds=full.rounds_run, dw_s=dw_s, admm_s=admm_s)


def site(args) -> tuple[str, dict]:
    (tariff, nb, limit, solar, cap), levels = args
    base = {"tariff": tariff, "n_batteries": nb, "grid": 50, "max_import_kw": limit,
            "solar_peak": solar, "batt_capacity": cap}
    return f"{tariff}-{nb}-{limit}-{solar}-{cap}", {lv: run({**base, "tank_levels": lv, "hvac_levels": lv})
                                                    for lv in (2, levels)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=13)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    out = {2: [], a.levels: []}
    with ProcessPoolExecutor(a.workers) as pool:          # the sites are independent
        for key, rows in pool.map(site, [(s, a.levels) for s in SITES]):
            line = key + ":"
            for lv, row in rows.items():
                out[lv].append(row)
                line += (f"  [{'on/off' if lv == 2 else f'{lv} levels'}] DW {row['dw']:.3f} ADMM {row['admm']:.3f}"
                         f" seed+polish {row['seed']:.3f}")
            print(line, flush=True)
    for lv, rows in out.items():
        m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
        # (times are per site with the other workers running alongside)
        print(f"{'on/off' if lv == 2 else f'{lv} duty levels':>15s}: mean above bound  DW {m('dw'):.3f}"
              f"  ADMM {m('admm'):.3f}  seed+polish {m('seed'):.3f} | ADMM rounds beat the seed on"
              f" {sum(r['beat_seed'] for r in rows)} of {len(rows)} sites, {m('rounds'):.1f} rounds"
              f" | time DW {m('dw_s'):.1f}s ADMM {m('admm_s'):.1f}s")
    gain = np.mean([b["bound"] - r["bound"] for b, r in zip(out[2], out[a.levels])])
    print(f"fractional control lowers the bound by {gain:.3f} on average")


if __name__ == "__main__":
    main()
