"""Can a better rho cut textbook ADMM's iterations? Sweep it.

    .venv/bin/python bench/rho_sweep.py          # exact QP device steps
    .venv/bin/python bench/rho_sweep.py --dp     # the testbed's DPs as device steps

On 12 of bench/admm_stability.py's sites (each tariff x 1 or 3 batteries x a
7 kW limit or none; 5 kW of PV, 10 kWh per battery, 48 hours):

- default: proximal message passing (bench/prior_art.py) with exact QP device
  steps on the relaxed devices, for a range of starting rho, with the paper's
  rho controller and with rho fixed: iterations to the paper's stopping test
  (cap 2000), time, and the relaxed objective reached (the problem is convex,
  so every converged run should reach the same value);
- --dp: the same loop with the testbed's DPs as device steps
  (bench/pmp_dp.py; 13 duty levels, cap 300): iterations, how many
  converged, and the best plan after the polish, above the DW bound.
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

from dw.webapi import _site_fc  # noqa: E402
from prior_art import build, pmp, relaxed_objective  # noqa: E402

SITES = [(t, nb, lim, 5, 10) for t, nb, lim in itertools.product(("dynamic", "day_night", "flat"), (1, 3), (7, None))]
RHOS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def qp_task(args):
    (tariff, nb, limit, solar, cap), rho0, adapt = args
    site, fc, _ = _site_fc({"tariff": tariff, "n_batteries": nb, "grid": 50, "max_import_kw": limit,
                            "solar_peak": solar, "batt_capacity": cap})
    devs, _ = build(site, fc)
    t = time.perf_counter()
    lam = 0.01 if adapt else 0.0
    p, u, rho, iters, ok = pmp(devs, site.horizon.steps, rho=rho0, lam=lam, mu=lam, max_iter=2000)
    return dict(site=(tariff, nb, limit), rho0=rho0, adapt=adapt, iters=iters, ok=ok,
                s=time.perf_counter() - t, obj=relaxed_objective(devs, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    if not a.dp:
        tasks = [(s, r, ad) for s in SITES for r in RHOS for ad in (True, False)]
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(qp_task, tasks))
        best = {}
        for r in rows:
            if r["ok"]:
                best[r["site"]] = min(best.get(r["site"], np.inf), r["obj"])
        for ad in (True, False):
            for rho0 in RHOS:
                sel = [r for r in rows if r["rho0"] == rho0 and r["adapt"] == ad]
                dev = max(abs(r["obj"] - best[r["site"]]) for r in sel if r["ok"] and r["site"] in best) \
                    if any(r["ok"] for r in sel) else float("nan")
                print(f"{'adaptive' if ad else 'fixed   '} rho0 {rho0:5.2f}: converged {sum(r['ok'] for r in sel):2d}/12"
                      f"  iterations median {np.median([r['iters'] for r in sel]):6.0f}"
                      f" max {max(r['iters'] for r in sel):5d}  time mean {np.mean([r['s'] for r in sel]):5.1f}s"
                      f"  | converged runs within {dev:.1e} of the best objective", flush=True)
    else:
        from pmp_dp import run
        variants = [(0.1, True), (0.03, False), (0.1, False), (0.3, False), (1.0, False)]
        tasks = [(s, 13, 300, r, ad) for s in SITES for r, ad in variants]
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(run, tasks))
        for i, (r0, ad) in enumerate(variants):
            sel = rows[i::len(variants)]
            print(f"{'adaptive' if ad else 'fixed   '} rho0 {r0:5.2f}: converged {sum(r['iters'] < 300 for r in sel):2d}/12"
                  f"  iterations median {np.median([r['iters'] for r in sel]):4.0f}"
                  f"  best plan + polish above the bound {np.mean([r['polished'] for r in sel]):.3f}"
                  f"  (before polish {np.mean([r['best'] for r in sel]):.3f})  time mean {np.mean([r['s'] for r in sel]):4.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
