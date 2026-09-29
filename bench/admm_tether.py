"""ADMM on the app's default site: how the tether (rho) trades speed for
quality, and whether over-relaxation (ADMM's momentum analog) helps.

    .venv/bin/python bench/admm_tether.py [--tariffs dynamic,day_night]

Part A runs hemspolicy.coordinate itself (current defaults: polish on, grid
step 4) with the starting rho swept, rho adaptation on (default) and off
(fixed rho). Part B uses bench/admm_variants.py's harness (the rounds only,
no polish) with over-relaxation alpha in the target and nu steps. Every plan
is scored against the DW lower bound on the same model (on/off tank).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "bench")]

import admm_variants as A  # noqa: E402
from dw.coordinator import Column, DWCoordinator  # noqa: E402
from dw.integrate import dw_plan  # noqa: E402
from dw.webapi import build_site  # noqa: E402
from hemspolicy import CoordinationConfig, coordinate, demo_forecasts  # noqa: E402

APP = {"n_batteries": 2, "batt_capacity": 10, "solar_peak": 5, "max_import_kw": 7,
       "hours": 24, "grid": 200, "terminal_mode": "linear"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tariffs", default="dynamic,day_night")
    args = ap.parse_args()
    for tariff in args.tariffs.split(","):
        site = build_site(APP)
        fc = demo_forecasts(site.horizon, tariff=tariff, solar_peak_kw=APP["solar_peak"])
        dw = dw_plan(site, fc)
        lb = dw.lower_bound
        co = DWCoordinator(site, fc)
        score = lambda devs: co.parts({k: Column(d.power if hasattr(d, "power") else d[0],  # noqa: E731
                                                 d.trajectory if hasattr(d, "trajectory") else d[1], 0.0, "x")
                                       for k, d in devs.items()})["total"]
        print(f"\n== {tariff}: DW plan {dw.plan_objective:.3f}, bound {lb:.3f} ({dw.rounds_run} it)")
        print("A. coordinate(), tether swept.  above bound after round 1/3/5/10/20/40 (best so far, before polish);"
              " returned (with polish); rounds; seconds")
        for adapt in (True, False):
            for rho in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
                cc = CoordinationConfig(rho=rho, max_rounds=40,
                                        rho_adapt_factor=1.5 if adapt else 1.0)
                t0 = time.perf_counter()
                res = coordinate(replace(site, coordination=cc), fc)
                s = time.perf_counter() - t0
                per = [score({k: (r.powers[k], r.trajectories[k]) for k in r.powers}) for r in res.rounds]
                best = np.minimum.accumulate(per)
                at = lambda i: f"{best[min(i, len(best)) - 1] - lb:6.3f}"  # noqa: E731
                print(f"   rho0={rho:4g} {'adaptive' if adapt else 'fixed   '}  "
                      + " ".join(at(i) for i in (1, 3, 5, 10, 20, 40))
                      + f"   returned {score(res.devices) - lb:6.3f}   {res.rounds_run:2d} rounds  {s:4.1f} s")
        print("B. over-relaxation (rounds only, no polish): best above bound in 40 rounds; round within 0.05 of it; stop round")
        for alpha in (1.0, 1.3, 1.6, 1.8):
            tr = A.run(site, fc, replace(A.AS_IMPLEMENTED, relax=alpha), rounds=40)
            b = np.minimum.accumulate([t["obj"] for t in tr])
            st = A.stop_round(tr, site.coordination, site.grid.active)
            print(f"   alpha={alpha:3.1f}  best {b[-1] - lb:6.3f}  within 0.05 at round {int(np.argmax(b <= b[-1] + 0.05)) + 1:2d}"
                  f"  stop at {st:2d} (above bound there {b[st - 1] - lb:6.3f})")


if __name__ == "__main__":
    main()
