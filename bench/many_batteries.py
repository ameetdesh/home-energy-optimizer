"""Why textbook ADMM does not settle with many batteries (docs/theory.tex,
Appendix on ADMM on the test sites).

    .venv/bin/python bench/many_batteries.py        # the exact-step run needs osqp

The site is the testbed's with N batteries (10 kWh, then larger units up to
28 kWh at N = 8), the tank and the HVAC, a dynamic tariff, a 7 kW import
limit, 24 hours.

1. Exact steps. Proximal message passing with every prox solved as a QP
   (bench/prior_art.py: continuous batteries, the tank and HVAC relaxed). Every
   cost is convex, so the convergence theorem applies - however many batteries
   share a price margin.
2. DP steps, the method as shipped (hemspolicy.exchange), on the batteries'
   50-state, 26-action grid, for N = 1, 2, 4, 8: the primal residual it stalls
   at (the mean over the last 20 iterations) against the stopping threshold,
   and, over those iterations, how many batteries change their plan at once
   in a slot where any does, and what share of them move the same way as the
   total.
3. DP steps at N = 8 on finer battery grids: the stall level against the
   action spacing.
4. LP steps (each battery's own LP, solved exactly: hemspolicy.battery_qp) at
   N = 8: the lockstep is gone and the loop converges; the on/off tank and
   HVAC still keep the plan from the optimum.
5. Batteries alone (three, no tank or HVAC) with LP steps and the bill
   unrounded: every step exact and convex, so the loop converges to the
   optimum - the DW master's, which holds the batteries exactly - to within
   whatever the stopping tolerance asks.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "bench")]

from dw.coordinator import Column, DWCoordinator  # noqa: E402
from dw.webapi import _site_fc  # noqa: E402
from hemspolicy.exchange import ExchangeRun  # noqa: E402
from hemspolicy.types import CoordinationConfig  # noqa: E402


def site(n_batteries: int, grid: int = 50, thermal: bool = True):
    return _site_fc({"tariff": "dynamic", "n_batteries": n_batteries, "hours": 24, "max_import_kw": 7,
                     "batt_capacity": 10, "solar_peak": 5, "grid": grid,
                     "enable_wh": thermal, "enable_hvac": thermal})[:2]


def above_bound(s, fc, devices) -> tuple[float, float]:
    co = DWCoordinator(s, fc)
    r = co.run(max_iter=40)
    val = co.parts({k: Column(d.power, d.trajectory, 0.0, "x") for k, d in devices.items()})["total"]
    return val - r.lower, r.upper - r.lower


def lp8(kink: float) -> str:
    s, fc = site(8)
    cc = CoordinationConfig(algorithm="exchange", exchange_battery_step="lp", kink_smoothing=kink)
    run = ExchangeRun(replace(s, coordination=cc), fc)
    run.step()
    res = run.result()
    dist, dw = above_bound(s, fc, res.devices)
    floor = float(np.mean([r.primal_res for r in run.records[-20:]]))
    return (f"LP steps,     8 batteries, kink rounded over {kink:.2f} kW: {run.stop_reason} after {run.k} | "
            f"residual, last 20: {floor:5.2f} | above the bound {dist:.3f} (DW {dw:.3f})")


def convex(eps: float) -> str:
    s, fc = site(3, thermal=False)
    cc = CoordinationConfig(algorithm="exchange", exchange_battery_step="lp", kink_smoothing=0.0,
                            exchange_eps=eps, exchange_rounds=3000, exchange_patience=3000)
    run = ExchangeRun(replace(s, coordination=cc), fc)
    run.step()
    dist, dw = above_bound(s, fc, run.result().devices)
    return (f"batteries alone, LP steps, stopping tolerance {eps:.0e}: {run.stop_reason} after {run.k:4d} | "
            f"above the optimum {dist:.4f} (DW gap {dw:.1e})")


def exact(n_batteries: int) -> str:
    from prior_art import build, pmp

    s, fc = site(n_batteries)
    devs, _ = build(s, fc)
    _, _, _, iters, ok = pmp(devs, s.horizon.steps, max_iter=2000)
    inexact = sum(getattr(d, "inexact", 0) for d in devs)
    return (f"exact steps,  {n_batteries} batteries: {'converged' if ok else 'NOT converged'} in {iters} iterations"
            + (f" ({inexact} proxes not solved to tolerance)" if inexact else ""))


def dp(args) -> str:
    n_batteries, grid = args
    s, fc = site(n_batteries, grid)
    run = ExchangeRun(replace(s, coordination=CoordinationConfig(algorithm="exchange")), fc)
    run.step()
    recs = run.records
    keys = [k for k in recs[0].powers if k.startswith("battery")]
    floor = float(np.mean([r.primal_res for r in recs[-20:]]))
    moves = np.array([[r.powers[k] for k in keys] for r in recs[-21:]])      # (21, N, slots)
    d = np.diff(moves, axis=0)                                                # (20, N, slots)
    moved = np.abs(d) > 1e-6
    agg = d.sum(axis=1)                                                       # (20, slots)
    busy = moved.any(axis=1)                                                  # slot-iterations with a move
    k = moved.sum(axis=1)[busy]                                               # batteries moving in each
    same = (np.sign(d) == np.sign(agg)[:, None, :]) & moved                   # moving with the aggregate
    with_agg = float(same.sum(axis=1)[busy].sum() / max(k.sum(), 1))
    b = s.battery_list[0]
    step = (b.p_charge_max_kw + b.p_discharge_max_kw) / (b.n_actions - 1)
    return (f"DP steps,     {n_batteries} batteries, {b.n_states:3d} states / {b.n_actions:3d} actions "
            f"(power step {step:.2f} kW): {run.stop_reason} after {run.k} | residual, last 20: {floor:5.2f} "
            f"(stop at {run.eps:.3f}) | where the plans move, {k.mean():.1f} of {n_batteries} batteries move at once, "
            f"{100 * with_agg:.0f}% of them the same way")


def main() -> None:
    with ProcessPoolExecutor(7) as pool:
        ex = pool.submit(exact, 8) if "--dp-only" not in sys.argv else None
        lps = [pool.submit(lp8, k) for k in (0.25, 0.0)]
        cvx = [pool.submit(convex, e) for e in (1e-3, 1e-4, 1e-5, 1e-6)]
        rows = list(pool.map(dp, [(1, 50), (2, 50), (4, 50), (8, 50), (8, 100), (8, 200)]))
        if ex is not None:
            print(ex.result(), flush=True)
        for r in rows:
            print(r, flush=True)
        for f in lps + cvx:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
