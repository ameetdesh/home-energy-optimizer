"""Accelerating Dantzig-Wolfe when batteries bid plans ("Batteries in LP" off).

    .venv/bin/python bench/dw_accel.py [--tariffs dynamic,day_night] [--iters 45]

The app's default site (2 x 10 kWh batteries, continuous tank, HVAC, 5 kW PV,
7 kW import limit, 24 h, 200-state grids) with the batteries as bidding
devices. Each variant runs to the iteration cap with the per-iteration
"plan if stopped here" on, and reports how fast the gap between the best
runnable plan and the lower bound closes.

Variants beyond the run() options:

* prox   - each iteration each battery also answers at the meter price with a
           tether to its current blend: best plan at pi, staying near where the
           blend is. Linear prices alone give all-or-nothing plans; the tether
           gives interior ones, which is where a battery's optimum usually is.
* warm   - seed the pool with the battery plans of a batteries-in-LP solve
           (fast). Only possible when the battery CAN be written as an LP.
* ema    - momentum on the prices: also ask at a moving average of past pi
           (heavy-ball damping of the price swings).
* nest   - Nesterov-style: also ask at pi_k + beta (pi_k - pi_{k-1}).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from home_energy_optimizer.dw.coordinator import Column, DWCoordinator  # noqa: E402
from home_energy_optimizer.dw.webapi import build_site  # noqa: E402
from home_energy_optimizer import demo_forecasts  # noqa: E402
from home_energy_optimizer.dp_battery import solve_battery  # noqa: E402

APP = {"n_batteries": 2, "batt_capacity": 10, "solar_peak": 5, "max_import_kw": 7,
       "hours": 24, "grid": 200, "terminal_mode": "linear"}


class Accel(DWCoordinator):
    def __init__(self, *a, seed_plans=None, prox_rho=0.0, momentum=None, beta=0.5, **k):
        super().__init__(*a, **k)
        self.seed_plans = seed_plans or {}
        self.prox_rho, self.momentum, self.beta = prox_rho, momentum, beta
        self._mix, self._prev_pi, self._ema = {}, None, None

    def seed(self):
        super().seed()
        for dev in self.devices:
            for col in self.seed_plans.get(dev.key, []):
                dev.add(Column(col.power.copy(), col.trajectory.copy(),
                               self.private_cost(dev, col.trajectory), "warm"))

    def mixed_power(self, weights):
        self._mix = super().mixed_power(weights)
        return self._mix

    def load_aware_oracle(self, dev, others, price_kwh):
        col = super().load_aware_oracle(dev, others, price_kwh)
        if self.prox_rho > 0 and dev.kind == "battery" and dev.key in self._mix:
            sol = solve_battery(dev.cfg, self.cfg.horizon, price_kwh, price_kwh, dp_load=None,
                                admm_target=self._mix[dev.key], admm_rho=self.prox_rho)
            dev.add(Column(sol.power.copy(), sol.trajectory.copy(),
                           self.private_cost(dev, sol.trajectory), "prox"))
        return col

    def extra_price_points(self, pi, center, it):
        out = []
        if self.momentum == "ema":
            self._ema = pi.copy() if self._ema is None else self.beta * self._ema + (1 - self.beta) * pi
            out.append(self._ema.copy())
        elif self.momentum == "nest" and self._prev_pi is not None:
            out.append(pi + self.beta * (pi - self._prev_pi))
        self._prev_pi = pi.copy()
        return out


def setup(tariff: str):
    site = build_site(APP)
    fc = demo_forecasts(site.horizon, tariff=tariff, solar_peak_kw=APP["solar_peak"])
    return site, fc


def warm_plans(site, fc):
    co = DWCoordinator(site, fc, battery_in_master=True, tank_in_master=True)
    r = co.run(max_iter=40)
    return {k: [c] for k, c in r.plan.items() if k.startswith("battery")}


VARIANTS = {
    "baseline (app defaults)": ({}, {}),
    "smoothing 0.8": ({}, {"smoothing": 0.8}),
    "no smoothing": ({}, {"smoothing": 0.0}),
    "pool: keep all": ({}, {"pool": "full"}),
    "plan + sensitivity": ({}, {"response": "both"}),
    "seed from ADMM": ({}, {"seed_admm": True}),
    "momentum: EMA 0.5": ({"momentum": "ema", "beta": 0.5}, {}),
    "momentum: EMA 0.8": ({"momentum": "ema", "beta": 0.8}, {}),
    "momentum: Nesterov 0.5": ({"momentum": "nest", "beta": 0.5}, {}),
    "tethered battery plans rho=1": ({"prox_rho": 1.0}, {}),
    "tethered battery plans rho=5": ({"prox_rho": 5.0}, {}),
    "warm start (batteries-in-LP plans)": ({"warm": True}, {}),
}


def first(history, lb_final, thr):
    for h in history:
        if h["best_plan"] - h["lb"] <= thr:
            return h["iter"], h["ms"] / 1000
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tariffs", default="dynamic,day_night")
    ap.add_argument("--iters", type=int, default=45)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for tariff in args.tariffs.split(","):
        site, fc = setup(tariff)
        ref = DWCoordinator(site, fc, battery_in_master=True, tank_in_master=True).run(max_iter=40)
        print(f"\n== {tariff}: batteries IN the LP (reference): plan {ref.upper:.3f}, bound {ref.lower:.3f}, "
              f"{ref.iterations} it, {ref.ms / 1000:.1f} s")
        print(f"{'variant':36s} {'plan':>7s} {'bound':>7s} {'gap':>6s} {'gap<=.05 @':>11s} {'gap<=.02 @':>11s} {'total s':>7s}")
        for name, (ck, rk) in VARIANTS.items():
            if args.only and args.only not in name:
                continue
            t0 = time.perf_counter()
            ck = dict(ck)
            if ck.pop("warm", False):
                ck["seed_plans"] = warm_plans(site, fc)
            co = Accel(site, fc, battery_in_master=False, tank_in_master=True, **ck)
            r = co.run(max_iter=args.iters, anytime=True, **rk)
            tot = time.perf_counter() - t0
            f5, f2 = first(r.history, r.lower, 0.05), first(r.history, r.lower, 0.02)
            fmt = lambda f: "never" if f[0] is None else f"{f[0]:3d} ({f[1]:4.1f}s)"  # noqa: E731
            print(f"{name:36s} {r.upper:7.3f} {r.lower:7.3f} {r.upper - r.lower:6.3f} {fmt(f5):>11s} {fmt(f2):>11s} {tot:7.1f}")


if __name__ == "__main__":
    main()
