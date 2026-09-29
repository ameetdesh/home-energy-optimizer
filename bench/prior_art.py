"""Proximal message passing (Kraning, Chu, Lavaei & Boyd) on the testbed's sites.

    .venv/bin/python bench/prior_art.py [--sites dynamic-3-7-5-10,...] [--iters 2000]

Needs osqp (pip install osqp): every device's prox is solved as an exact QP.

The sites are bench/admm_stability.py's 72 (48 hours each). The algorithm is
the paper's section 5.1, on one net - the house - whose terminals are
single-terminal devices as in its chapter 3:

  fixed load        p = load
  PV                p in [-g, 0], zero cost (curtailment is free to choose)
  grid connection   the paper's "external tie": import at the buy price,
                    export at the sell price, plus the breach price beyond a
                    grid limit, exactly as the objective scores it
  each battery      charge/discharge limits and efficiencies, capacity and
                    reserve, stored energy at the end valued at its price
  tank, HVAC        on/off and three-way devices, so - as the paper prescribes
                    for non-convex devices (sections 4.2 and 5.2) - message
                    passing runs on their convex relaxations: a modulating
                    element, and heat and cool duties h, k >= 0 with h + k <= 1

  1. prox updates   p_d <- argmin f_d(p) + (rho/2) |p - (p_d - pbar - u)|^2
  2. price update   u <- u + pbar
  3. rho update     rho <- rho exp(lam w + mu (w - w_prev)),
                    w = rho |r| / |s| - 1, u rescaled; held after `freeze`
  stop when |r| <= eps and |s| <= eps, eps = eps_abs sqrt(|terminals| T),
  r = pbar on every terminal, s = rho ((p - pbar) - (p_prev - pbar_prev)).

This converges to the relaxed optimum. The paper leaves open how to build a
runnable plan from it (section 4.2: "a starting point to help construct good,
local solutions"), so two ways are scored, and they are ours, not the paper's:

  lmp     each on/off device re-plans with its DP at the relaxed locational
          marginal price (rho u per kW-slot); the batteries keep their relaxed
          plans, which they can run as they are;
  polish  that, then the testbed's polish: each device in turn re-plans
          against the others with the limits priced, kept only if the
          objective falls.

Every plan - and ADMM's and Dantzig-Wolfe's for reference - is scored by
DWCoordinator.parts, the basis the DW lower bound is stated on.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dw.coordinator import Column, DWCoordinator, _PriceVector  # noqa: E402
from dw.webapi import _site_fc  # noqa: E402
from hemspolicy.coordinate import breach_price, coordinate, device_sell_price  # noqa: E402
from hemspolicy.dp_battery import solve_battery  # noqa: E402
from hemspolicy.dp_thermal import _relaxation, solve_hvac, solve_water_heater  # noqa: E402
from hemspolicy.meter import Limits  # noqa: E402
from hemspolicy.types import CoordinationConfig, SiteConfig  # noqa: E402

SITES = list(itertools.product(("dynamic", "day_night", "flat"), (1, 2, 3), (7, None), (5, 8), (10, 20)))


# ------------------------------------------------------------------ devices
class Fixed:
    """A fixed load: prox is the load itself."""

    def __init__(self, p):
        self.p = np.asarray(p, float)

    def prox(self, v, rho):
        return self.p


class PV:
    """Curtailable generation, zero cost: prox is projection onto [-g, 0]."""

    def __init__(self, g, curtail: bool):
        self.g, self.curtail = np.asarray(g, float), curtail

    def prox(self, v, rho):
        return np.clip(v, -self.g, 0.0) if self.curtail else -self.g


class Tie:
    """The grid connection. p = power delivered TO the grid (= -import).

    Cost per slot, in the import z = -p: dt (buy z+ - sell z- + c_br (z - L_imp)+
    + c_br (-z - L_exp)+), convex and piecewise linear, so its prox is closed
    form: on each linear piece the minimiser is v - slope / rho, clipped.
    """

    def __init__(self, buy, sell, dt, l_imp, l_exp, c_br):
        self.buy, self.sell, self.dt = np.asarray(buy, float), np.asarray(sell, float), dt
        self.l_imp, self.l_exp, self.c_br = l_imp, l_exp, c_br

    def cost(self, p):
        z = -p
        c = np.maximum(z, 0) * self.buy - np.maximum(-z, 0) * self.sell
        if self.l_imp is not None:
            c = c + self.c_br * np.maximum(z - self.l_imp, 0)
        if self.l_exp is not None:
            c = c + self.c_br * np.maximum(-z - self.l_exp, 0)
        return c * self.dt

    def slope(self, p):
        """d cost / d p on the open piece containing p (per slot)."""
        g = -self.buy * (p < 0) - self.sell * (p > 0)
        if self.l_imp is not None:
            g = g - self.c_br * (p < -self.l_imp)
        if self.l_exp is not None:
            g = g + self.c_br * (p > self.l_exp)
        return g * self.dt

    def prox(self, v, rho):
        # breakpoints in p (= -z): -L_imp, 0, L_exp
        bps = sorted([0.0] + ([-self.l_imp] if self.l_imp is not None else [])
                     + ([self.l_exp] if self.l_exp is not None else []))
        edges = [-np.inf] + bps + [np.inf]
        best, best_val = None, None
        for a, b in zip(edges[:-1], edges[1:]):
            inside = b - 1.0 if a == -np.inf else (a + 1.0 if b == np.inf else (a + b) / 2.0)
            x = np.clip(v - self.slope(np.full_like(v, inside)) / rho, a, b)
            val = self.cost(x) + 0.5 * rho * (x - v) ** 2
            if best is None:
                best, best_val = x, val
            else:
                m = val < best_val
                best, best_val = np.where(m, x, best), np.where(m, val, best_val)
        return best


class QPDevice:
    """A device whose prox is a QP:  min (1/2)|M x - v|^2 + (1/rho) c'x,  l <= A x <= u.

    Dividing the prox objective by rho keeps the quadratic fixed, so only the
    linear term changes between iterations (no refactorisation)."""

    def setup(self, M, c, A, lo, hi):
        self.M, self.c = sp.csc_matrix(M), np.asarray(c, float)
        P = (self.M.T @ self.M).tocsc()
        self.solver = osqp.OSQP()
        self.solver.setup(P=sp.triu(P).tocsc(), q=np.zeros(P.shape[0]), A=sp.csc_matrix(A), l=lo, u=hi,
                          eps_abs=1e-7, eps_rel=1e-7, polish=True, verbose=False, max_iter=100000)
        self.x = None
        self.inexact = 0          # proxes the QP solver did not finish (counted, reported)

    def prox(self, v, rho):
        self.solver.update(q=-(self.M.T @ v) + self.c / rho)
        res = self.solver.solve()
        if res.info.status == "maximum iterations reached":
            self.inexact += 1
        elif res.info.status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"{type(self).__name__} prox: {res.info.status}")
        self.x = res.x
        return self.M @ res.x


class Battery(QPDevice):
    """x = [charge c (n), discharge e (n), stored energy s_1..s_n (n)]; p = c - e."""

    def __init__(self, b, n, dt):
        self.n, s0 = n, b.capacity_kwh * b.soc_initial_frac
        self.s0 = s0
        I = sp.identity(n, format="csc")
        M = sp.hstack([I, -I, sp.csc_matrix((n, n))])
        c = np.zeros(3 * n)
        c[3 * n - 1] = -float(b.terminal_price)           # stored energy at the end is worth its price
        shift = sp.diags([np.ones(n - 1)], [-1], shape=(n, n))
        dyn = sp.hstack([-dt * b.eta_c * I, (dt / b.eta_d) * I, I - shift])   # s_t - s_{t-1} - ... = 0
        rhs = np.zeros(n)
        rhs[0] = s0
        A = sp.vstack([dyn, sp.identity(3 * n)])
        lo = np.concatenate([rhs, np.zeros(2 * n), np.full(n, b.soe_floor_kwh)])
        hi = np.concatenate([rhs, np.full(n, b.p_charge_max_kw), np.full(n, b.p_discharge_max_kw),
                             np.full(n, b.capacity_kwh)])
        self.setup(M, c, A, lo, hi)

    def trajectory(self):
        return np.concatenate([[self.s0], self.x[2 * self.n:]])


class Tank(QPDevice):
    """Relaxed tank (modulating element), the DW LP's dynamics exactly.
    x = [duty D (n), temperature T_1..T_n (n), shortfall S (n), end shortfall ST (1)]."""

    def __init__(self, cfg, n, dt, demand, ref):
        self.n, self.t0 = n, cfg.t_comfort
        C, P = cfg.heat_capacity_kwh_per_k, cfg.power_kw
        r, tinf = np.empty(n), np.empty(n)
        for t in range(n):
            rate, ti = _relaxation(cfg, float(demand[t]))
            r[t], tinf[t] = min(rate, 1.0 / dt), ti
        price_k = cfg.discomfort_price_per_kelvin_hour(ref)
        I = sp.identity(n, format="csc")
        Z = sp.csc_matrix((n, n))
        z1 = sp.csc_matrix((n, 1))
        M = sp.hstack([P * I, Z, Z, z1])
        c = np.concatenate([np.zeros(n), np.zeros(n), np.full(n, price_k * dt), [C * ref]])
        shift = sp.diags([-(1.0 - r[1:] * dt)], [-1], shape=(n, n))
        dyn = sp.hstack([-(P * dt / C) * I, I + shift, Z, z1])
        rhs = r * dt * tinf
        rhs[0] += (1.0 - r[0] * dt) * cfg.t_comfort
        comfort = sp.hstack([Z, I, I, z1])                              # T + S >= t_comfort
        term = sp.hstack([sp.csc_matrix((1, n)), sp.csc_matrix(([1.0], ([0], [n - 1])), shape=(1, n)),
                          sp.csc_matrix((1, n)), sp.csc_matrix([[1.0]])])  # T_n + ST >= t_comfort
        A = sp.vstack([dyn, comfort, term, sp.identity(3 * n + 1)])
        big = np.inf
        lo = np.concatenate([rhs, np.full(n, cfg.t_comfort), [cfg.t_comfort],
                             np.zeros(n), np.full(n, cfg.t_min - 100.0), np.zeros(n), [0.0]])
        hi = np.concatenate([rhs, np.full(n, big), [big],
                             np.ones(n), np.full(n, cfg.t_max), np.full(n, big), [big]])
        self.setup(M, c, A, lo, hi)

    def trajectory(self):
        return np.concatenate([[self.t0], self.x[self.n:2 * self.n]])


class Hvac(QPDevice):
    """Relaxed HVAC: heat and cool duties h, k >= 0, h + k <= 1 (the hull of the
    three-way action), the DP's Euler dynamics, linear discomfort.
    x = [h (n), k (n), T_1..T_n (n), too hot H (n), too cold L (n), end TH, TL]."""

    def __init__(self, cfg, n, dt, outdoor, ref):
        self.n, self.t0 = n, cfg.t_comfort_mid
        P, Cr, R = cfg.power_kw, cfg.c_room_kwh_per_k, cfg.r_wall_k_per_kw
        price_k = cfg.discomfort_price_per_kelvin_hour(ref)
        I = sp.identity(n, format="csc")
        Z = sp.csc_matrix((n, n))
        z1 = sp.csc_matrix((n, 1))
        M = sp.hstack([P * I, P * I, Z, Z, Z, z1, z1])
        c = np.concatenate([np.zeros(3 * n), np.full(2 * n, price_k * dt), [price_k, price_k]])
        a = dt / (Cr * R)
        shift = sp.diags([-(1.0 - a) * np.ones(n - 1)], [-1], shape=(n, n))
        heat, cool = dt / Cr * P * (cfg.cop + 1.0), dt / Cr * P * cfg.cop
        dyn = sp.hstack([-heat * I, cool * I, I + shift, Z, Z, z1, z1])
        rhs = a * np.asarray(outdoor, float)
        rhs[0] += (1.0 - a) * self.t0
        hot = sp.hstack([Z, Z, I, -I, Z, z1, z1])                       # T - H <= high
        cold = sp.hstack([Z, Z, I, Z, I, z1, z1])                       # T + L >= low
        e_n = sp.csc_matrix(([1.0], ([0], [n - 1])), shape=(1, n))
        zr = sp.csc_matrix((1, n))
        one = sp.csc_matrix([[1.0]])
        zero = sp.csc_matrix((1, 1))
        t_hot = sp.hstack([zr, zr, e_n, zr, zr, -one, zero])            # T_n - TH <= high
        t_cold = sp.hstack([zr, zr, e_n, zr, zr, zero, one])            # T_n + TL >= low
        duty = sp.hstack([I, I, Z, Z, Z, z1, z1])                       # h + k <= 1
        A = sp.vstack([dyn, hot, cold, t_hot, t_cold, duty, sp.identity(5 * n + 2)])
        big = np.inf
        lo = np.concatenate([rhs, np.full(n, -big), np.full(n, cfg.t_comfort_low), [-big], [cfg.t_comfort_low],
                             np.full(n, -big), np.zeros(2 * n), np.full(n, cfg.t_min), np.zeros(2 * n), [0.0, 0.0]])
        hi = np.concatenate([rhs, np.full(n, cfg.t_comfort_high), np.full(n, big), [cfg.t_comfort_high], [big],
                             np.ones(n), np.ones(2 * n), np.full(n, cfg.t_max), np.full(2 * n, big), [big, big]])
        self.setup(M, c, A, lo, hi)

    def trajectory(self):
        return np.concatenate([[self.t0], self.x[2 * self.n:3 * self.n]])


# ------------------------------------------------------------------ the algorithm
def pmp(devices, n, rho=0.1, lam=0.01, mu=0.01, eps_abs=1e-3, max_iter=2000, freeze=200):
    """Proximal message passing, the paper's steps 1-3. Returns (p, u, rho, iterations, converged)."""
    N = len(devices)
    p = np.zeros((N, n))
    u = np.zeros(n)
    pbar = p.mean(axis=0)
    prev_dev = p - pbar
    w_prev = 0.0
    eps = eps_abs * np.sqrt(N * n)
    for k in range(max_iter):
        v = p - pbar - u
        p = np.array([d.prox(v[i], rho) for i, d in enumerate(devices)])
        pbar = p.mean(axis=0)
        u = u + pbar
        dev = p - pbar
        r = np.sqrt(N) * np.linalg.norm(pbar)
        s = rho * np.linalg.norm(dev - prev_dev)
        prev_dev = dev
        if r <= eps and s <= eps:
            return p, u, rho, k + 1, True
        if k < freeze and s > 0:
            w = rho * r / s - 1.0
            new = rho * np.exp(lam * w + mu * (w - w_prev))
            w_prev = w
            u = u * (rho / new)
            rho = new
    return p, u, rho, max_iter, False


def build(site, fc):
    """The house as the paper's network: devices in a fixed order, with keys."""
    h, n, dt = site.horizon, site.horizon.steps, site.horizon.dt
    ref = float(np.mean(fc.buy))
    g = site.grid
    c_br = breach_price(g, fc.buy, fc.sell) if g.active else 0.0
    devs, keys = [Fixed(fc.load), PV(fc.solar, g.allow_curtailment),
                  Tie(fc.buy, fc.sell, dt, g.max_import_kw, g.max_export_kw, c_br)], [None, None, None]
    for i, b in enumerate(site.battery_list):
        if b.capacity_kwh > 0:
            devs.append(Battery(b, n, dt))
            keys.append(SiteConfig.battery_key(i))
    if site.water_heater is not None:
        devs.append(Tank(site.water_heater, n, dt, fc.hot_water_demand, ref))
        keys.append("water_heater")
    if site.hvac is not None:
        devs.append(Hvac(site.hvac, n, dt, fc.outdoor_temp, ref))
        keys.append("hvac")
    return devs, keys


def relaxed_objective(devs, p):
    """The relaxed objective of the final device plans, with the grid taking the
    balance (so the plan is feasible for the relaxed problem)."""
    tie = devs[2]
    x = -(p.sum(axis=0) - p[2])
    total = float(tie.cost(x).sum())
    for d in devs[3:]:
        total += float(d.c @ d.x)
        if isinstance(d, Battery):
            total += -float(d.c[-1]) * d.s0        # the value of the energy it started with
    return total


def recover(site, fc, co, devs, keys, p, u, rho, polish_sweeps=5):
    """The two runnable plans: DP best responses at the relaxed prices, then polish."""
    h, dt = site.horizon, site.horizon.dt
    price = _PriceVector(rho * u / dt, co.ref)               # rho u per kW-slot -> per kWh
    zero = np.zeros(h.steps)
    plan = {}
    for i, k in enumerate(keys):
        if k is None:
            continue
        d = devs[i]
        if isinstance(d, Battery):
            plan[k] = Column(p[i].copy(), d.trajectory(), 0.0, "pmp")
        elif k == "water_heater":
            sol = solve_water_heater(site.water_heater, h, price, price, fc.hot_water_demand, dp_load=zero)
            plan[k] = Column(sol.power, sol.trajectory, 0.0, "lmp")
        else:
            sol = solve_hvac(site.hvac, h, price, price, fc.outdoor_temp, dp_load=zero)
            plan[k] = Column(sol.power, sol.trajectory, 0.0, "lmp")
    lmp = co.parts(plan)["total"]

    g = site.grid
    limits = (Limits(g.max_import_kw, g.max_export_kw, breach_price(g, fc.buy, fc.sell), g.allow_curtailment)
              if g.active else None)
    sell = device_sell_price(fc.sell, g)
    batt = {SiteConfig.battery_key(i): b for i, b in enumerate(site.battery_list)}
    cur = lmp
    for _ in range(polish_sweeps):
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
    return lmp, cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="", help="comma-separated keys like dynamic-3-7-5-10 (default: all 72)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rho", type=float, default=0.1)
    a = ap.parse_args()
    wanted = set(filter(None, a.sites.split(",")))
    rows = []
    for tariff, nb, limit, solar, cap in SITES:
        key = f"{tariff}-{nb}-{limit}-{solar}-{cap}"
        if wanted and key not in wanted:
            continue
        site, fc, _ = _site_fc({"tariff": tariff, "n_batteries": nb, "grid": 50, "max_import_kw": limit,
                                "solar_peak": solar, "batt_capacity": cap})
        co = DWCoordinator(site, fc, tank_in_master=False)
        t = time.perf_counter()
        r = co.run(max_iter=40)
        dw_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        res = coordinate(replace(site, coordination=CoordinationConfig(rho=6.0, max_rounds=100)), fc)
        admm_ms = (time.perf_counter() - t) * 1000
        admm = co.parts({k: Column(d.power, d.trajectory, 0.0, "admm") for k, d in res.devices.items()})["total"]
        t = time.perf_counter()
        devs, keys = build(site, fc)
        p, u, rho, iters, ok = pmp(devs, site.horizon.steps, rho=a.rho, max_iter=a.iters)
        relax = relaxed_objective(devs, p)
        lmp, pol = recover(site, fc, co, devs, keys, p, u, rho)
        pmp_ms = (time.perf_counter() - t) * 1000
        row = dict(key=key, limit=limit is not None, bound=r.lower, dw=r.upper, admm=admm, admm_raw=res.total_objective,
                   relax=relax, lmp=lmp, polish=pol, iters=iters, ok=ok, dw_ms=dw_ms, admm_ms=admm_ms, pmp_ms=pmp_ms,
                   inexact=sum(getattr(d, "inexact", 0) for d in devs))
        rows.append(row)
        print(f"{key:24s} bound {r.lower:8.3f} | above bound: DW {r.upper - r.lower:6.3f}  ADMM {admm - r.lower:6.3f}"
              f"  PMP-lmp {lmp - r.lower:6.3f}  PMP-polish {pol - r.lower:6.3f} | relaxed {relax - r.lower:+7.3f}"
              f" | PMP {iters:4d} it{'' if ok else ' (cap)'} {pmp_ms / 1000:5.1f}s"
              f"{' (' + str(row['inexact']) + ' inexact proxes)' if row['inexact'] else ''}", flush=True)
    if len(rows) > 1:
        def m(f, sel=lambda r: True):
            return float(np.mean([f(r) for r in rows if sel(r)]))
        for name, sel in (("all", lambda r: True), ("limit", lambda r: r["limit"]), ("no limit", lambda r: not r["limit"])):
            print(f"mean above bound, {name:8s}: DW {m(lambda r: r['dw'] - r['bound'], sel):.3f}"
                  f"  ADMM {m(lambda r: r['admm'] - r['bound'], sel):.3f}"
                  f"  PMP-lmp {m(lambda r: r['lmp'] - r['bound'], sel):.3f}"
                  f"  PMP-polish {m(lambda r: r['polish'] - r['bound'], sel):.3f}"
                  f"  | relaxed - bound {m(lambda r: r['relax'] - r['bound'], sel):+.3f}")
        print(f"inexact proxes: {sum(r['inexact'] for r in rows)} in {sum(1 for r in rows if r['inexact'])} sites")
        print(f"PMP converged in {sum(r['ok'] for r in rows)} of {len(rows)}; iterations mean "
              f"{m(lambda r: r['iters']):.0f}; time mean PMP {m(lambda r: r['pmp_ms']) / 1000:.1f}s, "
              f"ADMM {m(lambda r: r['admm_ms']) / 1000:.1f}s, DW {m(lambda r: r['dw_ms']) / 1000:.1f}s; "
              f"ADMM scored on the bound's basis minus its own total_objective: mean "
              f"{m(lambda r: r['admm'] - r['admm_raw']):+.4f}, max {max(r['admm'] - r['admm_raw'] for r in rows):+.4f}")


if __name__ == "__main__":
    main()
