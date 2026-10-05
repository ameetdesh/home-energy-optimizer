"""An independent LP of a battery-only site with an electrical tree, written
out from the model's definition (docs/theory.tex, "Sub-meters and local
prices") with scipy's HiGHS - to check the Dantzig-Wolfe master and
submeter.site_meter against. Not imported by the package."""

from __future__ import annotations

import numpy as np

from home_energy_optimizer.coordinate import breach_price
from home_energy_optimizer.dp_battery import terminal_price
from home_energy_optimizer.types import MAIN, Forecasts, SiteConfig


def reference_lp(site: SiteConfig, fc: Forecasts) -> float:
    """The optimal objective of the site (batteries only), as one LP."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, dt = site.horizon.steps, site.horizon.dt
    var: list[tuple[float, float, float]] = []           # (cost, lb, ub) per variable, one slot each
    rows: list[list[tuple[int, float]]] = []             # equality rows: [(var, coef)]
    rhs: list[float] = []
    const = 0.0

    def new(cost: np.ndarray | float, lb: float, ub: np.ndarray | float) -> np.ndarray:
        idx = np.arange(len(var), len(var) + n)
        cost, ub = np.broadcast_to(cost, (n,)), np.broadcast_to(ub, (n,))
        for t in range(n):
            var.append((float(cost[t]), lb, float(ub[t])))
        return idx

    roots = [MAIN] + [c.name for c in site.connections]
    nodes = [sm.name for sm in site.submeters]
    bal = {b: [[] for _ in range(n)] for b in roots + nodes}
    bal_rhs = {b: np.zeros(n) for b in roots + nodes}
    bal_rhs[MAIN] = bal_rhs[MAIN] + fc.load
    allow = site.grid.allow_curtailment
    big = 1e4
    for key in fc.pv_keys:                       # PV: a fixed supply where it sits
        b = site.bus_of(key)
        bal_rhs[b] = bal_rhs[b] - fc.pv(key)
    for b in roots:
        if b == MAIN:
            g, buy, sell = site.grid, fc.buy, fc.sell
        else:
            g = next(c.grid for c in site.connections if c.name == b)
            buy, sell = (np.asarray(a, dtype=float) for a in fc.tariffs[b])
        B = breach_price(g, buy, sell) if g.active else 0.0
        imp = new(buy * dt, 0.0, big if g.max_import_kw is None else g.max_import_kw)
        exp = new(-sell * dt, 0.0, big if g.max_export_kw is None else g.max_export_kw)
        imp_over = new((buy + B) * dt, 0.0, big if g.max_import_kw is not None else 0.0)
        exp_over = new((B - sell) * dt, 0.0, big if g.max_export_kw is not None else 0.0)
        pv_here = sum((fc.pv(k) for k in fc.pv_keys if site.bus_of(k) == b), np.zeros(n))
        curt = new(0.0, 0.0, pv_here if g.allow_curtailment else np.zeros(n))
        for t in range(n):                       # import - export - curtail = draw on the bus
            bal[b][t] += [(imp[t], 1.0), (exp[t], -1.0), (imp_over[t], 1.0), (exp_over[t], -1.0), (curt[t], -1.0)]
    Bn = breach_price(site.grid, fc.buy, fc.sell)
    for sm in site.submeters:
        up = sm.parent or MAIN
        pv_here = sum((fc.pv(k) for k in fc.pv_keys if site.bus_of(k) == sm.name), np.zeros(n))
        out_cap = np.inf if sm.max_export_kw is None else sm.max_export_kw / sm.eta_export
        in_cap = np.inf if sm.max_import_kw is None else sm.max_import_kw * sm.eta_import
        send = new(0.0, 0.0, min(out_cap, big))
        send_over = new(Bn * dt, 0.0, big if np.isfinite(out_cap) else 0.0)
        take = new(0.0, 0.0, min(in_cap, big))
        take_over = new(Bn * dt, 0.0, big if np.isfinite(in_cap) else 0.0)
        clip = new(0.0, 0.0, pv_here if allow else np.maximum(pv_here - out_cap, 0.0))
        for t in range(n):
            # the node: what it sends minus what it takes = its PV - clip - its draw
            bal[sm.name][t] += [(send[t], -1.0), (send_over[t], -1.0), (take[t], 1.0), (take_over[t], 1.0),
                                (clip[t], -1.0)]
            # its parent: receives eta x sent, gives taken / eta
            bal[up][t] += [(send[t], sm.eta_export), (send_over[t], sm.eta_export),
                           (take[t], -1.0 / sm.eta_import), (take_over[t], -1.0 / sm.eta_import)]
    sets = {lim.name: [[] for _ in range(n)] for lim in site.set_limits}
    for lim in site.set_limits:
        imp = new(0.0, 0.0, big if lim.max_import_kw is None else lim.max_import_kw)
        exp = new(0.0, 0.0, big if lim.max_export_kw is None else lim.max_export_kw)
        imp_over = new(Bn * dt, 0.0, big if lim.max_import_kw is not None else 0.0)
        exp_over = new(Bn * dt, 0.0, big if lim.max_export_kw is not None else 0.0)
        for t in range(n):
            sets[lim.name][t] += [(imp[t], 1.0), (exp[t], -1.0), (imp_over[t], 1.0), (exp_over[t], -1.0)]
    soe_rows = []
    for i, b in enumerate(site.battery_list):
        key = SiteConfig.battery_key(i)
        tp = terminal_price(b, fc.buy)
        ch = new(0.0, 0.0, b.p_charge_max_kw)
        dis = new(0.0, 0.0, b.p_discharge_max_kw)
        soe = new(0.0, b.soe_floor_kwh, b.capacity_kwh)
        var[soe[-1]] = (-tp, b.soe_floor_kwh, b.capacity_kwh)
        s0 = b.capacity_kwh * b.soc_initial_frac
        const += tp * s0
        where = site.bus_of(key)
        for t in range(n):
            bal[where][t] += [(ch[t], -1.0), (dis[t], 1.0)]       # minus its draw
            for lim in site.set_limits:
                if key in lim.members:
                    sets[lim.name][t] += [(ch[t], -1.0), (dis[t], 1.0)]
            r = [(soe[t], 1.0), (ch[t], -dt * b.eta_c), (dis[t], dt / b.eta_d)]
            if t:
                r.append((soe[t - 1], -1.0))
            soe_rows.append((r, s0 if t == 0 else 0.0))
    for b in roots + nodes:
        for t in range(n):
            rows.append(bal[b][t])
            rhs.append(float(bal_rhs[b][t]))
    for name in sets:
        for t in range(n):
            rows.append(sets[name][t])
            rhs.append(0.0)
    for r, v in soe_rows:
        rows.append(r)
        rhs.append(v)
    ri = [i for i, r in enumerate(rows) for _ in r]
    ci = [j for r in rows for j, _ in r]
    vv = [c for r in rows for _, c in r]
    A = sparse.csr_matrix((vv, (ri, ci)), shape=(len(rows), len(var)))
    cost = np.array([v[0] for v in var])
    res = linprog(cost, A_eq=A, b_eq=np.array(rhs), bounds=[(v[1], v[2]) for v in var], method="highs")
    assert res.status == 0, res.message
    return float(res.fun + const)
