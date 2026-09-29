"""Stress test: the ADMM loop as implemented against textbook ADMM.

    .venv/bin/python bench/admm_variants.py            # full study
    .venv/bin/python bench/admm_variants.py --check    # harness == coordinate()?

docs/theory.tex ("Where the implementation departs from textbook ADMM") lists
four departures in hemspolicy.coordinate:

  1. the device step also prices the bill (others held at last round's plans);
  2. only storage is tethered - tank and HVAC best-respond, yet are counted in
     the target step;
  3. the two steps weigh the square differently: rho*dt/2 per slot in the
     device DP, rho/2 in the target step;
  4. nu is not rescaled when rho adapts;
  5. (found by this study) on the kink the target step takes g' = 0 and leaves
     the targets where the devices are heading. The minimiser instead puts the
     meter exactly at zero: all targets shift by (S + d)/m and g' = rz(S + d)/m.
     With g' = 0, every slot inside a band m*c/rho wide sends no price at all.

This harness re-implements the loop with each departure as a switch, so each
can be turned off alone and all together (= textbook scaled ADMM on the
sharing problem). With every switch at "as implemented" it reproduces
coordinate() round for round (--check). The library is not modified: the
tank and HVAC need a tether term for the textbook variants, so this file
carries copies of their DPs with one (python path).

Every plan is scored by DWCoordinator.parts - the basis the DW lower bound is
stated on - so "above bound" is a guaranteed optimality gap for any variant.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dw.coordinator import Column, DWCoordinator, _PriceVector  # noqa: E402
from hemspolicy import (  # noqa: E402
    BatteryConfig, CoordinationConfig, GridLimits, Horizon, HvacConfig, SiteConfig,
    WaterHeaterConfig, coordinate, demo_forecasts, plan,
)
from hemspolicy.coordinate import (  # noqa: E402
    apply_curtailment, breach_price, device_sell_price, total_objective,
)
from hemspolicy.dp_battery import solve_battery, terminal_price  # noqa: E402
from hemspolicy.dp_thermal import (  # noqa: E402
    HVAC_ACTIONS, _hvac_duty_cap, _hvac_heat_flow, _max_duty, _relaxation, _usable_outflow,
    baseline_hvac, baseline_water_heater, hvac_discomfort, rollout_hvac, rollout_water_heater,
    solve_hvac, solve_water_heater, wh_discomfort,
)
from hemspolicy.interp import interp_uniform  # noqa: E402


@dataclass(frozen=True)
class Flags:
    bill_in_device: bool = True       # 1: device step also prices the bill
    tether_thermal: bool = False      # 2: tank and HVAC get the square too
    thermal_in_zstep: bool = True     # 2: untethered tank/HVAC still counted in the target step
    consistent_rho: bool = False      # 3: target step weighs the square like the device step
    rescale_nu: bool = False          # 4: nu *= rho_old / rho_new when rho adapts
    exact_kink: bool = False          # 5: on the kink, shift targets so the meter sits at 0
    # grid-limit price mu (outside the ADMM steps):
    #   fixed  - as implemented: mu += step * excess, step = price_step * mean|buy|
    #   shrink - the same, with step / sqrt(round)
    #   rprop  - per slot: halve that slot's step when its excess changes sign,
    #            grow it by 1.2 while the sign holds
    mu_rule: str = "fixed"
    price_step: float | None = None   # None: cfg.grid.price_step
    mu_cap: bool = False              # never price a limit above the breach price
    relax: float = 1.0                # over-relaxation alpha (1 = none; ADMM's momentum analog)
    adaptive_rho: bool = True
    rho: float = 5.0


AS_IMPLEMENTED = Flags()


# ---------------------------------------------------------------------------
# Thermal DPs with a tether term: - (rho/2) dt (p - v_t)^2 in the stage reward.
# Copies of dp_thermal's python paths; `ref` pins the comfort price.
# ---------------------------------------------------------------------------

def wh_prox(cfg, h, buy, sell, demand, dp_load, target, rho, ref):
    n, dt, C = h.steps, h.dt, cfg.heat_capacity_kwh_per_k
    T = np.linspace(cfg.t_min, cfg.t_max, cfg.n_states)
    V = np.zeros((n + 1, cfg.n_states))
    POL = np.zeros((n, cfg.n_states), dtype=np.int64)
    V[n] = -C * ref * np.maximum(0.0, cfg.t_comfort - T)
    for t in range(n - 1, -1, -1):
        best = np.full(cfg.n_states, -np.inf)
        q_out = _usable_outflow(T, cfg, demand[t], dt)
        cap = _max_duty(T, cfg, q_out, dt)
        for ai, a in enumerate(cfg.duty_actions):
            q = cfg.power_kw * np.minimum(a, cap)
            Tn = T + (q - q_out) / C * dt
            idx, w = interp_uniform(V[t + 1], cfg.t_min, cfg.t_max, cfg.n_states, Tn)
            Vn = V[t + 1, idx] * (1 - w) + V[t + 1, np.minimum(idx + 1, cfg.n_states - 1)] * w
            imp = q + dp_load[t]
            Q = ((-buy[t] * np.maximum(imp, 0) + sell[t] * np.maximum(-imp, 0)) * dt
                 - wh_discomfort(cfg, Tn, ref, dt) + Vn
                 - (rho / 2) * dt * (q - target[t]) ** 2)
            better = Q > best
            best = np.where(better, Q, best)
            POL[t] = np.where(better, ai, POL[t])
        V[t] = best
    temp, on = rollout_water_heater(cfg, h, POL, demand, start_step=0, start_temp=cfg.t_comfort)
    return temp, on * cfg.power_kw


def hvac_prox(cfg, h, buy, sell, outdoor, dp_load, target, rho, ref):
    n, dt = h.steps, h.dt
    T = np.linspace(cfg.t_min, cfg.t_max, cfg.n_states)
    V = np.zeros((n + 1, cfg.n_states))
    POL = np.zeros((n, cfg.n_states), dtype=np.int64)
    V[n] = -hvac_discomfort(cfg, T, ref, 1.0)
    for t in range(n - 1, -1, -1):
        best = np.full(cfg.n_states, -np.inf)
        q_wall = (outdoor[t] - T) / cfg.r_wall_k_per_kw
        for ai, a in enumerate(HVAC_ACTIONS):
            duty = _hvac_duty_cap(T, cfg, q_wall, float(a), dt)
            Tn = T + (q_wall + _hvac_heat_flow(float(a), cfg) * duty) / cfg.c_room_kwh_per_k * dt
            idx, w = interp_uniform(V[t + 1], cfg.t_min, cfg.t_max, cfg.n_states, Tn)
            Vn = V[t + 1, idx] * (1 - w) + V[t + 1, np.minimum(idx + 1, cfg.n_states - 1)] * w
            p = cfg.power_kw * abs(a) * duty
            imp = p + dp_load[t]
            Q = ((-buy[t] * np.maximum(imp, 0) + sell[t] * np.maximum(-imp, 0)) * dt
                 - hvac_discomfort(cfg, Tn, ref, dt) + Vn
                 - (rho / 2) * dt * (p - target[t]) ** 2)
            better = Q > best
            best = np.where(better, Q, best)
            POL[t] = np.where(better, ai, POL[t])
        V[t] = best
    return rollout_hvac(cfg, h, POL, outdoor, start_step=0, start_temp=cfg.t_comfort_mid)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run(cfg: SiteConfig, fc, flags: Flags, rounds: int = 60):
    """Run `rounds` rounds without stopping early; return a per-round trace.

    Grid limits are priced by the same dual ascent as coordinate() (mu),
    outside the ADMM steps, for every variant alike.
    """
    h, n, dt, cc = cfg.horizon, cfg.horizon.steps, cfg.horizon.dt, cfg.coordination
    fixed = fc.net_fixed_demand.copy()
    ref = float(np.mean(fc.buy))
    sell_dev = device_sell_price(fc.sell, cfg.grid)
    batts = {SiteConfig.battery_key(i): b for i, b in enumerate(cfg.battery_list) if b.capacity_kwh > 0}
    thermal = [k for k, c in (("water_heater", cfg.water_heater), ("hvac", cfg.hvac)) if c is not None]
    keys = list(batts) + thermal
    tethered = list(batts) + (thermal if flags.tether_thermal else [])
    in_z = [k for k in keys if k in tethered or flags.thermal_in_zstep]
    m = max(len(in_z), 1)

    targets = {k: np.zeros(n) for k in keys}
    nu = {k: np.zeros(n) for k in keys}
    if cfg.water_heater is not None:
        targets["water_heater"] = baseline_water_heater(cfg.water_heater, h, fc.hot_water_demand)[1].copy()
    if cfg.hvac is not None:
        targets["hvac"] = baseline_hvac(cfg.hvac, h, fc.outdoor_temp)[1].copy()
    seed = fixed + sum(targets[k] for k in thermal) if thermal else fixed.copy()
    for k, b in batts.items():
        targets[k] = solve_battery(b, h, fc.buy, sell_dev, dp_load=seed, admm_rho=0.0,
                                   soc_gates=cfg.soc_gates if k == "battery" else ()).power.copy()
        seed = seed + targets[k]
    prev = {k: targets[k].copy() for k in keys}
    trajs = {}

    # textbook device step: private cost only - zero energy price, terminal
    # and comfort prices pinned to the tariff's
    zero = np.zeros(n)
    pinned = {k: replace(b, terminal_price=terminal_price(b, fc.buy)) for k, b in batts.items()}
    mu_imp, mu_exp = np.zeros(n), np.zeros(n)
    step0 = (flags.price_step if flags.price_step is not None else cfg.grid.price_step) \
        * max(float(np.mean(np.abs(fc.buy))), 1e-6)
    step_imp, step_exp = np.full(n, step0), np.full(n, step0)
    last_imp, last_exp = np.zeros(n), np.zeros(n)
    cap = breach_price(cfg.grid, fc.buy, fc.sell) if cfg.grid.active else np.inf
    best = (np.inf, None, None)
    co = DWCoordinator(cfg, fc)

    rho = flags.rho
    trace = []
    for r in range(rounds):
        t0 = time.perf_counter()
        buy = fc.buy + mu_imp
        sell = sell_dev - mu_exp

        def others(me):
            return fixed + sum(prev[k] for k in keys if k != me)

        powers = {}
        for k, b in batts.items():
            v = targets[k] - nu[k]
            if flags.bill_in_device:
                sol = solve_battery(b, h, buy, sell, dp_load=others(k), admm_target=v, admm_rho=rho,
                                    soc_gates=cfg.soc_gates if k == "battery" else ())
            else:
                sol = solve_battery(pinned[k], h, zero, zero, dp_load=zero, admm_target=v, admm_rho=rho,
                                    soc_gates=cfg.soc_gates if k == "battery" else ())
            powers[k], trajs[k] = sol.power, sol.trajectory
        for k in thermal:
            tethered_k = k in tethered
            v = targets[k] - nu[k]
            if not tethered_k and flags.bill_in_device:
                # exactly what coordinate() calls
                if k == "water_heater":
                    sol = solve_water_heater(cfg.water_heater, h, buy, sell, fc.hot_water_demand, dp_load=others(k))
                else:
                    sol = solve_hvac(cfg.hvac, h, buy, sell, fc.outdoor_temp, dp_load=others(k))
                trajs[k], powers[k] = sol.trajectory, sol.power
                continue
            if flags.bill_in_device:
                bb, ss, load = _PriceVector(buy, ref), _PriceVector(sell, ref), others(k)
            else:
                bb, ss, load = _PriceVector(zero, ref), _PriceVector(zero, ref), zero
            rr = rho if tethered_k else 0.0
            if k == "water_heater":
                trajs[k], powers[k] = wh_prox(cfg.water_heater, h, bb, ss, fc.hot_water_demand, load, v, rr, ref)
            else:
                trajs[k], powers[k] = hvac_prox(cfg.hvac, h, bb, ss, fc.outdoor_temp, load, v, rr, ref)

        # target step, per slot; the square weighs rho/2 (as implemented) or
        # rho*dt/2 (consistent with the device DP)
        rz = rho * dt if flags.consistent_rho else rho
        old = {k: targets[k].copy() for k in keys}
        outside = fixed + sum(powers[k] for k in keys if k not in in_z)
        # over-relaxation: the target step and nu see alpha*p + (1-alpha)*old target
        hat = {k: flags.relax * powers[k] + (1 - flags.relax) * old[k] for k in keys}
        for t in range(n):
            s = {k: hat[k][t] + nu[k][t] for k in in_z}
            S = sum(s.values())
            if S - m * fc.buy[t] * dt / rz + outside[t] > 0:
                g = fc.buy[t] * dt
            elif S + m * fc.sell[t] * dt / rz + outside[t] < 0:
                g = -fc.sell[t] * dt
            elif flags.exact_kink:
                # the minimiser puts the meter exactly on the kink: every
                # target moves by the same (S + d)/m, and the price g' lies
                # inside [-c_exp dt, c_imp dt] - it is not zero
                g = rz * (S + outside[t]) / m
            else:
                g = 0.0                                   # as implemented
            for k in in_z:
                targets[k][t] = s[k] - g / rz
        for k in in_z:
            nu[k] = nu[k] + hat[k] - targets[k]

        net, _ = apply_curtailment(fixed + sum(powers.values()), fc.solar, fc.sell, cfg.grid)
        if cfg.grid.active:
            def update(mu, stp, last, excess):
                if flags.mu_rule == "shrink":
                    stp = np.full(n, step0 / np.sqrt(r + 1))
                elif flags.mu_rule == "rprop":
                    flip = np.sign(excess) * np.sign(last) < 0
                    stp = np.where(flip, stp * 0.5, stp * 1.2)
                mu = np.maximum(0.0, mu + stp * excess)
                if flags.mu_cap:
                    mu = np.minimum(mu, cap)
                return mu, stp, excess
            if cfg.grid.max_import_kw is not None:
                mu_imp, step_imp, last_imp = update(mu_imp, step_imp, last_imp, net - cfg.grid.max_import_kw)
            if cfg.grid.max_export_kw is not None:
                mu_exp, step_exp, last_exp = update(mu_exp, step_exp, last_exp, -net - cfg.grid.max_export_kw)
        primal = sum(float(np.linalg.norm(powers[k] - targets[k])) for k in in_z)
        dual = sum(float(rho * np.linalg.norm(targets[k] - old[k])) for k in in_z)
        soe = {k: trajs[k] for k in batts} or None
        obj_lib = total_objective(cfg, net, fc, trajs.get("water_heater"), trajs.get("hvac"), soe)
        obj = co.parts({k: Column(powers[k], trajs[k], 0.0, "x") for k in keys})["total"]
        if obj < best[0]:
            best = (obj, {k: powers[k].copy() for k in keys}, {k: trajs[k].copy() for k in keys})
        breach_kwh = float(np.sum(np.maximum(net - cfg.grid.max_import_kw, 0))) * dt if cfg.grid.max_import_kw else 0.0
        trace.append({"obj": obj, "obj_lib": obj_lib, "primal": primal, "dual": dual, "rho": rho,
                      "breach_kwh": breach_kwh,
                      "ms": (time.perf_counter() - t0) * 1000})
        prev = {k: powers[k].copy() for k in keys}

        if flags.adaptive_rho:
            new = rho
            if primal > cc.rho_adapt_ratio * max(dual, 1e-6):
                new = rho * cc.rho_adapt_factor
            elif dual > cc.rho_adapt_ratio * max(primal, 1e-6):
                new = max(rho / cc.rho_adapt_factor, cc.rho_min)
            if flags.rescale_nu and new != rho:
                for k in keys:
                    nu[k] = nu[k] * rho / new
            rho = new
    run.last = {"powers": powers, "trajs": trajs, "nu": nu, "targets": targets, "rho": rho,
                "mu_imp": mu_imp, "mu_exp": mu_exp}
    run.best = best
    return trace


def polish(cfg: SiteConfig, fc, powers: dict, trajs: dict, mu_imp=None, mu_exp=None, sweeps: int = 3):
    """Gauss-Seidel best response from a finished plan, kept only if it helps.

    Each device in turn re-plans against the others' CURRENT plans, at the
    tariff (plus the final grid-limit prices, if any); the change is kept only
    if the full objective falls, so this can never make the plan worse.
    Returns (objective, sweeps used, powers, trajs).
    """
    h, n = cfg.horizon, cfg.horizon.steps
    co = DWCoordinator(cfg, fc)
    buy = fc.buy + (mu_imp if mu_imp is not None else 0.0)
    sell = device_sell_price(fc.sell, cfg.grid) - (mu_exp if mu_exp is not None else 0.0)
    batts = {SiteConfig.battery_key(i): b for i, b in enumerate(cfg.battery_list) if b.capacity_kwh > 0}
    score = lambda P, T: co.parts({k: Column(P[k], T[k], 0.0, "x") for k in P})["total"]  # noqa: E731
    powers, trajs = dict(powers), dict(trajs)
    cur = score(powers, trajs)
    used = 0
    for used in range(1, sweeps + 1):
        improved = False
        for k in list(powers):
            others = fc.net_fixed_demand + sum(powers[j] for j in powers if j != k)
            if k in batts:
                sol = solve_battery(batts[k], h, buy, sell, dp_load=others,
                                    soc_gates=cfg.soc_gates if k == "battery" else ())
            elif k == "water_heater":
                sol = solve_water_heater(cfg.water_heater, h, buy, sell, fc.hot_water_demand, dp_load=others)
            else:
                sol = solve_hvac(cfg.hvac, h, buy, sell, fc.outdoor_temp, dp_load=others)
            P2, T2 = dict(powers), dict(trajs)
            P2[k], T2[k] = sol.power, sol.trajectory
            v = score(P2, T2)
            if v < cur - 1e-9:
                powers, trajs, cur, improved = P2, T2, v, True
        if not improved:
            break
    return cur, used, powers, trajs


def stop_round(trace, cc, limited: bool = False) -> int:
    """Where coordinate()'s stopping rule would stop (1-based), on this trace:
    residuals small or objective settled - and, with a limit, only once the
    plan is within it. coordinate() caps a limited run at 40 rounds."""
    prev = np.inf
    cap = min(len(trace), 40) if limited else len(trace)
    for i, tr in enumerate(trace[:cap]):
        feasible = not limited or tr["breach_kwh"] <= 1e-6 * 0.25
        if feasible and i > 0 and tr["primal"] < cc.residual_tol and tr["dual"] < cc.residual_tol:
            return i + 1
        if feasible and i > 0 and abs(prev - tr["obj_lib"]) / max(abs(prev), 1e-6) < cc.converge_tol:
            return i + 1
        prev = tr["obj_lib"]
    return cap


# ---------------------------------------------------------------------------

def site(limit: float | None = None) -> SiteConfig:
    return SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0, p_charge_max_kw=5.0, p_discharge_max_kw=5.0,
                              n_states=100, n_actions=51),
        batteries=(BatteryConfig(capacity_kwh=6.0, p_charge_max_kw=3.0, p_discharge_max_kw=3.0,
                                 n_states=100, n_actions=51),),
        water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
        grid=GridLimits(max_import_kw=limit) if limit else GridLimits(),
    )


TEXTBOOK = Flags(bill_in_device=False, tether_thermal=True, consistent_rho=True,
                 rescale_nu=True, exact_kink=True)

VARIANTS = {
    "as implemented": AS_IMPLEMENTED,
    "fix 5 (exact kink)": replace(AS_IMPLEMENTED, exact_kink=True),
    "fix 3+4+5": replace(AS_IMPLEMENTED, consistent_rho=True, rescale_nu=True, exact_kink=True),
    "fix 3 (consistent rho)": replace(AS_IMPLEMENTED, consistent_rho=True),
    "fix 4 (rescale nu)": replace(AS_IMPLEMENTED, rescale_nu=True),
    "fix 3+4": replace(AS_IMPLEMENTED, consistent_rho=True, rescale_nu=True),
    "2: thermal out of target step": replace(AS_IMPLEMENTED, thermal_in_zstep=False),
    "2: tether thermal too": replace(AS_IMPLEMENTED, tether_thermal=True),
    "textbook, adaptive rho": TEXTBOOK,
    "textbook, kink as implemented": replace(TEXTBOOK, exact_kink=False),
    "textbook, fixed rho=0.03": replace(TEXTBOOK, adaptive_rho=False, rho=0.03),
    "textbook, fixed rho=0.3": replace(TEXTBOOK, adaptive_rho=False, rho=0.3),
    "textbook, fixed rho=1": replace(TEXTBOOK, adaptive_rho=False, rho=1.0),
    "textbook, fixed rho=5": replace(TEXTBOOK, adaptive_rho=False, rho=5.0),
    "textbook, fixed rho=20": replace(TEXTBOOK, adaptive_rho=False, rho=20.0),
}


MU_RULES = {
    "as implemented (fixed step 1)": AS_IMPLEMENTED,
    "fixed step 0.5": replace(AS_IMPLEMENTED, price_step=0.5),
    "fixed step 2": replace(AS_IMPLEMENTED, price_step=2.0),
    "fixed step 4": replace(AS_IMPLEMENTED, price_step=4.0),
    "shrinking step": replace(AS_IMPLEMENTED, mu_rule="shrink", price_step=2.0),
    "per-slot adaptive (rprop)": replace(AS_IMPLEMENTED, mu_rule="rprop"),
    "per-slot adaptive + cap": replace(AS_IMPLEMENTED, mu_rule="rprop", mu_cap=True),
}


def study_limits(rounds: int = 40) -> None:
    """Item 3 (grid-limit price rules) and item 1 (polish) on limit cases,
    plus polish alone on the no-limit cases."""
    cases = [(t, lim) for lim in (None, 3.0, 4.0, 5.0) for t in ("flat", "day_night", "dynamic")]
    rows: dict[str, list] = {}
    for tariff, lim in cases:
        cfg = site(lim)
        fc = demo_forecasts(cfg.horizon, tariff=tariff, solar_peak_kw=6.0)
        lb = plan(cfg, fc, fallback=False).lower_bound
        label = f"{tariff}, {'no limit' if lim is None else f'<= {lim:g} kW'}"
        print(f"\n== {label}: DW bound {lb:.4f}")
        print(f"{'rule':32s} {'@stop':>7s} {'stop':>5s} {'best40':>7s} {'breach':>7s} {'+polish':>8s} {'polish s':>8s}")
        for name, fl in (MU_RULES.items() if lim else [("as implemented", AS_IMPLEMENTED)]):
            tr = run(cfg, fc, fl, rounds=rounds)
            st = stop_round(tr, cfg.coordination, cfg.grid.active)
            b = np.minimum.accumulate([t["obj"] for t in tr])
            # polish the plan coordinate() would return: the best round up to the stop
            objs = [t["obj"] for t in tr[:st]]
            # re-run to the stop to get that plan (run keeps the overall best only)
            tr2 = run(cfg, fc, fl, rounds=st)
            _, P, T = run.best
            t0 = time.perf_counter()
            pol, _, _, Tp = polish(cfg, fc, P, T, run.last["mu_imp"], run.last["mu_exp"])
            ps = time.perf_counter() - t0
            breach = min(tr[:st], key=lambda t: t["obj"])["breach_kwh"]
            print(f"{name:32s} {min(objs) - lb:7.3f} {st:5d} {b[-1] - lb:7.3f} {breach:7.3f} {pol - lb:8.3f} {ps:8.2f}")
            rows.setdefault(name, []).append((label, min(objs) - lb, st, pol - lb))
    print("\nsummary: mean distance above the bound at the stop, without / with polish")
    for name, r in rows.items():
        print(f"  {name:32s} {np.mean([x[1] for x in r]):.3f} / {np.mean([x[3] for x in r]):.3f}"
              f"   over {len(r)} cases, mean stop round {np.mean([x[2] for x in r]):.1f}")


def check() -> None:
    """The harness with every switch 'as implemented' must be coordinate()."""
    cfg = replace(site(), coordination=CoordinationConfig(max_rounds=15, enable_baseline_fallback=False, polish=False))
    fc = demo_forecasts(cfg.horizon, tariff="day_night", solar_peak_kw=6.0)
    lib = coordinate(cfg, fc).round_objectives
    mine = [t["obj_lib"] for t in run(cfg, fc, AS_IMPLEMENTED, rounds=len(lib))]
    worst = max(abs(a - b) for a, b in zip(lib, mine))
    print(f"coordinate(): {len(lib)} rounds; worst per-round objective difference {worst:.2e}")
    assert worst < 1e-9, "harness does not reproduce coordinate()"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limits", action="store_true", help="grid-limit price rules + polish (items 1, 3)")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--limit", type=float, default=None, help="import limit, kW")
    ap.add_argument("--tariffs", default="flat,day_night,dynamic")
    args = ap.parse_args()
    if args.check:
        check()
        return
    if args.limits:
        study_limits()
        return
    for tariff in args.tariffs.split(","):
        cfg = site(args.limit)
        fc = demo_forecasts(cfg.horizon, tariff=tariff, solar_peak_kw=6.0)
        dw = plan(cfg, fc, fallback=False)
        lb = dw.lower_bound
        print(f"\n== {tariff}{'' if not args.limit else f', import <= {args.limit} kW'}: "
              f"DW plan {dw.plan_objective:.4f}, lower bound {lb:.4f}")
        print(f"{'variant':32s} {'best':>8s} {'above':>7s} {'@15':>7s} {'@60':>7s} "
              f"{'rounds to':>9s} {'stops@':>6s} {'above@stop':>10s} {'ms/rnd':>7s}")
        for name, fl in VARIANTS.items():
            tr = run(cfg, fc, fl, rounds=args.rounds)
            objs = np.array([t["obj"] for t in tr])
            best = np.minimum.accumulate(objs)
            final = best[-1]
            # first round whose best-so-far is within 0.05 of the variant's own best
            within = int(np.argmax(best <= final + 0.05)) + 1
            st = stop_round(tr, cfg.coordination, cfg.grid.active)
            print(f"{name:32s} {final:8.4f} {final - lb:7.4f} {best[min(14, len(best) - 1)] - lb:7.4f} "
                  f"{best[-1] - lb:7.4f} {within:9d} {st:6d} {best[st - 1] - lb:10.4f} "
                  f"{np.mean([t['ms'] for t in tr]):7.0f}")


if __name__ == "__main__":
    main()
