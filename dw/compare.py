"""ADMM coordinator (hemspolicy.coordinate) vs the Dantzig-Wolfe prototype
(dw/coordinator.py), side by side on identical instances.

Run:  python3 dw/compare.py [--milp] [--only ABC] [--iters N]

Every method is scored on ONE number, `extended_objective` (total_objective
plus the thermal horizon-edge terms every DP already optimises), so the rows
are directly comparable. DW additionally reports its Lagrangian lower bound,
which bounds EVERY method's plan on the same instance - including ADMM's.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from dw.coordinator import baseline_objective as baseline_obj, dw_coordinate, extended_objective
from bench.reference import joint_dp_battery_water_heater
from hemspolicy import (
    BatteryConfig, CoordinationConfig, GridLimits, Horizon, HvacConfig,
    SiteConfig, WaterHeaterConfig, coordinate, demo_forecasts,
)
from hemspolicy.coordinate import apply_curtailment

TARIFFS = ("flat", "day_night", "dynamic")


def admm_row(site, fc):
    t0 = time.perf_counter()
    r = coordinate(site, fc)
    ms = (time.perf_counter() - t0) * 1000
    trajs = {k: s.trajectory for k, s in r.devices.items()}
    return extended_objective(site, fc, r.net_grid, trajs), ms, r.rounds_run, r


def fmt_row(name, val, lb, base, ms, extra=""):
    cap = (base - val) / (base - lb) if base - lb > 1e-9 else float("nan")
    return f"    {name:<22}{val:>10.4f}{cap*100:>9.1f}%{ms:>9.0f} ms  {extra}"


def header():
    print(f"    {'method':<22}{'objective':>10}{'vs LB':>10}{'time':>12}")


def scenario_small(milp: bool, max_iter: int):
    """Battery (50x21) + tank: the bench_decomposition fixture, with exact refs."""
    print("\n== A. battery (50x21) + water heater, 24 h - the fixture with exact references ==")
    h = Horizon(dt=0.25, hours=24.0)
    batt = BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21)
    wh = WaterHeaterConfig()
    site = SiteConfig(horizon=h, battery=batt, water_heater=wh, hvac=None,
                      coordination=CoordinationConfig())
    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)
        base = baseline_obj(site, fc)
        a_val, a_ms, a_rounds, _ = admm_row(site, fc)
        dwc = dw_coordinate(site, fc, max_iter=max_iter)
        dwh = dw_coordinate(site, fc, battery_in_master=True, max_iter=max_iter)
        lb = max(dwc.lower, dwh.lower)

        t0 = time.perf_counter()
        j = joint_dp_battery_water_heater(batt, wh, h, fc.buy, fc.sell, fc.hot_water_demand,
                                          fc.net_fixed_demand)
        j_ms = (time.perf_counter() - t0) * 1000
        j_net, _ = apply_curtailment(j["p_batt"] + j["p_wh"] + fc.net_fixed_demand,
                                     fc.solar, fc.sell, site.grid)
        j_val = extended_objective(site, fc, j_net, {"battery": j["soe"], "water_heater": j["temp"]})

        print(f"  tariff={tariff}   baseline {base:.4f}   DW lower bound {lb:.4f}")
        header()
        print(fmt_row("ADMM", a_val, lb, base, a_ms, f"{a_rounds} rounds"))
        print(fmt_row("DW columns", dwc.upper, lb, base, dwc.ms,
                      f"{dwc.iterations} it, {dwc.n_columns} cols, relaxed {dwc.relaxed:.4f}, own LB {dwc.lower:.4f}"))
        print(fmt_row("DW hybrid", dwh.upper, lb, base, dwh.ms,
                      f"{dwh.iterations} it, {dwh.n_columns} cols, relaxed {dwh.relaxed:.4f}, own LB {dwh.lower:.4f}, pricing err {dwh.pricing_error:.4f}"))
        print(fmt_row("joint DP", j_val, lb, base, j_ms))
        if milp:
            from bench.milp_full import milp_battery_water_heater
            m = milp_battery_water_heater(batt, wh, h, fc.buy, fc.sell, fc.hot_water_demand,
                                          fc.net_fixed_demand, time_limit=120.0)
            m_net, _ = apply_curtailment(m["p_batt"] + m["p_wh"] + fc.net_fixed_demand,
                                         fc.solar, fc.sell, site.grid)
            m_val = extended_objective(site, fc, m_net, {"battery": m["soe"], "water_heater": m["temp"]})
            print(fmt_row("MILP", m_val, lb, base, m["solve_ms"], m["status"][:40]))


def scenario_shipped(max_iter: int):
    """Shipped grids, all three device types: where the joint DP is unaffordable."""
    print("\n== B. battery (200x81) + water heater + HVAC, 24 h - shipped grids, no exact reference ==")
    h = Horizon(dt=0.25, hours=24.0)
    site = SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0),
                      water_heater=WaterHeaterConfig(), hvac=HvacConfig())
    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)
        base = baseline_obj(site, fc)
        a_val, a_ms, a_rounds, _ = admm_row(site, fc)
        dwh = dw_coordinate(site, fc, battery_in_master=True, max_iter=max_iter)
        lb = dwh.lower
        print(f"  tariff={tariff}   baseline {base:.4f}   DW lower bound {lb:.4f}")
        header()
        print(fmt_row("ADMM", a_val, lb, base, a_ms, f"{a_rounds} rounds"))
        print(fmt_row("DW hybrid", dwh.upper, lb, base, dwh.ms,
                      f"{dwh.iterations} it, {dwh.n_columns} cols, relaxed {dwh.relaxed:.4f}, pricing err {dwh.pricing_error:.4f}"))


def scenario_limits(max_iter: int):
    """Two batteries, tank, HVAC, 9 kW PV, a binding import limit."""
    print("\n== C. 2 batteries + water heater + HVAC, day/night, 9 kW PV, import limit ==")
    h = Horizon(dt=0.25, hours=24.0)
    for limit in (6.0, 4.0, 3.0):
        site = SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0),
                          batteries=(BatteryConfig(capacity_kwh=10.0),),
                          water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
                          grid=GridLimits(max_import_kw=limit))
        fc = demo_forecasts(h, tariff="day_night", solar_peak_kw=9.0)
        base = baseline_obj(site, fc)
        a_val, a_ms, a_rounds, r = admm_row(site, fc)
        dwh = dw_coordinate(site, fc, battery_in_master=True, max_iter=max_iter)
        lb = dwh.lower
        net = fc.net_fixed_demand + sum(c.power for c in dwh.plan.values())
        net, _ = apply_curtailment(net, fc.solar, fc.sell, site.grid)
        dw_peak = float(net.max())
        print(f"  import limit {limit} kW   baseline {base:.4f}   DW lower bound {lb:.4f}")
        header()
        print(fmt_row("ADMM", a_val, lb, base, a_ms,
                      f"{a_rounds} rounds, peak {r.net_grid.max():.2f} kW, excess {r.grid_import_excess:.3f}"))
        print(fmt_row("DW hybrid", dwh.upper, lb, base, dwh.ms,
                      f"{dwh.iterations} it, peak {dw_peak:.2f} kW, relaxed {dwh.relaxed:.4f}, pricing err {dwh.pricing_error:.4f}"))


def scenario_ablation(max_iter: int):
    """Which ingredients carry the result? Shipped grids, 3 devices."""
    print("\n== D. ablation - battery (200x81) + water heater + HVAC, 24 h ==")
    h = Horizon(dt=0.25, hours=24.0)
    site = SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0),
                      water_heater=WaterHeaterConfig(), hvac=HvacConfig())
    variants = [
        ("hybrid (full)", dict(battery_in_master=True)),
        ("  no heuristic cols", dict(battery_in_master=True, heuristic_columns=False)),
        ("  no smoothing", dict(battery_in_master=True, smoothing=0.0)),
        ("  max-weight recovery", dict(battery_in_master=True, integer="maxweight")),
        ("  no polish", dict(battery_in_master=True, polish=False)),
        ("  bare (none of above)", dict(battery_in_master=True, heuristic_columns=False,
                                        smoothing=0.0, integer="maxweight", polish=False)),
        ("columns (full)", dict(battery_in_master=False)),
        ("  bare", dict(battery_in_master=False, heuristic_columns=False, smoothing=0.0,
                        integer="maxweight", polish=False)),
    ]
    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)
        base = baseline_obj(site, fc)
        results = [(name, dw_coordinate(site, fc, max_iter=max_iter, **kw)) for name, kw in variants]
        lb = max(r.lower for _, r in results)
        a_val, a_ms, a_rounds, _ = admm_row(site, fc)
        print(f"  tariff={tariff}   baseline {base:.4f}   best lower bound {lb:.4f}")
        header()
        print(fmt_row("ADMM", a_val, lb, base, a_ms, f"{a_rounds} rounds"))
        for name, r in results:
            print(fmt_row(name[:22], r.upper, lb, base, r.ms,
                          f"{r.iterations} it, {r.n_columns} cols, own gap {r.upper - r.lower:.4f}, "
                          f"relaxed {r.relaxed:.4f}, frac devs {r.fractional_devices}"))


def scenario_variants(max_iter: int):
    """The DW variants side by side: where the tank lives, pool policy, ADMM seed."""
    from dataclasses import replace as _replace
    from dw.coordinator import DWCoordinator
    print("\n== E. DW variants - 2 batteries + tank + HVAC, 7 kW import limit, 24 h ==")
    h = Horizon(dt=0.25, hours=24.0)
    base = SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0),
                      batteries=(BatteryConfig(capacity_kwh=12.0),),
                      water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
                      grid=GridLimits(max_import_kw=7.0, max_export_kw=14.5))

    def tank_levels(L):
        wh = _replace(base.water_heater, n_duty_levels=L)
        return _replace(base, water_heater=_replace(wh, n_states=wh.states_for_duty_levels(h.dt)))

    variants = [
        ("on/off tank as DP", tank_levels(2), {}, {}),
        ("  + seeded from ADMM", tank_levels(2), {}, {"seed_admm": True}),
        ("  + used+newest pool", tank_levels(2), {}, {"pool": "active"}),
        ("twelfths tank as DP", tank_levels(13), {}, {}),
        ("tank in master (LP)", base, {"tank_in_master": True}, {}),
        ("  + used+newest pool", base, {"tank_in_master": True}, {"pool": "active"}),
    ]
    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)
        base_obj = baseline_obj(base, fc)
        a_val, a_ms, a_rounds, _ = admm_row(tank_levels(2), fc)
        rows = []
        for name, site, ck, rk in variants:
            t0 = time.perf_counter()
            r = DWCoordinator(site, fc, battery_in_master=True, **ck).run(max_iter=max_iter, **rk)
            rows.append((name, r, (time.perf_counter() - t0) * 1000))
        lb = min(r.lower for _, r, _ in rows)   # the continuous tank's bound is the loosest model's
        print(f"  tariff={tariff}   baseline {base_obj:.4f}   lowest bound {lb:.4f}")
        header()
        print(fmt_row("ADMM (on/off tank)", a_val, lb, base_obj, a_ms, f"{a_rounds} rounds"))
        for name, r, ms in rows:
            extra = f"{r.iterations} it, {r.n_columns} cols, gap {r.upper - r.lower:.4f}, stop {r.stop_reason}"
            if r.admm_value is not None:
                extra += f", certifies ADMM within {r.admm_value - r.lower:.4f}"
            print(fmt_row(name, r.upper, lb, base_obj, ms, extra))


def scenario_sensitivity(max_iter: int):
    """Device response: proposals only, + DP price sensitivity, sensitivity only."""
    from dataclasses import replace as _replace
    from dw.coordinator import DWCoordinator
    print("\n== F. device response - 2 batteries (in master) + tank + HVAC, 7 kW import limit, 24 h ==")
    h = Horizon(dt=0.25, hours=24.0)
    base = SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0),
                      batteries=(BatteryConfig(capacity_kwh=12.0),),
                      water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
                      grid=GridLimits(max_import_kw=7.0, max_export_kw=14.5))

    def tank_levels(L):
        wh = _replace(base.water_heater, n_duty_levels=L)
        return _replace(base, water_heater=_replace(wh, n_states=wh.states_for_duty_levels(h.dt)))

    for tariff in ("day_night", "dynamic"):
        fc = demo_forecasts(h, tariff=tariff)
        for L in (2, 13):
            site = tank_levels(L)
            base_obj = baseline_obj(site, fc)
            rows = []
            for resp in ("proposals", "both", "sensitivity"):
                t0 = time.perf_counter()
                co = DWCoordinator(site, fc, battery_in_master=True)
                r = co.run(max_iter=max_iter, response=resp, record=True)
                rows.append((resp, r, (time.perf_counter() - t0) * 1000))
            lb = min(r.lower for _, r, _ in rows)
            print(f"  tariff={tariff}, tank levels={L}   baseline {base_obj:.4f}   lowest bound {lb:.4f}")
            header()
            for resp, r, ms in rows:
                peak = max(hh["columns"] for hh in r.history)
                print(fmt_row(resp, r.upper, lb, base_obj, ms,
                              f"{r.iterations} it, peak pool {peak}, gap {r.upper - r.lower:.4f}, stop {r.stop_reason}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milp", action="store_true", help="also run the (slow) HiGHS MILP reference")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--only", default="ABCDEF")
    a = ap.parse_args()
    print("capture 'vs LB' = (baseline - plan) / (baseline - DW lower bound); 100% = provably optimal")
    if "A" in a.only:
        scenario_small(a.milp, a.iters)
    if "B" in a.only:
        scenario_shipped(a.iters)
    if "C" in a.only:
        scenario_limits(a.iters)
    if "D" in a.only:
        scenario_ablation(a.iters)
    if "E" in a.only:
        scenario_variants(a.iters)
    if "F" in a.only:
        scenario_sensitivity(a.iters)


if __name__ == "__main__":
    main()
