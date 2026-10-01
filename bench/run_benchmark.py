"""Phase 1 gating experiment: how much does the fast path give up?

Run:  .venv/bin/python bench/run_benchmark.py

Reports two gaps separately, because they have different causes and different
fixes (see bench/reference.py):

  discretisation gap  - DP grid vs continuous LP, battery only
  decomposition gap   - ADMM/price coordination vs exact joint DP

The decision rule stated in optimsurvey_notes/poc-integration-analysis.md: if
the total gap is within ~1-2% the speed and lambda arguments carry the
proposal on their own. If it is ~10%, the pitch changes to "fast execution
tier underneath a MILP planner" - still useful, different story.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from bench.reference import joint_dp_battery_water_heater, lp_battery
from home_energy_optimizer import (
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    demo_forecasts,
)
from home_energy_optimizer.coordinate import baseline_solution, net_cost, total_objective
from home_energy_optimizer.dp_thermal import baseline_water_heater
from home_energy_optimizer.dp_battery import solve_battery, terminal_price
from home_energy_optimizer.dw.coordinator import extended_objective

TARIFFS = ("flat", "day_night", "dynamic")


def capture(baseline: float, fast: float, exact: float) -> float:
    """Fraction of the AVAILABLE savings that the fast path actually captures.

    Percent-of-objective is useless here: the objective is a net bill that
    routinely crosses zero on a PV site (net export), so a ratio against it
    explodes. The stable, meaningful denominator is how much there was to win
    in the first place - baseline minus the exact optimum.
    """
    available = baseline - exact
    if abs(available) < 1e-9:
        return 1.0
    return (baseline - fast) / available


def bench_discretisation(hours: float = 24.0) -> list[dict]:
    """Battery-only DP vs the exact continuous LP, identical model."""
    rows = []
    h = Horizon(dt=0.25, hours=hours)
    cfg = BatteryConfig(capacity_kwh=10.0)

    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)
        # The LP split needs sell <= buy; the dynamic fixture already satisfies it.
        dp_load = fc.net_fixed_demand
        tp = terminal_price(cfg, fc.buy)

        t0 = time.perf_counter()
        dp = solve_battery(cfg, h, fc.buy, fc.sell, dp_load=dp_load)
        dp_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        lp = lp_battery(cfg, h, fc.buy, fc.sell, dp_load)
        lp_ms = (time.perf_counter() - t0) * 1000

        dp_net = dp.power + dp_load
        dp_bill = net_cost(dp_net, fc.buy, fc.sell, h.dt)
        dp_obj = dp_bill - tp * (dp.trajectory[-1] - dp.trajectory[0])
        baseline = net_cost(dp_load, fc.buy, fc.sell, h.dt)

        rows.append(
            {
                "tariff": tariff,
                "baseline": baseline,
                "dp_obj": dp_obj,
                "ref_obj": lp["objective"],
                "gap": dp_obj - lp["objective"],
                "capture": capture(baseline, dp_obj, lp["objective"]),
                "dp_ms": dp_ms,
                "ref_ms": lp_ms,
            }
        )
    return rows


def bench_decomposition(hours: float = 24.0) -> list[dict]:
    """Battery + water heater: ADMM coordination vs the exact joint DP."""
    rows = []
    h = Horizon(dt=0.25, hours=hours)
    # Coarse grid on BOTH sides here: the joint reference is O(n_s * n_t * n_a)
    # per step, so the shipped 200x81 default would make the exact solve take
    # minutes. The comparison stays apples-to-apples because the decomposed
    # solver uses the same grid.
    batt = BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21)
    wh = WaterHeaterConfig()
    site = SiteConfig(
        horizon=h,
        battery=batt,
        water_heater=wh,
        hvac=None,
        coordination=CoordinationConfig(),
    )

    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)

        t0 = time.perf_counter()
        res = coordinate(site, fc)
        dec_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        joint = joint_dp_battery_water_heater(
            batt, wh, h, fc.buy, fc.sell, fc.hot_water_demand, fc.net_fixed_demand
        )
        joint_ms = (time.perf_counter() - t0) * 1000

        # Compare on TOTAL OBJECTIVE. Since comfort became a real currency
        # price (docs/NOTES.md 3a) the objective is an honest money number, so
        # this is now the right basis - and it is necessary: on a flat tariff
        # the exact solver correctly spends MORE on comfort, so a bill-only
        # comparison would score the decomposition as "better than exact".
        base_net, _ = baseline_solution(site, fc)
        wh_temp, _ = baseline_water_heater(wh, h, fc.hot_water_demand)
        idle = np.full(h.steps + 1, batt.capacity_kwh * batt.soc_initial_frac)
        base_obj = total_objective(site, base_net, fc, wh_temp, None, idle)

        rows.append(
            {
                "tariff": tariff,
                "baseline": base_obj,
                "dp_obj": res.total_objective,
                "ref_obj": joint["objective"],
                "gap": res.total_objective - joint["objective"],
                "capture": capture(base_obj, res.total_objective, joint["objective"]),
                "dp_ms": dec_ms,
                "ref_ms": joint_ms,
            }
        )
    return rows


def bench_milp(hours: float = 24.0) -> list[dict]:
    """Battery + water heater: textbook ADMM on the shipped battery grid vs a
    MILP over the same model (bench/milp_full.py), so both the grids and the
    decomposition count against the fast path."""
    from bench.milp_full import milp_battery_water_heater

    rows = []
    h = Horizon(dt=0.25, hours=hours)
    batt = BatteryConfig(capacity_kwh=10.0)                 # the shipped grid
    wh = WaterHeaterConfig()
    site = SiteConfig(horizon=h, battery=batt, water_heater=wh, hvac=None,
                      coordination=CoordinationConfig())
    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)
        t0 = time.perf_counter()
        res = coordinate(site, fc)
        dec_ms = (time.perf_counter() - t0) * 1000
        m = milp_battery_water_heater(batt, wh, h, fc.buy, fc.sell, fc.hot_water_demand,
                                      fc.net_fixed_demand)
        # One scoring function for every plan, including the tank's end-of-day
        # shortfall that all of them optimise (see make_theory_figs.fig_coordination).
        ref = extended_objective(site, fc, m["p_batt"] + m["p_wh"] + fc.net_fixed_demand,
                                 {"battery": m["soe"], "water_heater": m["temp"]})
        base_net, _ = baseline_solution(site, fc)
        wh_temp, _ = baseline_water_heater(wh, h, fc.hot_water_demand)
        idle = np.full(h.steps + 1, batt.capacity_kwh * batt.soc_initial_frac)
        base_obj = extended_objective(site, fc, base_net, {"battery": idle, "water_heater": wh_temp})
        obj = extended_objective(site, fc, res.net_grid, {k: d.trajectory for k, d in res.devices.items()})
        rows.append({"tariff": tariff, "baseline": base_obj, "dp_obj": obj,
                     "ref_obj": ref, "gap": obj - ref,
                     "capture": capture(base_obj, obj, ref),
                     "available": base_obj - ref,
                     "dp_ms": dec_ms, "ref_ms": m["solve_ms"], "mip_gap": m["mip_gap"]})
    return rows


def report(title: str, rows: list[dict], note: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'tariff':<12}{'baseline':>10}{'fast':>10}{'exact':>10}"
        f"{'gap':>9}{'capture':>9}{'fast ms':>9}{'exact ms':>9}"
    )
    for r in rows:
        print(
            f"{r['tariff']:<12}{r['baseline']:>10.4f}{r['dp_obj']:>10.4f}{r['ref_obj']:>10.4f}"
            f"{r['gap']:>9.4f}{100 * r['capture']:>8.1f}%"
            f"{r['dp_ms']:>9.1f}{r['ref_ms']:>9.1f}"
        )
    print(f"  {note}")


def main() -> None:
    print("home-energy-optimizer :: Phase 1 gating experiment")
    print("Objective is a COST (lower is better); a positive gap means the fast path loses.")

    disc = bench_discretisation()
    report(
        "1. Discretisation gap  (battery-only DP vs exact continuous LP)",
        disc,
        "cause: finite state/action grids. tunable - see docs/NOTES.md 1a for the convergence table.",
    )

    dec = bench_decomposition()
    report(
        "2. Decomposition gap  (ADMM coordination vs exact joint DP)",
        dec,
        "cause: per-device 1-D DPs coordinated by price. at 1-2 devices, solve jointly instead.",
    )

    milp = bench_milp()
    report(
        "3. Against a MILP over the same model  (textbook ADMM, shipped battery grid)",
        milp,
        "both gaps at once: the grids and the decomposition. MILP relative gap "
        + ", ".join(f"{r['mip_gap']:.0e}" for r in milp) + ".",
    )

    worst_d = min(r["capture"] for r in disc)
    worst_c = min(r["capture"] for r in dec)
    print(f"\nworst savings capture, discretisation: {100 * worst_d:.1f}%")
    print(f"worst savings capture, decomposition:  {100 * worst_c:.1f}%")
    worst = min(worst_d, worst_c)
    verdict = (
        "PASS - speed and lambda carry the proposal"
        if worst > 0.98
        else "MARGINAL - pitch as a fast tier under an exact planner"
        if worst > 0.90
        else "FAIL - the heuristic costs too much to lead with"
    )
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
