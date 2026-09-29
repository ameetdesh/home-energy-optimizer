"""Figures for docs/theory.tex appendix: what the suboptimality actually looks like.

Run:  .venv/bin/python bench/make_theory_figs.py

Produces, in docs/figs/:
  gap_discretisation.pdf  LP (continuous) vs DP (gridded), battery only
  gap_decomposition.pdf   exact joint DP vs ADMM + per-device DP
  gap_summary.pdf         captured fraction of available savings, both gaps
  scaling.pdf             solve cost vs device count, 1 to 20 DERs

The point of the first two is visual: the trajectories are hard to tell apart,
because the gap is a few percent of the SAVINGS, not of the bill. The third
plot says what that is numerically, so the reader is not left to eyeball it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.reference import joint_dp_battery_water_heater, lp_battery
from bench.run_benchmark import bench_milp, capture
from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    demo_forecasts,
)
from hemspolicy.coordinate import baseline_solution, net_cost, total_objective
from hemspolicy.dp_battery import solve_battery, terminal_price
from hemspolicy.dp_thermal import baseline_water_heater

FIGS = ROOT / "docs" / "figs"
TARIFFS = ("flat", "day_night", "dynamic")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "axes.titlesize": 8.5,
    }
)

LP_KW = dict(color="#333333", lw=2.2, alpha=0.55, label="LP (continuous optimum)")
DP_KW = dict(color="#c0392b", lw=1.0, label="DP (gridded, shipped)")
EX_KW = dict(color="#333333", lw=2.2, alpha=0.55, label="exact joint DP")
AD_KW = dict(color="#1f6fb4", lw=1.0, label="ADMM + per-device DP")


# --------------------------------------------------------------------------
# figure 1: discretisation gap (battery only, DP vs LP)
# --------------------------------------------------------------------------
def fig_discretisation(hours: float = 24.0) -> list[dict]:
    h = Horizon(dt=0.25, hours=hours)
    cfg = BatteryConfig(capacity_kwh=10.0)
    t = np.arange(h.steps) * h.dt
    ts = np.arange(h.steps + 1) * h.dt

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.5), sharex=True)
    rows = []

    for j, tariff in enumerate(TARIFFS):
        fc = demo_forecasts(h, tariff=tariff)
        dp = solve_battery(cfg, h, fc.buy, fc.sell, fc.net_fixed_demand)
        lp = lp_battery(cfg, h, fc.buy, fc.sell, fc.net_fixed_demand)

        soe0 = cfg.capacity_kwh * cfg.soc_initial_frac
        tp = terminal_price(cfg, fc.buy)
        base = net_cost(fc.net_fixed_demand, fc.buy, fc.sell, h.dt)
        dp_bill = net_cost(dp.power + fc.net_fixed_demand, fc.buy, fc.sell, h.dt)
        dp_obj = dp_bill - tp * (dp.trajectory[-1] - dp.trajectory[0])

        rows.append(
            {
                "tariff": tariff,
                "capture": capture(base, dp_obj, lp["objective"]),
                "gap": dp_obj - lp["objective"],
                "available": base - lp["objective"],
            }
        )

        ax = axes[0, j]
        ax.plot(ts, lp["soe"], **LP_KW)
        ax.plot(ts, dp.trajectory, **DP_KW)
        ax.set_title(f"{tariff}   capture {rows[-1]['capture']*100:.1f}%")
        if j == 0:
            ax.set_ylabel("state of energy\n[kWh]")

        ax = axes[1, j]
        ax.plot(t, lp["power"], **LP_KW)
        ax.plot(t, dp.power, **DP_KW)
        ax.set_xlabel("hour")
        ax.set_xticks(np.arange(0, hours + 1, 6))
        if j == 0:
            ax.set_ylabel("battery power\n[kW]  (+ charge)")

    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(pad=0.5)
    fig.savefig(FIGS / "gap_discretisation.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# figure 2: decomposition gap (battery + water heater, ADMM vs joint DP)
# --------------------------------------------------------------------------
def fig_decomposition(hours: float = 24.0) -> list[dict]:
    h = Horizon(dt=0.25, hours=hours)
    batt = BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21)
    wh = WaterHeaterConfig()
    site = SiteConfig(
        horizon=h,
        battery=batt,
        water_heater=wh,
        hvac=None,
        coordination=CoordinationConfig(algorithm="exchange"),
    )
    t = np.arange(h.steps) * h.dt
    ts = np.arange(h.steps + 1) * h.dt

    fig, axes = plt.subplots(3, 3, figsize=(7.4, 4.9), sharex=True)
    rows = []

    for j, tariff in enumerate(TARIFFS):
        fc = demo_forecasts(h, tariff=tariff)
        res = coordinate(site, fc)
        joint = joint_dp_battery_water_heater(
            batt, wh, h, fc.buy, fc.sell, fc.hot_water_demand, fc.net_fixed_demand
        )

        base_net, _ = baseline_solution(site, fc)
        wh_temp, _ = baseline_water_heater(wh, h, fc.hot_water_demand)
        idle = np.full(h.steps + 1, batt.capacity_kwh * batt.soc_initial_frac)
        base_obj = total_objective(site, base_net, fc, wh_temp, None, idle)

        rows.append(
            {
                "tariff": tariff,
                "capture": capture(base_obj, res.total_objective, joint["objective"]),
                "gap": res.total_objective - joint["objective"],
                "available": base_obj - joint["objective"],
            }
        )

        ax = axes[0, j]
        ax.plot(ts, joint["soe"], **EX_KW)
        ax.plot(ts, res.devices["battery"].trajectory, **AD_KW)
        ax.set_title(f"{tariff}   capture {rows[-1]['capture']*100:.1f}%")
        if j == 0:
            ax.set_ylabel("battery SoE\n[kWh]")

        ax = axes[1, j]
        ax.plot(ts, joint["temp"], **EX_KW)
        ax.plot(ts, res.devices["water_heater"].trajectory, **AD_KW)
        if j == 0:
            ax.set_ylabel("tank temp\n[$^\\circ$C]")

        ax = axes[2, j]
        ax.plot(t, joint["p_batt"] + joint["p_wh"] + fc.net_fixed_demand, **EX_KW)
        ax.plot(t, res.net_grid, **AD_KW)
        ax.axhline(0, color="#888", lw=0.5, ls=":")
        ax.set_xlabel("hour")
        ax.set_xticks(np.arange(0, hours + 1, 6))
        if j == 0:
            ax.set_ylabel("net grid\n[kW]  (+ import)")

    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(pad=0.5)
    fig.savefig(FIGS / "gap_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# figure 4: how the solve scales with the number of devices
# --------------------------------------------------------------------------
def fig_scaling() -> list[dict]:
    from bench.run_scale import collect

    rows = collect()
    n = np.array([r["ders"] for r in rows], dtype=float)
    plan = np.array([r["plan_ms"] for r in rows]) / 1000.0     # seconds
    infer = np.array([r["infer_us"] for r in rows])

    # Affine in the number of STORAGE units: the two thermal devices are fixed
    # overhead and belong in the intercept, not smeared across the units.
    b, a = np.polyfit(n, plan, 1)
    resid = plan - (a + b * n)
    r2 = 1.0 - resid.var() / plan.var()

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))

    ax = axes[0]
    ax.plot(n, plan, "o", color="#1f6fb4", ms=4, label="measured")
    ax.plot(n, a + b * n, "-", color="#888", lw=1.0,
            label=f"${a:.1f} + {b:.2f}\\,n$ s")
    ax.set_xlabel("storage units $n$")
    ax.set_ylabel("planning solve [s]")
    ax.set_title(f"plan: one per forecast ($R^2={r2:.4f}$)")
    ax.legend(loc="upper left")

    # The linearity claim, stated the only way that cannot flatter itself:
    # what does ONE more storage unit cost?
    ax = axes[1]
    marginal = np.diff(plan) / np.diff(n)
    ax.plot(n[1:], marginal, "o-", color="#c0392b", lw=1.2, ms=3.5)
    ax.axhline(b, color="#888", ls="--", lw=1.0, label=f"fit slope {b:.2f} s")
    ax.set_ylim(0, max(marginal) * 1.35)
    ax.set_xlabel("storage units $n$")
    ax.set_ylabel("marginal cost [s/unit]")
    ax.set_title("cost of one more unit")
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.plot(n, infer / n, "o-", color="#2e8b57", lw=1.2, ms=3.5)
    ax.set_ylim(0, max(infer / n) * 1.35)
    ax.set_xlabel("storage units $n$")
    ax.set_ylabel(r"[$\mu$s] per unit")
    ax.set_title("infer: one per state change")

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGS / "scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    for r, m in zip(rows[1:], marginal):
        r["marginal_ms"] = float(m) * 1000.0
    rows[0]["fit"] = (float(a), float(b), float(r2))
    return rows


# --------------------------------------------------------------------------
# figure 3: what the percentages mean in money
# --------------------------------------------------------------------------
def fig_summary(disc: list[dict], dec: list[dict], milp: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5))
    x = np.arange(len(TARIFFS))
    w = 0.27
    series = ((disc, -w, "#c0392b", "DP vs LP (discretisation)"),
              (dec, 0.0, "#1f6fb4", "ADMM+DP vs joint DP (decomposition)"),
              (milp, w, "#2e8b57", "ADMM+DP vs MILP (both)"))

    ax = axes[0]
    for rows, off, color, label in series:
        ax.bar(x + off, [r["capture"] * 100 for r in rows], w, color=color, label=label)
    ax.axhline(100, color="#333", lw=0.8, ls="--")
    ax.set_ylim(80, 104)
    ax.set_xticks(x)
    ax.set_xticklabels(TARIFFS)
    ax.set_ylabel("% of available savings\ncaptured")
    ax.legend(loc="lower left", fontsize=6.2)

    ax = axes[1]
    for i, (rows, off, color, _) in enumerate(series):
        ax.bar(x + off, [r["available"] for r in rows], w, color=color, alpha=0.30,
               label="available savings (baseline $-$ reference)" if i == 0 else None)
        ax.bar(x + off, [r["gap"] for r in rows], w, color=color,
               label="forgone by the fast path" if i == 0 else None)
    # The first bar of each group is the battery-only site; the other two are
    # battery + water heater (the decomposition's on the joint DP's coarse
    # grid, the MILP's on the shipped one), so the "available" totals differ.
    ax.axhline(0, color="#333", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(TARIFFS)
    ax.set_ylabel("currency units / day")
    ax.legend(loc="upper left", fontsize=6.2)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGS / "gap_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    disc = fig_discretisation()
    print("discretisation (DP vs LP):")
    for r in disc:
        print(f"  {r['tariff']:10s} capture {r['capture']*100:6.2f}%  "
              f"gap {r['gap']:+.4f}  of available {r['available']:.4f}")

    dec = fig_decomposition()
    print("decomposition (ADMM+DP vs joint DP):")
    for r in dec:
        print(f"  {r['tariff']:10s} capture {r['capture']*100:6.2f}%  "
              f"gap {r['gap']:+.4f}  of available {r['available']:.4f}")

    milp = bench_milp()
    print("against the MILP (ADMM+DP on the shipped grid vs exact):")
    for r in milp:
        print(f"  {r['tariff']:10s} capture {r['capture']*100:6.2f}%  "
              f"gap {r['gap']:+.4f}  of available {r['available']:.4f}  MILP {r['ref_ms']/1000:.1f} s")

    fig_summary(disc, dec, milp)

    rows = fig_scaling()
    print("scaling (plan / infer):")
    for r in rows:
        print(f"  {r['ders']:3d} units  plan {r['plan_ms']:7.1f} ms  "
              f"infer {r['infer_us']:7.1f} us  ({r['infer_us']/r['ders']:5.1f} us/unit)")

    print(f"\nwrote 4 figures to {FIGS} in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
