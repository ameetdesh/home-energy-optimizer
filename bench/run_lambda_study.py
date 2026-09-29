"""Settle the lambda question: is dV/ds a real shadow price, and does it
survive decomposition?

Run:  .venv/bin/python bench/run_lambda_study.py

Two experiments, deliberately separate:

  A. DP dV/ds  vs  LP dual from HiGHS      - is it the right object at all?
  B. decomposed dV/ds vs joint DP dV/ds    - does coordination preserve it?

A is the ground-truth check: the LP's multiplier on the SoE recursion is the
discrete costate, computed to solver accuracy, and it obeys a known analytic
bound - lambda must sit inside [sell/eta, buy*eta], the no-arbitrage band for a
lossy store. If the DP's numerical gradient tracks that, dV/ds is the shadow
price and not merely something shaped like one.

B is the open question carried from optimsurvey_notes/poc-integration-analysis.md
section 8.4, and it is the one that matters for the proposal: the decomposed
battery solves with the other devices as EXOGENOUS load, so its dV/ds is a
marginal value conditional on their plans, not a full system dual.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from bench.duals import agreement, joint_costate, lp_battery_with_duals
from bench.reference import joint_dp_battery_water_heater
from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    PolicySnapshot,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    demo_forecasts,
    marginal_value,
)

TARIFFS = ("flat", "day_night", "dynamic")


def no_arbitrage_band(cfg: BatteryConfig, buy: np.ndarray, sell: np.ndarray) -> tuple[float, float]:
    """[sell/eta, buy*eta]: acquiring a stored kWh from forgone export costs
    sell/eta; delivering one to the house saves buy*eta. Any price outside that
    band would be an arbitrage the optimiser should already have taken."""
    return float(sell.min() / cfg.eta_c), float(buy.max() * cfg.eta_d)


def experiment_a() -> list[dict]:
    """DP dV/ds vs the LP costate, battery only, evaluated on the SAME states."""
    rows = []
    h = Horizon(dt=0.25, hours=24.0)
    cfg = BatteryConfig(capacity_kwh=10.0)

    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)
        dp_load = fc.net_fixed_demand

        lp = lp_battery_with_duals(cfg, h, fc.buy, fc.sell, dp_load)
        site = SiteConfig(
            horizon=h, battery=cfg, water_heater=None, hvac=None,
            coordination=CoordinationConfig(max_rounds=8),
        )
        snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))

        # Evaluate the DP price at the LP's own optimal states: a dual is a
        # local object, so comparing it anywhere else is not a like-for-like.
        lam_dp = np.array(
            [marginal_value(snap, t, float(lp["soe"][t])) for t in range(h.steps)]
        )
        lam_lp = lp["lambda"]

        lo, hi = no_arbitrage_band(cfg, fc.buy, fc.sell)
        stats = agreement(lam_dp, lam_lp)
        rows.append(
            {
                "tariff": tariff,
                "band": (lo, hi),
                "lp_in_band": float(np.mean((lam_lp >= lo - 1e-6) & (lam_lp <= hi + 1e-6))),
                "dp_in_band": float(np.mean((lam_dp >= lo - 1e-6) & (lam_dp <= hi + 1e-6))),
                **stats,
            }
        )
    return rows


def experiment_b() -> list[dict]:
    """Decomposed dV/ds vs the exact joint costate, battery + water heater."""
    rows = []
    h = Horizon(dt=0.25, hours=24.0)
    # Coarse grid: the joint reference is O(ns*nt*na) per step.
    batt = BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21)
    wh = WaterHeaterConfig()
    site = SiteConfig(
        horizon=h, battery=batt, water_heater=wh, hvac=None,
        coordination=CoordinationConfig(max_rounds=15),
    )

    for tariff in TARIFFS:
        fc = demo_forecasts(h, tariff=tariff)

        res = coordinate(site, fc)
        snap = PolicySnapshot.from_result(site, fc, res)
        joint = joint_dp_battery_water_heater(
            batt, wh, h, fc.buy, fc.sell, fc.hot_water_demand, fc.net_fixed_demand
        )

        # Both priced on the joint solution's own trajectory, so any difference
        # is the pricing model rather than a difference in where we stand.
        lam_joint = joint_costate(joint, h)
        lam_dec = np.array(
            [marginal_value(snap, t, float(joint["soe"][t])) for t in range(h.steps)]
        )

        lo, hi = no_arbitrage_band(batt, fc.buy, fc.sell)
        stats = agreement(lam_dec, lam_joint)
        rows.append(
            {
                "tariff": tariff,
                "band": (lo, hi),
                "joint_in_band": float(
                    np.mean((lam_joint >= lo - 1e-6) & (lam_joint <= hi + 1e-6))
                ),
                "dec_in_band": float(np.mean((lam_dec >= lo - 1e-6) & (lam_dec <= hi + 1e-6))),
                **stats,
            }
        )
    return rows


def report(title: str, rows: list[dict], a_name: str, b_name: str, band_keys: tuple[str, str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'tariff':<11}{'band':>16}{'mean ' + a_name:>13}{'mean ' + b_name:>13}"
        f"{'MAE':>9}{'max|d|':>9}{'corr':>7}{'MAE/sd':>8}"
    )
    for r in rows:
        lo, hi = r["band"]
        print(
            f"{r['tariff']:<11}{f'[{lo:.3f},{hi:.3f}]':>16}"
            f"{r['mean_a']:>13.4f}{r['mean_b']:>13.4f}"
            f"{r['mae']:>9.4f}{r['max_abs']:>9.4f}{r['corr']:>7.3f}{r['mae_over_sd']:>8.2f}"
        )
    ka, kb = band_keys
    print(
        "  inside the no-arbitrage band:  "
        + ",  ".join(
            f"{r['tariff']} {a_name} {100 * r[ka]:.0f}% / {b_name} {100 * r[kb]:.0f}%" for r in rows
        )
    )


def main() -> None:
    print("hems-policy :: lambda validation study")
    print("lambda is a marginal value of STORED energy, in currency/kWh.")
    print("Theory says it must lie in [sell/eta, buy*eta] - the no-arbitrage band for a lossy store.")

    a = experiment_a()
    report(
        "A. DP dV/ds vs LP dual (HiGHS), battery only, priced on the LP's states",
        a, "DP", "LP", ("dp_in_band", "lp_in_band"),
    )
    print("  -> if these agree, dV/ds IS the shadow price, not just something shaped like one.")

    b = experiment_b()
    report(
        "B. Decomposed dV/ds vs exact joint dV/ds, battery + water heater",
        b, "dec", "joint", ("dec_in_band", "joint_in_band"),
    )
    print("  -> this is the open question from poc-integration-analysis.md section 8.4.")

    worst_a = max(r["mae"] for r in a)
    worst_b = max(r["mae"] for r in b)
    tariff_scale = 0.30
    print(f"\nworst MAE vs LP dual:        {worst_a:.4f} /kWh  ({100 * worst_a / tariff_scale:.1f}% of a 0.30 tariff)")
    print(f"worst MAE vs joint costate:  {worst_b:.4f} /kWh  ({100 * worst_b / tariff_scale:.1f}% of a 0.30 tariff)")
    verdict = (
        "PASS - lambda is a real, decomposition-robust price"
        if max(worst_a, worst_b) < 0.02
        else "USABLE - directionally right, quantitatively loose"
        if max(worst_a, worst_b) < 0.05
        else "FAIL - lambda does not survive; do not ship it as a price"
    )
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
