"""How does the coordinated solve scale with the number of devices?

Run:  .venv/bin/python bench/run_scale.py

The claim the decomposition exists to support is that cost grows LINEARLY in
the device count, where the exact joint recursion grows as the product of the
state spaces. This measures it directly, from 1 to 20 storage units plus the
two thermal devices.

Two things are timed separately, because they scale differently and are used
at different cadences:

  plan    a full `coordinate()` - the slow tier, once per forecast
  infer   `fleet_action()` - the fast tier, once per state change

Reported per-round as well as in total, since the round count is adaptive and
a raw wall-clock total confounds "slower rounds" with "more rounds".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    HvacConfig,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    demo_forecasts,
)
from hemspolicy.policy import PolicySnapshot, fleet_action

COUNTS = (1, 2, 3, 5, 8, 12, 16, 20)


def site_with(n_batteries: int, thermal: bool, hours: float = 24.0) -> SiteConfig:
    """n storage units, sized so the fleet is not trivially over-provisioned."""
    h = Horizon(dt=0.25, hours=hours)
    unit = lambda: BatteryConfig(capacity_kwh=10.0, n_states=100, n_actions=41)
    return SiteConfig(
        horizon=h,
        battery=unit(),
        batteries=tuple(unit() for _ in range(n_batteries - 1)),
        water_heater=WaterHeaterConfig() if thermal else None,
        hvac=HvacConfig() if thermal else None,
        coordination=CoordinationConfig(),
    )


def time_plan(site: SiteConfig, fc, repeats: int = 1):
    """Best-of-N wall clock for a full coordinate(), rounds run, and the result.
    One run by default: a textbook ADMM plan takes seconds, long enough to
    time once."""
    best, rounds, res = float("inf"), 0, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = coordinate(site, fc)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
        rounds = res.rounds_run
    return best, rounds, res


SWEEPS = 4


def time_infer(site: SiteConfig, fc, res, repeats: int = 200) -> float:
    """Microseconds for one coordinated real-time decision across the fleet.

    WORST case: `tol=0` forces all SWEEPS to run. Left at the default the loop
    exits as soon as nothing is moving, which makes the column measure sweep
    count rather than fleet size and is useless for a latency budget.

    Caveat: every unit is handed the same snapshot, since PolicySnapshot only
    captures battery 0. That is fine for timing - the work per action() call is
    identical - but the REPLIES from this call are not physically meaningful,
    because the shared dp_load already contains the siblings' plans. Only the
    clock is being measured here.
    """
    snap = PolicySnapshot.from_result(site, fc, res)
    units = [(f"b{i}", snap, 5.0) for i in range(len(site.battery_list))]
    load = float(snap.dp_load[40]) + 4.0
    call = lambda: fleet_action(units, 40, measured_load_kw=load,
                                rounds=SWEEPS, tol=0.0)
    for _ in range(20):  # warm up
        call()
    t0 = time.perf_counter()
    for _ in range(repeats):
        call()
    return (time.perf_counter() - t0) * 1e6 / repeats


def collect(counts=COUNTS) -> list[dict]:
    """Run the sweep and return the rows, so figures can be built from it."""
    h24 = Horizon(dt=0.25, hours=24.0)
    fc = demo_forecasts(h24, tariff="dynamic")
    out = []
    for n in counts:
        site = site_with(n, thermal=True)
        ms, rounds, res = time_plan(site, fc)
        out.append(
            {
                "ders": n,
                "devices": len(site.battery_list) + 2,
                "plan_ms": ms,
                "rounds": rounds,
                "ms_per_round": ms / max(rounds, 1),
                "infer_us": time_infer(site, fc, res),
            }
        )
    return out


def main() -> None:
    h24 = Horizon(dt=0.25, hours=24.0)
    fc = demo_forecasts(h24, tariff="dynamic")

    print("hems-policy :: scale with device count")
    print(f"96 slots (24 h at 15 min), 100 SoE states x 41 actions per unit\n")
    print(f"{'DERs':>5}{'devices':>9}{'plan ms':>10}{'rounds':>8}"
          f"{'ms/round':>10}{'us/unit/rd':>12}{'infer us':>10}{'us/unit':>9}")
    print("-" * 73)

    rows = []
    for n in COUNTS:
        site = site_with(n, thermal=True)
        n_dev = len(site.battery_list) + 2
        ms, rounds, res = time_plan(site, fc)
        infer_us = time_infer(site, fc, res)
        per_round = ms / max(rounds, 1)
        rows.append((n, n_dev, ms, rounds, per_round, infer_us))
        print(f"{n:5d}{n_dev:9d}{ms:10.1f}{rounds:8d}{per_round:10.2f}"
              f"{per_round * 1000 / n_dev:12.1f}{infer_us:10.1f}{infer_us / n:9.1f}")

    # Linearity check: cost per device per round should be roughly flat. That
    # is the whole claim - if it grows, the decomposition is not buying what it
    # is supposed to buy.
    per_unit = [r[4] * 1000 / r[1] for r in rows]
    print(f"\nper-device-per-round cost: {min(per_unit):.1f}-{max(per_unit):.1f} us "
          f"(spread {max(per_unit) / min(per_unit):.2f}x across a {max(COUNTS)}x "
          f"range in device count)")

    first, last = rows[0], rows[-1]
    print(f"total plan time grew {last[2] / first[2]:.1f}x for "
          f"{last[1] / first[1]:.1f}x the devices")

    print("\nfor contrast, the exact joint recursion over the product state")
    print("space would need 100^n states at n units: 10^4 at 2, 10^6 at 3,")
    print("10^40 at 20 - which is the reason the decomposition exists.")


if __name__ == "__main__":
    main()
