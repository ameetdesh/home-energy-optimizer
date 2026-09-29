"""Stream lambda into a live Home Assistant, simulating a day as it goes.

    HA_URL=http://127.0.0.1:8123 HA_TOKEN=... \
        python tools/ha-lambda-demo/run.py --speed 60

This is Phase 5 step 1 of docs/PLAN.md: put the marginal value of a stored kWh
in front of a user, in the one place where an automation can already act on it.

The two-tier structure the whole package argues for is visible here:

* the **slow tier** re-plans on `--resolve-every` simulated minutes, the
  cadence forecasts actually change on - with Dantzig-Wolfe by default
  (`--method admm` for ADMM, each re-solve starting from the last one's state);
* the **fast tier** publishes on every simulated slot from the stored value
  function - a table lookup, no solver involved.

The battery is advanced by the policy's own action each slot, so the state
drifts the way a real one would and the published lambda tracks it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from home_energy_optimizer import (  # noqa: E402
    BatteryConfig,
    Horizon,
    HvacConfig,
    PolicySnapshot,
    SiteConfig,
    WaterHeaterConfig,
    action,
    demo_forecasts,
    plan,
)
from home_energy_optimizer.ha import HomeAssistant, publish_policy, publish_site  # noqa: E402


def build(hours: float, tariff: str, capacity: float, grid: int, thermal: bool) -> tuple:
    site = SiteConfig(
        horizon=Horizon(dt=0.25, hours=hours),
        battery=BatteryConfig(
            capacity_kwh=capacity, n_states=grid, n_actions=max(9, grid // 2 + 1)
        ),
        water_heater=WaterHeaterConfig() if thermal else None,
        hvac=HvacConfig() if thermal else None,
    )
    return site, demo_forecasts(site.horizon, tariff=tariff)


def describe(res) -> str:
    """One line on what planned, and how good the plan is known to be."""
    if res.gap is not None:
        return f"(dw: plan {res.plan_objective:.3f}, bound {res.lower_bound:.3f}, gap {res.gap:.3f})"
    return f"({res.method}{': ' + res.note if res.note else ''}; no certificate)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("HA_URL", "http://127.0.0.1:8123"))
    ap.add_argument("--token", default=os.environ.get("HA_TOKEN", ""))
    ap.add_argument("--tariff", default="day_night", choices=["flat", "day_night", "dynamic"])
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--capacity", type=float, default=10.0)
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument(
        "--speed", type=float, default=60.0,
        help="simulated minutes per real second (60 = a day in 24 s)",
    )
    ap.add_argument(
        "--resolve-every", type=float, default=180.0,
        help="simulated minutes between full re-solves (the slow tier)",
    )
    ap.add_argument("--prefix", default="hems")
    ap.add_argument(
        "--method", default="dw", choices=["dw", "admm"],
        help="coordinator for the slow tier (default: Dantzig-Wolfe)",
    )
    ap.add_argument(
        "--no-thermal", action="store_true",
        help="battery only; by default the water heater and HVAC are included",
    )
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("set HA_TOKEN (Profile -> Long-lived access tokens)")

    ha = HomeAssistant(base_url=args.url, token=args.token)
    if not ha.ping():
        raise SystemExit(f"cannot reach Home Assistant at {args.url}")

    site, fc = build(
        args.hours, args.tariff, args.capacity, args.grid, not args.no_thermal
    )
    dt_min = site.horizon.dt * 60.0
    steps = site.horizon.steps

    devices = ", ".join(
        n for n, d in (("battery", site.battery), ("tank", site.water_heater),
                       ("hvac", site.hvac)) if d is not None
    )
    print(f"solving {steps} slots ({args.hours} h, {args.tariff}; {devices}; {args.method})...")
    t0 = time.perf_counter()
    res = plan(site, fc, method=args.method)
    snap = PolicySnapshot.from_result(site, fc, res)
    print(f"  slow tier: {1000 * (time.perf_counter() - t0):.0f} ms  {describe(res)}")

    soe = site.battery.capacity_kwh * site.battery.soc_initial_frac
    since_solve = 0.0
    sleep_s = dt_min / max(args.speed, 1e-6)

    print(f"streaming to {args.url} as sensor.{args.prefix}_* "
          f"({dt_min:.0f} sim-min per {sleep_s:.2f} s)\n")
    print(
        f"{'sim time':>9}{'price':>8}{'lambda':>9}{'action':>9}{'soe':>8}"
        f"{'pv':>7}{'net':>8}  tier"
    )

    for t in range(steps):
        if since_solve >= args.resolve_every:
            # Slow tier: re-solve from the CURRENT state, as an MPC would.
            resolve = replace(
                site,
                battery=replace(
                    site.battery,
                    soc_initial_frac=float(
                        np.clip(soe / site.battery.capacity_kwh, 0.0, 1.0)
                    ),
                ),
            )
            # ADMM starts from where the last solve stood (the horizon has not
            # moved - the same day is replayed - so there is nothing to shift).
            res = plan(resolve, fc, method=args.method, warm=res.warm_start)
            snap = PolicySnapshot.from_result(resolve, fc, res)
            since_solve = 0.0
            tier = "SOLVE"
        else:
            tier = "lookup"

        published = publish_policy(ha, snap, t, soe, prefix=args.prefix)
        publish_site(ha, site, fc, res, t, prefix=args.prefix)

        hh = int(t * site.horizon.dt)
        mm = int((t * site.horizon.dt - hh) * 60)
        print(
            f"{hh:>7d}:{mm:02d}{published['import_price']:>8.3f}"
            f"{published['lambda']:>9.4f}{published['action_kw']:>9.2f}"
            f"{soe:>8.2f}{fc.solar[t]:>7.2f}{res.net_grid[t]:>8.2f}  {tier}"
        )

        # Advance the battery by the action the policy just chose.
        a = action(snap, t, soe)
        eff = a * site.battery.eta_c if a > 0 else a / site.battery.eta_d
        soe = float(np.clip(soe + eff * site.horizon.dt, 0.0, site.battery.capacity_kwh))
        since_solve += dt_min
        time.sleep(sleep_s)

    print("\ndone - the sensors keep their final values in Home Assistant")


if __name__ == "__main__":
    main()
