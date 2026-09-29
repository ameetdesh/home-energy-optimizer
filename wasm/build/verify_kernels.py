#!/usr/bin/env python3
"""Assert the compiled kernels reproduce the Python reference exactly.

Not "close to": identical. A DP stores an ARGMAX, so a value that differs in
the last bit can flip a stored action, and that action is what the execution
tier replays for the next fifteen minutes. Bitwise on the value function is
the only check worth running.

Run after building the extension in place:

    cd wasm/build
    python3 -m venv .venv && ./.venv/bin/pip install setuptools cython numpy
    CFLAGS=-ffp-contract=off ./.venv/bin/python setup.py build_ext --inplace
    ./.venv/bin/python verify_kernels.py

The flag matters natively: arm64 and x86 compilers fuse a*b + c into one
rounding, which numpy does not do and wasm32 cannot, so a native build without
it differs from the reference in the last bit.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from home_energy_optimizer import (  # noqa: E402
    BatteryConfig, Horizon, HvacConfig, WaterHeaterConfig,
    solve_battery, solve_hvac, solve_water_heater,
)
from home_energy_optimizer import _kernels, dp_battery, dp_thermal  # noqa: E402
from home_energy_optimizer.profiles import (  # noqa: E402
    day_night_tariff, dynamic_tariff, flat_tariff,
    hot_water_demand_profile, outdoor_temp_profile,
)

if not _kernels.HAVE_KERNELS:
    sys.exit("dp_kernels not importable; build it first (see the docstring)")


def without_kernels():
    """Force the Python path by making the dispatch report failure - in the
    modules that call it, which hold their own reference to each function."""
    saved = (dp_battery.kernel_battery, dp_thermal.kernel_water_heater, dp_thermal.kernel_hvac)
    dp_battery.kernel_battery = lambda *a: False
    dp_thermal.kernel_water_heater = lambda *a: False
    dp_thermal.kernel_hvac = lambda *a: False
    return saved


def restore(saved):
    (dp_battery.kernel_battery, dp_thermal.kernel_water_heater, dp_thermal.kernel_hvac) = saved


def compare(name, ref, got) -> bool:
    ok = True
    for field in ("value", "policy", "trajectory", "power"):
        a, b = getattr(ref, field), getattr(got, field)
        if a.shape != b.shape:
            print(f"  {name}.{field}: SHAPE {a.shape} vs {b.shape}")
            ok = False
            continue
        if np.array_equal(a, b):
            continue
        d = np.abs(a.astype(float) - b.astype(float))
        print(f"  {name}.{field}: DIFFERS  max |d| = {d.max():.3e} "
              f"at {int((d == d.max()).sum())} of {d.size} entries")
        ok = False
    print(f"  {name}: {'identical' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    h = Horizon(dt=0.25, hours=24.0)
    tariffs = {"flat": flat_tariff(h), "day_night": day_night_tariff(h),
               "dynamic": dynamic_tariff(h)}
    demand = hot_water_demand_profile(h)
    outdoor = outdoor_temp_profile(h)
    n = h.steps
    rng = np.random.default_rng(0)
    load = rng.normal(0.6, 0.9, n)

    ok = True
    for label, (buy, sell) in tariffs.items():
        print(f"[{label}]")
        for rho, target in ((0.0, None), (5.0, rng.normal(0.0, 1.5, n))):
            tag = f"battery(rho={rho})"
            args = dict(dp_load=load, admm_target=target, admm_rho=rho)
            saved = without_kernels()
            ref = solve_battery(BatteryConfig(n_states=100, n_actions=51), h, buy, sell, **args)
            restore(saved)
            got = solve_battery(BatteryConfig(n_states=100, n_actions=51), h, buy, sell, **args)
            ok &= compare(tag, ref, got)

        for mult in (1.0, 8.0):
            saved = without_kernels()
            ref = solve_water_heater(WaterHeaterConfig(), h, buy, sell, demand * mult, load)
            restore(saved)
            got = solve_water_heater(WaterHeaterConfig(), h, buy, sell, demand * mult, load)
            ok &= compare(f"water_heater(x{mult:g})", ref, got)

        for mean in (24.0, 34.0):
            od = outdoor_temp_profile(h, mean_c=mean)
            saved = without_kernels()
            ref = solve_hvac(HvacConfig(), h, buy, sell, od, load)
            restore(saved)
            got = solve_hvac(HvacConfig(), h, buy, sell, od, load)
            ok &= compare(f"hvac(mean={mean:g})", ref, got)

        # A comfort band that changes hour by hour (as dragged in the DW app)
        # reaches the kernel one run of equal-band slots at a time.
        pts = np.arange(n + 1) * h.dt
        hours = np.arange(25)
        banded = HvacConfig(
            comfort_low_profile=tuple(np.interp(pts, hours, 21.0 + 2.0 * (hours % 3 == 0))),
            comfort_high_profile=tuple(np.interp(pts, hours, 25.0 + 2.0 * (hours >= 12))))
        od = outdoor_temp_profile(h, mean_c=28.0)
        saved = without_kernels()
        ref = solve_hvac(banded, h, buy, sell, od, load)
        restore(saved)
        got = solve_hvac(banded, h, buy, sell, od, load)
        ok &= compare("hvac(hourly band)", ref, got)

    print("\nALL IDENTICAL" if ok else "\nMISMATCHES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
