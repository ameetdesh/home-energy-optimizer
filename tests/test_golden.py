"""Golden-file tests: pin the solver's output so refactors are detectable.

The behavioural tests elsewhere assert economics ("charges when cheap"). These
assert *this exact solver* still produces *these exact numbers*, which is what
makes the Cython/WASM port in Phase 3 safe: the compiled path must reproduce
the reference path to tolerance, and any drift shows up here first.

Regenerate deliberately with:  python -m tests.test_golden --update
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    HvacConfig,
    PolicySnapshot,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    demo_forecasts,
    marginal_value,
)

GOLDEN = pathlib.Path(__file__).parent / "golden.json"
TOL = 1e-6


def scenarios() -> dict[str, tuple[SiteConfig, str]]:
    h = Horizon(dt=0.25, hours=24.0)
    full = SiteConfig(
        horizon=h,
        battery=BatteryConfig(capacity_kwh=10.0),
        water_heater=WaterHeaterConfig(),
        hvac=HvacConfig(),
        coordination=CoordinationConfig(max_rounds=8),
    )
    return {
        "battery_only_day_night": (full.without("water_heater", "hvac"), "day_night"),
        "battery_only_dynamic": (full.without("water_heater", "hvac"), "dynamic"),
        "all_devices_dynamic": (full, "dynamic"),
        "no_battery_day_night": (full.without("battery"), "day_night"),
    }


def summarise(cfg: SiteConfig, tariff: str) -> dict:
    fc = demo_forecasts(cfg.horizon, tariff=tariff)
    res = coordinate(cfg, fc)

    out = {
        "net_cost": round(res.net_cost, 9),
        "import_cost": round(res.import_cost, 9),
        "export_revenue": round(res.export_revenue, 9),
        "total_objective": round(res.total_objective, 9),
        "baseline_cost": round(res.baseline_cost, 9),
        "rounds_run": res.rounds_run,
        "net_grid_sum": round(float(res.net_grid.sum()), 9),
        "devices": sorted(res.devices),
    }
    for name, sol in sorted(res.devices.items()):
        out[f"{name}_energy"] = round(float(sol.power.sum() * cfg.horizon.dt), 9)
        out[f"{name}_throughput"] = round(float(np.abs(sol.power).sum() * cfg.horizon.dt), 9)
        out[f"{name}_traj_end"] = round(float(sol.trajectory[-1]), 9)
    if "battery" in res.devices:
        snap = PolicySnapshot.from_result(cfg, fc, res)
        out["lambda_at_12"] = round(marginal_value(snap, 12, 5.0), 9)
        out["lambda_at_76"] = round(marginal_value(snap, 76, 5.0), 9)
    return out


def generate() -> dict:
    return {name: summarise(cfg, tariff) for name, (cfg, tariff) in scenarios().items()}


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden.json not generated yet")
@pytest.mark.parametrize("name", list(scenarios()))
def test_matches_golden(name):
    expected = json.loads(GOLDEN.read_text())[name]
    cfg, tariff = scenarios()[name]
    actual = summarise(cfg, tariff)

    assert set(actual) == set(expected), f"key set changed for {name}"
    for key, exp in expected.items():
        act = actual[key]
        if isinstance(exp, (int, str, list)):
            assert act == exp, f"{name}.{key}: {act!r} != {exp!r}"
        else:
            assert act == pytest.approx(exp, abs=TOL), f"{name}.{key}: {act} != {exp}"


def test_solver_is_deterministic():
    """Two identical runs must agree exactly. No RNG, no dict-order leakage."""
    cfg, tariff = scenarios()["all_devices_dynamic"]
    assert summarise(cfg, tariff) == summarise(cfg, tariff)


if __name__ == "__main__":
    GOLDEN.write_text(json.dumps(generate(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")
