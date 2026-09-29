"""The HiGHS MILP reference - "what a solver would give you today".

`bench/reference.py` measures the fast path against the exact optimum of its own
model. This measures it against a **mixed-integer solver**, which is the bar
EMHASS sets in Home Assistant (CVXPY defaulting to `cp.HIGHS`) and what evcc's
cloud optimizer uses. Same model, different algorithm.

Only expressible because comfort is priced linearly now; the POC's `w * degC^2`
penalty has no MILP form.

Small horizons throughout: the binary tree grows fast (24 steps ~0.4 s,
48 steps ~1.8 s), which is itself one of the findings.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="MILP reference needs scipy/HiGHS")

from bench.milp_full import milp_battery_water_heater  # noqa: E402
from home_energy_optimizer import (  # noqa: E402
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    demo_forecasts,
)
from home_energy_optimizer.coordinate import total_objective  # noqa: E402


@pytest.fixture(scope="module")
def small():
    h = Horizon(dt=0.25, hours=8.0)
    return h, BatteryConfig(capacity_kwh=10.0), WaterHeaterConfig()


def _solve(h, batt, wh, tariff):
    fc = demo_forecasts(h, tariff=tariff)
    return fc, milp_battery_water_heater(
        batt, wh, h, fc.buy, fc.sell, fc.hot_water_demand, fc.net_fixed_demand, time_limit=60
    )


def test_milp_solves_to_proven_optimality(small):
    h, batt, wh = small
    _, m = _solve(h, batt, wh, "day_night")
    assert m["mip_gap"] < 1e-3, "should close the gap on a problem this size"
    assert "successfully" in m["status"].lower()


def test_milp_respects_the_physical_bounds(small):
    h, batt, wh = small
    _, m = _solve(h, batt, wh, "day_night")
    assert m["soe"].min() >= -1e-6
    assert m["soe"].max() <= batt.capacity_kwh + 1e-6
    assert m["temp"].min() >= wh.t_min - 1e-6
    assert m["temp"].max() <= wh.t_max + 1e-6
    assert m["p_batt"].max() <= batt.p_charge_max_kw + 1e-6
    assert m["p_batt"].min() >= -batt.p_discharge_max_kw - 1e-6


def test_water_heater_stays_binary(small):
    """The whole reason this needs a MILP rather than an LP."""
    h, batt, wh = small
    _, m = _solve(h, batt, wh, "day_night")
    on = m["p_wh"] / wh.power_kw
    assert np.all((np.abs(on) < 1e-6) | (np.abs(on - 1) < 1e-6))


def test_milp_soe_recursion_is_self_consistent(small):
    """Guards the split-efficiency convention in the MILP, independently of
    the DP - if these two disagree the comparison is meaningless."""
    h, batt, wh = small
    _, m = _solve(h, batt, wh, "day_night")
    s = m["soe"][0]
    for t in range(h.steps):
        a = m["p_batt"][t]
        eff = a * batt.eta_c if a > 0 else a / batt.eta_d
        s = s + eff * h.dt
        assert s == pytest.approx(m["soe"][t + 1], abs=1e-6)


def test_milp_energy_balance(small):
    h, batt, wh = small
    fc, m = _solve(h, batt, wh, "day_night")
    net = m["p_batt"] + m["p_wh"] + fc.net_fixed_demand
    imp = np.maximum(net, 0.0)
    exp = np.maximum(-net, 0.0)
    bill = float(np.sum(imp * fc.buy * h.dt) - np.sum(exp * fc.sell * h.dt))
    assert bill == pytest.approx(m["bill"], abs=1e-6)


def test_milp_is_at_least_as_good_as_the_decomposition(small):
    """The MILP is exact for this model, so the heuristic cannot beat it.

    If it ever does, the two are not optimising the same objective - which is
    exactly the bug that made the first decomposition benchmark meaningless
    (docs/NOTES.md 1b).
    """
    h, batt, wh = small
    site = SiteConfig(
        horizon=h, battery=batt, water_heater=wh, hvac=None,
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    for tariff in ("flat", "day_night"):
        fc, m = _solve(h, batt, wh, tariff)
        res = coordinate(site, fc)
        assert res.total_objective >= m["objective"] - 1e-3, (
            f"{tariff}: heuristic {res.total_objective} beat the exact MILP {m['objective']}"
        )


def test_milp_rejects_a_tariff_that_needs_direction_binaries(small):
    """sell > buy would let the LP relaxation inflate import and export
    together; refuse rather than return a wrong number."""
    h, batt, wh = small
    fc = demo_forecasts(h, tariff="day_night")
    with pytest.raises(ValueError, match="sell <= buy"):
        milp_battery_water_heater(
            batt, wh, h, fc.buy, fc.buy * 2.0, fc.hot_water_demand, fc.net_fixed_demand
        )


def test_milp_requires_linear_comfort(small):
    """The POC's quadratic penalty has no MILP form; say so rather than
    silently optimising a different objective."""
    from dataclasses import replace

    h, batt, wh = small
    fc = demo_forecasts(h, tariff="day_night")
    with pytest.raises(ValueError, match="linear comfort"):
        milp_battery_water_heater(
            batt, replace(wh, comfort_mode="quadratic"), h,
            fc.buy, fc.sell, fc.hot_water_demand, fc.net_fixed_demand,
        )


def test_objective_matches_the_packages_own_accounting(small):
    """The MILP's reported objective must be computed the same way
    `coordinate.total_objective` computes it, or the comparison is apples to
    oranges."""
    h, batt, wh = small
    site = SiteConfig(horizon=h, battery=batt, water_heater=wh, hvac=None)
    fc, m = _solve(h, batt, wh, "day_night")

    net = m["p_batt"] + m["p_wh"] + fc.net_fixed_demand
    recomputed = total_objective(site, net, fc, m["temp"], None, m["soe"])
    assert recomputed == pytest.approx(m["objective"], abs=1e-4)
