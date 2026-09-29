"""The Dantzig-Wolfe prototype (dw/): bound ordering, and parity with ADMM.

The DW master runs on dw/lpsolver.py (numpy only), so no scipy is needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dw.coordinator import DWCoordinator, dw_coordinate, extended_objective  # noqa: E402
from hemspolicy import (  # noqa: E402
    BatteryConfig, GridLimits, Horizon, HvacConfig, SiteConfig, WaterHeaterConfig,
    coordinate, demo_forecasts,
)


def small_site(**kw):
    h = Horizon(dt=0.25, hours=24.0)
    return SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21),
                      water_heater=WaterHeaterConfig(), hvac=None, **kw)


@pytest.mark.parametrize("tariff", ["flat", "day_night", "dynamic"])
def test_bounds_are_ordered(tariff):
    site = small_site()
    fc = demo_forecasts(site.horizon, tariff=tariff)
    r = dw_coordinate(site, fc, battery_in_master=True)
    # lower bound <= convexified optimum <= implementable plan
    assert r.lower <= r.relaxed + 1e-6
    assert r.relaxed <= r.upper + 1e-6


@pytest.mark.parametrize("tariff", ["flat", "day_night", "dynamic"])
def test_not_worse_than_admm(tariff):
    site = small_site()
    fc = demo_forecasts(site.horizon, tariff=tariff)
    a = coordinate(site, fc)
    a_val = extended_objective(site, fc, a.net_grid, {k: s.trajectory for k, s in a.devices.items()})
    r = dw_coordinate(site, fc, battery_in_master=True)
    assert r.upper <= a_val + 1e-9


def test_import_limit_is_met_exactly():
    h = Horizon(dt=0.25, hours=24.0)
    site = SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21),
                      water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
                      grid=GridLimits(max_import_kw=3.0))
    fc = demo_forecasts(h, tariff="day_night", solar_peak_kw=9.0)
    co = DWCoordinator(site, fc, battery_in_master=True)
    r = co.run()
    net = fc.net_fixed_demand + sum(c.power for c in r.plan.values())
    assert net.max() <= 3.0 + 1e-6


def test_master_price_lies_in_the_tariff_band_without_limits():
    """With no grid limit the meter's marginal price must sit between the
    export and import tariffs - it is the kink's subgradient."""
    site = small_site()
    fc = demo_forecasts(site.horizon, tariff="dynamic")
    r = dw_coordinate(site, fc, battery_in_master=True)
    assert np.all(r.prices >= np.minimum(fc.sell, 0.0) - 1e-9)
    assert np.all(r.prices <= fc.buy + 1e-9)


# ---------------------------------------------------------------- variants

def limit_site():
    h = Horizon(dt=0.25, hours=24.0)
    return SiteConfig(horizon=h, battery=BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21),
                      water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
                      grid=GridLimits(max_import_kw=4.0))


def test_tank_in_master_follows_the_tank_physics():
    """The LP tank must be the DP's tank with continuous duty: re-running its
    duty through the physics reproduces its temperature exactly."""
    site = limit_site()
    fc = demo_forecasts(site.horizon, tariff="day_night", solar_peak_kw=9.0)
    co = DWCoordinator(site, fc, battery_in_master=True, tank_in_master=True)
    r = co.run()
    tank = r.plan["water_heater"]
    from hemspolicy.dp_thermal import _max_duty, _usable_outflow
    cfg, dt = site.water_heater, site.horizon.dt
    T = [cfg.t_comfort]
    for t in range(site.horizon.steps):
        q = float(_usable_outflow(T[-1], cfg, fc.hot_water_demand[t], dt))
        duty = min(tank.power[t] / cfg.power_kw, float(_max_duty(T[-1], cfg, q, dt)))
        T.append(T[-1] + (cfg.power_kw * duty - q) / cfg.heat_capacity_kwh_per_k * dt)
    assert np.max(np.abs(np.array(T) - tank.trajectory)) < 1e-6
    assert r.lower <= r.relaxed + 1e-6


def test_tank_in_master_is_at_least_as_good_as_any_duty_grid():
    """Continuous duty contains every grid, so its bound sits below theirs."""
    site = limit_site()
    fc = demo_forecasts(site.horizon, tariff="dynamic")
    lp = DWCoordinator(site, fc, battery_in_master=True, tank_in_master=True).run()
    grid = DWCoordinator(site, fc, battery_in_master=True).run()
    assert lp.lower <= grid.lower + 1e-6
    assert lp.upper <= grid.upper + 1e-6


def test_admm_seed_is_never_worse_than_admm():
    site = limit_site()
    fc = demo_forecasts(site.horizon, tariff="day_night", solar_peak_kw=9.0)
    r = DWCoordinator(site, fc, battery_in_master=True).run(seed_admm=True)
    assert r.admm_value is not None
    assert r.upper <= r.admm_value + 1e-9


def test_active_pool_stays_small_and_ordered():
    site = small_site()
    fc = demo_forecasts(site.horizon, tariff="day_night")
    full = DWCoordinator(site, fc, battery_in_master=True).run(pool="full")
    act = DWCoordinator(site, fc, battery_in_master=True).run(pool="active")
    assert act.n_columns <= full.n_columns
    assert act.lower <= act.relaxed + 1e-6 <= act.upper + 2e-6


# ------------------------------------------------------- DP price sensitivity

def test_sensitivity_variants_follow_the_tank_physics():
    """Every "force one step, then re-plan" variant is a plan the tank can run:
    re-simulating its power reproduces its temperature."""
    from dw.coordinator import Device
    from dw.sensitivity import variants
    from hemspolicy.dp_thermal import _max_duty, _usable_outflow
    site = limit_site()
    fc = demo_forecasts(site.horizon, tariff="day_night")
    co = DWCoordinator(site, fc, battery_in_master=True)
    dev = next(d for d in co.devices if d.kind == "water_heater")
    _, sol = co.price_solve(dev, fc.buy)
    found, _ = variants(co, dev, sol)
    assert len(found) > 50
    cfg, dt = site.water_heater, site.horizon.dt
    for t0, _d, pw, tr in found[::17]:
        T = [cfg.t_comfort]
        for t in range(site.horizon.steps):
            q = float(_usable_outflow(T[-1], cfg, fc.hot_water_demand[t], dt))
            duty = min(pw[t] / cfg.power_kw, float(_max_duty(T[-1], cfg, q, dt)))
            T.append(T[-1] + (cfg.power_kw * duty - q) / cfg.heat_capacity_kwh_per_k * dt)
        assert np.max(np.abs(np.array(T) - tr)) < 1e-6


@pytest.mark.parametrize("response", ["both", "sensitivity"])
def test_sensitivity_modes_keep_the_bound_honest(response):
    site = limit_site()
    fc = demo_forecasts(site.horizon, tariff="day_night", solar_peak_kw=9.0)
    r = DWCoordinator(site, fc, battery_in_master=True).run(response=response, anytime=True)
    assert r.lower <= r.relaxed + 1e-6
    # the plan may undercut the bound only by DP discretisation, not by a blend
    # an on/off device cannot run
    assert r.upper >= r.lower - 5e-3
    for k, c in r.plan.items():
        assert c.source != "aggregate" or k == "water_heater" and site.water_heater.n_duty_levels > 2


def test_sensitivity_only_bounds_memory():
    site = limit_site()
    fc = demo_forecasts(site.horizon, tariff="day_night")
    both = DWCoordinator(site, fc, battery_in_master=True).run(response="both", record=True, pool="full")
    sens = DWCoordinator(site, fc, battery_in_master=True).run(response="sensitivity", record=True)
    assert max(h["columns"] for h in sens.history) < max(h["columns"] for h in both.history)


# ---------------------------------------------------------------------------
# A battery that bids plans runs its blend (it can modulate); an EV with a
# charger minimum picks one plan, unless it may duty-cycle within a slot.
# ---------------------------------------------------------------------------

def _two_batteries_as_plans():
    from dw.webapi import build_site
    site = build_site({"n_batteries": 2, "max_import_kw": 7, "terminal_mode": "linear", "grid": 60})
    return site, demo_forecasts(site.horizon, tariff="dynamic", solar_peak_kw=5.0)


def test_a_battery_bidding_plans_runs_its_blend():
    """Forcing one proposed plan per battery returned 14.18 here - a plan far
    over the import limit - where the LP's blend was -2.17."""
    site, fc = _two_batteries_as_plans()
    r = DWCoordinator(site, fc, battery_in_master=False, tank_in_master=True).run(max_iter=40)
    assert r.plan["battery"].source == "blend"
    assert r.plan_parts["breach"] < 1e-6
    assert r.upper <= r.relaxed + 0.1


def test_a_battery_blend_is_run_through_its_physics():
    site, fc = _two_batteries_as_plans()
    co = DWCoordinator(site, fc, battery_in_master=False)
    co.seed()
    dev = next(d for d in co.devices if d.kind == "battery")
    w = np.full(len(dev.columns), 1.0 / len(dev.columns))
    col = co.blend_plan(dev, w)
    cap = dev.cfg.capacity_kwh
    assert col.trajectory.min() >= -1e-9 and col.trajectory.max() <= cap + 1e-9
    mixed = sum(wi * c.power for wi, c in zip(w, dev.columns))
    # the meter sees the blend, except where it would overfill or empty the store
    moved = np.abs(col.power - mixed) > 1e-9
    at_edge = (col.trajectory[1:][moved] >= cap - 1e-9) | (col.trajectory[1:][moved] <= 1e-9)
    assert at_edge.all()


def test_an_ev_with_a_charger_minimum_picks_one_plan_unless_it_may_duty_cycle():
    ev = BatteryConfig(capacity_kwh=60.0, p_discharge_max_kw=1e-9, charge_deadband_kw=4.14)
    site = SiteConfig(horizon=Horizon(dt=0.25, hours=24.0), battery=BatteryConfig(capacity_kwh=10.0),
                      batteries=(ev,), water_heater=None, hvac=None)
    fc = demo_forecasts(site.horizon)
    ev_dev = lambda co: next(d for d in co.devices if d.key == SiteConfig.battery_key(1))  # noqa: E731
    assert not ev_dev(DWCoordinator(site, fc)).modulating
    assert ev_dev(DWCoordinator(site, fc, ev_duty_cycle=True)).modulating
