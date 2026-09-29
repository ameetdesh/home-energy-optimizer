"""Grid limits as a priced coupling constraint.

A grid limit binds the SUM of every device's power plus the inflexible load, so
no per-device DP can enforce it on its own - unlike a battery power rating,
which is local. The coordinator prices it instead: a per-timestep multiplier
that raises the effective import price until the plan fits. That multiplier is
a shadow price on the constraint, the same kind of object lambda is for stored
energy.

Dual ascent on a non-convex problem carries no guarantee, so these tests map
where it does and does not work rather than asserting it always does.
See docs/NOTES.md section 1 for the measured numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    Forecasts,
    GridLimits,
    Horizon,
    SiteConfig,
    coordinate,
)
from hemspolicy.profiles import day_night_tariff


def two_battery_site(limit: float | None = None, **grid_kw):
    """Two identical batteries and a big evening load, so both are needed."""
    h = Horizon(dt=0.25, hours=12.0)
    b = BatteryConfig(
        capacity_kwh=15.0, p_charge_max_kw=8.0, p_discharge_max_kw=8.0,
        soc_initial_frac=0.05, n_states=60, n_actions=25,
    )
    grid = GridLimits(max_import_kw=limit, **grid_kw)
    return SiteConfig(
        horizon=h, battery=b, batteries=(b,), water_heater=None, hvac=None,
        grid=grid, coordination=CoordinationConfig(exchange_rounds=20),
    )


def big_evening_load(h: Horizon, peak_kw: float = 10.0, rise: float = 0.0) -> Forecasts:
    """`rise`: the peak price climbs by this much per hour after 07:00, so the
    batteries discharge late and the house imports early - which makes the
    unconstrained optimum unique (with a flat peak price, flat and peaky
    import cost the same)."""
    n = h.steps
    buy, sell = day_night_tariff(h, peak=0.60, offpeak=0.10, sell=0.05)
    hours = h.times()
    buy = buy + rise * np.maximum(hours - 7.0, 0.0)
    return Forecasts(
        buy=buy, sell=sell,
        load=np.where(hours >= 7, peak_kw, 0.5),
        solar=np.zeros(n), outdoor_temp=np.full(n, 20.0),
        hot_water_demand=np.zeros(n),
    )


def physical_floor(site: SiteConfig, fc: Forecasts) -> float:
    """Lowest peak import any plan could achieve, from energy alone.

    The batteries can deliver `usable * eta_d` into the peak window; whatever is
    left must be imported, and the best case spreads it perfectly flat.
    """
    h = site.horizon
    hours = h.times()
    peak = hours >= 7
    peak_energy = float(fc.load[peak].sum() * h.dt)
    delivered = sum(
        (b.capacity_kwh - b.capacity_kwh * b.soc_initial_frac) * b.eta_d
        for b in site.battery_list
    )
    return (peak_energy - delivered) / float(peak.sum() * h.dt)


# --------------------------------------------------------------------------
# Two batteries, no limit
# --------------------------------------------------------------------------


def test_two_batteries_share_the_work_when_there_is_enough_load():
    """Identical batteries should be used symmetrically. Asymmetry would mean
    the sequential warm start is starving the later ones."""
    site = two_battery_site()
    fc = big_evening_load(site.horizon, peak_kw=10.0)
    res = coordinate(site, fc)

    tp = [float(np.abs(s.power).sum() * site.horizon.dt) for s in res.devices.values()]
    assert min(tp) > 0.5 * max(tp), f"lopsided use of identical batteries: {tp}"


# --------------------------------------------------------------------------
# The limit is actually enforced
# --------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [8.0, 6.0])
def test_binding_import_limits_are_met(limit):
    site = two_battery_site(limit)
    fc = big_evening_load(site.horizon, rise=0.01)
    res = coordinate(site, fc)

    assert res.grid_import_excess <= 1e-6, (
        f"limit {limit}: still {res.grid_import_excess:.3f} kW over"
    )
    assert res.net_grid.max() <= limit + 1e-6


def test_an_unconstrained_plan_would_breach_the_limit():
    """Guards the tests above from being vacuous."""
    site = two_battery_site()
    fc = big_evening_load(site.horizon, rise=0.01)
    assert coordinate(site, fc).net_grid.max() > 8.0 + 1.0


def test_cost_degrades_gracefully_as_the_limit_tightens():
    fc = big_evening_load(Horizon(dt=0.25, hours=12.0))
    unlimited = coordinate(two_battery_site(), fc)
    tight = coordinate(two_battery_site(8.0), fc)
    assert tight.net_cost >= unlimited.net_cost - 1e-6
    assert tight.net_cost < unlimited.net_cost * 1.15, "should not cost the earth"


def test_a_slack_limit_changes_nothing():
    fc = big_evening_load(Horizon(dt=0.25, hours=12.0))
    base = coordinate(two_battery_site(), fc)
    slack = coordinate(two_battery_site(100.0), fc)
    assert slack.net_cost == pytest.approx(base.net_cost, rel=0.02)
    assert slack.grid_import_excess == 0.0


# --------------------------------------------------------------------------
# Where it stops working, and why
# --------------------------------------------------------------------------


def test_failures_sit_at_or_below_the_physical_floor():
    """The honest boundary.

    Measured: converges at 6.0 kW, leaves a residual breach at 5.0 and below.
    The floor for this site is ~4.87 kW, so the failures are at the edge of what
    ANY method could achieve - the right answer there is "the fuse is too small",
    not "the solver is broken".
    """
    site = two_battery_site(5.0)
    fc = big_evening_load(site.horizon)
    floor = physical_floor(site, fc)

    assert 4.5 < floor < 5.5, f"floor moved to {floor:.2f}; revisit this test"
    res = coordinate(site, fc)
    # It fails, and that is expected this close to the floor...
    assert res.grid_import_excess > 0
    # ...but the failure is reported, never hidden.
    assert res.grid_import_excess == pytest.approx(
        max(0.0, res.net_grid.max() - 5.0), abs=1e-6
    )


def test_residual_breach_is_reported_not_swallowed():
    """A plan that cannot meet the limit must say so. Silently shipping one is
    the failure mode that matters for a fuse."""
    site = two_battery_site(2.0)
    fc = big_evening_load(site.horizon)
    res = coordinate(site, fc)
    assert res.grid_import_excess > 1.0
    assert res.net_grid.max() > 2.0


def _pv_site(pv_hours: float, export_limit: float | None):
    h = Horizon(dt=0.25, hours=12.0)
    n = h.steps
    hours = h.times()
    buy, sell = day_night_tariff(h, peak=0.40, offpeak=0.20, sell=0.15)
    fc = Forecasts(
        buy=buy, sell=sell, load=np.full(n, 0.3),
        solar=np.where((hours > 5) & (hours < 5 + pv_hours), 9.0, 0.0),
        outdoor_temp=np.full(n, 20.0), hot_water_demand=np.zeros(n),
    )
    b = BatteryConfig(capacity_kwh=20.0, p_charge_max_kw=6.0, p_discharge_max_kw=6.0,
                      soc_initial_frac=0.05, n_states=60, n_actions=25)
    site = SiteConfig(
        horizon=h, battery=b, water_heater=None, hvac=None,
        grid=GridLimits(max_export_kw=export_limit) if export_limit else GridLimits(),
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    return site, fc, b


def test_export_limit_is_enforced_while_the_battery_has_headroom():
    """The mirror case: PV spilling into a capped feed-in connection.

    Surplus is 8.7 kW and the battery absorbs 6 kW, so export sits at 2.7 kW -
    under a 5 kW cap - for as long as there is somewhere to put the energy.
    """
    site, fc, _ = _pv_site(pv_hours=2.0, export_limit=5.0)
    res = coordinate(site, fc)
    assert res.grid_export_excess <= 1e-6, f"{res.grid_export_excess:.3f} kW over"
    assert -res.net_grid.min() <= 5.0 + 1e-6


def test_curtailment_holds_the_export_cap_once_the_battery_saturates():
    """Without curtailment this was unenforceable, and that was a real gap.

    Over a sustained PV window (51.8 kWh of production against 19.0 kWh of
    storage) the battery fills, and every remaining watt of surplus MUST go to
    the grid - no price can prevent it, because nothing can respond. Throwing
    the PV away is the only lever left, and it reduces export 1:1.
    """
    site, fc, b = _pv_site(pv_hours=6.0, export_limit=5.0)
    res = coordinate(site, fc)

    soe = res.devices["battery"].trajectory
    assert soe.max() >= b.capacity_kwh * 0.99, "expected the battery to fill"
    assert res.grid_export_excess <= 1e-6, f"{res.grid_export_excess:.3f} kW over"
    assert -res.net_grid.min() <= 5.0 + 1e-6
    assert res.curtailed_kwh > 0, "the cap can only be met by curtailing here"


def test_without_curtailment_the_cap_is_unenforceable():
    """Kept so the reason curtailment exists stays testable rather than folklore."""
    h = Horizon(dt=0.25, hours=12.0)
    site, fc, _ = _pv_site(pv_hours=6.0, export_limit=5.0)
    site = SiteConfig(
        horizon=site.horizon, battery=site.battery, water_heater=None, hvac=None,
        grid=GridLimits(max_export_kw=5.0, allow_curtailment=False),
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    res = coordinate(site, fc)
    assert res.grid_export_excess > 0
    assert res.curtailed_kwh == 0.0


def test_curtailment_never_exceeds_available_pv():
    site, fc, _ = _pv_site(pv_hours=6.0, export_limit=5.0)
    res = coordinate(site, fc)
    assert res.curtailment is not None
    assert np.all(res.curtailment >= -1e-9)
    assert np.all(res.curtailment <= fc.solar + 1e-9)


def test_reported_excess_matches_the_actual_series():
    """These were computed from different arrays once - the fallback rebuilt
    net_grid without curtailing - so net_grid breached the cap while the
    reported excess read 0.000."""
    for allow in (False, True):
        site, fc, _ = _pv_site(pv_hours=6.0, export_limit=5.0)
        site = SiteConfig(
            horizon=site.horizon, battery=site.battery, water_heater=None, hvac=None,
            grid=GridLimits(max_export_kw=5.0, allow_curtailment=allow),
            coordination=CoordinationConfig(exchange_rounds=20),
        )
        res = coordinate(site, fc)
        assert res.grid_export_excess == pytest.approx(
            max(0.0, -res.net_grid.min() - 5.0), abs=1e-6
        )


def test_negative_export_price_triggers_curtailment():
    """Being paid to stop generating beats paying someone to take the energy."""
    h = Horizon(dt=0.25, hours=12.0)
    n = h.steps
    hours = h.times()
    buy, sell = day_night_tariff(h, peak=0.40, offpeak=0.20, sell=0.15)
    sell = np.where((hours > 8) & (hours < 10), -0.05, sell)
    fc = Forecasts(
        buy=buy, sell=sell, load=np.full(n, 0.3),
        solar=np.where((hours > 5) & (hours < 11), 9.0, 0.0),
        outdoor_temp=np.full(n, 20.0), hot_water_demand=np.zeros(n),
    )
    b = BatteryConfig(capacity_kwh=20.0, p_charge_max_kw=6.0, p_discharge_max_kw=6.0,
                      soc_initial_frac=0.05, n_states=60, n_actions=25)

    def run(allow):
        return coordinate(SiteConfig(
            horizon=h, battery=b, water_heater=None, hvac=None,
            grid=GridLimits(allow_curtailment=allow),
            coordination=CoordinationConfig(exchange_rounds=20)), fc)

    off, on = run(False), run(True)
    neg = sell < 0
    paid_off = float(np.sum(np.maximum(-off.net_grid[neg], 0) * sell[neg] * h.dt))
    paid_on = float(np.sum(np.maximum(-on.net_grid[neg], 0) * sell[neg] * h.dt))

    assert paid_off < 0, "should be paying to export without curtailment"
    assert paid_on == pytest.approx(0.0, abs=1e-9), "and not paying with it"
    assert on.curtailed_kwh > 0
    assert on.net_cost < off.net_cost, "curtailing must be cheaper here"


def test_no_curtailment_when_there_is_no_reason():
    """No export cap and a positive export price: never throw PV away."""
    site, fc, _ = _pv_site(pv_hours=6.0, export_limit=None)
    res = coordinate(site, fc)
    assert res.curtailed_kwh == 0.0


def test_a_slack_export_limit_is_met_trivially():
    site, fc, _ = _pv_site(pv_hours=6.0, export_limit=20.0)
    assert coordinate(site, fc).grid_export_excess == 0.0
