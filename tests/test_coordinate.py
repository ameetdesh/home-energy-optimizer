"""Coordination loop behaviour.

No optimality assertions here - the ADMM/price heuristic has no certificate,
which is precisely why Phase 1 (benchmark against EMHASS's MILP) exists. What
IS asserted: it terminates, it never returns something worse than doing
nothing, and the accounting is internally consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from home_energy_optimizer import (
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    HvacConfig,
    SiteConfig,
    SocGate,
    WaterHeaterConfig,
    baseline_hvac,
    baseline_solution,
    baseline_water_heater,
    coordinate,
    demo_forecasts,
    net_cost,
)
from home_energy_optimizer.coordinate import comfort_penalty, total_objective


@pytest.fixture
def site() -> SiteConfig:
    return SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0),
        water_heater=WaterHeaterConfig(),
        hvac=HvacConfig(),
        coordination=CoordinationConfig(exchange_rounds=20),
    )


def test_terminates_and_reports_rounds(site):
    fc = demo_forecasts(site.horizon)
    res = coordinate(site, fc)
    assert 1 <= res.rounds_run <= site.coordination.exchange_rounds
    assert len(res.round_objectives) == res.rounds_run
    assert len(res.timings) == res.rounds_run


def test_all_devices_present(site):
    res = coordinate(site, demo_forecasts(site.horizon))
    assert set(res.devices) == {"battery", "water_heater", "hvac"}
    for sol in res.devices.values():
        assert np.all(np.isfinite(sol.power))


def test_cost_accounting_is_consistent(site):
    fc = demo_forecasts(site.horizon)
    res = coordinate(site, fc)
    assert res.net_cost == pytest.approx(res.import_cost - res.export_revenue, abs=1e-9)
    recomputed = net_cost(res.net_grid, fc.buy, fc.sell, site.horizon.dt)
    assert res.net_cost == pytest.approx(recomputed, abs=1e-9)


def test_net_grid_matches_device_powers(site):
    fc = demo_forecasts(site.horizon)
    res = coordinate(site, fc)
    rebuilt = fc.net_fixed_demand.copy()
    for sol in res.devices.values():
        rebuilt = rebuilt + sol.power
    assert np.allclose(rebuilt, res.net_grid, atol=1e-9)


def test_never_worse_than_doing_nothing_on_total_objective(site):
    """The baseline-fallback safety net must actually work.

    Coordination is a heuristic, so it CAN produce a bad round. What it must
    never do is ship one: every device's thermostat baseline is available as a
    fallback and is kept whenever it scores better on TOTAL OBJECTIVE.

    Note the invariant is on the objective (bill + comfort), not the bill. See
    test_bill_can_regress_when_comfort_improves below - that is deliberate, and
    the weighting between the two is an open design question (docs/NOTES.md).
    """
    for tariff in ("flat", "day_night", "dynamic"):
        fc = demo_forecasts(site.horizon, tariff=tariff)
        res = coordinate(site, fc)
        base_net, _ = baseline_solution(site, fc)
        wh_temp, _ = baseline_water_heater(site.water_heater, site.horizon, fc.hot_water_demand)
        hvac_temp, _ = baseline_hvac(site.hvac, site.horizon, fc.outdoor_temp)
        idle_soe = np.full(site.horizon.steps + 1, site.battery.capacity_kwh * 0.5)
        base_obj = total_objective(site, base_net, fc, wh_temp, hvac_temp, idle_soe)
        assert res.total_objective <= base_obj + 1e-6, f"regressed vs baseline on {tariff}"


def test_linear_comfort_pricing_does_not_regress_the_bill(site):
    """Regression for the fix to the arbitrary comfort weight.

    Under the POC's quadratic weight the optimiser RAISED the flat-tariff bill
    (5.733 -> 5.886) because it spent money preheating ahead of draw events,
    and that was defensible only because the comfort "penalty" fell from ~608
    to ~0.01 - in units that were not currency. A user comparing bills called
    it a regression, correctly.

    With discomfort priced in currency per kelvin-hour, derived from the cost
    of restoring the setpoint, the trade becomes visible to the optimiser and
    the bill goes DOWN on every tariff.
    """
    for tariff in ("flat", "day_night", "dynamic"):
        fc = demo_forecasts(site.horizon, tariff=tariff)
        res = coordinate(site, fc)
        _, base_cost = baseline_solution(site, fc)
        assert res.net_cost < base_cost, f"bill regressed on {tariff}"


def test_linear_comfort_still_beats_the_thermostat_on_comfort(site):
    """Cheaper AND more comfortable - not cheaper by abandoning comfort."""
    fc = demo_forecasts(site.horizon, tariff="flat")
    res = coordinate(site, fc)
    ref = float(np.mean(fc.buy))

    wh_temp, _ = baseline_water_heater(site.water_heater, site.horizon, fc.hot_water_demand)
    hvac_temp, _ = baseline_hvac(site.hvac, site.horizon, fc.outdoor_temp)
    base_comfort = comfort_penalty(site, wh_temp, hvac_temp, ref)
    opt_comfort = comfort_penalty(
        site, res.devices["water_heater"].trajectory, res.devices["hvac"].trajectory, ref
    )
    assert opt_comfort < base_comfort


def test_comfort_penalty_is_in_currency(site):
    """Sanity on the units: discomfort must be the same order as a bill, not
    two orders larger. The old weight scored ~608 against a ~6 bill."""
    fc = demo_forecasts(site.horizon, tariff="flat")
    ref = float(np.mean(fc.buy))
    wh_temp, _ = baseline_water_heater(site.water_heater, site.horizon, fc.hot_water_demand)
    hvac_temp, _ = baseline_hvac(site.hvac, site.horizon, fc.outdoor_temp)
    _, base_cost = baseline_solution(site, fc)

    base_comfort = comfort_penalty(site, wh_temp, hvac_temp, ref)
    assert base_comfort < base_cost, "discomfort should not dwarf the bill it is added to"


def test_quadratic_mode_reproduces_the_old_bill_regression(site):
    """The old behaviour is still reachable, and still wrong. Kept so the
    reason for the default is testable rather than folklore."""
    from dataclasses import replace

    quad = replace(
        site,
        water_heater=replace(site.water_heater, comfort_mode="quadratic"),
        hvac=replace(site.hvac, comfort_mode="quadratic"),
    )
    fc = demo_forecasts(site.horizon, tariff="flat")
    res = coordinate(quad, fc)
    lin = coordinate(site, fc)
    # The pathology is that an arbitrary kelvin-squared weight buys comfort at
    # any price, so the BILL goes up relative to pricing discomfort in currency.
    # Stated against the linear mode rather than against the baseline: once the
    # element stopped being billed for full slots it no longer had to run past
    # the untuned baseline for the effect to be visible, but the overspend is
    # the same object either way.
    assert res.net_cost > lin.net_cost


def test_savings_are_reported(site):
    fc = demo_forecasts(site.horizon, tariff="day_night")
    res = coordinate(site, fc)
    assert res.baseline_cost > 0
    assert res.savings == pytest.approx(res.baseline_cost - res.net_cost)


def test_price_spread_creates_more_savings_than_flat(site):
    """No arbitrage opportunity on a flat tariff; plenty on a spread one."""
    flat = coordinate(site, demo_forecasts(site.horizon, tariff="flat"))
    spread = coordinate(site, demo_forecasts(site.horizon, tariff="day_night"))
    assert spread.savings > flat.savings


def test_battery_only_site_still_works(site):
    cfg = site.without("water_heater", "hvac")
    res = coordinate(cfg, demo_forecasts(cfg.horizon, tariff="day_night"))
    assert set(res.devices) == {"battery"}


def test_no_devices_at_all(site):
    cfg = site.without("battery", "water_heater", "hvac")
    fc = demo_forecasts(cfg.horizon)
    res = coordinate(cfg, fc)
    assert res.devices == {}
    assert np.allclose(res.net_grid, fc.net_fixed_demand)


def test_soc_gate_disables_the_idle_battery_fallback(site):
    """With a gate present, swapping in an idle battery would violate it."""
    cfg = SiteConfig(
        horizon=site.horizon,
        battery=BatteryConfig(capacity_kwh=10.0),
        water_heater=None,
        hvac=None,
        soc_gates=(SocGate(hour=7.0, soc_frac=0.9),),
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    res = coordinate(cfg, demo_forecasts(cfg.horizon, tariff="day_night"))
    step = int(7.0 / cfg.horizon.dt)
    assert res.devices["battery"].trajectory[step] >= 0.8 * cfg.battery.capacity_kwh


def test_forecast_validation_rejects_bad_input(site):
    fc = demo_forecasts(site.horizon)
    bad = type(fc)(
        buy=fc.buy[:-1],
        sell=fc.sell,
        load=fc.load,
        solar=fc.solar,
        outdoor_temp=fc.outdoor_temp,
        hot_water_demand=fc.hot_water_demand,
    )
    with pytest.raises(ValueError, match="buy has shape"):
        coordinate(site, bad)


def test_nan_input_is_rejected(site):
    fc = demo_forecasts(site.horizon)
    buy = fc.buy.copy()
    buy[3] = np.nan
    bad = type(fc)(
        buy=buy,
        sell=fc.sell,
        load=fc.load,
        solar=fc.solar,
        outdoor_temp=fc.outdoor_temp,
        hot_water_demand=fc.hot_water_demand,
    )
    with pytest.raises(ValueError, match="non-finite"):
        coordinate(site, bad)


# --------------------------------------------------------------------------
# per-round records: what the coordinator kept, and why it kept that one
# --------------------------------------------------------------------------


def test_every_round_is_recorded_and_one_is_marked_selected(site):
    res = coordinate(site, demo_forecasts(site.horizon))

    assert len(res.rounds) == res.rounds_run
    assert [r.index for r in res.rounds] == list(range(len(res.rounds)))
    assert sum(r.selected for r in res.rounds) == 1
    assert res.rounds[res.selected_round].selected

    # the result is the retained iteration's plan after the baseline fallback
    # and the polish, which can only lower its objective
    sel = res.rounds[res.selected_round]
    assert res.total_objective <= sel.total_objective + 1e-9
    assert set(sel.powers) == set(res.devices)


def test_the_retained_round_is_the_one_with_the_lowest_objective(site):
    """The ranking key, asserted directly: total objective, nothing else.

    Not the last round and not the cheapest bill - the objective carries the
    comfort and grid-breach costs too, so it is the only comparison needed.
    """
    res = coordinate(site, demo_forecasts(site.horizon))
    # score_objective is the value as it stood when the round was ranked. The
    # retained round's plan may be re-pointed afterwards by the baseline
    # fallback, which is why the ranking is asserted on it and not on the live
    # field.
    keys = [r.score_objective for r in res.rounds]
    assert keys[res.selected_round] == min(keys)
    # ties break toward the earlier round, so nothing before it may match
    assert all(k > keys[res.selected_round] for k in keys[: res.selected_round])

    # every round that was NOT re-pointed has the two in exact agreement
    for r in res.rounds:
        if not r.fallback_applied:
            assert r.score_objective == r.total_objective


def test_a_breaching_round_does_not_win_by_having_a_smaller_bill(site):
    """With a hard import limit the loop passes through cheap-but-breaching
    plans. The breach is priced into the objective, so those must not win.

    This is the property the pricing exists for: ranking on the raw bill picks
    the plan that overdraws hardest, because overdrawing is what made it cheap.
    """
    from home_energy_optimizer import GridLimits
    from home_energy_optimizer.coordinate import grid_penalty

    limited = SiteConfig(
        horizon=site.horizon,
        battery=site.battery,
        water_heater=site.water_heater,
        hvac=site.hvac,
        grid=GridLimits(max_import_kw=1.5),
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    fc = demo_forecasts(limited.horizon, tariff="day_night")
    res = coordinate(limited, fc)
    sel = res.rounds[res.selected_round]

    # any round with a smaller BILL than the retained one pays for it
    # elsewhere in the objective - a breach, or comfort - so it did not win
    for r in res.rounds:
        if r.net_cost < sel.net_cost:
            assert r.score_objective >= sel.score_objective

    # and the penalty is a real, priced quantity - not a flag
    if sel.violation > 0:
        assert grid_penalty(limited, sel.net_grid, fc.buy) > 0


def test_the_breach_price_is_what_stops_the_trade(site):
    """Turn the price off and the cheapest-but-worst-breaching plan wins.

    The contrast is the argument for pricing it at all: at multiplier 0 the
    objective is blind to the limit, so selection reverts to the raw bill.
    """
    from home_energy_optimizer import GridLimits

    def solve(mult):
        cfg = SiteConfig(
            horizon=site.horizon,
            battery=site.battery,
            water_heater=site.water_heater,
            hvac=site.hvac,
            grid=GridLimits(max_import_kw=1.5, breach_price_multiplier=mult),
            coordination=CoordinationConfig(exchange_rounds=20),
        )
        return coordinate(cfg, demo_forecasts(cfg.horizon, tariff="dynamic"))

    priced, blind = solve(10.0), solve(0.0)
    b_sel = blind.rounds[blind.selected_round]
    p_sel = priced.rounds[priced.selected_round]
    assert b_sel.violation >= p_sel.violation, (
        "pricing the breach should not make the retained plan breach MORE"
    )


def test_rounds_carry_enough_to_reprice_a_round_that_was_not_retained(site):
    """A round keeps its residual load but not its value function - that is
    what lets the pricing tier re-derive lambda for any round on demand
    without the solve holding every V in memory."""
    res = coordinate(site, demo_forecasts(site.horizon))
    for r in res.rounds:
        assert r.battery_dp_load is not None
        assert r.battery_dp_load.shape == (site.horizon.steps,)
        assert not hasattr(r, "value")


def test_the_round_views_line_up(site):
    res = coordinate(site, demo_forecasts(site.horizon))
    assert res.round_objectives == [r.total_objective for r in res.rounds]
    assert res.round_costs == [r.net_cost for r in res.rounds]
    assert len(res.timings) == len(res.rounds)
    assert res.timings[0]["round_ms"] > 0
