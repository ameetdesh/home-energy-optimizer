"""Policy layer: fast lookup, marginal value, counterfactuals, hard clamps.

This is the module that justifies using DP over MILP, so these tests are the
load-bearing ones. They check the three claims made in
optimsurvey_notes/poc-integration-analysis.md.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    HardLimits,
    Horizon,
    PolicySnapshot,
    SiteConfig,
    action,
    clamp,
    coordinate,
    demo_forecasts,
    evaluate,
    marginal_value,
    marginal_value_curve,
    price_signal,
    q_values,
    rollout,
)


@pytest.fixture
def snap() -> PolicySnapshot:
    site = SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0),
        water_heater=None,
        hvac=None,
        coordination=CoordinationConfig(max_rounds=6),
    )
    fc = demo_forecasts(site.horizon, tariff="day_night")
    return PolicySnapshot.from_result(site, fc, coordinate(site, fc))


# --------------------------------------------------------------------------
# Claim 1: acting optimally is a lookup, not a re-solve
# --------------------------------------------------------------------------


def test_action_is_within_power_limits(snap):
    for soe in (0.0, 2.5, 5.0, 7.5, 10.0):
        a = action(snap, 10, soe)
        assert -snap.battery.p_discharge_max_kw - 1e-9 <= a <= snap.battery.p_charge_max_kw + 1e-9


def test_action_respects_energy_limits(snap):
    """An empty battery cannot discharge; a full one cannot charge."""
    assert action(snap, 10, 0.0) >= -1e-9
    assert action(snap, 10, snap.battery.capacity_kwh) <= 1e-9


def test_action_works_off_grid_points(snap):
    """States between grid points are the normal case in reality.

    Q is recomputed from V rather than read from POL precisely so that a
    measured SoE of 3.77 kWh gets a real answer, not the nearest grid node's.
    """
    fine = [action(snap, 20, s) for s in np.linspace(0.05, 9.95, 40)]
    assert all(np.isfinite(fine))


def test_action_lookup_is_fast(snap):
    """The latency claim. Budget is deliberately loose - this is pure numpy,
    not the eventual compiled path - but it must be far under a control cycle."""
    t0 = time.perf_counter()
    for _ in range(1000):
        action(snap, 30, 4.2)
    per_call_ms = (time.perf_counter() - t0) * 1000.0 / 1000
    assert per_call_ms < 1.0, f"{per_call_ms:.3f} ms/call is too slow for a fast tier"


def test_action_matches_the_solved_trajectory_at_its_own_states(snap):
    """Following the policy from the solved trajectory reproduces its actions."""
    soe = snap.value  # unused, but documents intent
    sol_soe, sol_power = rollout(snap, 0, snap.battery.capacity_kwh * 0.5)
    for t in range(0, 40, 5):
        assert action(snap, t, sol_soe[t]) == pytest.approx(sol_power[t], abs=1e-9)


# --------------------------------------------------------------------------
# Claim 2: dV/ds is a usable price signal
# --------------------------------------------------------------------------


def test_marginal_value_is_positive(snap):
    """Stored energy is worth something. Negative would mean holding charge hurts."""
    vals = [marginal_value(snap, 12, s) for s in np.linspace(1.0, 9.0, 20)]
    assert np.mean(vals) > 0


def test_marginal_value_has_sane_magnitude(snap):
    """It is a price, so it should live in the neighbourhood of the tariff."""
    lam = marginal_value(snap, 12, 5.0)
    assert 0.0 < lam < 10 * snap.buy.max(), f"lambda={lam} is not a plausible price"


def test_marginal_value_is_higher_when_energy_is_scarcer(snap):
    """Diminishing returns: the first kWh is worth more than the last.

    This is what makes lambda a useful substitute for evcc's hand-tuned
    bufferSoc/prioritySoc thresholds - it encodes the same intuition as a
    number instead of a constant.
    """
    low = marginal_value(snap, 12, 1.0)
    high = marginal_value(snap, 12, 9.0)
    assert low >= high


def test_marginal_value_curve_shape(snap):
    curve = marginal_value_curve(snap, 12)
    assert curve.shape == snap.states.shape
    assert np.all(np.isfinite(curve))


def test_price_signal_payload(snap):
    sig = price_signal(snap, 12, 5.0)
    assert set(sig) == {
        "step",
        "lambda_per_kwh",
        "import_price",
        "export_price",
        # What one more kWh at the METER costs - a different object from
        # lambda, and the one a load should gate on. See test_meter_price.py.
        "meter_price",
        "worth_running_above",
        # Grid trading thresholds - see test_reservation_prices.py.
        "import_below",
        "export_above",
        "spread",
    }
    assert sig["import_price"] == pytest.approx(snap.buy[12])


def test_lambda_tracks_the_cheap_window():
    """Overnight, when the battery is about to be filled cheaply, a stored kWh
    should be worth less than during the expensive evening peak."""
    site = SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0),
        water_heater=None,
        hvac=None,
        coordination=CoordinationConfig(max_rounds=6),
    )
    fc = demo_forecasts(site.horizon, tariff="day_night")
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))

    cheap_step = int(3.0 / site.horizon.dt)  # 03:00, offpeak
    peak_step = int(19.0 / site.horizon.dt)  # 19:00, peak
    assert marginal_value(snap, cheap_step, 5.0) <= marginal_value(snap, peak_step, 5.0)


# --------------------------------------------------------------------------
# Claim 3: counterfactuals are cheap
# --------------------------------------------------------------------------


def test_opportunity_cost_of_the_optimal_action_is_zero(snap):
    cf = evaluate(snap, 12, 5.0, action(snap, 12, 5.0))
    assert cf.opportunity_cost == pytest.approx(0.0, abs=1e-6)


def test_opportunity_cost_is_never_negative(snap):
    """No forced action can beat the optimum. Guards the Q/V consistency."""
    for t in (5, 20, 50, 80):
        for forced in np.linspace(-5.0, 5.0, 11):
            cf = evaluate(snap, t, 5.0, float(forced))
            assert cf.opportunity_cost >= -1e-6, f"t={t} a={forced} gave {cf.opportunity_cost}"


def test_forcing_a_bad_action_costs_money(snap):
    """Charging hard at the evening peak should be measurably worse than optimal."""
    peak = int(19.0 / snap.horizon.dt)
    cf = evaluate(snap, peak, 5.0, snap.battery.p_charge_max_kw)
    assert cf.opportunity_cost > 0


def test_counterfactual_is_fast(snap):
    cf = evaluate(snap, 30, 4.0, 2.0)
    assert cf.elapsed_ms < 1.0


def test_rollout_from_an_unplanned_state(snap):
    """The forecast was wrong and we are at 20% instead of 60% - still works.

    A MILP trajectory has nothing to say about a state it did not predict.
    """
    soe, power = rollout(snap, 40, 2.0)
    assert len(soe) == snap.horizon.steps - 40 + 1
    assert len(power) == snap.horizon.steps - 40
    assert soe[0] == pytest.approx(2.0)
    assert soe.min() >= -1e-9
    assert soe.max() <= snap.battery.capacity_kwh + 1e-9


def test_rollout_at_the_horizon_edge(snap):
    soe, power = rollout(snap, snap.horizon.steps, 5.0)
    assert len(power) == 0
    assert len(soe) == 1


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_snapshot_roundtrip(snap, tmp_path):
    path = str(tmp_path / "snap.npz")
    snap.save(path)
    loaded = PolicySnapshot.load(path)

    assert np.allclose(loaded.value, snap.value)
    assert np.allclose(loaded.states, snap.states)
    assert loaded.battery.capacity_kwh == snap.battery.capacity_kwh
    # The whole point: a process that never runs the solver can still act.
    assert action(loaded, 30, 4.2) == pytest.approx(action(snap, 30, 4.2))
    assert marginal_value(loaded, 30, 4.2) == pytest.approx(marginal_value(snap, 30, 4.2))


# --------------------------------------------------------------------------
# Hard limits
# --------------------------------------------------------------------------


def test_clamp_is_a_noop_without_limits():
    a, hit = clamp(3.0, other_load_kw=1.0, limits=HardLimits())
    assert a == 3.0 and hit == []


def test_clamp_enforces_import_limit():
    """Fuse protection. Must never depend on a big-M penalty inside the DP."""
    a, hit = clamp(5.0, other_load_kw=4.0, limits=HardLimits(max_import_kw=6.0))
    assert a == pytest.approx(2.0)
    assert "max_import_kw" in hit


def test_clamp_enforces_export_limit():
    a, hit = clamp(-5.0, other_load_kw=-2.0, limits=HardLimits(max_export_kw=3.0))
    assert a == pytest.approx(-1.0)
    assert "max_export_kw" in hit


def test_clamp_enforces_device_ratings():
    a, hit = clamp(9.0, 0.0, HardLimits(max_charge_kw=5.0))
    assert a == 5.0 and "max_charge_kw" in hit
    a, hit = clamp(-9.0, 0.0, HardLimits(max_discharge_kw=4.0))
    assert a == -4.0 and "max_discharge_kw" in hit


def test_clamp_enforces_ramp_limit():
    """Fast control must mean smoother setpoints, not faster relay cycling."""
    a, hit = clamp(5.0, 0.0, HardLimits(max_ramp_kw_per_step=1.0), previous_kw=0.0)
    assert a == pytest.approx(1.0)
    assert "max_ramp_kw_per_step" in hit


def test_clamp_reports_every_binding_limit():
    a, hit = clamp(
        9.0, other_load_kw=4.0, limits=HardLimits(max_charge_kw=6.0, max_import_kw=5.0)
    )
    assert a == pytest.approx(1.0)
    assert set(hit) == {"max_charge_kw", "max_import_kw"}


# --------------------------------------------------------------------------
# clamp_fleet: one shared grid limit, several storage units
# --------------------------------------------------------------------------


def test_clamp_fleet_does_not_hand_the_same_headroom_to_everyone():
    """The bug this function exists to prevent.

    Two batteries, 6 kW import limit, 1 kW of house load. Clamping each
    independently tells both that 5 kW is free, so the pair draws 11 kW.
    """
    from hemspolicy.policy import clamp_fleet

    naive = [clamp(5.0, other_load_kw=1.0, limits=HardLimits(max_import_kw=6.0))[0]
             for _ in range(2)]
    assert 1.0 + sum(naive) == pytest.approx(11.0)  # the wrong answer

    granted, _ = clamp_fleet(
        [("b0", 5.0), ("b1", 5.0)], other_load_kw=1.0,
        limits=HardLimits(max_import_kw=6.0),
    )
    assert 1.0 + sum(granted.values()) == pytest.approx(6.0)


def test_clamp_fleet_reports_which_unit_was_bound():
    from hemspolicy.policy import clamp_fleet

    _, why = clamp_fleet(
        [("b0", 5.0), ("b1", 5.0)], other_load_kw=1.0,
        limits=HardLimits(max_import_kw=6.0),
    )
    assert why["b0"] == []                       # first in line, unconstrained
    assert "max_import_kw" in why["b1"]


def test_clamp_fleet_order_is_the_priority():
    from hemspolicy.policy import clamp_fleet

    a, _ = clamp_fleet([("b0", 5.0), ("b1", 5.0)], 1.0, HardLimits(max_import_kw=6.0))
    b, _ = clamp_fleet([("b1", 5.0), ("b0", 5.0)], 1.0, HardLimits(max_import_kw=6.0))
    assert a["b0"] == pytest.approx(5.0) and a["b1"] == pytest.approx(0.0)
    assert b["b1"] == pytest.approx(5.0) and b["b0"] == pytest.approx(0.0)


def test_clamp_fleet_holds_on_export_too():
    from hemspolicy.policy import clamp_fleet

    # Both want to dump 5 kW; PV already exporting 2 kW; export cap 4 kW.
    granted, _ = clamp_fleet(
        [("b0", -5.0), ("b1", -5.0)], other_load_kw=-2.0,
        limits=HardLimits(max_export_kw=4.0),
    )
    assert -2.0 + sum(granted.values()) == pytest.approx(-4.0)


def test_clamp_fleet_applies_per_unit_limits_separately():
    from hemspolicy.policy import clamp_fleet

    granted, why = clamp_fleet(
        [("small", 5.0), ("big", 5.0)], other_load_kw=0.0,
        limits=HardLimits(max_import_kw=100.0),
        unit_limits={"small": HardLimits(max_charge_kw=2.0)},
    )
    assert granted["small"] == pytest.approx(2.0)
    assert granted["big"] == pytest.approx(5.0)   # shared limit is slack
    assert "max_charge_kw" in why["small"]


def test_clamp_fleet_is_feasible_for_any_fleet_size():
    from hemspolicy.policy import clamp_fleet

    for n in (1, 3, 8):
        granted, _ = clamp_fleet(
            [(f"b{i}", 4.0) for i in range(n)], other_load_kw=0.5,
            limits=HardLimits(max_import_kw=5.0),
        )
        assert 0.5 + sum(granted.values()) <= 5.0 + 1e-9


# --------------------------------------------------------------------------
# real-time exogenous input at inference, and the double-braking it invites
# --------------------------------------------------------------------------


def test_action_reacts_to_a_measured_load_the_forecast_never_saw(snap):
    """dp_load_kw makes the stage cost exact for the step about to be taken."""
    t, soe = 40, 5.0
    blind = action(snap, t, soe)
    spike = float(snap.dp_load[t]) + 4.0
    aware = action(snap, t, soe, dp_load_kw=spike)
    assert aware < blind          # more import on the meter -> discharge harder
    assert action(snap, t, soe, dp_load_kw=float(snap.dp_load[t])) == blind


def test_independent_replies_to_one_measurement_overshoot(snap):
    """The failure mode fleet_action exists to prevent.

    Three units each answering the same measurement each cover the whole
    deviation, so the fleet covers it three times and drives the meter past
    zero in the opposite direction.
    """
    from hemspolicy.policy import fleet_action

    t, soe = 40, 5.0
    spike = float(snap.dp_load[t]) + 4.0
    units = [(f"b{i}", snap, soe) for i in range(3)]

    naive_total = sum(action(s, t, so, dp_load_kw=spike) for _, s, so in units)
    one = action(snap, t, soe, dp_load_kw=spike)
    assert naive_total == pytest.approx(3 * one)        # each replied in full
    assert spike + naive_total < 0.0                    # meter driven past zero

    coord, _ = fleet_action(units, t, measured_load_kw=spike)
    assert abs(spike + sum(coord.values())) < abs(spike + naive_total)


def test_fleet_response_is_about_one_units_worth_regardless_of_size(snap):
    """A deviation should be covered once, not once per battery."""
    from hemspolicy.policy import fleet_action

    t, soe = 40, 5.0
    spike = float(snap.dp_load[t]) + 4.0
    one = action(snap, t, soe, dp_load_kw=spike) - action(snap, t, soe)

    for m in (2, 3, 5):
        coord, _ = fleet_action([(f"b{i}", snap, soe) for i in range(m)], t, spike)
        moved = sum(coord.values()) - m * action(snap, t, soe)
        # the fleet as a whole moves on the order of a single unit's response,
        # not m times it
        assert abs(moved) < 2.0 * abs(one)


def test_fleet_action_allocates_asymmetrically_by_priority(snap):
    """Earlier units absorb more, because later ones see less residual."""
    from hemspolicy.policy import fleet_action

    t, soe = 40, 5.0
    spike = float(snap.dp_load[t]) + 4.0
    coord, _ = fleet_action([("first", snap, soe), ("second", snap, soe),
                             ("third", snap, soe)], t, spike)
    assert abs(coord["first"]) > abs(coord["third"])


def test_fleet_action_respects_its_round_budget(snap):
    from hemspolicy.policy import fleet_action

    _, sweeps = fleet_action([("b0", snap, 5.0)], 40, 0.0, rounds=3)
    assert sweeps <= 3


def test_fleet_action_matches_single_action_for_one_unit(snap):
    """With one unit there is nothing to coordinate; the fixed point is the
    ordinary reply."""
    from hemspolicy.policy import fleet_action

    t, soe = 40, 5.0
    spike = float(snap.dp_load[t]) + 4.0
    coord, _ = fleet_action([("only", snap, soe)], t, spike, rounds=30, damping=0.7)
    assert coord["only"] == pytest.approx(action(snap, t, soe, dp_load_kw=spike), abs=0.2)


def test_persistent_deviation_is_flagged_for_replanning(snap):
    from hemspolicy.policy import deviation_is_persistent

    t = 40
    assert not deviation_is_persistent(snap, t, float(snap.dp_load[t]) + 0.1)
    assert deviation_is_persistent(snap, t, float(snap.dp_load[t]) + 4.0)
