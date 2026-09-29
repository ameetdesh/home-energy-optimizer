"""Three different prices, and which one a load should use.

This package computes several things that all look like "the price of energy"
and are not interchangeable:

* **lambda** (`marginal_value`) - what a kWh sitting INSIDE a battery is worth.
  There is one per battery, and they can differ.
* **meter price** (`meter_price`) - what one more kWh drawn AT THE METER costs.
  A single site-wide number, and the one a flexible load should gate on.
* **the tariff** (`buy[t]`) - what the utility charges, which with storage
  present is often not what an extra kWh actually costs.

Validated against an LP finite difference, which is the ground truth: perturb
the exogenous demand at one slot and measure the change in optimal cost.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="the finite-difference ground truth needs scipy")

from bench.duals import lp_battery_with_duals  # noqa: E402
from home_energy_optimizer import (  # noqa: E402
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    PolicySnapshot,
    SiteConfig,
    coordinate,
    demo_forecasts,
    marginal_value,
    meter_price,
    price_signal,
)


@pytest.fixture(scope="module")
def case():
    h = Horizon(dt=0.25, hours=24.0)
    cfg = BatteryConfig(capacity_kwh=10.0, n_states=200, n_actions=81)
    fc = demo_forecasts(h, tariff="day_night")
    site = SiteConfig(
        horizon=h, battery=cfg, water_heater=None, hvac=None,
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
    lp = lp_battery_with_duals(cfg, h, fc.buy, fc.sell, fc.net_fixed_demand)
    return h, cfg, fc, snap, lp


# --------------------------------------------------------------------------
# The ground truth, and the sign that was wrong
# --------------------------------------------------------------------------


def test_meter_dual_matches_a_finite_difference(case):
    """The LP's meter dual must equal d(cost)/d(demand), measured directly.

    This caught a sign error: the dual was being negated, which is correct for
    the SoE costate (free energy REDUCES cost) and wrong here (extra demand
    INCREASES it).
    """
    h, cfg, fc, _, lp = case
    eps = 0.01
    for t in (8, 48, 60, 84):
        pert = fc.net_fixed_demand.copy()
        pert[t] += eps
        lp2 = lp_battery_with_duals(cfg, h, fc.buy, fc.sell, pert)
        fd = (lp2["objective"] - lp["objective"]) / (eps * h.dt)
        assert lp["meter_lambda"][t] == pytest.approx(fd, abs=1e-3), (
            f"t={t}: dual {lp['meter_lambda'][t]:.4f} vs finite difference {fd:.4f}"
        )


def test_the_meter_price_is_positive(case):
    """Extra demand costs money. A negative value here means a flipped sign."""
    _, _, _, _, lp = case
    assert np.all(lp["meter_lambda"] > 0)


# --------------------------------------------------------------------------
# The DP's meter price against that ground truth
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tariff,mean_tol,max_tol",
    [("day_night", 0.005, 0.01), ("dynamic", 0.01, 0.06), ("flat", 0.01, 0.05)],
)
def test_dp_meter_price_tracks_the_lp(tariff, mean_tol, max_tol):
    """`meter_price` is a min over three routes, not a solved dual, so it is an
    approximation. Measured against the LP: mean 0.0007 (day_night), 0.0046
    (dynamic), 0.0053 (flat); worst case 0.043. Tolerances sit just above.
    """
    h = Horizon(dt=0.25, hours=24.0)
    cfg = BatteryConfig(capacity_kwh=10.0, n_states=200, n_actions=81)
    fc = demo_forecasts(h, tariff=tariff)
    site = SiteConfig(horizon=h, battery=cfg, water_heater=None, hvac=None,
                      coordination=CoordinationConfig(exchange_rounds=20))
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
    lp = lp_battery_with_duals(cfg, h, fc.buy, fc.sell, fc.net_fixed_demand)

    dp = np.array([meter_price(snap, t, float(lp["soe"][t])) for t in range(h.steps)])
    err = np.abs(dp - lp["meter_lambda"])
    assert err.mean() < mean_tol, f"{tariff}: mean error {err.mean():.4f}/kWh"
    assert err.max() < max_tol, f"{tariff}: max error {err.max():.4f}/kWh"


def test_export_route_is_keyed_on_surplus_not_on_the_battery():
    """Regression for two failed attempts.

    `net < 0` flips on the discrete action grid; widening it to a tolerance
    band let the export branch fire at 21:00 with no PV at all, pricing a kWh
    at 0.08 when it really cost 0.167. The condition must be a property of
    everything OTHER than the battery.
    """
    h = Horizon(dt=0.25, hours=24.0)
    cfg = BatteryConfig(capacity_kwh=10.0, n_states=200, n_actions=81)
    fc = demo_forecasts(h, tariff="day_night")
    site = SiteConfig(horizon=h, battery=cfg, water_heater=None, hvac=None,
                      coordination=CoordinationConfig(exchange_rounds=20))
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))

    evening = int(21.0 / h.dt)
    assert snap.dp_load[evening] > 0, "no PV at 21:00, so no export route"
    assert meter_price(snap, evening, 5.0) > float(fc.sell[evening]) + 1e-6

    midday = int(14.0 / h.dt)
    assert snap.dp_load[midday] < 0, "PV spilling at 14:00"
    assert meter_price(snap, midday, 5.0) == pytest.approx(float(fc.sell[midday]), abs=1e-6)


def test_meter_price_is_bounded_by_the_tariff(case):
    """It can be far BELOW the tariff - storage serves the load - but never
    above it, because buying is always an option."""
    h, _, fc, snap, lp = case
    for t in range(0, h.steps, 4):
        m = meter_price(snap, t, float(lp["soe"][t]))
        assert m <= fc.buy[t] + 1e-9, f"t={t}: {m:.4f} > tariff {fc.buy[t]:.4f}"
        assert m > 0


def test_meter_price_falls_well_below_the_tariff_at_the_peak(case):
    """The headline case. At the evening peak the meter reads 0.40 while an
    extra kWh really costs ~0.167 - the battery serves it and refills
    overnight. A load gating on buy[t] would wrongly refuse to run."""
    h, _, fc, snap, lp = case
    t = int(21.0 / h.dt)
    assert fc.buy[t] == pytest.approx(0.40)
    m = meter_price(snap, t, float(lp["soe"][t]))
    assert m < 0.5 * fc.buy[t], f"expected well below tariff, got {m:.4f}"


# --------------------------------------------------------------------------
# lambda and the meter price are DIFFERENT objects
# --------------------------------------------------------------------------


def test_lambda_and_meter_price_differ_by_the_discharge_efficiency(case):
    """Delivering 1 kWh to a load drains 1/eta_d from store, so the meter price
    is lambda/eta_d whenever the battery is the marginal source."""
    h, cfg, fc, snap, lp = case
    t = int(21.0 / h.dt)
    soe = float(lp["soe"][t])
    lam = marginal_value(snap, t, soe)
    assert meter_price(snap, t, soe) == pytest.approx(lam / cfg.eta_d, abs=1e-6)
    assert meter_price(snap, t, soe) > lam, "the meter price exceeds lambda by the losses"


def test_price_signal_reports_both(case):
    h, _, _, snap, lp = case
    sig = price_signal(snap, 84, float(lp["soe"][84]))
    assert set(sig) >= {"lambda_per_kwh", "meter_price", "worth_running_above"}
    # A load gates on the meter price, not on lambda.
    assert sig["worth_running_above"] == pytest.approx(sig["meter_price"])
    assert sig["meter_price"] != pytest.approx(sig["lambda_per_kwh"])


# --------------------------------------------------------------------------
# Per-battery lambda
# --------------------------------------------------------------------------


def test_two_batteries_can_have_different_lambdas():
    """One lambda per battery, not one for the site.

    A kWh in a lossy battery is worth less than one in an efficient battery,
    because less of it comes back out. Measured mean gap 0.015/kWh, max 0.044
    for eta 0.95 vs 0.85.
    """
    from home_energy_optimizer.dp_battery import solve_battery

    h = Horizon(dt=0.25, hours=24.0)
    fc = demo_forecasts(h, tariff="day_night")
    b0 = BatteryConfig(capacity_kwh=8.0, p_charge_max_kw=5.0, p_discharge_max_kw=5.0,
                       eta=0.95, n_states=100, n_actions=51)
    b1 = BatteryConfig(capacity_kwh=25.0, p_charge_max_kw=3.0, p_discharge_max_kw=3.0,
                       eta=0.85, n_states=100, n_actions=51)
    site = SiteConfig(horizon=h, battery=b0, batteries=(b1,), water_heater=None,
                      hvac=None, coordination=CoordinationConfig(exchange_rounds=20))
    res = coordinate(site, fc)

    def lam_for(cfg, key):
        others = fc.net_fixed_demand.copy()
        for k, sol in res.devices.items():
            if k != key:
                others = others + sol.power
        s = solve_battery(cfg, h, fc.buy, fc.sell, dp_load=others, admm_rho=0.0)
        traj = res.devices[key].trajectory
        return np.array([
            float(np.interp(traj[t], s.states, np.gradient(s.value[t], s.states)))
            for t in range(h.steps)
        ])

    l0, l1 = lam_for(b0, "battery"), lam_for(b1, "battery1")
    assert np.abs(l0 - l1).max() > 0.01, "different batteries should price differently"
    assert np.abs(l0 - l1).mean() < 0.10, "but not wildly - they share a meter"


def test_the_published_snapshot_prices_battery_zero_only():
    """Documents a known limitation rather than asserting it is right.

    PolicySnapshot carries battery 0's value function, so `marginal_value` is
    battery 0's lambda. With several batteries the others have their own, and
    nothing surfaces them yet.
    """
    h = Horizon(dt=0.25, hours=12.0)
    fc = demo_forecasts(h, tariff="day_night")
    b = BatteryConfig(capacity_kwh=10.0, n_states=60, n_actions=25)
    site = SiteConfig(horizon=h, battery=b, batteries=(b,), water_heater=None,
                      hvac=None, coordination=CoordinationConfig(exchange_rounds=20))
    res = coordinate(site, fc)
    snap = PolicySnapshot.from_result(site, fc, res)

    assert len(res.devices) == 2
    assert snap.battery.capacity_kwh == site.battery.capacity_kwh
    assert snap.value.shape == (h.steps + 1, b.n_states)
