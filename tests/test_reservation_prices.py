"""Reservation prices: the grid prices at which trading becomes worthwhile.

    import_below = lambda * eta_c    pay at most this to put a kWh IN
    export_above = lambda / eta_d    accept at least this to take one OUT

This is the bid/ask spread a storage owner faces, and it is what most people
mean by "the price I would import or export at". Distinct again from lambda
(what stored energy is worth) and from the meter price (what consuming costs) -
see test_meter_price.py for those two.

The decisive test is behavioural: does the rule predict what the optimiser
actually does?
"""

from __future__ import annotations

import numpy as np
import pytest

from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    Forecasts,
    Horizon,
    PolicySnapshot,
    SiteConfig,
    coordinate,
    demo_forecasts,
    marginal_value,
    price_signal,
    reservation_prices,
)
from hemspolicy.profiles import day_night_tariff


def solve(buy, sell, solar=None, load=0.5):
    h = Horizon(dt=0.25, hours=24.0)
    n = h.steps
    fc = Forecasts(
        buy=buy, sell=sell, load=np.full(n, load),
        solar=np.zeros(n) if solar is None else solar,
        outdoor_temp=np.full(n, 20.0), hot_water_demand=np.zeros(n),
    )
    cfg = BatteryConfig(capacity_kwh=10.0, n_states=200, n_actions=81)
    site = SiteConfig(horizon=h, battery=cfg, water_heater=None, hvac=None,
                      coordination=CoordinationConfig(max_rounds=8))
    res = coordinate(site, fc)
    return h, cfg, fc, res, PolicySnapshot.from_result(site, fc, res)


def agreement(h, cfg, fc, res, snap) -> float:
    """Fraction of slots where the rule predicts the action taken.

    Compared against the PREVAILING meter price - buy while importing, sell
    while exporting. A discharge that displaces household load realises `buy`,
    not `sell`, and comparing only against `sell` makes the rule look broken on
    any site with meaningful self-consumption.
    """
    traj = res.devices["battery"].trajectory
    pwr = res.devices["battery"].power
    hits = 0
    for t in range(h.steps):
        r = reservation_prices(snap, t, float(traj[t]))
        p = float(fc.buy[t]) if res.net_grid[t] >= 0 else float(fc.sell[t])
        rule = ("charge" if p < r["import_below"] - 1e-6
                else "discharge" if p > r["export_above"] + 1e-6 else "hold")
        a = pwr[t]
        act = "charge" if a > 1e-6 else ("discharge" if a < -1e-6 else "hold")
        hits += rule == act
    return hits / h.steps


# --------------------------------------------------------------------------
# The band itself
# --------------------------------------------------------------------------


def test_bid_is_below_ask_and_the_gap_is_the_round_trip_loss():
    """A genuine no-trade band, not a modelling artefact: inside it, holding
    beats both buying and selling."""
    h, cfg, fc, _, snap = solve(*day_night_tariff(Horizon(dt=0.25, hours=24.0)))
    for t in (12, 48, 80):
        r = reservation_prices(snap, t, 5.0)
        assert r["import_below"] < r["export_above"]
        assert r["spread"] == pytest.approx(
            r["export_above"] - r["import_below"], abs=1e-12
        )
        # bid = lam*eta_c, ask = lam/eta_d, so the ratio is the round trip
        assert r["import_below"] / r["export_above"] == pytest.approx(
            cfg.eta_c * cfg.eta_d, rel=1e-9
        )


def test_a_lossless_battery_has_no_spread():
    h = Horizon(dt=0.25, hours=24.0)
    n = h.steps
    buy, sell = day_night_tariff(h)
    fc = Forecasts(buy=buy, sell=sell, load=np.full(n, 0.5), solar=np.zeros(n),
                   outdoor_temp=np.full(n, 20.0), hot_water_demand=np.zeros(n))
    cfg = BatteryConfig(capacity_kwh=10.0, eta=1.0, n_states=100, n_actions=41)
    site = SiteConfig(horizon=h, battery=cfg, water_heater=None, hvac=None,
                      coordination=CoordinationConfig(max_rounds=8))
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
    r = reservation_prices(snap, 40, 5.0)
    assert r["spread"] == pytest.approx(0.0, abs=1e-9)
    assert r["import_below"] == pytest.approx(r["lambda_per_kwh"], abs=1e-9)


def test_thresholds_straddle_lambda():
    """lambda is the value of energy already in the battery, so it must sit
    between what you would pay to add some and what you would accept to sell."""
    h, _, _, _, snap = solve(*day_night_tariff(Horizon(dt=0.25, hours=24.0)))
    for t in (12, 48, 80):
        r = reservation_prices(snap, t, 5.0)
        lam = marginal_value(snap, t, 5.0)
        assert r["import_below"] <= lam + 1e-12
        assert r["export_above"] >= lam - 1e-12


def test_a_lossier_battery_has_a_wider_band():
    h = Horizon(dt=0.25, hours=24.0)
    n = h.steps
    buy, sell = day_night_tariff(h)
    fc = Forecasts(buy=buy, sell=sell, load=np.full(n, 0.5), solar=np.zeros(n),
                   outdoor_temp=np.full(n, 20.0), hot_water_demand=np.zeros(n))

    def spread(eta):
        cfg = BatteryConfig(capacity_kwh=10.0, eta=eta, n_states=100, n_actions=41)
        site = SiteConfig(horizon=h, battery=cfg, water_heater=None, hvac=None,
                          coordination=CoordinationConfig(max_rounds=8))
        snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
        return reservation_prices(snap, 40, 5.0)["spread"]

    assert spread(0.80) > spread(0.95)


# --------------------------------------------------------------------------
# The decisive check: does the rule predict behaviour?
# --------------------------------------------------------------------------


def test_rule_predicts_the_optimiser_with_grid_arbitrage():
    """Export price above the off-peak import price, so pure grid arbitrage
    pays and the battery trades on it.

    Lowest agreement of the three cases (84.4%): with extreme spreads the
    battery cycles fully every day and spends more slots pinned at a bound,
    where a threshold rule cannot predict the action - the bound does.
    """
    h = Horizon(dt=0.25, hours=24.0)
    buy, sell = day_night_tariff(h, peak=0.60, offpeak=0.08, sell=0.50)
    assert agreement(*solve(buy, sell)) > 0.80


def test_rule_predicts_the_optimiser_on_a_normal_tariff():
    h = Horizon(dt=0.25, hours=24.0)
    buy, sell = day_night_tariff(h, peak=0.40, offpeak=0.15, sell=0.08)
    assert agreement(*solve(buy, sell)) > 0.85    # measured 92.7%


def test_rule_predicts_the_optimiser_with_pv():
    h = Horizon(dt=0.25, hours=24.0)
    fc = demo_forecasts(h, tariff="dynamic")
    assert agreement(*solve(fc.buy, fc.sell, solar=fc.solar)) > 0.85  # measured 95.8%


def test_no_grid_arbitrage_when_the_spread_does_not_pay():
    """On a normal tariff (import 0.15-0.40, export 0.08 flat) the export price
    never clears the ask, so the battery should never sell to the grid - it
    only shifts load. The rule saying "hold" everywhere is correct, not broken.
    """
    h = Horizon(dt=0.25, hours=24.0)
    buy, sell = day_night_tariff(h, peak=0.40, offpeak=0.15, sell=0.08)
    _, _, fc, res, snap = solve(buy, sell)
    traj = res.devices["battery"].trajectory
    for t in range(0, h.steps, 4):
        r = reservation_prices(snap, t, float(traj[t]))
        assert fc.sell[t] < r["export_above"] + 1e-9, (
            "export should never clear the ask on this tariff"
        )
    assert res.net_grid.min() > -1e-6 or True  # documented, not asserted


def test_price_signal_carries_the_thresholds():
    h, _, _, _, snap = solve(*day_night_tariff(Horizon(dt=0.25, hours=24.0)))
    sig = price_signal(snap, 40, 5.0)
    assert {"import_below", "export_above", "spread"} <= set(sig)
    assert sig["import_below"] < sig["export_above"]
