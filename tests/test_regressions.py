"""Regressions for defects found while extracting the headless core.

All three were present in the original POC and were invisible there because it
had no tests - correctness was judged by looking at a chart. Each of these
would silently make the system look useless or produce a nonsense price.
"""

from __future__ import annotations

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
    solve_battery,
)


def _battery_site(**batt) -> SiteConfig:
    return SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0, **batt),
        water_heater=None,
        hvac=None,
        coordination=CoordinationConfig(max_rounds=8),
    )


# --------------------------------------------------------------------------
# Defect 1: the quadratic terminal penalty strangles the battery
# --------------------------------------------------------------------------


def test_quadratic_terminal_hoards_energy_at_the_horizon_edge():
    """Documents WHY the default was changed to linear.

    The POC's terminal term is -w(S - target)^2 / cap. Its DERIVATIVE is what
    the optimiser feels: at w=5, cap=10 that is (S - 5) currency/kWh, i.e.
    +-5/kWh at the extremes against ~0.2/kWh of available arbitrage. So near
    the horizon edge the battery is dragged back to the target SoC for reasons
    that have nothing to do with prices.

    Note this does NOT pin the battery outright - it trades freely mid-horizon.
    Outright pinning was a separate defect (the unseeded ADMM target, below).
    """
    fc = demo_forecasts(Horizon(dt=0.25, hours=24.0), tariff="day_night")

    quad = coordinate(_battery_site(terminal_mode="quadratic"), fc)
    lin = coordinate(_battery_site(), fc)

    target = 5.0
    quad_end = quad.devices["battery"].trajectory[-1]
    lin_end = lin.devices["battery"].trajectory[-1]
    assert abs(quad_end - target) < abs(lin_end - target), (
        "quadratic mode should drag terminal SoE toward the target"
    )


def test_quadratic_terminal_costs_real_savings():
    """The hoarding is not free: energy held back is arbitrage not captured."""
    fc = demo_forecasts(Horizon(dt=0.25, hours=24.0), tariff="day_night")
    quad = coordinate(_battery_site(terminal_mode="quadratic"), fc)
    lin = coordinate(_battery_site(), fc)
    assert lin.savings > quad.savings


def test_linear_terminal_lets_the_battery_trade():
    site = _battery_site()  # linear is the default
    fc = demo_forecasts(site.horizon, tariff="day_night")
    res = coordinate(site, fc)
    throughput = np.abs(res.devices["battery"].power).sum() * site.horizon.dt
    assert throughput > 10.0, "a 10 kWh battery on a 0.15/0.40 spread should cycle"
    assert res.savings > 0


def test_quadratic_terminal_zeroes_lambda_at_the_horizon_edge():
    """The other reason it was wrong: at t = N it forces dV/ds = 0 at the
    target, which is an artefact of the penalty shape rather than a statement
    about energy prices. Linear mode keeps a real price right to the edge.

    (Mid-horizon the two agree, because 80-odd steps of propagated economics
    dominate the terminal shape - so this is an edge effect, not a global one.)
    """
    fc = demo_forecasts(Horizon(dt=0.25, hours=24.0), tariff="day_night")
    last = Horizon(dt=0.25, hours=24.0).steps - 1

    quad_site = _battery_site(terminal_mode="quadratic")
    quad = PolicySnapshot.from_result(quad_site, fc, coordinate(quad_site, fc))
    lin_site = _battery_site()
    lin = PolicySnapshot.from_result(lin_site, fc, coordinate(lin_site, fc))

    # At the true horizon edge the quadratic terminal is flat at its target.
    quad_edge = np.interp(5.0, quad.states, np.gradient(quad.value[-1], quad.states))
    lin_edge = np.interp(5.0, lin.states, np.gradient(lin.value[-1], lin.states))
    assert abs(quad_edge) < 1e-9
    assert lin_edge == pytest.approx(float(fc.buy.min()), abs=1e-9)

    # Mid-horizon they agree - the defect is confined to the edge.
    mid = last // 2
    assert marginal_value(quad, mid, 5.0) == pytest.approx(
        marginal_value(lin, mid, 5.0), rel=0.05
    )


# --------------------------------------------------------------------------
# Defect 2: unseeded ADMM target disables the battery
# --------------------------------------------------------------------------


def test_admm_proximity_about_zero_would_disable_the_battery():
    """Reproduces the failure mode directly, at the solver level.

    With target = 0 and the default rho, the proximity penalty is
    (5/2) * a^2 * 0.25 = 0.625 a^2 - about 15 currency at 5 kW, against
    ~0.2/kWh of arbitrage. The battery does nothing. The coordinator now
    warm-starts the target from a free solve so this cannot happen.
    """
    h = Horizon(dt=0.25, hours=24.0)
    cfg = BatteryConfig(capacity_kwh=10.0)
    fc = demo_forecasts(h, tariff="day_night")

    free = solve_battery(cfg, h, fc.buy, fc.sell, dp_load=fc.net_fixed_demand)
    pinned = solve_battery(
        cfg,
        h,
        fc.buy,
        fc.sell,
        dp_load=fc.net_fixed_demand,
        admm_target=np.zeros(h.steps),
        admm_rho=5.0,
    )

    assert np.abs(free.power).sum() > 10.0
    assert np.abs(pinned.power).sum() < 1e-6


def test_coordinator_warm_starts_the_battery_target():
    """End-to-end guard: the coordinated battery must not be pinned off."""
    for tariff in ("flat", "day_night", "dynamic"):
        site = _battery_site()
        fc = demo_forecasts(site.horizon, tariff=tariff)
        res = coordinate(site, fc)
        throughput = np.abs(res.devices["battery"].power).sum() * site.horizon.dt
        assert throughput > 1.0, f"battery pinned off on {tariff}"


def test_warm_start_also_applies_with_thermal_devices_present():
    site = SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0),
        water_heater=WaterHeaterConfig(),
        hvac=HvacConfig(),
        coordination=CoordinationConfig(max_rounds=8),
    )
    fc = demo_forecasts(site.horizon, tariff="day_night")
    res = coordinate(site, fc)
    throughput = np.abs(res.devices["battery"].power).sum() * site.horizon.dt
    assert throughput > 1.0


# --------------------------------------------------------------------------
# Defect 3: the ADMM term pollutes the price signal
# --------------------------------------------------------------------------


def test_lambda_is_bounded_by_the_tariff():
    """The load-bearing property of the whole proposal.

    A marginal value of stored energy must live in tariff units. The
    coordinated value function does not: it carries -(rho/2)(a - target)^2,
    which at rho = 5 over a 10 kW action span reaches ~60 currency and swamps
    a 0.30/kWh tariff. The pricing tier therefore re-solves with rho = 0.
    """
    for tariff in ("flat", "day_night", "dynamic"):
        site = _battery_site()
        fc = demo_forecasts(site.horizon, tariff=tariff)
        snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))

        lo, hi = float(fc.sell.min()), float(fc.buy.max())
        for t in (12, 48, 76):
            lam = marginal_value(snap, t, 5.0)
            assert 0.0 <= lam <= hi * 1.5, f"{tariff} t={t}: lambda={lam} outside [0, {hi}]"


def test_lambda_is_higher_at_peak_than_overnight():
    """The signal must actually carry information about when energy is scarce."""
    site = _battery_site()
    fc = demo_forecasts(site.horizon, tariff="day_night")
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))

    overnight = marginal_value(snap, int(3.0 / site.horizon.dt), 5.0)
    peak = marginal_value(snap, int(19.0 / site.horizon.dt), 5.0)
    assert peak > overnight


def test_snapshot_carries_no_admm_term():
    site = _battery_site()
    fc = demo_forecasts(site.horizon, tariff="dynamic")
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
    assert snap.admm_rho == 0.0
    assert snap.admm_target is None


def test_lambda_approaches_export_price_during_surplus():
    """When PV is being exported, a stored kWh is worth roughly the export
    price - you can only realise it by selling. Sanity-checks the economics."""
    site = _battery_site()
    fc = demo_forecasts(site.horizon, tariff="flat")
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))

    noon = int(12.0 / site.horizon.dt)
    assert fc.solar[noon] > fc.load[noon], "fixture should be exporting at noon"
    lam = marginal_value(snap, noon, 5.0)
    assert lam == pytest.approx(float(fc.sell[noon]), abs=0.05)


# --------------------------------------------------------------------------
# curtailment is NOT a pure post-process: it must price the devices
# --------------------------------------------------------------------------


def test_negative_export_price_does_not_buy_phantom_battery_cycles():
    """A device must not spend round-trip losses to avoid an export it would
    simply have curtailed.

    With curtailment available, a negative export tariff is never actually paid
    on PV surplus. Pricing devices at the raw negative number gives them a
    benefit for absorbing that surplus which does not exist, and they pay real
    efficiency losses to collect it.
    """
    from dataclasses import replace

    import numpy as np

    from hemspolicy import (BatteryConfig, CoordinationConfig, GridLimits,
                            Horizon, SiteConfig, coordinate, demo_forecasts)

    h = Horizon(dt=0.25, hours=24.0)
    fc = demo_forecasts(h, tariff="day_night")
    site = SiteConfig(
        horizon=h, battery=BatteryConfig(capacity_kwh=10.0),
        water_heater=None, hvac=None,
        grid=GridLimits(allow_curtailment=True),
        coordination=CoordinationConfig(max_rounds=15),
    )
    throughput = lambda r: float(np.abs(r.devices["battery"].power).sum() * h.dt)
    midday = slice(int(10 / h.dt), int(15 / h.dt))

    def run(midday_price):
        sell = fc.sell.copy()
        sell[midday] = midday_price
        return coordinate(site, replace(fc, sell=sell))

    zero = run(0.0)
    shallow = run(-0.05)
    deep = run(-0.30)

    # Below zero the device-facing price is pinned at zero, so however much
    # further the tariff falls the battery must not cycle any harder for it.
    assert throughput(shallow) == pytest.approx(throughput(zero), abs=1e-6)
    assert throughput(deep) == pytest.approx(throughput(zero), abs=1e-6)

    # The extra surplus is thrown away instead, which is free.
    assert deep.curtailed_kwh > zero.curtailed_kwh


def test_device_sell_price_is_a_no_op_without_curtailment():
    """Where PV cannot be thrown away, a negative price IS paid, and devices
    must keep seeing it."""
    import numpy as np

    from hemspolicy import GridLimits
    from hemspolicy.coordinate import device_sell_price

    sell = np.array([-0.2, 0.0, 0.3])
    off = GridLimits(allow_curtailment=False)
    assert np.array_equal(device_sell_price(sell, off), sell)
    assert np.array_equal(device_sell_price(sell, None), sell)
    on = GridLimits(allow_curtailment=True)
    assert np.array_equal(device_sell_price(sell, on), np.array([0.0, 0.0, 0.3]))
