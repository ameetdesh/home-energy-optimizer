"""Sub-meters (types.SubMeter): a hybrid inverter, and a shared breaker.

What they must do: change nothing when they do not bind; respect their
ratings in every planner (Dantzig-Wolfe, ADMM, the battery DP); price the
devices behind them at their bus's own price; and keep the saving split
adding up. On a battery-only site the whole problem is one LP, so the DW plan
is checked against an independent LP of the same site (scipy's HiGHS).
"""

from dataclasses import replace

import numpy as np
import pytest

from home_energy_optimizer import (
    BatteryConfig,
    GridLimits,
    Horizon,
    HvacConfig,
    PolicySnapshot,
    SiteConfig,
    WaterHeaterConfig,
    action,
    coordinate,
    group_limit,
    hybrid_inverter,
)
from home_energy_optimizer.coordinate import apply_curtailment
from home_energy_optimizer.dp_battery import terminal_price
from home_energy_optimizer.dw.attribution import ledger
from home_energy_optimizer.dw.coordinator import DWCoordinator
from home_energy_optimizer.dw.integrate import dw_plan
from home_energy_optimizer.meter import bus_flow
from home_energy_optimizer.profiles import demo_forecasts
from home_energy_optimizer.submeter import site_meter

H = Horizon(dt=0.5, hours=24)
FC = demo_forecasts(H, solar_peak_kw=9.0)


def _site(limit, eta=0.95, curtail=True, **kw):
    subs = () if limit is False else (hybrid_inverter(("battery",), limit, limit, eta, eta),)
    return SiteConfig(horizon=H, battery=BatteryConfig(), water_heater=kw.pop("water_heater", None),
                      hvac=kw.pop("hvac", None), grid=GridLimits(allow_curtailment=curtail), submeters=subs, **kw)


def _reference_lp(b: BatteryConfig, limit: float | None, eta: float, curtail: bool) -> float:
    """The hybrid site as one LP, written out independently: battery charge c,
    discharge e, stored energy s; inverter send x and take y (DC side); PV
    clipped k; import i, export o."""
    sp = pytest.importorskip("scipy.optimize")
    from scipy import sparse
    n, dt, fc = H.steps, H.dt, FC
    c, e, s, x, y, k, i, o = (np.arange(n) + j * n for j in range(8))
    nv = 8 * n
    cost = np.zeros(nv)
    cost[i], cost[o] = fc.buy * dt, -fc.sell * dt
    tp = terminal_price(b, fc.buy)
    cost[s[-1]] = -tp
    cap_dc = np.inf if limit is None else limit / eta
    ub = np.full(nv, np.inf)
    ub[c], ub[e], ub[s] = b.p_charge_max_kw, b.p_discharge_max_kw, b.capacity_kwh
    ub[x], ub[y] = cap_dc, np.inf if limit is None else limit * eta
    ub[k] = fc.solar if curtail else np.maximum(fc.solar - cap_dc, 0.0)
    lb = np.zeros(nv)
    lb[s] = b.soe_floor_kwh
    rows, cols, vals, rhs = [], [], [], []

    def eq(row, col, val):
        rows.append(row), cols.append(col), vals.append(np.broadcast_to(val, row.shape).astype(float))
    t = np.arange(n)
    eq(t, i, 1), eq(t, o, -1), eq(t, x, eta), eq(t, y, -1 / eta)           # meter: i - o + flow = load
    rhs.append(fc.load)
    r = n + t
    eq(r, k, -1), eq(r, c, -1), eq(r, e, 1), eq(r, x, -1), eq(r, y, 1)      # bus: pv = k + c - e + x - y
    rhs.append(-fc.solar)
    r = 2 * n + t
    eq(r, s, 1), eq(r[1:], s[:-1], -1), eq(r, c, -dt * b.eta_c), eq(r, e, dt / b.eta_d)
    r0 = np.zeros(n)
    r0[0] = b.capacity_kwh * b.soc_initial_frac
    rhs.append(r0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, nv))
    res = sp.linprog(cost, A_eq=A, b_eq=np.concatenate(rhs), bounds=list(zip(lb, ub)), method="highs")
    assert res.status == 0, res.message
    return float(res.fun + tp * r0[0])


@pytest.mark.parametrize("limit, eta, curtail", [(4.0, 0.97, True), (4.0, 0.97, False), (2.5, 0.95, True),
                                                 (None, 0.9, True)])
def test_dw_matches_an_independent_lp_of_the_hybrid_site(limit, eta, curtail):
    """A battery and the PV on an inverter's DC bus: DW's plan, its bound and
    an LP written out independently agree, and the AC side stays within the
    rating."""
    site = _site(limit, eta, curtail)
    r = DWCoordinator(site, FC).run()
    ref = _reference_lp(site.battery, limit, eta, curtail)
    assert r.upper == pytest.approx(ref, abs=1e-4)
    assert r.lower == pytest.approx(ref, abs=1e-4)
    flows = site_meter(site, FC, {k: c.power for k, c in r.plan.items()})
    if limit is not None:
        assert flows.ac["inverter"].max() <= limit + 1e-6
        assert flows.ac["inverter"].min() >= -limit - 1e-6
    assert flows.breach_kwh.sum() == pytest.approx(0.0, abs=1e-9)


def test_a_lossless_unlimited_inverter_changes_nothing():
    """With no rating and no losses, the PV and battery behind an inverter
    plan exactly as on the AC bus."""
    a = DWCoordinator(_site(False), FC).run()
    b = DWCoordinator(_site(None, eta=1.0), FC).run()
    assert b.upper == pytest.approx(a.upper, abs=1e-6)        # the LP solver's tolerance
    # the same plan scores the same either way (the plans themselves may be
    # different optima of the same cost)
    p = {"battery": a.plan["battery"].power}
    assert np.allclose(site_meter(_site(None, 1.0), FC, p).net, site_meter(_site(False), FC, p).net, atol=1e-12)


def test_no_submeter_meter_is_the_old_formula():
    """site_meter with no sub-meter: the load, minus the PV, plus the
    devices, then the curtailment rule - the formula it replaced."""
    site = _site(False)
    p = {"battery": np.sin(np.arange(H.steps) / 5.0) * 3.0}
    old, curt = apply_curtailment(FC.load - FC.solar + p["battery"], FC.solar, FC.sell, site.grid)
    new = site_meter(site, FC, p)
    assert np.array_equal(new.net, old) and np.array_equal(new.curtail, curt)


def test_the_bus_price_falls_to_zero_while_pv_is_clipped():
    """Where the inverter is full and PV is being clipped, one more kWh on
    the DC bus is worth nothing - the bus's price is 0 - while the meter's
    is still the export price."""
    site = _site(2.5)
    r = DWCoordinator(site, FC).run()
    flows = site_meter(site, FC, {k: c.power for k, c in r.plan.items()})
    clipped = flows.curtail > 0.1
    assert clipped.any()
    assert np.all(np.abs(r.local_prices["inverter"][clipped]) < 1e-6)
    assert np.all(r.prices[clipped] > 0.01)


def test_a_battery_bidding_plans_behind_the_inverter_keeps_a_valid_bound():
    """The battery as a DW block (its DP, priced at the bus's price): the
    bound stays below the plan, and the plan within the rating."""
    site = _site(4.0, water_heater=WaterHeaterConfig())
    r = DWCoordinator(site, FC, battery_in_master=False).run()
    assert r.lower <= r.upper + 1e-6
    flows = site_meter(site, FC, {k: c.power for k, c in r.plan.items()})
    assert flows.ac["inverter"].max() <= 4.0 + 1e-6


def test_a_shared_breaker_holds_the_tank_and_heat_pump_together():
    """A 3 kW breaker shared by the tank (3 kW) and the heat pump: they never
    draw more than 3 kW together, and the plan pays no breach."""
    site = replace(_site(False, water_heater=WaterHeaterConfig(), hvac=HvacConfig()),
                   submeters=(group_limit("breaker", ("water_heater", "hvac"), max_kw=3.0),))
    r = DWCoordinator(site, FC).run()
    both = r.plan["water_heater"].power + r.plan["hvac"].power
    assert both.max() <= 3.0 + 1e-6
    assert r.plan_parts["breach_kwh"] == pytest.approx(0.0, abs=1e-9)


def test_the_saving_split_adds_up_with_local_prices():
    """Aumann-Shapley at each bus's price: the shares and reimbursements add
    up to the house's saving exactly, with the inverter binding."""
    site = _site(2.5, water_heater=WaterHeaterConfig())
    dark = replace(FC, solar=np.zeros(H.steps))
    co, co_dark = DWCoordinator(site, FC), DWCoordinator(site, dark)
    led = ledger(co, co.run().plan, co_dark, co_dark.run().plan)
    tot = led["totals"]
    assert tot["net_gain"] + tot["private_cost_change"] == pytest.approx(
        tot["house_bill_before"] - tot["house_bill_after"], abs=1e-9)


def test_admm_respects_the_rating_and_the_dw_bound():
    """ADMM with the inverter as a two-terminal device: a plan within the
    rating, no cheaper than DW's lower bound."""
    site = _site(2.5)
    a = coordinate(site, FC)
    flows = site_meter(site, FC, {k: v.power for k, v in a.devices.items()})
    assert flows.ac["inverter"].max() <= 2.5 + 1e-6
    assert flows.breach_kwh.sum() < 1e-6
    assert a.total_objective >= DWCoordinator(site, FC).run().lower - 1e-6


def test_the_fast_tier_acts_through_the_inverter():
    """The policy snapshot of a hybrid site: its battery DP carries the bus,
    survives a save and load, and never asks the inverter for more than its
    rating at midday."""
    site = _site(2.5)
    res = dw_plan(site, FC)
    snap = PolicySnapshot.from_result(site, FC, res)
    assert snap.bus is not None
    t = int(np.argmax(FC.solar))
    a = action(snap, t, 5.0)
    flow, _, over, _ = bus_flow(a + snap.bus.others[t], snap.bus.clip_cap[t], snap.bus.eta_export,
                                snap.bus.eta_import, snap.bus.export_cap, snap.bus.import_cap)
    assert float(flow) <= 2.5 + 1e-6 and float(over) == 0.0


def test_a_snapshot_with_a_bus_round_trips(tmp_path):
    site = _site(2.5)
    snap = PolicySnapshot.from_result(site, FC, dw_plan(site, FC))
    path = tmp_path / "snap.npz"
    snap.save(str(path))
    back = PolicySnapshot.load(str(path))
    assert back.bus is not None and np.array_equal(back.bus.others, snap.bus.others)
    assert back.bus.export_cap == snap.bus.export_cap
    t = int(np.argmax(FC.solar))
    assert action(back, t, 5.0) == action(snap, t, 5.0)


def test_bus_flow_clips_pv_before_it_breaches():
    """6 kW of PV surplus into a 4 kW (bus-side) connection: 2 kW clipped,
    no breach. A battery discharging into it with no PV to clip: a breach."""
    flow, clip, over, _ = bus_flow(np.array([-6.0]), np.array([6.0]), 1.0, 1.0, 4.0, 4.0)
    assert flow[0] == pytest.approx(4.0) and clip[0] == pytest.approx(2.0) and over[0] == 0.0
    flow, clip, over, _ = bus_flow(np.array([-6.0]), np.array([0.0]), 1.0, 1.0, 4.0, 4.0)
    assert clip[0] == 0.0 and over[0] == pytest.approx(2.0)


def test_validation():
    with pytest.raises(ValueError, match="behind two"):
        replace(_site(False), submeters=(group_limit("a", ("battery",), max_kw=1.0),
                                         group_limit("b", ("battery",), max_kw=1.0))).validate()
    with pytest.raises(ValueError, match="no device"):
        DWCoordinator(replace(_site(False), submeters=(group_limit("a", ("ev",), max_kw=1.0),)), FC)
    with pytest.raises(ValueError, match="force the members"):
        group_limit("a", ("battery",), min_kw=1.0)


@pytest.mark.parametrize("min_kw", [-1.0, 0.0])
def test_a_group_limits_export_holds_a_battery_behind_it(min_kw):
    """min_kw <= 0 is how much a group may push back to the house: a battery
    behind a -1 kW limit discharges at most 1 kW; behind 0, not at all (no
    backfeed). The plan meets its bound, as an LP must."""
    fc = demo_forecasts(H, tariff="day_night")
    site = SiteConfig(horizon=H, battery=BatteryConfig(), water_heater=None, hvac=None,
                      submeters=(group_limit("panel", ("battery",), min_kw=min_kw),))
    r = DWCoordinator(site, fc).run()
    assert r.plan["battery"].power.min() >= min_kw - 1e-6
    assert r.upper == pytest.approx(r.lower, abs=1e-6)
