"""The site's electrical tree beyond one level (types.SubMeter with parents,
GridConnection, SetLimit, PV arrays): nested limits, a second hybrid inverter
behind the first, several PV arrays, a second grid connection with its own
tariff, and a limit on a set of devices across the tree.

On battery-only sites the problem is one LP, so the Dantzig-Wolfe plan - scored
by submeter.site_meter - is checked against an LP written out independently
(tests/topology_reference.py). The other planners are checked for what they
promise: a valid bound, plans within the ratings, shares that add up.
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
from home_energy_optimizer.dw.attribution import dark, ledger
from home_energy_optimizer.dw.coordinator import DWCoordinator
from home_energy_optimizer.dw.integrate import dw_plan
from home_energy_optimizer.profiles import demo_forecasts
from home_energy_optimizer.submeter import site_meter, topology
from home_energy_optimizer.types import GridConnection, SetLimit, SubMeter

pytest.importorskip("scipy", reason="the reference LP needs scipy/HiGHS")
from topology_reference import reference_lp  # noqa: E402

H = Horizon(dt=0.5, hours=24)
FC = demo_forecasts(H, solar_peak_kw=8.0)
B2 = BatteryConfig(capacity_kwh=6.0, p_charge_max_kw=3.0, p_discharge_max_kw=3.0)
ARRAYS = replace(FC, pv_arrays={"carport": 0.5 * FC.solar, "garage_roof": 0.3 * FC.solar})


def _site(**kw):
    return SiteConfig(horizon=H, battery=BatteryConfig(), batteries=kw.pop("batteries", (B2,)),
                      water_heater=None, hvac=None, **kw)


SITES = {
    "nested limits": (_site(submeters=(
        hybrid_inverter(("battery",), 4.0, 4.0, 0.97, 0.97),
        SubMeter("garage", (), max_import_kw=3.0, max_export_kw=0.0),
        SubMeter("ev", ("battery1",), max_import_kw=2.0, parent="garage"))), FC),
    "a second hybrid inverter on the first's backup port": (_site(submeters=(
        SubMeter("inv1_ac", (), max_import_kw=11.5, max_export_kw=11.5),
        SubMeter("inv1_dc", ("battery", "pv"), max_import_kw=5.0, max_export_kw=5.0,
                 eta_export=0.97, eta_import=0.97, parent="inv1_ac"),
        SubMeter("backup", (), max_import_kw=7.0, max_export_kw=7.0, parent="inv1_ac"),
        SubMeter("inv2_dc", ("battery1", "garage_roof"), max_import_kw=3.0, max_export_kw=3.0,
                 eta_export=0.96, eta_import=0.96, parent="backup"))), ARRAYS),
    "two PV arrays, curtailment off": (_site(batteries=(), grid=GridLimits(allow_curtailment=False),
                                             submeters=(hybrid_inverter(("battery",), 3.0, 3.0, 0.97, 0.97),)),
                                       replace(FC, pv_arrays={"carport": 0.5 * FC.solar})),
    "a second grid connection": (_site(connections=(GridConnection("hp_meter", GridLimits(max_import_kw=4.0),
                                                                   members=("battery1",)),)),
                                 replace(FC, tariffs={"hp_meter": (0.7 * FC.buy, np.minimum(0.7 * FC.buy, 0.0))})),
    "a limit on a set across the tree": (_site(
        submeters=(hybrid_inverter(("battery",), 4.0, 4.0, 0.97, 0.97),),
        set_limits=(SetLimit("L1", ("battery", "battery1"), max_import_kw=3.0, max_export_kw=2.0),)), FC),
}


@pytest.mark.parametrize("name", list(SITES))
def test_dw_matches_an_independent_lp(name):
    """Dantzig-Wolfe's plan, scored through the tree, its bound and an
    independent LP of the same site all agree; nothing is breached."""
    site, fc = SITES[name]
    r = DWCoordinator(site, fc).run()
    ref = reference_lp(site, fc)
    assert r.upper == pytest.approx(ref, abs=1e-4)
    assert r.lower == pytest.approx(ref, abs=1e-4)
    flows = site_meter(site, fc, {k: c.power for k, c in r.plan.items()})
    assert flows.breach_kwh.sum() == pytest.approx(0.0, abs=1e-6)
    for sm in site.submeters:                 # every connection within its ratings
        if sm.max_export_kw is not None:
            assert flows.ac[sm.name].max() <= sm.max_export_kw + 1e-6
        if sm.max_import_kw is not None:
            assert -flows.ac[sm.name].min() <= sm.max_import_kw + 1e-6


def test_a_set_limit_binds_and_is_priced():
    site, fc = SITES["a limit on a set across the tree"]
    r = DWCoordinator(site, fc).run()
    total = r.plan["battery"].power + r.plan["battery1"].power
    assert total.max() <= 3.0 + 1e-6 and total.min() >= -2.0 - 1e-6
    assert r.limit_prices["L1"].max() > 0.0          # it binds somewhere, at a premium


@pytest.mark.parametrize("solver", ["numpy", "highs"])
def test_a_panel_that_cannot_export_prices_its_loads_at_the_meter(solver):
    """Loads behind a no-backfeed panel, a breaker inside it: while nothing
    there draws, any price from the meter's minus the breach price up to the
    meter's is an optimal dual, and a solver returns one end. The loop also
    prices at the same optimum's duals nearest the pass-through prices, so it
    closes its gap (it stalled for 40 iterations on the bottom of that range),
    and those are the local prices it reports. (Exactness behind such a panel
    is test_dw_matches_an_independent_lp's "nested limits": the thermal DPs
    are exact only up to their grid.)"""
    site = SiteConfig(horizon=H, battery=BatteryConfig(), water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
                      submeters=(hybrid_inverter(("battery",), 4.0, 4.0, 0.97, 0.97),
                                 group_limit("garage", ("hvac",), max_kw=7.4, min_kw=0.0),
                                 group_limit("heat", ("water_heater",), max_kw=3.5, parent="garage")))
    r = DWCoordinator(site, FC, solver=solver).run()
    assert r.stop_reason == "converged" and len(r.history) < 15
    assert r.relaxed - r.lower < 1e-6
    for node in ("garage", "heat"):
        assert np.all(r.local_prices[node] >= r.prices - 1e-6)


@pytest.mark.parametrize("name", list(SITES))
def test_bidding_batteries_keep_a_valid_bound(name):
    """Batteries as DP blocks, priced at their bus's price plus their set
    limits': the bound stays below the plan."""
    site, fc = SITES[name]
    r = DWCoordinator(replace(site, water_heater=WaterHeaterConfig()), fc, battery_in_master=False).run()
    assert r.lower <= r.upper + 1e-6


@pytest.mark.parametrize("name", list(SITES))
def test_the_saving_split_adds_up(name):
    site, fc = SITES[name]
    co, co_dark = DWCoordinator(site, fc), DWCoordinator(site, dark(fc))
    tot = ledger(co, co.run().plan, co_dark, co_dark.run().plan)["totals"]
    assert tot["net_gain"] + tot["private_cost_change"] == pytest.approx(
        tot["house_bill_before"] - tot["house_bill_after"], abs=1e-9)


@pytest.mark.parametrize("name", ["nested limits", "a limit on a set across the tree"])
def test_admm_steers_by_the_tree(name):
    """ADMM with a net per bus and per set limit: no cheaper than DW's bound,
    and (nearly) within every limit - its iterations price them, not only
    the final score."""
    site, fc = SITES[name]
    a = coordinate(site, fc)
    flows = site_meter(site, fc, {k: v.power for k, v in a.devices.items()})
    r = DWCoordinator(site, fc).run()
    assert a.total_objective >= r.lower - 1e-6
    assert flows.breach_kwh.sum() < 0.05
    assert a.total_objective < r.upper + 1.0       # in ADMM's usual range of the optimum


def test_the_fast_tier_follows_the_chain():
    """The battery behind the second inverter, behind the backup panel: the
    pricing solve and the policy carry a three-level chain, which survives a
    save and load."""
    site, fc = SITES["a second hybrid inverter on the first's backup port"]
    site = replace(site, battery=B2, batteries=(BatteryConfig(),),
                   submeters=tuple(replace(sm, members=tuple("battery" if m == "battery1" else
                                                            "battery1" if m == "battery" else m
                                                            for m in sm.members)) for sm in site.submeters))
    res = dw_plan(site, fc)
    snap = PolicySnapshot.from_result(site, fc, res)
    assert snap.bus is not None and len(snap.bus.levels) == 3     # inv2_dc, backup, inv1_ac
    assert np.all(np.isfinite([action(snap, t, 3.0) for t in range(H.steps)]))


def test_a_battery_on_a_second_connection_meets_its_tariff():
    site, fc = SITES["a second grid connection"]
    site = replace(site, connections=(GridConnection("hp_meter", GridLimits(max_import_kw=4.0), members=("battery",)),))
    res = dw_plan(site, fc)
    assert res.battery_tariff is not None
    assert np.allclose(res.battery_tariff[0], fc.tariffs["hp_meter"][0])
    snap = PolicySnapshot.from_result(site, fc, res)
    assert np.allclose(snap.buy, fc.tariffs["hp_meter"][0])


def test_the_topology_is_read_once_and_right():
    site, fc = SITES["a second hybrid inverter on the first's backup port"]
    topo = topology(site, fc)
    assert topo.order.index("inv2_dc") < topo.order.index("backup") < topo.order.index("inv1_ac")
    assert topo.chain("inv2_dc") == ("inv2_dc", "backup", "inv1_ac")
    assert topo.pv_at["inv1_dc"] == ("pv",) and topo.pv_at["inv2_dc"] == ("garage_roof",)
    assert topo.pv_at["grid"] == ("carport",)


@pytest.mark.parametrize("bad, message", [
    ((SubMeter("a", (), parent="b"), SubMeter("b", (), parent="a")), "loop"),
    ((SubMeter("a", (), parent="nowhere"),), "no parent"),
    ((SubMeter("a", ("battery",)), SubMeter("b", ("battery",))), "behind two"),
])
def test_validation(bad, message):
    with pytest.raises(ValueError, match=message):
        _site(submeters=bad).validate()
