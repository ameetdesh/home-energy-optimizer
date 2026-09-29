"""Dantzig-Wolfe behind the Home Assistant and evcc integrations.

`hemspolicy.plan()` must hand both integrations the same object ADMM does - a
CoordinationResult a PolicySnapshot can be built from - plus the certificate
and the meter price. And an EV must stay an EV under DW: its charge floor
(p_demand), charger minimum (c_min) and SoC goal (s_goal) are honoured by the
device DP, so an EV bids plans rather than sitting in the master as a plain
LP battery.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dw.coordinator import DWCoordinator, dw_coordinate  # noqa: E402
from hemspolicy import (  # noqa: E402
    BatteryConfig, GridLimits, Horizon, HvacConfig, PolicySnapshot, SiteConfig, SocGate,
    WaterHeaterConfig, demo_forecasts, marginal_value, plan,
)
from hemspolicy.coordinate import apply_curtailment  # noqa: E402
from hemspolicy.evcc import ContractError, optimize_charge_schedule  # noqa: E402
from hemspolicy.ha import publish_policy, publish_site  # noqa: E402
from test_evcc_contract import make_request  # noqa: E402
from test_ha import FakeHA  # noqa: E402


def site_with(**kw) -> SiteConfig:
    base = dict(horizon=Horizon(dt=0.25, hours=24.0),
                battery=BatteryConfig(capacity_kwh=10.0, n_states=60, n_actions=31),
                water_heater=WaterHeaterConfig(), hvac=None)
    base.update(kw)
    return SiteConfig(**base)


@pytest.fixture(scope="module")
def dw_run():
    site = site_with()
    fc = demo_forecasts(site.horizon, tariff="day_night")
    return site, fc, plan(site, fc)


# --------------------------------------------------------------------------
# plan(): one entry point, one result type
# --------------------------------------------------------------------------


def test_dw_result_carries_a_certificate(dw_run):
    _, _, res = dw_run
    assert res.method == "dw"
    assert res.lower_bound <= res.plan_objective + 1e-9
    assert res.gap == pytest.approx(res.plan_objective - res.lower_bound)


def test_dw_result_is_self_consistent(dw_run):
    site, fc, res = dw_run
    raw = fc.net_fixed_demand + sum(s.power for s in res.devices.values())
    net, _ = apply_curtailment(raw, fc.solar, fc.sell, site.grid)
    np.testing.assert_allclose(res.net_grid, net)
    assert res.meter_price.shape == (site.horizon.steps,)


def test_the_meter_price_stays_in_the_tariff_box(dw_run):
    """With no grid limit, one more kWh at the meter can cost no more than
    importing it and no less than not exporting it."""
    _, fc, res = dw_run
    assert np.all(res.meter_price <= fc.buy + 1e-6)
    assert np.all(res.meter_price >= np.minimum(fc.sell, 0.0) - 1e-6)


def test_the_fast_tier_works_on_a_dw_plan(dw_run):
    site, fc, res = dw_run
    snap = PolicySnapshot.from_result(site, fc, res)
    lam = [marginal_value(snap, t, 5.0) for t in range(0, site.horizon.steps, 8)]
    assert np.all(np.isfinite(lam))


def test_admm_is_still_one_argument_away():
    site = site_with(water_heater=None)
    fc = demo_forecasts(site.horizon, tariff="day_night")
    res = plan(site, fc, method="admm")
    assert res.method == "admm" and res.gap is None and res.meter_price is None


def test_a_non_convex_tariff_falls_back_to_admm_and_says_so():
    """Export above import makes the master's meter cost non-convex; DW
    refuses it, and a home-automation loop must still get a plan."""
    site = site_with(water_heater=None)
    fc = demo_forecasts(site.horizon, tariff="day_night")
    fc = replace(fc, sell=fc.buy + 0.05)
    res = plan(site, fc)
    assert res.method == "admm"
    assert "DW unavailable" in res.note
    with pytest.raises(ValueError):
        plan(site, fc, fallback=False)


def test_unknown_method_is_rejected():
    site = site_with(water_heater=None)
    with pytest.raises(ValueError):
        plan(site, demo_forecasts(site.horizon), method="milp")


# --------------------------------------------------------------------------
# Which batteries sit in the master
# --------------------------------------------------------------------------


def test_a_plain_battery_is_lp_and_an_ev_bids_plans():
    ev = BatteryConfig(capacity_kwh=60.0, p_discharge_max_kw=1e-9, charge_deadband_kw=4.14)
    site = site_with(water_heater=None, batteries=(ev,))
    co = DWCoordinator(site, demo_forecasts(site.horizon))
    assert [k for k, _ in co.lp_batts] == ["battery"]
    assert [d.key for d in co.devices] == [SiteConfig.battery_key(1)]


def test_soc_gates_send_the_battery_to_the_bidding_side():
    """The master's LP battery carries no gate penalty; the DP does."""
    site = site_with(water_heater=None, soc_gates=(SocGate(hour=18.0, soc_frac=0.8),))
    co = DWCoordinator(site, demo_forecasts(site.horizon))
    assert co.lp_batts == [] and [d.key for d in co.devices] == ["battery"]


def test_a_gated_battery_meets_its_gate_under_dw():
    site = site_with(water_heater=None, soc_gates=(SocGate(hour=18.0, soc_frac=0.8),))
    fc = demo_forecasts(site.horizon, tariff="day_night")
    res = plan(site, fc, fallback=False)
    assert res.devices["battery"].trajectory[72] >= 0.8 * 10.0 - 0.3


# --------------------------------------------------------------------------
# EVs through evcc, planned by DW
# --------------------------------------------------------------------------


def ev_request(limit_w: float = 17_250.0) -> dict:
    """Home battery + a loadpoint with a 4.14 kW charger floor and a goal of
    40 kWh from 07:00; PV at midday."""
    n = 96
    req = make_request(slots=n, solar_peak_wh=1500.0)
    goal = [0.0] * 28 + [40_000.0] * (n - 28)
    req["batteries"].append({
        "c_max": 11_040.0, "c_min": 4_140.0, "d_max": 0.0,
        "s_initial": 18_000.0, "s_min": 0.0, "s_max": 60_000.0, "s_capacity": 60_000.0,
        "s_goal": goal, "charge_from_grid": True,
    })
    req["grid"]["p_max_imp"] = limit_w
    return req


@pytest.fixture(scope="module")
def ev_dw():
    return optimize_charge_schedule(ev_request(5_000.0), method="dw")


def test_dw_meets_the_ev_goal(ev_dw):
    soc = np.array(ev_dw["batteries"][1]["state_of_charge"])
    assert soc[27] >= 40_000.0 - 500.0, "the EV must reach 40 kWh by 07:00"


def test_dw_never_plans_below_the_charger_minimum(ev_dw):
    kw = np.array(ev_dw["batteries"][1]["charging_power"]) / 1000.0 / 0.25
    assert np.sum((kw > 1e-6) & (kw < 4.14 - 1e-6)) == 0


def test_dw_holds_the_import_limit(ev_dw):
    kw = np.array(ev_dw["grid_import"]) / 1000.0 / 0.25
    assert kw.max() <= 5.0 + 1e-6
    assert ev_dw["limit_violations"]["grid_import_limit_exceeded"] is False


def test_dw_response_carries_the_certificate(ev_dw):
    cert = ev_dw["_hems_policy_certificate"]
    assert cert["method"] == "dw"
    assert cert["lower_bound"] <= cert["plan_cost"] + 1e-6
    assert any("enforced in the plan" in n for n in ev_dw["_hems_policy_notes"])


def test_p_demand_is_honoured_under_dw():
    """The pool used to be seeded with an idle plan, which an EV with a
    charge floor cannot run - and the master picked it."""
    n = 48
    req = make_request(slots=n)
    demand = [0.0] * n
    for t in range(8, 20):
        demand[t] = 2_000.0
    req["batteries"].append({
        "c_max": 11_000.0, "c_min": 0.0, "d_max": 0.0, "s_initial": 0.0,
        "s_min": 0.0, "s_max": 60_000.0, "s_capacity": 60_000.0, "p_demand": demand,
    })
    c = np.array(optimize_charge_schedule(req, method="dw")["batteries"][1]["charging_power"])
    assert c[8:20].min() >= 2_000.0 - 1.0


def test_hems_method_selects_the_coordinator(monkeypatch):
    req = make_request(slots=32)
    monkeypatch.setenv("HEMS_METHOD", "admm")
    assert "_hems_policy_certificate" not in optimize_charge_schedule(req)
    monkeypatch.setenv("HEMS_METHOD", "dw")
    assert "_hems_policy_certificate" in optimize_charge_schedule(req)
    monkeypatch.setenv("HEMS_METHOD", "simplex")
    with pytest.raises(ContractError):
        optimize_charge_schedule(req)


# --------------------------------------------------------------------------
# Home Assistant
# --------------------------------------------------------------------------


def test_home_assistant_gets_the_meter_price_and_the_gap(dw_run):
    site, fc, res = dw_run
    ha = FakeHA()
    publish_site(ha, site, fc, res, t=10)
    mp = ha.states["sensor.hems_meter_price"]
    assert mp["state"] == pytest.approx(round(float(res.meter_price[10]), 4))
    assert len(mp["attributes"]["forecast"]) == site.horizon.steps - 10
    gap = ha.states["sensor.hems_plan_gap"]
    assert gap["state"] == pytest.approx(round(res.gap, 4))
    assert gap["attributes"]["method"] == "dw"


def test_home_assistant_says_admm_has_no_certificate():
    site = site_with(water_heater=None)
    fc = demo_forecasts(site.horizon, tariff="day_night")
    ha = FakeHA()
    publish_site(ha, site, fc, plan(site, fc, method="admm"), t=0)
    assert "sensor.hems_meter_price" not in ha.states
    assert ha.states["sensor.hems_plan_gap"]["state"] == "unknown"


@pytest.mark.parametrize("method", ["dw", "admm"])
def test_home_assistant_runs_with_either_coordinator(method):
    """The HA runner's two tiers (tools/ha-lambda-demo/run.py) with each
    coordinator: plan, snapshot, publish the policy and the site, then re-solve
    from a new state - ADMM starting from where the last solve stood."""
    site = site_with(water_heater=None)
    fc = demo_forecasts(site.horizon, tariff="day_night")
    res = plan(site, fc, method=method)
    assert res.method == method and res.battery_pricing is not None
    snap = PolicySnapshot.from_result(site, fc, res)
    ha = FakeHA()
    soe = site.battery.capacity_kwh * site.battery.soc_initial_frac
    out = publish_policy(ha, snap, 0, soe)
    publish_site(ha, site, fc, res, t=0)
    assert np.isfinite(out["lambda"]) and np.isfinite(out["action_kw"])
    assert {"sensor.hems_lambda", "sensor.hems_battery_action", "sensor.hems_plan_gap"} <= set(ha.states)
    assert ("sensor.hems_meter_price" in ha.states) == (method == "dw")
    assert (res.warm_start is not None) == (method == "admm")
    moved = replace(site, battery=replace(site.battery, soc_initial_frac=0.8))
    again = plan(moved, fc, method=method, warm=res.warm_start)
    assert again.method == method
    assert np.isfinite(publish_policy(ha, PolicySnapshot.from_result(moved, fc, again), 4, 8.0)["lambda"])


# --------------------------------------------------------------------------
# The numpy master on the shipped grids
# --------------------------------------------------------------------------


def test_numpy_master_solves_the_shipped_grid_dynamic_day():
    """This master LP stalled a hair short of tolerance: the in-place
    regularisation of the normal equations, scaled by their largest
    diagonal, swamped the small rows late in the solve. HiGHS: -0.18899."""
    site = SiteConfig(horizon=Horizon(dt=0.25, hours=24.0),
                      battery=BatteryConfig(capacity_kwh=10.0, n_states=100, n_actions=51),
                      water_heater=WaterHeaterConfig(), hvac=HvacConfig(), grid=GridLimits())
    r = dw_coordinate(site, demo_forecasts(site.horizon, tariff="dynamic"))
    assert r.upper == pytest.approx(-0.18899, abs=2e-3)
    assert r.lower <= r.upper


# --------------------------------------------------------------------------
# The app's dragged prices and load (dw/webapi.py apply_edits)
# --------------------------------------------------------------------------


def test_dragged_points_reach_the_solve_and_export_stays_below_import():
    from dw.webapi import solve

    buy = [0.3] * 12 + [0.6] * 13
    r = solve({"hours": 24, "grid": 60, "tariff": "flat", "buy_h": buy,
               "sell_h": [0.45] * 25, "load_h": [2.0] * 25})
    assert "error" not in r
    b, s = np.array(r["buy"]), np.array(r["sell"])
    assert b[0] == pytest.approx(0.3) and b[-1] == pytest.approx(0.6)
    assert np.all(s <= b + 1e-12)
    assert np.allclose(r["load"], 2.0)


def test_export_equal_to_import_everywhere_solves():
    """Import-then-export costs nothing then, so the master's optimal set is
    unbounded without finite meter bounds - the numpy master used to stall."""
    from dw.webapi import solve

    r = solve({"hours": 24, "grid": 60, "tariff": "flat", "buy_h": [0.3] * 12 + [0.6] * 13,
               "sell_h": [0.9] * 25, "load_h": [1.0] * 25})
    assert "error" not in r
    assert r["summary"]["lower"] <= r["summary"]["upper"] + 1e-6


# --------------------------------------------------------------------------
# A grid limit is not cheaper to break when energy is cheap
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["dw", "admm"])
@pytest.mark.parametrize("price", [0.0, -0.05])
def test_a_free_or_negative_price_does_not_buy_a_breach(method, price):
    """The breach used to cost a multiple of the slot's import price: free at
    a zero price, a reward at a negative one - measured 9-16 kWh over a 3 kW
    fuse in three hours. Now it is one constant price for the horizon.

    DW meets the limit exactly. ADMM steers by raising a price step by step
    and keeps a residual (measured 0.20-0.27 kWh, peak 3.15-3.2 kW; more
    rounds do not remove it) - the dual-ascent caveat of docs/theory.tex."""
    site = SiteConfig(horizon=Horizon(dt=0.25, hours=24.0),
                      battery=BatteryConfig(capacity_kwh=20.0, p_charge_max_kw=8.0, p_discharge_max_kw=8.0,
                                            soc_initial_frac=0.1, n_states=60, n_actions=31),
                      water_heater=None, hvac=None, grid=GridLimits(max_import_kw=3.0))
    fc = demo_forecasts(site.horizon, tariff="day_night", solar_peak_kw=0.0)
    buy = fc.buy.copy()
    buy[40:52] = price
    fc = replace(fc, buy=buy, sell=np.minimum(fc.sell, buy))
    res = plan(site, fc, method=method)
    over_kwh = float(np.maximum(res.net_grid - 3.0, 0.0).sum() * 0.25)
    assert over_kwh < (1e-3 if method == "dw" else 0.4), f"{over_kwh:.2f} kWh over the limit"


def test_the_breach_price_is_constant_and_scales_with_the_currency():
    from hemspolicy.coordinate import breach_price

    buy = np.array([0.1, -0.2, 0.4])
    assert breach_price(GridLimits(max_import_kw=3.0), buy) == pytest.approx(4.0)
    assert breach_price(GridLimits(max_import_kw=3.0), buy * 80) == pytest.approx(320.0)
    assert breach_price(GridLimits(max_import_kw=3.0, breach_price=2.5), buy) == 2.5
    assert breach_price(GridLimits(max_import_kw=3.0), np.zeros(3)) == pytest.approx(10.0)


def test_the_app_reports_each_iteration_then_finishing():
    """The Solve button reads these: pushed to a callback (the in-browser
    worker) and kept for polling under the request's id (the served page)."""
    from dw import webapi

    seen, polled = [], []

    def cb(it, n, *metrics):
        seen.append((it, n, *metrics))
        polled.append(webapi.call("progress", {"id": "t1"})["progress"])

    r = webapi.call("solve", {"hours": 24, "grid": 60, "tariff": "day_night", "tank_levels": 2,
                              "max_iter": 12, "_progress_id": "t1"}, cb)
    assert "error" not in r
    starts = [a[0] for a in seen if len(a) == 2]         # as each iteration starts, then finishing
    assert starts[:-1] == list(range(1, r["summary"]["iterations"] + 1)) and starts[-1] == -1
    assert all(a[1] == 12 for a in seen)
    # what a poll sees: the counter, with the last iteration's metrics kept
    assert all(p[:2] == [a[0], a[1]] for p, a in zip(polled, seen))
    assert webapi.call("progress", {"id": "t1"})["progress"] is None   # cleared afterwards


# --------------------------------------------------------------------------
# Dashboard: the DW preset when batteries bid plans, live metrics, ADMM mode
# --------------------------------------------------------------------------


def test_batteries_bidding_plans_get_the_fast_preset():
    """Keep every plan and smooth harder when a plain battery bids plans
    (bench/dw_accel.py); otherwise the lean defaults stay."""
    from dw.webapi import build_site
    site = build_site({"n_batteries": 2, "terminal_mode": "linear", "grid": 60})
    fc = demo_forecasts(site.horizon)
    off = DWCoordinator(site, fc, battery_in_master=False, tank_in_master=True)
    off.run(max_iter=2)
    on = DWCoordinator(site, fc, tank_in_master=True)
    on.run(max_iter=2)
    assert (off.pool_used, off.smoothing_used) == ("full", 0.8)
    assert (on.pool_used, on.smoothing_used) == ("active", 0.5)


def test_progress_carries_the_live_metrics():
    from dw import webapi
    seen = []
    webapi.call("solve", {"grid": 60, "max_iter": 6, "battery_in_master": False, "_progress_id": "m"},
                lambda *a: seen.append(a))
    ended = [a for a in seen if len(a) == 5]
    assert ended and all(a[0] >= 1 for a in ended)
    blend, bound, best = ended[-1][2:]
    assert bound <= blend + 1e-9 and best is not None


def test_the_dashboard_can_plan_with_admm_and_its_rho():
    from dw import webapi
    seen = []
    tight = webapi.call("solve", {"method": "admm", "grid": 60, "xrho": 0.5, "rho_adapt": False, "max_iter": 15,
                                  "_progress_id": "a"}, lambda *a: seen.append(a))
    loose = webapi.call("solve", {"method": "admm", "grid": 60, "xrho": 0.03, "rho_adapt": False, "max_iter": 15})
    for r in (tight, loose):
        assert "error" not in r and r["summary"]["method"] == "admm"
        assert r["summary"]["lower"] is None and r["plan"]["price"] is None
        assert [i["index"] for i in r["iterations"]] == list(range(1, r["summary"]["iterations"] + 1))
    assert {i["rho"] for i in tight["iterations"]} == {0.5}
    assert {i["rho"] for i in loose["iterations"]} == {0.03}
    assert any(len(a) == 5 and a[3] is None for a in seen)       # no bound for ADMM


def test_a_dragged_hot_water_draw_reaches_the_tank():
    """draw_h (kW of heat, one point per hour) replaces the draw forecast; a
    bigger evening draw makes the tank heat more."""
    from dw.webapi import solve
    base = {"hours": 24, "grid": 60, "tariff": "flat", "tank_levels": 2, "enable_hvac": False}
    r0 = solve(base)
    heavy = [0.0] * 18 + [6.0, 6.0] + [0.0] * 5
    r1 = solve({**base, "draw_h": heavy})
    assert "error" not in r1 and len(r1["hot_water"]) == r1["steps"]
    assert max(r1["hot_water"]) == pytest.approx(6.0) and min(r1["hot_water"]) >= 0
    tank = lambda r: sum(r["plan"]["devices"]["water_heater"]["power"]) * r["dt"]  # noqa: E731
    assert tank(r1) > tank(r0)


# --------------------------------------------------------------------------
# Who saves what (dw/attribution.py)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["dw", "admm"])
def test_the_ledger_adds_up(method):
    """Against a no-PV baseline: players' bills sum to the house bill before and
    after; the load keeps its bill; solar starts at zero and has no private
    cost; the net gains are the total saving, and the coordination part is the
    saving over thermostats with PV, on the objective."""
    from dw.attribution import baseline
    from dw.coordinator import Column
    from dw.webapi import _site_fc, solve
    p = {"method": method, "grid": 60, "tariff": "day_night", "n_batteries": 2, "tank_levels": 2,
         "max_export_kw": 2.0, "ledger": True, "hours": 24, "max_iter": 30}
    r = solve(p)
    L, t = r["ledger"], r["ledger"]["totals"]
    rows = {row["player"]: row for row in L["rows"]}
    assert t["bill_before"] == pytest.approx(t["house_bill_before"], abs=2e-3)
    assert t["bill_after"] == pytest.approx(t["house_bill_after"], abs=2e-3)
    assert rows["household load"]["bill_change"] == pytest.approx(0, abs=1e-9)
    assert rows["solar"]["bill_before"] == 0 and rows["solar"]["private_cost_change"] == 0
    assert rows["solar"]["net_gain"] > 0
    assert t["net_gain"] == pytest.approx(t["house_bill_before"] - t["house_bill_after"]
                                          - t["private_cost_change"], abs=2e-3)
    # coordination alone = thermostat-with-PV objective - plan objective
    site, fc, _ = _site_fc({**p, "tank_levels": 2})
    co = DWCoordinator(site, fc)
    keys = [k for k in rows if k not in ("household load", "solar")]
    base = {k: baseline(co, k) for k in keys}
    plan_cost = r["summary"]["plan_parts"]["total"]
    base_cost = co.parts({k: Column(pw, tr, 0.0, "b") for k, (pw, tr) in base.items()})["total"]
    assert t["coordination_saving"] == pytest.approx(base_cost - plan_cost, abs=2e-3)


@pytest.mark.parametrize("method", ["dw", "admm"])
def test_min_soe_is_a_floor(method):
    """The reserve holds in every plan, and the page is told where it is."""
    from dw.webapi import solve
    r = solve({"method": method, "grid": 60, "tariff": "dynamic", "n_batteries": 2, "soe_min": 30,
               "hours": 24, "max_iter": 30})
    for k, cap, floor in zip(r["battery_keys"], r["battery_capacities"], r["battery_floors"]):
        assert floor == pytest.approx(0.3 * cap)
        assert min(r["plan"]["devices"][k]["trajectory"]) >= floor - 1e-6


def test_stored_energy_is_valued_at_the_average_import_price():
    from dw.webapi import _site_fc
    site, fc, _ = _site_fc({"tariff": "day_night", "n_batteries": 2})
    assert all(b.terminal_price == pytest.approx(float(np.mean(fc.buy))) for b in site.battery_list)


def test_every_control_the_page_script_wires_exists():
    """The page wires controls by id (sliders through `fmt`, selects through an
    onchange list, the rest through $('#id')); an id with no element throws at
    load and leaves the page dead."""
    import re
    html = (Path(__file__).resolve().parents[1] / "dw" / "gui" / "index.html").read_text()
    ids = set(re.findall(r'\bid="([\w-]+)"', html))
    fmt = re.search(r"const fmt = \{(.*?)\};", html, re.S).group(1)
    sliders = re.findall(r"(\w+):v=>", fmt)
    selects = re.findall(r"'(\w+)'", re.search(r"\[([^\]]*)\]\.forEach\(id => \$\('#'\+id\)\.onchange", html).group(1))
    direct = re.findall(r"\$\('#([\w-]+)'\)", html)
    missing = sorted({i for i in sliders + selects + direct if i not in ids}
                     | {"o_" + i for i in sliders if "o_" + i not in ids})
    assert not missing, missing
