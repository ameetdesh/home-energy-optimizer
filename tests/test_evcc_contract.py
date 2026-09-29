"""evcc optimizer wire contract.

The contract is transcribed from the generated Go client in
`github.com/evcc-io/optimizer/client` (pinned by commit in evcc's go.mod), so
these tests are what stands between "we implemented a JSON API" and "a stock
evcc binary can actually talk to it".

The unit conversions are the whole risk here: evcc speaks W/Wh and per-Wh
prices, this package speaks kW/kWh and per-kWh prices, and the arrays are
energies per slot rather than powers despite `charging_power`'s name.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemspolicy.evcc import (
    ContractError,
    W_PER_KW,
    health,
    horizon_from_dt,
    optimize_charge_schedule,
    request_to_site,
)


def make_request(
    slots: int = 96,
    dt: int = 900,
    first_slot: int | None = None,
    capacity_wh: float = 10_000.0,
    peak_price: float = 0.40,
    offpeak_price: float = 0.15,
    solar_peak_wh: float = 1_000.0,
    export_price: float = 0.08,
    **overrides,
) -> dict:
    """A realistic OptimizationInput, shaped the way evcc builds one."""
    dts = [dt] * slots
    if first_slot is not None:
        dts[0] = first_slot  # evcc's first slot is the rest of the quarter-hour

    hours = np.arange(slots) * dt / 3600.0
    p_n = np.where(hours % 24 < 7, offpeak_price, peak_price) / W_PER_KW
    p_e = np.full(slots, export_price / W_PER_KW)
    ft = solar_peak_wh * np.clip(np.sin((hours % 24 - 6) / 12 * np.pi), 0, None)
    gt = np.full(slots, 250.0)

    req = {
        "batteries": [
            {
                "c_max": 5000.0,
                "c_min": 0.0,
                "d_max": 5000.0,
                "s_initial": capacity_wh * 0.5,
                "s_min": 0.0,
                "s_max": capacity_wh,
                "s_capacity": capacity_wh,
                "p_a": offpeak_price / W_PER_KW,
            }
        ],
        "eta_c": 0.95,
        "eta_d": 0.95,
        "grid": {"p_max_imp": 17250.0, "p_max_exp": 10000.0},
        "time_series": {
            "dt": dts,
            "ft": [float(v) for v in ft],
            "gt": [float(v) for v in gt],
            "p_N": [float(v) for v in p_n],
            "p_E": [float(v) for v in p_e],
        },
    }
    req.update(overrides)
    return req


# --------------------------------------------------------------------------
# Horizon / dt handling
# --------------------------------------------------------------------------


def test_uniform_dt_is_detected():
    h, dt_h, uniform = horizon_from_dt([900] * 96)
    assert uniform
    assert dt_h == pytest.approx(0.25)
    assert h.steps == 96


def test_evccs_short_first_slot_is_reported_not_hidden():
    """evcc's first slot is the remainder of the current quarter-hour.

    `Horizon` is uniform, so this cannot be represented exactly. It must be
    surfaced rather than silently averaged away - a plan that reports Optimal
    with every slot offset is precisely the failure EMHASS's AGENTS.md warns
    about.
    """
    _, dt_h, uniform = horizon_from_dt([420] + [900] * 95)
    assert not uniform
    assert dt_h == pytest.approx(0.25), "modal step, not the odd first one"


def test_empty_or_bad_dt_is_rejected():
    with pytest.raises(ContractError, match="empty"):
        horizon_from_dt([])
    with pytest.raises(ContractError, match="non-positive"):
        horizon_from_dt([900, 0, 900])


# --------------------------------------------------------------------------
# Unit conversion — the main risk
# --------------------------------------------------------------------------


def test_prices_convert_from_per_wh_to_per_kwh():
    req = make_request(peak_price=0.40, offpeak_price=0.15)
    _, fc, _ = request_to_site(req)
    assert fc.buy.max() == pytest.approx(0.40)
    assert fc.buy.min() == pytest.approx(0.15)
    assert fc.sell[0] == pytest.approx(0.08)


def test_energy_per_slot_converts_to_average_kw():
    """250 Wh in a 900 s slot is 1 kW, not 0.25 kW."""
    req = make_request(slots=8, dt=900)
    _, fc, _ = request_to_site(req)
    assert fc.load[0] == pytest.approx(1.0)


def test_a_short_first_slot_is_not_read_as_a_demand_dip():
    """Same Wh in a shorter slot means HIGHER average power.

    Dividing by a fixed dt instead of each slot's own duration would make
    evcc's partial first slot look like a lull in demand, and the plan would
    start by under-serving the house.
    """
    req = make_request(slots=8, dt=900, first_slot=450)
    _, fc, _ = request_to_site(req)
    assert fc.load[0] == pytest.approx(2.0)
    assert fc.load[1] == pytest.approx(1.0)


def test_battery_converts_from_wh_to_kwh():
    req = make_request(capacity_wh=12_000.0)
    site, _, meta = request_to_site(req)
    assert site.battery.capacity_kwh == pytest.approx(12.0)
    assert site.battery.p_charge_max_kw == pytest.approx(5.0)
    assert site.battery.soc_initial_frac == pytest.approx(0.5)
    assert meta["s_capacity_kwh"] == pytest.approx(12.0)


def test_split_efficiency_is_carried_through():
    req = make_request()
    req["eta_c"], req["eta_d"] = 0.97, 0.93
    site, _, _ = request_to_site(req)
    assert site.battery.eta_c == pytest.approx(0.97)
    assert site.battery.eta_d == pytest.approx(0.93)
    assert site.battery.round_trip == pytest.approx(0.97 * 0.93)


def test_p_a_becomes_the_linear_terminal_price():
    """evcc's "monetary value of stored energy per Wh at end of horizon" is
    exactly this package's linear terminal price, once rescaled."""
    req = make_request()
    req["batteries"][0]["p_a"] = 0.22 / W_PER_KW
    site, _, _ = request_to_site(req)
    assert site.battery.terminal_mode == "linear"
    assert site.battery.terminal_price == pytest.approx(0.22)


def test_s_min_offset_is_applied_and_undone():
    """A battery usable only between 1 and 9 kWh has 8 kWh of usable capacity,
    and the SoC reported back must be absolute again."""
    req = make_request(capacity_wh=10_000.0)
    req["batteries"][0].update({"s_min": 1000.0, "s_max": 9000.0, "s_initial": 5000.0})
    site, _, meta = request_to_site(req)
    assert site.battery.capacity_kwh == pytest.approx(8.0)
    assert site.battery.soc_initial_frac == pytest.approx(0.5)

    resp = optimize_charge_schedule(req)
    soc = np.array(resp["batteries"][0]["state_of_charge"])
    assert soc.min() >= 1000.0 - 1e-3
    assert soc.max() <= 9000.0 + 1e-3


# --------------------------------------------------------------------------
# Contract validation
# --------------------------------------------------------------------------


def test_missing_required_fields_are_rejected():
    for field in ("batteries", "time_series"):
        req = make_request()
        del req[field]
        with pytest.raises(ContractError, match=field):
            request_to_site(req)


def test_wrong_series_length_is_rejected():
    req = make_request(slots=96)
    req["time_series"]["gt"] = [1.0] * 95
    with pytest.raises(ContractError, match="gt has 95"):
        request_to_site(req)


def test_multi_battery_is_optimised():
    """evcc sends one BatteryConfig per stationary battery AND one per
    loadpoint (core/site_optimizer.go:532,564), so anyone with a charger sends
    at least two. Refusing them made the integration a no-op for evcc's core
    audience."""
    req = make_request(slots=48)
    req["batteries"].append(dict(req["batteries"][0], s_initial=2000.0))
    site, _, meta = request_to_site(req)

    assert meta["n_batteries"] == 2
    assert len(site.battery_list) == 2
    resp = optimize_charge_schedule(req)
    assert len(resp["batteries"]) == 2
    for b in resp["batteries"]:
        assert len(b["state_of_charge"]) == 48


def test_multi_battery_results_keep_request_order():
    """evcc matches results back to its own devices positionally
    (applyOptimizerResult walks details index-for-index), so ordering is
    load-bearing. Give the two batteries different capacities and check each
    result stays inside its own bounds."""
    req = make_request(slots=48)
    req["batteries"][0].update(s_max=10_000.0, s_capacity=10_000.0, s_initial=5_000.0)
    req["batteries"].append(
        dict(req["batteries"][0], s_max=60_000.0, s_capacity=60_000.0, s_initial=30_000.0)
    )
    resp = optimize_charge_schedule(req)

    soc0 = np.array(resp["batteries"][0]["state_of_charge"])
    soc1 = np.array(resp["batteries"][1]["state_of_charge"])
    assert soc0.max() <= 10_000.0 + 1e-3, "first result must be the 10 kWh battery"
    assert soc1.max() > 10_000.0, "second result must be the 60 kWh battery"


def test_loadpoint_shaped_battery_is_accepted():
    """A loadpoint arrives as a battery that only charges: d_max = 0, no V2G."""
    req = make_request(slots=48)
    req["batteries"].append(
        {
            "c_max": 11_000.0, "c_min": 1_400.0, "d_max": 0.0,
            "s_initial": 10_000.0, "s_min": 0.0, "s_max": 60_000.0,
            "s_capacity": 60_000.0, "charge_from_grid": True,
        }
    )
    resp = optimize_charge_schedule(req)
    d = np.array(resp["batteries"][1]["discharging_power"])
    assert d.max() < 1e-3, "a d_max=0 battery must never discharge"


# --------------------------------------------------------------------------
# p_demand and s_goal - how evcc expresses "the car needs charging"
# --------------------------------------------------------------------------


def test_p_demand_forces_charging():
    """Without honouring p_demand an EV would simply never charge, because a
    loadpoint carries its entire requirement there."""
    n = 48
    req = make_request(slots=n)
    demand = [0.0] * n
    for t in range(8, 20):
        demand[t] = 2_000.0  # Wh per 900 s slot -> 8 kW
    req["batteries"].append(
        {
            "c_max": 11_000.0, "c_min": 0.0, "d_max": 0.0,
            "s_initial": 0.0, "s_min": 0.0, "s_max": 60_000.0,
            "s_capacity": 60_000.0, "p_demand": demand,
        }
    )
    resp = optimize_charge_schedule(req)
    c = np.array(resp["batteries"][1]["charging_power"])
    assert c[8:20].min() >= 2_000.0 - 1.0, "must charge at least the demanded energy"
    assert c[:8].sum() + c[20:].sum() >= 0.0


def test_p_demand_is_energy_per_slot_not_power():
    """2000 Wh in a 900 s slot is 8 kW. Reading it as 2 kW would under-charge
    by 4x and the car would miss its deadline."""
    n = 16
    req = make_request(slots=n)
    demand = [0.0] * n
    demand[4] = 2_000.0
    req["batteries"].append(
        {
            "c_max": 11_000.0, "c_min": 0.0, "d_max": 0.0,
            "s_initial": 0.0, "s_min": 0.0, "s_max": 60_000.0,
            "s_capacity": 60_000.0, "p_demand": demand,
        }
    )
    site, _, _ = request_to_site(req)
    ev = site.battery_list[1]
    assert ev.min_charge_kw[4] == pytest.approx(8.0)
    assert ev.min_charge_kw[3] == pytest.approx(0.0)


def test_c_min_is_a_semi_continuous_floor_not_a_minimum():
    """A wallbox is OFF or at >= c_min; it cannot modulate below.

    Captured from a live evcc: a loadpoint arrives with c_min = 4140 W
    (Voltage x minCurrent x minPhases). Planning 2 kW into it produces a
    schedule the hardware cannot execute.
    """
    n = 48
    req = make_request(slots=n)
    req["batteries"].append(
        {
            "c_max": 11_040.0, "c_min": 4_140.0, "d_max": 0.0,
            "s_initial": 18_000.0, "s_min": 0.0, "s_max": 60_000.0,
            "s_capacity": 60_000.0, "charge_from_grid": True,
        }
    )
    site, _, _ = request_to_site(req)
    assert site.battery_list[1].charge_deadband_kw == pytest.approx(4.14)

    resp = optimize_charge_schedule(req)
    c = np.array(resp["batteries"][1]["charging_power"])
    slot_h = 900 / 3600
    kw = c / 1000.0 / slot_h
    forbidden = kw[(kw > 1e-6) & (kw < 4.14 - 1e-6)]
    assert forbidden.size == 0, f"{forbidden.size} setpoints the charger cannot reach"


def test_zero_c_min_leaves_the_grid_continuous():
    """A stationary battery sends c_min = 0 and must keep full modulation."""
    req = make_request(slots=32)
    assert req["batteries"][0]["c_min"] == 0.0
    site, _, _ = request_to_site(req)
    assert site.battery.charge_deadband_kw == 0.0


def test_s_goal_pulls_the_state_of_charge_up():
    """s_goal is how evcc says '60% by 07:00'. Soft, so an unreachable goal
    degrades instead of returning infeasible."""
    n = 48
    req = make_request(slots=n)
    goal = [0.0] * 24 + [8_000.0] * 24
    req["batteries"][0].update(s_initial=2_000.0, s_min=0.0, s_max=10_000.0, p_demand=None)
    req["batteries"][0]["s_goal"] = goal

    resp = optimize_charge_schedule(req)
    soc = np.array(resp["batteries"][0]["state_of_charge"])
    assert soc[30] > 6_000.0, f"expected the goal to pull SoC up, got {soc[30]}"


def test_s_goal_is_shifted_into_the_usable_window():
    """s_goal is absolute Wh; the model works in [s_min, s_max] coordinates."""
    n = 16
    req = make_request(slots=n)
    req["batteries"][0].update(s_min=1_000.0, s_max=9_000.0, s_initial=5_000.0)
    req["batteries"][0]["s_goal"] = [6_000.0] * n
    site, _, _ = request_to_site(req)
    assert site.battery.soc_goal_kwh[0] == pytest.approx(5.0)  # 6 kWh - 1 kWh offset


def test_per_slot_arrays_must_match_the_horizon():
    for field in ("p_demand", "s_goal"):
        req = make_request(slots=48)
        req["batteries"][0][field] = [1.0] * 47
        with pytest.raises(ContractError, match=f"{field} has 47"):
            request_to_site(req)


def test_s_initial_above_capacity_is_rejected():
    req = make_request()
    req["batteries"][0]["s_initial"] = 99_000.0
    with pytest.raises(ContractError, match="exceeds s_capacity"):
        request_to_site(req)


def test_no_batteries_is_rejected():
    req = make_request()
    req["batteries"] = []
    with pytest.raises(ContractError, match="nothing to optimise"):
        request_to_site(req)


# --------------------------------------------------------------------------
# Response shape
# --------------------------------------------------------------------------


def test_response_matches_the_optimization_result_schema():
    resp = optimize_charge_schedule(make_request())
    for key in (
        "status",
        "objective_value",
        "batteries",
        "flow_direction",
        "grid_import",
        "grid_export",
        "grid_import_overshoot",
        "grid_export_overshoot",
        "limit_violations",
    ):
        assert key in resp, f"missing {key}"
    assert resp["status"] in ("Optimal", "Feasible", "Infeasible", "Unbounded", "Undefined", "Not Solved")

    b = resp["batteries"][0]
    n = len(make_request()["time_series"]["dt"])
    for key in ("charging_power", "discharging_power", "state_of_charge"):
        assert len(b[key]) == n, f"{key} length"


def test_charge_and_discharge_are_non_negative_and_disjoint():
    """evcc splits the signed series into two non-negative ones."""
    resp = optimize_charge_schedule(make_request())
    c = np.array(resp["batteries"][0]["charging_power"])
    d = np.array(resp["batteries"][0]["discharging_power"])
    assert c.min() >= 0.0 and d.min() >= 0.0
    assert np.all((c < 1e-9) | (d < 1e-9)), "never charging and discharging at once"


def test_flow_direction_encoding():
    """0 = import, 1 = export, and it must agree with the grid arrays."""
    resp = optimize_charge_schedule(make_request())
    flow = np.array(resp["flow_direction"])
    imp = np.array(resp["grid_import"])
    exp = np.array(resp["grid_export"])
    assert set(np.unique(flow)).issubset({0, 1})
    assert np.all(imp[flow == 1] < 1e-6)
    assert np.all(exp[flow == 0] < 1e-6)


def test_objective_is_a_benefit_not_a_cost():
    """evcc documents objective_value as an economic benefit; ours is a cost,
    so the sign must flip or evcc will read a good plan as a bad one."""
    req = make_request(peak_price=0.60, offpeak_price=0.05)
    resp = optimize_charge_schedule(req)
    assert resp["objective_value"] == pytest.approx(-_net_cost_of(req), abs=1e-4)


def _net_cost_of(req: dict) -> float:
    from hemspolicy import coordinate

    site, fc, _ = request_to_site(req)
    return coordinate(site, fc).net_cost


def test_energy_balance_holds_on_the_wire():
    """import - export must equal load - solar + charge - discharge, in Wh."""
    req = make_request()
    resp = optimize_charge_schedule(req)
    ts = req["time_series"]
    gt = np.array(ts["gt"])
    ft = np.array(ts["ft"])
    c = np.array(resp["batteries"][0]["charging_power"])
    d = np.array(resp["batteries"][0]["discharging_power"])
    imp = np.array(resp["grid_import"])
    exp = np.array(resp["grid_export"])
    residual = (imp - exp) - (gt - ft + c - d)
    assert np.abs(residual).max() < 1e-3, f"max residual {np.abs(residual).max()} Wh"


def test_short_first_slot_scales_its_energy_back_correctly():
    """Round-trip: a 450 s first slot must report Wh for 450 s, not 900 s."""
    req = make_request(slots=16, first_slot=450)
    resp = optimize_charge_schedule(req)
    ts = req["time_series"]
    gt, ft = np.array(ts["gt"]), np.array(ts["ft"])
    c = np.array(resp["batteries"][0]["charging_power"])
    d = np.array(resp["batteries"][0]["discharging_power"])
    imp = np.array(resp["grid_import"])
    exp = np.array(resp["grid_export"])
    residual = (imp - exp) - (gt - ft + c - d)
    assert np.abs(residual[0]) < 1e-3, "first slot must balance too"


def test_approximations_are_reported():
    req = make_request(first_slot=450)
    req["batteries"][0]["c_priority"] = 2
    resp = optimize_charge_schedule(req)
    notes = " ".join(resp["_hems_policy_notes"])
    assert "non-uniform dt" in notes
    assert "c_priority" in notes


def test_grid_overshoot_is_reported_when_limits_bind():
    req = make_request(slots=32)
    req["grid"]["p_max_imp"] = 100.0  # absurdly tight, forces a breach
    resp = optimize_charge_schedule(req)
    assert resp["limit_violations"]["grid_import_limit_exceeded"] is True
    assert np.array(resp["grid_import_overshoot"]).sum() > 0


def test_export_above_import_is_an_arbitrage_and_is_taken():
    """A tariff where export pays more than import is free money, and the
    optimiser is right to take it.

    Worth pinning because it looks like a bug: with a cheap night rate and a
    flat feed-in tariff the plan discharges overnight rather than saving the
    energy for the evening peak. It is the tariff that is odd, not the plan.
    (bench/reference.py's LP refuses such inputs outright, since the grid split
    would need binaries to stay exact.)
    """
    req = make_request(slots=48, offpeak_price=0.05, peak_price=0.60, export_price=0.08)
    resp = optimize_charge_schedule(req)
    d = np.array(resp["batteries"][0]["discharging_power"])
    hours = np.arange(48) * 0.25
    assert d[hours < 7].sum() > 0, "should exploit sell > buy overnight"


def test_health_endpoint():
    h = health()
    assert h["status"] == "ok"
    assert "hems-policy" in h["message"]


def test_plan_is_economically_sensible_without_pv():
    """End-to-end textbook check: charge off-peak, discharge on-peak.

    Deliberately PV-free. With solar in the mix the optimal plan is NOT the
    textbook one - see test_pv_makes_overnight_discharge_optimal - and the
    export price must also stay below the off-peak import price or the tariff
    itself contains an arbitrage.
    """
    req = make_request(
        slots=96, offpeak_price=0.20, peak_price=0.60, export_price=0.05, solar_peak_wh=0.0
    )
    resp = optimize_charge_schedule(req)
    c = np.array(resp["batteries"][0]["charging_power"])
    d = np.array(resp["batteries"][0]["discharging_power"])
    hours = np.arange(96) * 0.25
    cheap = hours < 7
    assert c[cheap].sum() > c[~cheap].sum(), "should charge mostly off-peak"
    assert d[~cheap].sum() > d[cheap].sum(), "should discharge mostly on-peak"


def test_pv_makes_overnight_discharge_optimal():
    """Documents a plan that looks wrong and is right.

    With abundant midday PV the optimiser drains the battery overnight to cover
    load at 0.20/kWh, then refills from solar surplus whose opportunity cost is
    only the 0.05 export price - cheaper than buying at 0.20. So it discharges
    during the CHEAP window, which reads as a bug until you price the refill.

    Pinned because the obvious "charge cheap, discharge expensive" assertion
    fails here, and the temptation is to call the solver broken.
    """
    req = make_request(
        slots=96, offpeak_price=0.20, peak_price=0.60, export_price=0.05, solar_peak_wh=1_000.0
    )
    resp = optimize_charge_schedule(req)
    d = np.array(resp["batteries"][0]["discharging_power"])
    c = np.array(resp["batteries"][0]["charging_power"])
    hours = np.arange(96) * 0.25

    assert d[hours < 7].sum() > 0, "expected overnight discharge"
    solar_window = (hours >= 8) & (hours <= 16)
    assert c[solar_window].sum() > 0, "expected a midday refill from PV"
    # And it must still be profitable overall.
    assert resp["objective_value"] > 0


# --------------------------------------------------------------------------
# Against a request captured from a LIVE evcc instance
# --------------------------------------------------------------------------


def _real_request() -> dict:
    import json
    import pathlib

    p = pathlib.Path(__file__).parent / "fixtures" / "evcc_request_real.json"
    return json.loads(p.read_text())


def test_real_evcc_request_is_handled():
    """The fixture is a verbatim capture from a running evcc (see docs/PLAN.md
    Phase 4), not something we invented. Everything below is a property of what
    evcc actually sent, so it is the strongest available check on the mapping.
    """
    req = _real_request()
    ts = req["time_series"]

    # Properties of the real request, asserted so a future fixture swap is visible.
    assert len(ts["dt"]) == 672, "evcc sent 7 days of quarter-hours"
    assert ts["dt"][0] == 810, "and a SHORT first slot - the rest of the quarter"
    assert req["grid"] == {}, "grid limits may be entirely absent"
    b = req["batteries"][0]
    assert b["s_min"] == 2000 and b["s_max"] == 9500 and b["s_capacity"] == 10000

    resp = optimize_charge_schedule(req)
    assert resp["status"] == "Optimal"
    for key in ("charging_power", "discharging_power", "state_of_charge"):
        assert len(resp["batteries"][0][key]) == 672


def test_real_request_respects_the_reported_soc_window():
    """evcc's s_min/s_max are a usable window inside a larger pack. The SoC we
    return must be absolute Wh inside that window, or evcc's forecast extremes
    (batteryForecastSocExtremes) will report nonsense."""
    resp = optimize_charge_schedule(_real_request())
    soc = np.array(resp["batteries"][0]["state_of_charge"])
    assert soc.min() >= 2000 - 1e-3
    assert soc.max() <= 9500 + 1e-3


def test_real_request_solves_fast_enough_for_the_slot_cadence():
    """evcc re-optimises once per 15-minute slot, and a self-hosted optimizer
    gets the FULL forecast length rather than the 2-day cap evcc applies only
    to its own cloud endpoint - so 672 slots is the realistic size, not 192."""
    import time

    t0 = time.perf_counter()
    optimize_charge_schedule(_real_request())
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"{elapsed:.1f}s is too slow for a 15-minute cadence"


def test_real_request_energy_balance():
    req = _real_request()
    resp = optimize_charge_schedule(req)
    ts = req["time_series"]
    residual = (
        np.array(resp["grid_import"])
        - np.array(resp["grid_export"])
        - (
            np.array(ts["gt"])
            - np.array(ts["ft"])
            + np.array(resp["batteries"][0]["charging_power"])
            - np.array(resp["batteries"][0]["discharging_power"])
        )
    )
    assert np.abs(residual).max() < 1e-2


def test_real_loadpoint_request_is_handled():
    """Second live capture, this time with a charger + vehicle configured.

    evcc modelled the LOADPOINT as a battery - `d_max: 0` (no V2G),
    `charge_from_grid: true`, `c_min: 4140` (the charger's physical floor) and
    `s_initial: 18000` (30% of a 60 kWh car). This is the shape every evcc user
    with a charger sends, and refusing it made the integration a no-op for them.
    """
    import json
    import pathlib

    req = json.loads(
        (pathlib.Path(__file__).parent / "fixtures" / "evcc_request_loadpoint.json").read_text()
    )
    b = req["batteries"][0]
    assert b["d_max"] == 0, "a loadpoint cannot discharge"
    assert b["c_min"] > 0, "a loadpoint has a physical minimum charge power"
    assert b["charge_from_grid"] is True

    site, _, _ = request_to_site(req)
    assert site.battery.charge_deadband_kw == pytest.approx(b["c_min"] / 1000.0)
    assert site.battery.p_discharge_max_kw < 1e-6

    resp = optimize_charge_schedule(req)
    assert resp["status"] == "Optimal"
    d = np.array(resp["batteries"][0]["discharging_power"])
    assert d.max() < 1e-3, "must never discharge a car"
