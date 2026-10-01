"""Participants: devices answered through the interface (interface.py) rather
than modelled by the coordinator, plus the master's export ceiling and
end-of-day energy targets."""

from __future__ import annotations

import json

import numpy as np
import pytest

from home_energy_optimizer import BatteryConfig, Horizon, SiteConfig, WaterHeaterConfig
from home_energy_optimizer.dp_thermal import baseline_water_heater, solve_water_heater, wh_discomfort
from home_energy_optimizer.dw.coordinator import DWCoordinator, thermal_terminal
from home_energy_optimizer.interface import (
    Answer,
    Query,
    answer_from_dict,
    answer_to_dict,
    query_from_dict,
    query_to_dict,
    schema,
)
from home_energy_optimizer.profiles import demo_forecasts
from home_energy_optimizer.types import CoordinationConfig, GridLimits


class TankParticipant:
    """The package's own on/off tank, answered through the interface: the same
    DP and the same costs the coordinator uses for a built-in tank, so a plan
    through it must equal the built-in plan."""

    modulating = False

    def __init__(self, cfg: WaterHeaterConfig, horizon: Horizon, fc):
        self.key, self.cfg, self.h, self.fc = "water_heater", cfg, horizon, fc
        self.max_power_kw = cfg.power_kw
        self.ref = float(np.mean(fc.buy))

    def _answer(self, temp, power) -> Answer:
        """An Answer for a tank plan: `temp` (degC, one point more than the
        plan) and `power` (kW); its private cost is discomfort plus the
        end-of-day term, as the coordinator prices a built-in tank."""
        cost = (float(np.sum(wh_discomfort(self.cfg, temp[1:], self.ref, self.h.dt)))
                + thermal_terminal(self.cfg, "water_heater", temp[-1], self.ref))
        return Answer(power, cost, temp)

    def respond(self, q: Query) -> Answer:
        """The tank's DP at the query's prices, metered under the rest of the
        house for a best response."""
        load = q.residual_kw if q.kind == "best_response" else None
        sol = solve_water_heater(self.cfg, self.h, q.price_draw, q.price_supply, self.fc.hot_water_demand,
                                 dp_load=load, ref_price=self.ref)
        return self._answer(sol.trajectory, sol.power)

    def baseline(self) -> Answer:
        """The thermostat plan, as the built-in tank's seed."""
        temp, power = baseline_water_heater(self.cfg, self.h, self.fc.hot_water_demand)
        return self._answer(temp, power)


@pytest.fixture
def day():
    """A day at 15-minute steps, a 10 kWh battery and the default tank."""
    h = Horizon(dt=0.25, hours=24.0)
    return h, BatteryConfig(capacity_kwh=10.0), WaterHeaterConfig()


@pytest.mark.parametrize("tariff", ["day_night", "dynamic"])
def test_a_participant_plans_exactly_as_the_same_device_built_in(day, tariff):
    """The tank as a participant (through Query/Answer) gives the same plan and
    objective as the tank modelled inside the coordinator."""
    h, batt, wh = day
    fc = demo_forecasts(h, tariff=tariff)
    native = DWCoordinator(SiteConfig(horizon=h, battery=batt, water_heater=wh, hvac=None,
                                      coordination=CoordinationConfig()), fc).run()
    via = DWCoordinator(SiteConfig(horizon=h, battery=batt, water_heater=None, hvac=None,
                                   coordination=CoordinationConfig()), fc,
                        participants=[TankParticipant(wh, h, fc)]).run()
    assert via.upper == pytest.approx(native.upper, abs=1e-9)
    assert np.allclose(via.plan["water_heater"].power, native.plan["water_heater"].power)


def test_the_export_ceiling_is_a_hard_cap(day):
    """With export capped at the PV surplus and no curtailment, the house never
    exports more than that surplus in any slot (the battery cannot discharge
    to the grid)."""
    h, batt, _ = day
    fc = demo_forecasts(h, tariff="dynamic", solar_peak_kw=6.0)
    ceiling = np.maximum(fc.solar - fc.load, 0.0)            # export no more than the PV surplus
    site = SiteConfig(horizon=h, battery=batt, water_heater=None, hvac=None,
                      grid=GridLimits(allow_curtailment=False))
    r = DWCoordinator(site, fc, export_ceiling=ceiling).run()
    net = fc.net_fixed_demand + sum(c.power for c in r.plan.values())
    assert np.all(-net <= ceiling + 1e-6)


def test_a_battery_in_the_master_ends_at_its_target(day):
    """A battery held in the master LP with an end-of-day target of 8 kWh
    ends the plan at exactly 8 kWh."""
    h, batt, _ = day
    fc = demo_forecasts(h, tariff="day_night")
    site = SiteConfig(horizon=h, battery=batt, water_heater=None, hvac=None)
    r = DWCoordinator(site, fc, soe_targets={"battery": 8.0}).run()
    assert r.plan["battery"].trajectory[-1] == pytest.approx(8.0, abs=1e-6)


def test_queries_and_answers_round_trip_as_json():
    """A Query and an Answer survive dict -> JSON text -> dict -> object."""
    q = Query("price_response", np.array([0.2, 0.3]), np.array([0.2, 0.3]))
    d = json.loads(json.dumps(query_to_dict(q, step_minutes=15, start="2026-10-01T00:00:00Z")))
    back = query_from_dict(d)
    assert back.kind == q.kind and np.array_equal(back.price_draw, q.price_draw)
    a = Answer(np.array([1.0, -0.5]), 0.42, np.array([5.0, 5.2, 5.0]))
    b = answer_from_dict(json.loads(json.dumps(answer_to_dict(a, solver="test"))))
    assert b.private_cost == a.private_cost and np.array_equal(b.trajectory, a.trajectory)


def test_the_json_follows_the_packaged_schemas():
    """The JSON a query and an answer convert to validates against the
    packaged schemas, and a price response without prices does not."""
    jsonschema = pytest.importorskip("jsonschema")
    q = query_to_dict(Query("best_response", np.ones(3), np.ones(3), residual_kw=np.zeros(3)), step_minutes=30)
    jsonschema.validate(q, schema("device-query"))
    jsonschema.validate(answer_to_dict(Answer(np.zeros(3), 0.0, np.zeros(4))), schema("device-answer"))
    with pytest.raises(jsonschema.ValidationError):            # a price response needs its prices
        jsonschema.validate({"version": 1, "kind": "price_response", "horizon": {"step_minutes": 15, "slots": 2}},
                            schema("device-query"))


def test_a_participant_gets_the_same_share_as_the_same_device_built_in(day):
    """The saving split (dw.attribution.ledger) gives the tank the same bill,
    private-cost change and net gain whether it is built in or a participant."""
    from dataclasses import replace

    from home_energy_optimizer.dw.attribution import ledger

    h, batt, wh = day
    fc = demo_forecasts(h, tariff="dynamic")
    dark = replace(fc, solar=np.zeros_like(fc.solar))

    def split(as_participant):
        out = []
        for f in (fc, dark):
            site = SiteConfig(horizon=h, battery=batt, water_heater=None if as_participant else wh, hvac=None,
                              coordination=CoordinationConfig())
            parts = [TankParticipant(wh, h, f)] if as_participant else []
            co = DWCoordinator(site, f, participants=parts)
            out += [co, co.run().plan]
        return {r["player"]: r for r in ledger(*out)["rows"]}

    native, via = split(False), split(True)
    for col in ("bill_before", "bill_after", "private_cost_change", "net_gain"):
        assert via["water_heater"][col] == pytest.approx(native["water_heater"][col], abs=1e-6)
        assert via["battery"][col] == pytest.approx(native["battery"][col], abs=1e-6)
