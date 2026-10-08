"""Remote participants (remote.py): the participant API over HTTP, between a
coordinator (RemoteParticipant) and a solver (ParticipantService, served
here by `serve` on a local port)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from home_energy_optimizer import BatteryConfig, Horizon, SiteConfig, WaterHeaterConfig
from home_energy_optimizer.dw.coordinator import DWCoordinator
from home_energy_optimizer.interface import Answer, Query, schema
from home_energy_optimizer.profiles import demo_forecasts
from home_energy_optimizer.remote import (
    ParticipantService,
    PlanWindow,
    RemoteError,
    RemoteParticipant,
    RemoteUnavailable,
    WindowRefused,
    serve,
    urllib_transport,
)
from home_energy_optimizer.types import CoordinationConfig
from test_participants import TankParticipant

H = Horizon(dt=0.25, hours=24.0)
WINDOW = PlanWindow(step_minutes=15, slots=H.steps, start="2026-10-07T00:00:00+00:00")
FC = demo_forecasts(H, tariff="dynamic")
WH = WaterHeaterConfig()


class Recorder:
    """What the service saw, and what it committed."""

    def __init__(self) -> None:
        self.kinds: list[str] = []
        self.committed: list[Answer] = []


def _service(rec: Recorder | None = None, token: str | None = None, steps: int = H.steps,
             kinds: tuple[str, ...] = ("price_response",)) -> ParticipantService:
    """The package's tank as a remote: built per horizon, refusing any other
    slot length, answering `kinds`; commits recorded."""
    rec = rec or Recorder()

    def build(window: PlanWindow) -> TankParticipant:
        if window.step_minutes != 15 or window.slots != steps:
            raise WindowRefused(f"this tank plans {steps} slots of 15 minutes")
        tank = TankParticipant(WH, H, FC)
        respond = tank.respond

        def recorded(q: Query) -> Answer:
            rec.kinds.append(q.kind)
            return respond(q)
        tank.respond = recorded                          # type: ignore[method-assign]
        return tank

    def commit(participant, answer: Answer) -> tuple[bool, str]:
        rec.committed.append(answer)
        return True, ""

    return ParticipantService(lambda: {"key": "water_heater", "devices": ["water_heater"],
                                       "max_power_kw": WH.power_kw, "modulating": False, "onoff": True,
                                       "kinds": list(kinds)},
                              build, commit, token=token)


@pytest.fixture
def remote():
    rec = Recorder()
    server = serve(_service(rec))
    yield server, rec
    server.shutdown()


def _site() -> SiteConfig:
    return SiteConfig(horizon=H, battery=BatteryConfig(capacity_kwh=10.0), water_heater=None, hvac=None,
                      coordination=CoordinationConfig())


def test_a_remote_plans_exactly_as_the_same_participant_in_process():
    """The tank behind HTTP, answering the same kinds of query, gives the
    coordinator the same plan and objective as the same tank in-process: the
    wire loses nothing."""
    server = serve(_service(kinds=("price_response", "best_response")))
    try:
        here = DWCoordinator(_site(), FC, participants=[TankParticipant(WH, H, FC)]).run()
        there = DWCoordinator(_site(), FC,
                              participants=[RemoteParticipant("water_heater", server.url, WINDOW)]).run()
    finally:
        server.shutdown()
    assert there.upper == here.upper and there.lower == here.lower
    assert np.array_equal(there.plan["water_heater"].power, here.plan["water_heater"].power)


def test_what_goes_over_the_wire_follows_the_schemas(remote):
    jsonschema = pytest.importorskip("jsonschema")
    server, _ = remote
    seen: list[tuple[str, bytes | None, bytes]] = []
    send = urllib_transport(10)

    def recording(method, url, headers, body):
        status, raw = send(method, url, headers, body)
        seen.append((url, body, raw))
        return status, raw

    p = RemoteParticipant("water_heater", server.url, WINDOW, transport=recording)
    p.respond(Query("price_response", FC.buy, FC.buy))
    p.baseline()
    for url, body, raw in seen:
        if url.endswith("/v1/query"):
            jsonschema.validate(json.loads(body), schema("device-query"))
        if url.endswith(("/v1/query", "/v1/baseline")):
            jsonschema.validate(json.loads(raw), schema("device-answer"))


def test_a_remote_without_the_right_token_is_refused():
    server = serve(_service(token="s3cret"))
    try:
        with pytest.raises(RemoteError) as err:
            RemoteParticipant("water_heater", server.url, WINDOW, token="wrong")
        assert err.value.status == 401
        assert RemoteParticipant("water_heater", server.url, WINDOW, token="s3cret").max_power_kw == WH.power_kw
    finally:
        server.shutdown()


def test_an_unreachable_remote_is_said_so():
    server = serve(_service())
    url = server.url
    server.shutdown()
    server.server_close()
    with pytest.raises(RemoteError) as err:
        RemoteParticipant("water_heater", url, WINDOW, timeout=2)
    assert err.value.status == 0


def test_a_remote_that_stops_answering_stands_in_with_its_last_plan():
    """After one answer, a remote that goes away answers with that plan,
    marked fallback, so the coordinator proves no bound from it; before any
    answer it is unavailable."""
    server = serve(_service())
    p = RemoteParticipant("water_heater", server.url, WINDOW, timeout=2)
    first = p.respond(Query("price_response", FC.buy, FC.buy))
    server.shutdown()
    server.server_close()
    later = p.respond(Query("price_response", 2 * FC.buy, 2 * FC.buy))
    assert later.status == "fallback" and np.array_equal(later.plan_kw, first.plan_kw)
    server = serve(_service())
    q = RemoteParticipant("water_heater", server.url, WINDOW, timeout=2)
    server.shutdown()
    server.server_close()
    with pytest.raises(RemoteUnavailable):
        q.baseline()


def test_a_horizon_the_remote_cannot_plan_is_refused_with_its_reason(remote):
    server, _ = remote
    p = RemoteParticipant("water_heater", server.url, PlanWindow(30, 48), timeout=5)
    with pytest.raises(RemoteUnavailable) as err:
        p.respond(Query("price_response", np.full(48, 0.2), np.full(48, 0.2)))
    assert err.value.status == 409 and "15 minutes" in str(err.value)


def test_only_the_plan_chosen_is_committed_and_only_as_answered(remote):
    """A commit names the answer the coordinator chose; the remote runs that
    answer (its own, with what it kept), and refuses a plan it did not give."""
    server, rec = remote
    p = RemoteParticipant("water_heater", server.url, WINDOW)
    r = DWCoordinator(_site(), FC, participants=[p]).run()
    chosen = r.plan["water_heater"]
    assert rec.committed == []                                   # queries have no effect
    assert p.commit(chosen.detail, chosen.power) == (True, "")
    assert np.array_equal(rec.committed[-1].plan_kw, chosen.power)
    ok, why = p.commit(chosen.detail, chosen.power + 1.0)
    assert not ok and "409" in why and len(rec.committed) == 1


def test_a_remote_is_not_sent_the_house_s_load_unless_it_asks(remote):
    """A best response carries the rest of the house's load; a remote that
    does not list best_response gets the same prices as a price response."""
    server, rec = remote
    p = RemoteParticipant("water_heater", server.url, WINDOW)
    p.respond(Query("best_response", FC.buy, FC.sell, residual_kw=FC.load))
    assert rec.kinds == ["price_response"]
