"""The EMHASS adapter's translation of EMHASS's battery (integrations/emhass.py).
Needs no EMHASS install: the adapter imports EMHASS only when it plans."""

import logging

import numpy as np
import pytest

from home_energy_optimizer.integrations.emhass import (
    _battery,
    _linear_battery,
    _site,
    optimize,
    plan_status,
    unsupported,
)
from home_energy_optimizer.types import SetLimit

PLANT = {"battery_nominal_energy_capacity": 10000, "battery_minimum_state_of_charge": 0.3,
         "battery_maximum_state_of_charge": 0.9, "battery_charge_power_max": 5000,
         "battery_discharge_power_max": 4000, "battery_charge_efficiency": 0.95,
         "battery_discharge_efficiency": 0.9}


def test_power_limits_are_emhass_limits_at_the_meter():
    """EMHASS bounds meter-side power by each limit directly and through the
    efficiency; the tighter one holds: charging 5 kW, discharging 0.9 x 4 kW."""
    b = _battery(PLANT, soc_init=0.5, buy=np.full(4, 0.2))
    assert b.p_charge_max_kw == pytest.approx(5.0)
    assert b.p_discharge_max_kw == pytest.approx(3.6)


def test_the_soc_window_becomes_the_store():
    """A 10 kWh battery used between 30% and 90%: a 9 kWh store with a 3 kWh
    floor, starting at 5 kWh, with leftover energy valued at the mean tariff."""
    b = _battery(PLANT, soc_init=0.5, buy=np.array([0.1, 0.3]))
    assert b.capacity_kwh == pytest.approx(9.0)
    assert b.soe_floor_kwh == pytest.approx(3.0)
    assert b.capacity_kwh * b.soc_initial_frac == pytest.approx(5.0)
    assert b.terminal_price == pytest.approx(0.2)


def test_a_battery_participant_without_an_emhass_battery_falls_back():
    """`site` may list the battery while EMHASS's own is switched off.
    There is then no state of charge to start from; the adapter must decline,
    so EMHASS runs its own solver, rather than raise into EMHASS."""

    class Opt:                    # what optimize() reads before it plans
        optim_conf = {"optimization_backend": "dantzig_wolfe", "set_use_battery": False,
                      "number_of_deferrable_loads": 0,
                      "site": [{"id": "battery", "parent": "grid", "solver": "home_energy_optimizer"}]}
        plant_conf: dict = {}
        costfun = "profit"
        time_step = 0.5
        logger = logging.getLogger("emhass-test")

    reason = unsupported(Opt.optim_conf, Opt.plant_conf, Opt.costfun, {})
    assert reason is not None
    n = 4
    opt = Opt()
    assert optimize(opt, list(range(n)), np.zeros(n), np.full(n, 500.0),
                    np.full(n, 0.3), np.full(n, 0.1)) is None
    # ...and say why, so EMHASS can report which solver made the plan.
    assert opt.fed_fallback_reason == f"not available for {reason}"


def test_a_plan_is_optimal_only_when_its_gap_is_closed():
    """"Optimal" is a proof: the plan is within 0.1% of its lower bound. A
    plan the coordinator stopped on with the bound still open is runnable,
    and EMHASS publishes it, but it is reported "Optimal_Inaccurate"."""
    assert plan_status(10.0, 10.0) == "Optimal"
    assert plan_status(10.0, 9.995) == "Optimal"           # 0.05% of 10
    assert plan_status(10.0, 9.9) == "Optimal_Inaccurate"   # 1%
    assert plan_status(0.0, -0.0005) == "Optimal"          # absolute below 1
    assert plan_status(0.0, -0.01) == "Optimal_Inaccurate"
    assert plan_status(-200.0, -200.1) == "Optimal"         # a profit: scale by |upper|


@pytest.mark.parametrize("extra, oc_extra, blocked", [
    ({}, {}, None),
    ({}, {"set_nodischarge_to_grid": True}, "set_nodischarge_to_grid"),
    ({"inverter_stress_cost": 0.05}, {}, "inverter_stress_cost"),
    ({"inverter_ac_output_max": None, "pv_inverter_model": "SMA_Sunny_Boy"}, {}, "pv_inverter_model"),
    ({"inverter_ac_output_max": None, "pv_inverter_model": 4000}, {}, None),
])
def test_a_hybrid_inverter_is_planned_unless_an_option_ties_it_down(extra, oc_extra, blocked):
    """A hybrid inverter is a sub-meter the coordinator models exactly; the
    options that tie it to the meter's direction, price its stress, or rate
    it by a CEC model name still fall back, by name."""
    pc = {**PLANT, "inverter_is_hybrid": True, "inverter_ac_output_max": 5000, **extra}
    oc = {"set_use_battery": True, "number_of_deferrable_loads": 0, **oc_extra}
    reason = unsupported(oc, pc, "profit", {})
    if blocked is None:
        assert reason is None
    else:
        assert reason is not None and blocked in reason


# ---------------------------------------------------------------- the site
DEVICES = ["battery", "deferrable0", "deferrable1"]          # what EMHASS plans here
HEO = "home_energy_optimizer"
FOUR_DER = [
    {"id": "grid", "max_import": 9000, "max_export": 5000},
    {"id": "inverter", "parent": "grid", "type": "hybrid_inverter", "max_import": 4000, "max_export": 4000,
     "efficiency_import": 0.97, "efficiency_export": 0.97},
    {"id": "garage", "parent": "grid", "type": "panel", "max_import": 7400, "max_export": 0},
    {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 3500},
    {"id": "l1", "type": "limit", "max_import": 5000},
    {"id": "pv", "parent": "inverter"},
    {"id": "battery", "parent": "inverter", "solver": HEO, "limits": ["l1"]},
    {"id": "deferrable0", "parent": "garage", "group": "loads"},
    {"id": "deferrable1", "parent": "garage", "group": "loads"},
    {"id": "water_heater", "parent": "heat", "solver": HEO, "config": {"power_kw": 3.0}},
    {"id": "hvac", "parent": "heat", "solver": HEO, "limits": ["l1"]},
]


def _layout(site, oc=None, pc=None, devices=DEVICES):
    return _site({"site": site, **(oc or {})}, pc or {}, devices)


def test_the_site_becomes_groups_nodes_and_set_limits():
    """One list: nested nodes (W to kW, efficiencies each way), the devices and
    the PV placed on them (a group as its id), the hybrid inverter named, a
    limit over the devices tagged with it, and the main meter's limits."""
    layout, reason = _layout(FOUR_DER)
    assert reason is None and layout.inverter == "inverter" and layout.notes == ()
    assert {g["key"]: g["devices"] for g in layout.groups} == {
        "battery": ["battery"], "loads": ["deferrable0", "deferrable1"],
        "water_heater": ["water_heater"], "hvac": ["hvac"]}
    assert next(g for g in layout.groups if g["key"] == "water_heater")["config"] == {"power_kw": 3.0}
    inv, garage, heat = layout.submeters
    assert inv.keys == ("battery", "pv") and inv.max_export_kw == 4.0 and inv.eta_export == 0.97
    assert garage.members == ("loads",) and garage.max_export_kw == 0.0 and garage.parent is None
    assert heat.parent == "garage" and set(heat.members) == {"water_heater", "hvac"} and heat.max_import_kw == 3.5
    assert layout.set_limits == (SetLimit("l1", ("battery", "hvac"), max_import_kw=5.0),)
    assert layout.grid_w == (9000.0, 5000.0)


def test_devices_site_leaves_out_are_emhass_s_alone_on_the_meter():
    layout, reason = _layout([{"id": "battery", "parent": "grid", "solver": HEO}])
    assert reason is None and layout.submeters == ()
    assert {g["key"]: (g["solver"], g["devices"]) for g in layout.groups} == {
        "battery": (HEO, ["battery"]), "deferrable0": ("emhass", ["deferrable0"]),
        "deferrable1": ("emhass", ["deferrable1"])}


def test_a_load_group_inside_one_group_stays_in_its_model():
    """EMHASS's own deferrable_load_groups, all its loads in one EMHASS group:
    that group's model holds it (mutual exclusion too); the coordinator holds nothing."""
    oc = {"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}]}
    layout, reason = _layout(FOUR_DER, oc)
    assert reason is None and [s.name for s in layout.set_limits] == ["l1"]
    assert layout.own_groups == {"loads": oc["deferrable_load_groups"]}


def test_a_load_group_across_groups_is_a_set_limit():
    """Across groups, a shared max_power (W) is a set limit (kW), named after its loads."""
    oc = {"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "max_power": 3000}]}
    layout, reason = _layout([], oc)
    assert reason is None and layout.own_groups == {}
    assert layout.set_limits == (SetLimit("deferrable0+deferrable1", ("deferrable0", "deferrable1"), 3.0),)


def test_without_a_node_the_inverter_keys_still_describe_it():
    pc = {"inverter_is_hybrid": True, "inverter_ac_output_max": 5000, "inverter_efficiency_dc_ac": 0.96}
    layout, reason = _layout([{"id": "battery", "parent": "grid", "solver": HEO}], pc=pc)
    assert reason is None and layout.inverter == "inverter"
    (inv,) = layout.submeters
    assert inv.keys == ("battery", "pv") and inv.max_import_kw == 5.0 and inv.eta_export == 0.96


@pytest.mark.parametrize("site, oc, problem", [
    ([], {"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}]},
     "mutual exclusion across participants"),
    ([{"id": "a", "parent": "grid"}, {"id": "deferrable0", "parent": "a", "group": "g"}, {"id": "deferrable1", "parent": "grid", "group": "g"}],
     {}, "sits on 'a' and 'grid'"),
    ([{"id": "l", "type": "limit", "max_import": 1}, {"id": "deferrable0", "parent": "grid", "group": "g", "limits": ["l"]},
      {"id": "deferrable1", "parent": "grid", "group": "g"}], {}, "splits the group 'g'"),
    ([{"id": "hvac", "parent": "b", "solver": HEO}], {}, "is no node"),
    ([{"id": "hvac", "parent": "grid", "solver": HEO, "limits": ["nope"]}], {}, "which is no limit"),
    ([{"id": "pv", "parent": "grid", "solver": "emhass"}], {}, "planned by no solver"),
    ([{"id": "hvac", "parent": "grid", "solver": "cbc"}], {}, "unknown solver"),
    ([{"id": "deferrable0", "parent": "grid", "solver": HEO}], {}, "has no solver for 'deferrable0'"),
    ([{"id": "deferrable0", "parent": "grid", "config": {"power_kw": 1}}], {}, "only home_energy_optimizer reads"),
    ([{"id": "water_heater", "parent": "grid", "solver": HEO, "group": "heat"}, {"id": "hvac", "parent": "grid", "solver": HEO, "group": "heat"}],
     {}, "one solver for all"),
    ([{"id": "deferrable0", "parent": "grid", "group": "battery"}], {}, "has the name of an element"),
    ([{"id": "hvac", "parent": "grid", "solver": HEO}, {"id": "hvac", "parent": "grid", "solver": HEO}], {}, "lists 'hvac' twice"),
    ([{"id": "deferrable5", "parent": "grid"}], {}, "which EMHASS does not plan here"),
    ([{"id": "deferrable0"}], {}, "'deferrable0' needs a parent"),
    ([{"id": "garage", "type": "panel", "max_import": 7400}], {}, "'garage' needs a parent"),
    ([{"id": "garage_emhass", "solver": {"url": "http://h:5000/participant"}}], {}, "'garage_emhass' needs a parent"),
    ([{"id": "garage_emhass", "parent": "shed", "solver": {"url": "http://h:5000/participant"}}], {},
     "is no node"),
    ([{"id": "deferrable0", "parent": "grid", "group": "r"},
      {"id": "r", "parent": "grid", "solver": {"url": "http://h:5000/participant"}}], {},
     "names two elements"),
])
def test_a_site_the_coordinator_cannot_hold_is_refused_by_name(site, oc, problem):
    """Each refusal names its reason, and EMHASS's own solver plans instead."""
    _, reason = _layout(site, oc)
    assert reason is not None and problem in reason


def test_a_remote_solver_is_one_participant_where_it_is_wired():
    """A solver on the network is one participant, keyed by its id, on its
    parent, counted in the limits it is tagged with; its devices are its own."""
    remote = {"id": "garage_emhass", "parent": "garage", "limits": ["l1"],
              "solver": {"url": "http://emhass-garage:5000/participant", "token_secret": "garage_token"}}
    layout, reason = _layout(FOUR_DER + [remote])
    assert reason is None
    g = next(g for g in layout.groups if g["key"] == "garage_emhass")
    assert g["solver"] == "remote" and g["devices"] == ["garage_emhass"]
    assert g["config"] == {"url": "http://emhass-garage:5000/participant", "token_secret": "garage_token"}
    garage = next(sm for sm in layout.submeters if sm.name == "garage")
    assert set(garage.members) == {"loads", "garage_emhass"}
    assert layout.set_limits == (SetLimit("l1", ("battery", "garage_emhass", "hvac"), max_import_kw=5.0),)


def test_what_does_not_stop_the_plan_is_noted():
    layout, reason = _layout([{"id": "spare", "parent": "grid", "type": "breaker", "max_import": 2000},
                              {"id": "l9", "type": "limit", "max_import": 1}])
    assert reason is None
    assert layout.notes == ("the limit 'l9' has no device tagged with it", "the node 'spare' holds nothing")


@pytest.mark.parametrize("site, nodes", [
    # a hybrid inverter on another's backup port
    ([{"id": "inv1", "parent": "grid", "type": "hybrid_inverter", "max_import": 5000, "max_export": 5000},
      {"id": "backup", "type": "panel", "parent": "inv1", "max_import": 7000, "max_export": 7000},
      {"id": "inv2", "type": "inverter", "parent": "backup", "max_import": 3000, "max_export": 3000,
       "efficiency_export": 0.96},
      {"id": "pv", "parent": "inv1"}, {"id": "battery", "parent": "inv2"}],
     {"inv1": ((), ["pv"]), "backup": ("inv1", []), "inv2": ("backup", ["battery"])}),
    # AC-coupled: a string PV inverter and a battery inverter
    ([{"id": "pv_inv", "parent": "grid", "type": "inverter", "max_import": 0, "max_export": 6000},
      {"id": "bat_inv", "parent": "grid", "type": "inverter", "max_import": 5000, "max_export": 5000},
      {"id": "pv", "parent": "pv_inv"}, {"id": "battery", "parent": "bat_inv"}],
     {"pv_inv": ((), ["pv"]), "bat_inv": ((), ["battery"])}),
    # an islanded backup subpanel under a hybrid inverter
    ([{"id": "inv", "parent": "grid", "type": "hybrid_inverter", "max_import": 4000, "max_export": 4000},
      {"id": "island", "type": "panel", "parent": "inv", "max_import": 0, "max_export": 0},
      {"id": "pv", "parent": "inv"}, {"id": "battery", "parent": "inv"},
      {"id": "deferrable0", "parent": "island"}],
     {"inv": ((), ["battery", "pv"]), "island": ("inv", ["deferrable0"])}),
    # a device on the main meter, said explicitly; a sub-meter with no rating
    ([{"id": "ev_meter", "parent": "grid", "type": "meter"}, {"id": "deferrable0", "parent": "ev_meter"},
      {"id": "deferrable1", "parent": "grid"}],
     {"ev_meter": ((), ["deferrable0"])}),
])
def test_houses_from_the_stress_test_map_as_drawn(site, nodes):
    layout, reason = _layout(site)
    assert reason is None
    got = {s.name: ((s.parent,) if s.parent else (), sorted(s.keys)) for s in layout.submeters}
    want = {k: ((p,) if isinstance(p, str) else p, m) for k, (p, m) in nodes.items()}
    assert got == want


def test_deferrable_load_groups_no_longer_falls_back_by_itself():
    oc = {"set_use_battery": False, "number_of_deferrable_loads": 2,
          "deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "max_power": 3000}]}
    assert unsupported(oc, {}, "profit", {}) is None


def test_a_battery_and_pv_on_one_node_cannot_take_nodischarge_to_grid():
    oc = {"set_nodischarge_to_grid": True, "set_use_battery": True, "site": FOUR_DER}
    reason = unsupported(oc, {}, "profit", {})
    assert reason is not None and "set_nodischarge_to_grid" in reason


@pytest.mark.parametrize("old", ["participants", "electrical_topology", "group_limits"])
def test_keys_from_earlier_drafts_fall_back_by_name(old):
    """0.2.6 / 0.2.7 read these; planning without what they said would be worse."""
    reason = unsupported({"set_use_battery": False, old: [{"x": 1}]}, {}, "profit", {})
    assert reason is not None and old in reason and "site" in reason


def _remotes_only(urls: list[str]):
    """What optimize() reads from EMHASS for a house of remote solvers alone."""
    class Opt:
        optim_conf = {"optimization_backend": "dantzig_wolfe", "set_use_battery": False,
                      "number_of_deferrable_loads": 0,
                      "site": [{"id": f"r{i}", "parent": "grid", "solver": {"url": u}} for i, u in enumerate(urls)]}
        plant_conf: dict = {}
        retrieve_hass_conf: dict = {}
        costfun = "profit"
        time_step = 0.25
        logger = logging.getLogger("emhass-test")

        @staticmethod
        def _prepare_power_limit_array(value, name, n):
            return np.full(n, float(value))
    return Opt()


def _plan_with_remotes(monkeypatch, events: list, raise_for: str | None = None):
    """Plan a house of two remote tanks; record when the saving is split and
    when each remote is asked to commit, `raise_for`'s commit raising.
    Needs pandas, as EMHASS does (the adapter's results are a DataFrame)."""
    pd = pytest.importorskip("pandas")

    import home_energy_optimizer.integrations.emhass as adapter
    from home_energy_optimizer.remote import RemoteParticipant, serve
    from test_remote import FC, H, Recorder, _service

    recs = [Recorder(), Recorder()]
    servers = [serve(_service(rec)) for rec in recs]
    real_commit, real_shares = RemoteParticipant.commit, adapter.shares

    def commit(self, detail, plan_kw):
        events.append(f"commit {self.key}")
        if self.key == raise_for:
            raise RuntimeError("the remote's reply made no sense")
        return real_commit(self, detail, plan_kw)

    def split(*args, **kwargs):
        events.append("shares")
        return real_shares(*args, **kwargs)

    monkeypatch.setattr(RemoteParticipant, "commit", commit)
    monkeypatch.setattr(adapter, "shares", split)
    try:
        opt = _remotes_only([s.url for s in servers])
        idx = pd.date_range("2026-10-07", periods=H.steps, freq="15min", tz="UTC")
        res = optimize(opt, pd.DataFrame(index=idx), np.zeros(H.steps), np.full(H.steps, 500.0),
                       FC.buy, np.minimum(FC.sell, FC.buy))
    finally:
        for s in servers:
            s.shutdown()
    return res, recs


def test_remotes_commit_only_once_the_whole_plan_is_made(monkeypatch):
    """Once a remote runs its part, nothing may send EMHASS to its default
    solver: that would plan the house again around a remote already running
    this plan. So the remotes commit last, after the saving's split."""
    events: list[str] = []
    res, recs = _plan_with_remotes(monkeypatch, events)
    assert res is not None
    assert events == ["shares", "commit r0", "commit r1"]
    assert [len(rec.committed) for rec in recs] == [1, 1]
    assert set(res["fed_remote_status_r0"]) == set(res["fed_remote_status_r1"]) == {"committed"}


def test_a_commit_that_raises_is_a_refusal_and_the_plan_stands(monkeypatch):
    """A commit that fails in any way is that remote's refusal: the plan is
    still published (not handed to the default solver), and the others commit."""
    events: list[str] = []
    res, recs = _plan_with_remotes(monkeypatch, events, raise_for="r0")
    assert res is not None
    assert events == ["shares", "commit r0", "commit r1"]
    assert [len(rec.committed) for rec in recs] == [0, 1]
    (r0,) = set(res["fed_remote_status_r0"])
    assert r0.startswith("refused: RuntimeError") and set(res["fed_remote_status_r1"]) == {"committed"}


@pytest.mark.parametrize("conf", ["optim_conf", "plant_conf"])
@pytest.mark.parametrize("key", ["battery_soc_deficit_cost", "battery_soc_surplus_cost"])
def test_a_soc_penalty_keeps_the_battery_in_emhass_s_own_model(conf, key):
    """EMHASS reads the SoC penalties from optim_conf: a positive one means
    its own battery model, which prices them; the linear one would drop them."""
    oc: dict = {"weight_battery_discharge": 0, "weight_battery_charge": 0}
    pc: dict = {}
    assert _linear_battery(oc, pc)
    (oc if conf == "optim_conf" else pc)[key] = 0.05
    assert not _linear_battery(oc, pc)


def test_a_remote_that_cannot_plan_the_horizon_is_said_to_decline(monkeypatch):
    """A remote that answers, but cannot plan this horizon (409: another slot
    length, another first slot), is planned without, and the plan says why:
    not "unreachable", which would send someone to check the network."""
    pd = pytest.importorskip("pandas")
    from home_energy_optimizer.remote import serve
    from test_remote import FC, H, _service

    server = serve(_service(steps=H.steps // 2))          # it plans half a day only
    try:
        idx = pd.date_range("2026-10-07", periods=H.steps, freq="15min", tz="UTC")
        res = optimize(_remotes_only([server.url]), pd.DataFrame(index=idx), np.zeros(H.steps),
                       np.full(H.steps, 500.0), FC.buy, np.minimum(FC.sell, FC.buy))
    finally:
        server.shutdown()
    assert res is not None and "fed_remote_power_r0" not in res
    (status,) = set(res["fed_remote_status_r0"])
    assert status.startswith("declined: ") and "409" in status and "48 slots" in status
