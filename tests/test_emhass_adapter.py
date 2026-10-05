"""The EMHASS adapter's translation of EMHASS's battery (integrations/emhass.py).
Needs no EMHASS install: the adapter imports EMHASS only when it plans."""

import logging

import numpy as np
import pytest

from home_energy_optimizer.integrations.emhass import _battery, _tree, optimize, plan_status, unsupported
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
    """A participant may name the battery while EMHASS's own is switched off.
    There is then no state of charge to start from; the adapter must decline,
    so EMHASS runs its own solver, rather than raise into EMHASS."""

    class Opt:                    # what optimize() reads before it plans
        optim_conf = {"optimization_backend": "dantzig_wolfe", "set_use_battery": False,
                      "number_of_deferrable_loads": 0,
                      "participants": [{"devices": ["battery"], "solver": "home_energy_optimizer"}]}
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


# ---------------------------------------------------------------- the tree
EMHASS_EACH = [{"devices": ["battery"], "solver": "emhass"}, {"devices": ["deferrable0"], "solver": "emhass"},
               {"devices": ["deferrable1"], "solver": "emhass"}]
GROUPED = [{"devices": ["battery"], "solver": "home_energy_optimizer"},
           {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},
           {"devices": ["water_heater"], "solver": "home_energy_optimizer"},
           {"devices": ["hvac"], "solver": "home_energy_optimizer"}]
FOUR_DER = {
    "nodes": [
        {"id": "inverter", "type": "hybrid_inverter", "max_import": 4000, "max_export": 4000,
         "efficiency_from_parent": 0.97, "efficiency_to_parent": 0.97},
        {"id": "garage", "type": "panel", "max_import": 7400, "max_export": 0},
        {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 3500},
    ],
    "devices": {"pv": "inverter", "battery": "inverter", "water_heater": "heat", "hvac": "heat",
                "deferrable0": "garage", "deferrable1": "garage"},
    "constraints": [{"name": "l1", "devices": ["battery", "hvac"], "max_import": 5000}],
}


def test_the_topology_becomes_nodes_and_set_limits():
    """electrical_topology: nested nodes (W to kW, efficiencies each way), the
    devices and the PV placed on them (an EMHASS participant as its key), the
    hybrid inverter named, and a constraint as a set limit."""
    tree, reason = _tree({"electrical_topology": FOUR_DER}, {}, GROUPED)
    assert reason is None and tree.inverter == "inverter"
    inv, garage, heat = tree.submeters
    assert inv.keys == ("battery", "pv") and inv.max_export_kw == 4.0 and inv.eta_export == 0.97
    assert garage.members == ("deferrable0+deferrable1",) and garage.max_export_kw == 0.0
    assert heat.parent == "garage" and set(heat.members) == {"water_heater", "hvac"} and heat.max_import_kw == 3.5
    assert tree.set_limits == (SetLimit("l1", ("battery", "hvac"), max_import_kw=5.0),)


def test_a_load_group_inside_one_participant_stays_in_its_model():
    """EMHASS's own deferrable_load_groups, all its loads in one participant:
    that participant's EMHASS model holds it (mutual exclusion too); the
    coordinator holds nothing."""
    oc = {"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}]}
    tree, reason = _tree(oc, {}, GROUPED)
    assert reason is None and tree.set_limits == ()
    assert tree.own_groups == {"deferrable0+deferrable1": oc["deferrable_load_groups"]}


def test_a_load_group_across_participants_is_a_set_limit():
    """Across participants, a shared max_power (W) is a set limit (kW), named after its loads."""
    oc = {"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "max_power": 3000}]}
    tree, reason = _tree(oc, {}, EMHASS_EACH)
    assert reason is None and tree.own_groups == {}
    assert tree.set_limits == (SetLimit("deferrable0+deferrable1", ("deferrable0", "deferrable1"), 3.0),)


def test_without_a_topology_the_inverter_keys_still_describe_it():
    pc = {"inverter_is_hybrid": True, "inverter_ac_output_max": 5000, "inverter_efficiency_dc_ac": 0.96}
    tree, reason = _tree({}, pc, GROUPED)
    assert reason is None and tree.inverter == "inverter"
    (inv,) = tree.submeters
    assert inv.keys == ("battery", "pv") and inv.max_import_kw == 5.0 and inv.eta_export == 0.96


@pytest.mark.parametrize("oc, groups, problem", [
    ({"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}]},
     EMHASS_EACH, "mutual exclusion across participants"),
    ({"electrical_topology": {"nodes": [{"id": "a"}], "devices": {"deferrable0": "a"}}},
     GROUPED, "splits the participant"),
    ({"electrical_topology": {"nodes": [{"id": "a"}], "devices": {"ev": "a"}}}, GROUPED, "nothing plans"),
    ({"electrical_topology": {"nodes": [{"id": "a"}], "devices": {"hvac": "b"}}}, GROUPED, "which is no node"),
    ({"electrical_topology": {"constraints": [{"name": "c", "devices": ["deferrable1"], "max_import": 1}]}},
     GROUPED, "splits the participant"),
    ({"electrical_topology": {"constraints": [{"name": "c", "devices": ["hvac"], "max_import": 1},
                                              {"name": "c", "devices": ["battery"], "max_import": 1}]}},
     GROUPED, "two limits are named"),
])
def test_a_tree_the_coordinator_cannot_hold_is_refused_by_name(oc, groups, problem):
    """Each refusal names its reason, and EMHASS's own solver plans instead."""
    _, reason = _tree(oc, {}, groups)
    assert reason is not None and problem in reason


def test_deferrable_load_groups_no_longer_falls_back_by_itself():
    oc = {"set_use_battery": False, "number_of_deferrable_loads": 2,
          "deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "max_power": 3000}]}
    assert unsupported(oc, {}, "profit", {}) is None


def test_a_battery_and_pv_on_one_node_cannot_take_nodischarge_to_grid():
    oc = {"set_nodischarge_to_grid": True, "electrical_topology": FOUR_DER}
    reason = unsupported(oc, {}, "profit", {})
    assert reason is not None and "set_nodischarge_to_grid" in reason


def test_group_limits_from_an_earlier_draft_fall_back_by_name():
    """0.2.6 read group_limits; planning without them would drop the limits."""
    oc = {"set_use_battery": False, "group_limits": [{"name": "garage", "devices": ["deferrable0"],
                                                      "max_power": 3000}]}
    reason = unsupported(oc, {}, "profit", {})
    assert reason is not None and "electrical_topology" in reason
