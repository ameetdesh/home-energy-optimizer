"""The EMHASS adapter's translation of EMHASS's battery (integrations/emhass.py).
Needs no EMHASS install: the adapter imports EMHASS only when it plans."""

import logging

import numpy as np
import pytest

from home_energy_optimizer.integrations.emhass import Limit, _battery, _limits, optimize, plan_status, unsupported

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


# ---------------------------------------------------------------- shared limits
EMHASS_EACH = [{"devices": ["battery"], "solver": "emhass"}, {"devices": ["deferrable0"], "solver": "emhass"},
               {"devices": ["deferrable1"], "solver": "emhass"}]
GROUPED = [{"devices": ["battery"], "solver": "home_energy_optimizer"},
           {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},
           {"devices": ["water_heater", "hvac"], "solver": "home_energy_optimizer"}]


def test_a_load_group_inside_one_participant_stays_in_its_model():
    """EMHASS's own deferrable_load_groups, all its loads in one participant:
    that participant's EMHASS model holds it (mutual exclusion too); the
    coordinator holds nothing."""
    oc = {"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}]}
    held, own, reason = _limits(oc, GROUPED, False)
    assert reason is None and held == []
    assert own == {"deferrable0+deferrable1": oc["deferrable_load_groups"]}


def test_a_load_group_across_participants_is_a_group_limit():
    """Across participants, a shared max_power (W) is the coordinator's
    group limit (kW), named after its loads."""
    oc = {"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "max_power": 3000}]}
    held, own, reason = _limits(oc, EMHASS_EACH, False)
    assert reason is None and own == {}
    assert held == [Limit("deferrable0+deferrable1", ("deferrable0", "deferrable1"), None, 3.0)]


def test_group_limits_name_the_package_devices():
    """group_limits: devices EMHASS does not model, in W, min_power <= 0."""
    oc = {"group_limits": [{"name": "garage", "devices": ["water_heater", "hvac"], "max_power": 3000,
                            "min_power": 0}]}
    held, _, reason = _limits(oc, GROUPED, False)
    assert reason is None
    assert held == [Limit("garage", ("hvac", "water_heater"), 0.0, 3.0)]


@pytest.mark.parametrize("oc, hybrid, problem", [
    ({"deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "mutual_exclusion": True}]},
     False, "mutual exclusion across participants"),
    ({"group_limits": [{"name": "g", "devices": ["deferrable0", "water_heater"], "max_power": 3000}]},
     False, "splits the participant"),
    ({"group_limits": [{"name": "g", "devices": ["ev"], "max_power": 3000}]}, False, "nothing plans"),
    ({"group_limits": [{"name": "g", "devices": ["battery"], "max_power": 3000}]}, True, "hybrid inverter"),
    ({"group_limits": [{"name": "g", "devices": ["hvac"], "min_power": 500}]}, False, "min_power must be <= 0"),
    ({"group_limits": [{"name": "a", "devices": ["hvac"], "max_power": 1000},
                       {"name": "b", "devices": ["hvac", "water_heater"], "max_power": 3000}]},
     False, "under two limits"),
])
def test_a_limit_the_coordinator_cannot_hold_is_refused_by_name(oc, hybrid, problem):
    """Each refusal names its reason, and EMHASS's own solver plans instead."""
    groups = EMHASS_EACH if "mutual" in problem else GROUPED
    _, _, reason = _limits(oc, groups, hybrid)
    assert reason is not None and problem in reason


def test_deferrable_load_groups_no_longer_falls_back_by_itself():
    oc = {"set_use_battery": False, "number_of_deferrable_loads": 2,
          "deferrable_load_groups": [{"names": ["deferrable0", "deferrable1"], "max_power": 3000}]}
    assert unsupported(oc, {}, "profit", {}) is None
