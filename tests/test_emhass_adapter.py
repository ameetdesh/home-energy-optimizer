"""The EMHASS adapter's translation of EMHASS's battery (integrations/emhass.py).
Needs no EMHASS install: the adapter imports EMHASS only when it plans."""

import logging

import numpy as np
import pytest

from home_energy_optimizer.integrations.emhass import _battery, optimize, plan_status, unsupported

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
