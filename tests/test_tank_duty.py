"""Fractional tank element: WaterHeaterConfig.n_duty_levels."""

import numpy as np
import pytest

from home_energy_optimizer import Horizon, WaterHeaterConfig, demo_forecasts
from home_energy_optimizer.dp_thermal import solve_water_heater


@pytest.fixture
def day():
    h = Horizon(dt=0.25, hours=24.0)
    return h, demo_forecasts(h, tariff="day_night")


def test_on_off_is_the_default():
    assert WaterHeaterConfig().n_duty_levels == 2
    assert np.array_equal(WaterHeaterConfig().duty_actions, [0.0, 1.0])


def test_more_levels_never_value_the_tank_worse(day):
    """0, 1/4 .. 1 contains off and on, so every V is at least the on/off V."""
    h, fc = day
    v2 = solve_water_heater(WaterHeaterConfig(), h, fc.buy, fc.sell, fc.hot_water_demand).value
    v5 = solve_water_heater(WaterHeaterConfig(n_duty_levels=5), h, fc.buy, fc.sell,
                            fc.hot_water_demand).value
    assert np.all(v5 >= v2 - 1e-12)


def test_fractional_plan_uses_partial_duty(day):
    h, fc = day
    wh = WaterHeaterConfig(n_duty_levels=5)
    p = solve_water_heater(wh, h, fc.buy, fc.sell, fc.hot_water_demand).power
    partial = (p > 1e-6) & (p < wh.power_kw - 1e-6)
    assert partial.any()


def test_grid_is_refined_to_resolve_a_duty_step(day):
    """Too coarse a grid lets the DP plan moves its rollout cannot follow:
    its claimed value and the cost of what it executes drift apart."""
    h, fc = day
    wh = WaterHeaterConfig(n_duty_levels=13)
    wh = WaterHeaterConfig(n_duty_levels=13, n_states=wh.states_for_duty_levels(h.dt))
    assert wh.n_states > 68
    s = solve_water_heater(wh, h, fc.buy, fc.sell, fc.hot_water_demand)
    claimed = -np.interp(wh.t_comfort, s.states, s.value[0])
    # bill + comfort of the executed plan, as the DP scores it
    from home_energy_optimizer.dp_thermal import wh_discomfort
    ref = float(np.mean(fc.buy))
    real = float(np.sum(s.power * fc.buy * h.dt) + np.sum(wh_discomfort(wh, s.trajectory[1:], ref, h.dt))
                 + wh.heat_capacity_kwh_per_k * ref * max(0.0, wh.t_comfort - s.trajectory[-1]))
    assert abs(real - claimed) < 0.02


def test_levels_below_two_are_rejected():
    with pytest.raises(ValueError):
        WaterHeaterConfig(n_duty_levels=1).validate()
