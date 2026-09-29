"""Thermal DP correctness, and the V/POL export that the POC dropped."""

from __future__ import annotations

import numpy as np
import pytest

from home_energy_optimizer import (
    Horizon,
    HvacConfig,
    WaterHeaterConfig,
    baseline_hvac,
    baseline_water_heater,
    rollout_water_heater,
    solve_hvac,
    solve_water_heater,
)
from home_energy_optimizer.dp_thermal import _draw_factor, _relaxation, _usable_outflow
from home_energy_optimizer.profiles import (
    day_night_tariff,
    hot_water_demand_profile,
    outdoor_temp_profile,
)


@pytest.fixture
def horizon() -> Horizon:
    return Horizon(dt=0.25, hours=24.0)


# --------------------------------------------------------------------------
# Water heater
# --------------------------------------------------------------------------


def test_water_heater_returns_value_and_policy(horizon):
    """The regression this package exists to fix.

    The POC's compiled solvers returned only (temp, on), so two-thirds of the
    value functions were discarded at the WASM boundary and the marginal-value
    machinery only ever worked for the battery.
    """
    cfg = WaterHeaterConfig()
    buy, sell = day_night_tariff(horizon)
    sol = solve_water_heater(cfg, horizon, buy, sell, hot_water_demand_profile(horizon))

    assert sol.value.shape == (horizon.steps + 1, cfg.n_states)
    assert sol.policy.shape == (horizon.steps, cfg.n_states)
    assert np.all(np.isfinite(sol.value))
    assert set(np.unique(sol.policy)).issubset({0, 1})


def test_water_heater_temperature_within_bounds(horizon):
    cfg = WaterHeaterConfig()
    buy, sell = day_night_tariff(horizon)
    sol = solve_water_heater(cfg, horizon, buy, sell, hot_water_demand_profile(horizon))
    assert sol.trajectory.min() >= cfg.t_min - 1e-9
    assert sol.trajectory.max() <= cfg.t_max + 1e-9


def test_water_heater_action_is_binary_but_the_last_slot_may_be_partial(horizon):
    """The DECISION is on/off; the realised duty is capped by the cut-out.

    The element is commanded on or off - the policy stores an index into a
    two-element action set - but the slot in which the tank reaches t_max runs
    the element only as long as the cut-out allows. Billing a full slot there
    was the old model's pessimism, and clamping the state afterwards was how it
    got away with it.
    """
    cfg = WaterHeaterConfig()
    buy, sell = day_night_tariff(horizon)
    sol = solve_water_heater(cfg, horizon, buy, sell, hot_water_demand_profile(horizon))

    assert set(np.unique(sol.policy)).issubset({0, 1})
    duty = sol.power / cfg.power_kw
    assert duty.min() >= -1e-12 and duty.max() <= 1.0 + 1e-12
    # full-power and off slots still dominate; partial ones are the exception
    full_or_off = np.isclose(duty, 0.0) | np.isclose(duty, 1.0)
    assert full_or_off.mean() > 0.9


def test_water_heater_preheats_into_the_cheap_window(horizon):
    """Optimised heating should be more overnight-weighted than a thermostat."""
    cfg = WaterHeaterConfig()
    buy, sell = day_night_tariff(horizon, peak=0.40, offpeak=0.05)
    demand = hot_water_demand_profile(horizon)

    sol = solve_water_heater(cfg, horizon, buy, sell, demand)
    _, base_power = baseline_water_heater(cfg, horizon, demand)

    cheap = buy < 0.10
    opt_share = sol.power[cheap].sum() / max(sol.power.sum(), 1e-9)
    base_share = base_power[cheap].sum() / max(base_power.sum(), 1e-9)
    assert opt_share > base_share


def test_water_heater_meets_demand_roughly(horizon):
    """It should not simply refuse to heat: comfort is penalised."""
    cfg = WaterHeaterConfig()
    buy, sell = day_night_tariff(horizon)
    sol = solve_water_heater(cfg, horizon, buy, sell, hot_water_demand_profile(horizon))
    assert sol.power.sum() > 0
    # Mean temperature should sit near or above comfort, not collapse.
    assert sol.trajectory.mean() > cfg.t_comfort - 10.0


# --------------------------------------------------------------------------
# HVAC
# --------------------------------------------------------------------------


def test_hvac_returns_value_and_policy(horizon):
    cfg = HvacConfig()
    buy, sell = day_night_tariff(horizon)
    sol = solve_hvac(cfg, horizon, buy, sell, outdoor_temp_profile(horizon))

    assert sol.value.shape == (horizon.steps + 1, cfg.n_states)
    assert sol.policy.shape == (horizon.steps, cfg.n_states)
    assert set(np.unique(sol.policy)).issubset({0, 1, 2})


def test_hvac_temperature_within_bounds(horizon):
    cfg = HvacConfig()
    buy, sell = day_night_tariff(horizon)
    sol = solve_hvac(cfg, horizon, buy, sell, outdoor_temp_profile(horizon))
    assert sol.trajectory.min() >= cfg.t_min - 1e-9
    assert sol.trajectory.max() <= cfg.t_max + 1e-9


def test_hvac_power_is_never_negative(horizon):
    """Electrical draw is |action| * power - cooling still consumes."""
    cfg = HvacConfig()
    buy, sell = day_night_tariff(horizon)
    sol = solve_hvac(cfg, horizon, buy, sell, outdoor_temp_profile(horizon))
    assert sol.power.min() >= 0.0
    assert sol.power.max() <= cfg.power_kw + 1e-9


def test_hvac_works_harder_when_it_is_hotter(horizon):
    cfg = HvacConfig()
    buy, sell = day_night_tariff(horizon)
    mild = solve_hvac(cfg, horizon, buy, sell, outdoor_temp_profile(horizon, mean_c=24.0))
    hot = solve_hvac(cfg, horizon, buy, sell, outdoor_temp_profile(horizon, mean_c=36.0))
    assert hot.power.sum() > mild.power.sum()


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_baselines_hold_comfort(horizon):
    wh_cfg = WaterHeaterConfig()
    temp, power = baseline_water_heater(wh_cfg, horizon, hot_water_demand_profile(horizon))
    assert temp.shape == (horizon.steps + 1,)
    assert power.shape == (horizon.steps,)
    # A bang-bang thermostat overshoots by at most one full step of heating:
    # 3 kW * 0.25 h / (150 L * 4.186 / 3600 kWh/K) ~= 4.3 K at these defaults.
    # Coarse, but that is the point - it is the dumb baseline to beat.
    max_overshoot = wh_cfg.power_kw * horizon.dt / wh_cfg.heat_capacity_kwh_per_k
    assert temp.max() <= wh_cfg.t_comfort + max_overshoot + 1e-6

    hv_cfg = HvacConfig()
    temp, power = baseline_hvac(hv_cfg, horizon, outdoor_temp_profile(horizon))
    assert temp.shape == (horizon.steps + 1,)
    assert power.min() >= 0.0


# --------------------------------------------------------------------------
# the state grid floor must not invent a temperature the tank never held
# --------------------------------------------------------------------------


def test_grid_floor_sits_at_or_below_thermal_equilibrium():
    """A floor above the equilibrium is a wall the model invents.

    The tank relaxes toward a flow-weighted mixture of t_ambient (through the
    insulation) and t_inlet (through the draw), so the lowest temperature it
    can ever reach is min(t_ambient, t_inlet). A floor above that would have to
    clamp the state at a temperature that was never reached.
    """
    cfg = WaterHeaterConfig()
    assert cfg.t_min <= min(cfg.t_ambient, cfg.t_inlet)

    with pytest.raises(ValueError, match="at or below the thermal equilibrium"):
        WaterHeaterConfig(t_min=40.0).validate()


def test_a_drained_tank_is_allowed_to_go_properly_cold():
    """With a draw the heater cannot meet, the tank must be free to fall well
    below the comfort target instead of being pinned just under it.

    The old floor was 40 C, fifteen kelvin ABOVE t_comfort=55, so a tank being
    drained flat was recorded at 40 C however cold it really got - fabricating
    energy and capping the discomfort the objective is supposed to price.
    """
    cfg = WaterHeaterConfig()
    horizon = Horizon(dt=0.25, hours=24.0)
    buy, sell = day_night_tariff(horizon)
    heavy = hot_water_demand_profile(horizon) * 8.0

    sol = solve_water_heater(cfg, horizon, buy, sell, heavy)
    assert sol.trajectory.min() < 40.0, "tank cannot represent being properly cold"
    # ...and the mixing temperature is an exact floor, reached by bounding the
    # OUTFLOW rather than by projecting the state. Nothing clamps here any
    # more. Under a heavy draw that floor tends to the mains temperature,
    # which is what a tank run flat actually settles at.
    assert sol.trajectory.min() >= cfg.t_inlet - 1e-9


def test_discomfort_keeps_growing_once_the_tank_is_very_cold():
    """The penalty must not saturate. Under the old 40 C floor it did: the
    measured shortfall stopped increasing no matter how much was drawn."""
    cfg = WaterHeaterConfig()
    horizon = Horizon(dt=0.25, hours=24.0)
    buy, sell = day_night_tariff(horizon)
    demand = hot_water_demand_profile(horizon)

    shortfall = []
    for mult in (4.0, 8.0, 16.0):
        sol = solve_water_heater(cfg, horizon, buy, sell, demand * mult)
        shortfall.append(
            float(np.maximum(cfg.t_comfort - sol.trajectory, 0.0).sum())
        )
    assert shortfall[0] < shortfall[1] < shortfall[2]


def test_the_state_bounds_hold_without_any_projection():
    """Both ends are respected by construction, for the tank and the room.

    The transitions do not project onto [t_min, t_max]. The ceiling holds
    because the action set is narrowed to what the cut-out permits; the floor
    holds because the outflow is bounded by what the state can supply - the
    tank may relax all the way to the mixing temperature in one slot, never
    past it. Neither is enforced by clamping the state afterwards.
    """
    horizon = Horizon(dt=0.25, hours=24.0)
    buy, sell = day_night_tariff(horizon)
    wh = WaterHeaterConfig()
    for mult in (1.0, 8.0, 200.0):
        sol = solve_water_heater(
            wh, horizon, buy, sell, hot_water_demand_profile(horizon) * mult
        )
        assert sol.trajectory.min() >= min(wh.t_ambient, wh.t_inlet) - 1e-9
        assert sol.trajectory.max() <= wh.t_max + 1e-9

    hv = HvacConfig()
    outdoor = outdoor_temp_profile(horizon)
    sol = solve_hvac(hv, horizon, buy, sell, outdoor)
    assert sol.trajectory.min() >= hv.t_min - 1e-9
    assert sol.trajectory.max() <= hv.t_max + 1e-9


def test_comfort_is_priced_on_the_temperature_the_tank_actually_has():
    """With the projection gone the shortfall must keep growing with the draw
    instead of saturating at (t_comfort - floor)."""
    horizon = Horizon(dt=0.25, hours=24.0)
    buy, sell = day_night_tariff(horizon)
    cfg = WaterHeaterConfig()
    POL = np.zeros((horizon.steps, cfg.n_states), dtype=np.int64)  # heater off

    shortfall = []
    for mult in (1.0, 4.0, 16.0):
        temp, _ = rollout_water_heater(
            cfg, horizon, POL, hot_water_demand_profile(horizon) * mult, 0, cfg.t_comfort
        )
        shortfall.append(float(np.maximum(cfg.t_comfort - temp, 0.0).sum()))
    assert shortfall[0] < shortfall[1] < shortfall[2]


# --------------------------------------------------------------------------
# the draw references the mains, not the air around the tank
# --------------------------------------------------------------------------


def test_the_draw_factor_is_one_at_the_setpoint():
    """`hot_water_demand` is carried in kW-at-setpoint, so the factor that
    rescales it must be exactly 1 there - that is what makes the forecast
    interpretable without knowing the tank's state."""
    cfg = WaterHeaterConfig()
    assert _draw_factor(cfg.t_comfort, cfg) == pytest.approx(1.0)


def test_the_draw_is_referenced_to_the_mains_not_to_ambient():
    """The insulation leaks to the air around the tank; the tap replaces what
    it takes with mains water. Two different reservoirs, two different terms.

    A tank sitting exactly at ambient still loses heat when it is drawn from,
    because mains water is colder than room air. Referencing the draw to
    ambient would make that outflow zero, and the tank would be unable to fall
    below room temperature however much was drawn.
    """
    cfg = WaterHeaterConfig()
    assert cfg.t_inlet < cfg.t_ambient, "fixture must keep the two distinct"

    # no draw: the only loss is the standing one, which vanishes at ambient
    assert _usable_outflow(cfg.t_ambient, cfg, 0.0, 0.25) == pytest.approx(0.0)
    # with a draw: strictly positive at the same temperature
    assert _usable_outflow(cfg.t_ambient, cfg, 4.0, 0.25) > 0.0


def test_the_equilibrium_mixes_ambient_and_inlet_by_flow():
    """With no draw the tank settles at ambient; under a heavy one it settles
    at the mains temperature. In between it is a flow-weighted mixture."""
    cfg = WaterHeaterConfig()
    _, idle = _relaxation(cfg, 0.0)
    _, light = _relaxation(cfg, 1.0)
    _, heavy = _relaxation(cfg, 1000.0)

    assert idle == pytest.approx(cfg.t_ambient)
    assert heavy == pytest.approx(cfg.t_inlet, abs=0.05)
    assert cfg.t_inlet < light < cfg.t_ambient


def test_the_euler_rate_cap_stops_the_state_overshooting_the_equilibrium():
    """An explicit step at rate*dt > 1 would carry the tank past the
    temperature it is relaxing toward. The cap in _usable_outflow is what makes
    the floor a property of the model rather than of the parameters."""
    cfg = WaterHeaterConfig()
    dt = 0.25
    absurd = 400.0  # rate * dt ~ 14 at the fixture
    rate, t_inf = _relaxation(cfg, absurd)
    assert rate * dt > 1.0, "pick a draw big enough to make the cap bind"

    start = cfg.t_comfort
    q_out = float(_usable_outflow(start, cfg, absurd, dt))
    landed = start - q_out / cfg.heat_capacity_kwh_per_k * dt
    assert landed == pytest.approx(t_inf), "should relax exactly to equilibrium"
    assert landed >= cfg.t_inlet - 1e-9


def test_fractional_hvac_duty_never_costs_more(horizon):
    """n_duty_levels > 2 lets the HVAC run a fraction of a slot in either
    direction - a superset of off / full cool / full heat - so at the same
    prices its plan costs no more (private cost plus energy), and every action
    it takes is one of its duty levels."""
    from dataclasses import replace

    from home_energy_optimizer.dp_thermal import hvac_discomfort

    buy, sell = day_night_tariff(horizon)
    out = outdoor_temp_profile(horizon)
    ref = float(np.mean(buy))

    def cost(cfg):
        sol = solve_hvac(cfg, horizon, buy, sell, out)
        comfort = float(np.sum(hvac_discomfort(cfg, sol.trajectory[1:], ref, horizon.dt)))
        return comfort + float(buy @ sol.power * horizon.dt), sol

    three, _ = cost(HvacConfig())
    frac = replace(HvacConfig(), n_duty_levels=5)
    frac = replace(frac, n_states=frac.states_for_duty_levels(horizon.dt))
    quarters, sol = cost(frac)
    assert quarters <= three + 1e-6
    assert len(frac.duty_actions) == 9 and set(np.unique(sol.policy)) <= set(range(9))
    assert np.allclose(HvacConfig().duty_actions, [0.0, -1.0, 1.0])     # the default is off / cool / heat


def test_thermal_tether_pulls_toward_the_target(horizon):
    """The optional ADMM tether on the thermal DPs: a large rho pulls the plan
    onto a target it can follow; rho = 0 is the untethered plan."""
    buy, sell = day_night_tariff(horizon)
    demand = hot_water_demand_profile(horizon)
    free = solve_water_heater(WaterHeaterConfig(), horizon, buy, sell, demand)
    target = np.roll(free.power, 8)                     # the same energy, two hours later
    pulled = solve_water_heater(WaterHeaterConfig(), horizon, buy, sell, demand,
                                admm_target=target, admm_rho=100.0)
    assert np.abs(pulled.power - target).sum() < np.abs(free.power - target).sum()
