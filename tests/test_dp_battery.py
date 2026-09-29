"""Battery DP correctness.

These assert economic behaviour, not numbers copied from a run. The POC had no
tests at all - correctness was verified by looking at a chart - so these are
written to fail loudly if the physics or the sign conventions drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from home_energy_optimizer import BatteryConfig, Horizon, SocGate, solve_battery


@pytest.fixture
def horizon() -> Horizon:
    return Horizon(dt=0.25, hours=24.0)


@pytest.fixture
def cfg() -> BatteryConfig:
    return BatteryConfig(capacity_kwh=10.0, p_charge_max_kw=5.0, p_discharge_max_kw=5.0)


def test_shapes_and_grids(horizon, cfg):
    n = horizon.steps
    buy = np.full(n, 0.30)
    sell = np.full(n, 0.05)
    sol = solve_battery(cfg, horizon, buy, sell)

    assert sol.value.shape == (n + 1, cfg.n_states)
    assert sol.policy.shape == (n, cfg.n_states)
    assert sol.trajectory.shape == (n + 1,)
    assert sol.power.shape == (n,)
    assert sol.states[0] == 0.0
    assert sol.states[-1] == pytest.approx(cfg.capacity_kwh)
    assert np.all(np.isfinite(sol.value))


def test_soe_stays_within_bounds(horizon, cfg):
    n = horizon.steps
    buy = np.linspace(0.05, 0.60, n)
    sol = solve_battery(cfg, horizon, buy, buy * 0.3)
    assert sol.trajectory.min() >= -1e-9
    assert sol.trajectory.max() <= cfg.capacity_kwh + 1e-9


def test_power_respects_rating(horizon, cfg):
    n = horizon.steps
    buy = np.linspace(0.05, 0.60, n)
    sol = solve_battery(cfg, horizon, buy, buy * 0.3)
    assert sol.power.max() <= cfg.p_charge_max_kw + 1e-9
    assert sol.power.min() >= -cfg.p_discharge_max_kw - 1e-9


def test_soe_recursion_is_self_consistent(horizon, cfg):
    """Replaying the returned power must reproduce the returned trajectory.

    Guards the efficiency convention: charge stores eta*a, discharge drains
    a/eta. Getting this backwards is a classic sign-convention bug and would
    otherwise only show up as slightly-wrong money.
    """
    n = horizon.steps
    buy = np.linspace(0.05, 0.60, n)
    sol = solve_battery(cfg, horizon, buy, buy * 0.3)

    s = sol.trajectory[0]
    for t in range(n):
        a = sol.power[t]
        eff = a * cfg.eta_c if a > 0 else a / cfg.eta_d
        s = float(np.clip(s + eff * horizon.dt, 0.0, cfg.capacity_kwh))
        assert s == pytest.approx(sol.trajectory[t + 1], abs=1e-9)


def test_soe_bounds_need_no_projection(horizon, cfg):
    """Energy must be conserved at the bounds, not clipped into shape.

    The action limits carry the efficiencies: charging at `a` adds `eta_c*a*dt`,
    discharging `|a|` drains `|a|*dt/eta_d`. Using the un-adjusted `(cap-s)/dt`
    and `(0-s)/dt` instead lets a nearly empty battery discharge past zero,
    where the projection onto [0, cap] silently absorbs the deficit and the plan
    delivers energy that never existed - 0.111 kWh in a single slot at
    s = 1 kWh with eta = 0.9. It also under-charges near full, so the store can
    never quite reach capacity.

    With correct bounds `s + eff*dt` lands inside [0, cap] by construction.
    """
    from home_energy_optimizer.dp_battery import _feasible_actions

    dt, cap = horizon.dt, cfg.capacity_kwh
    grid = np.linspace(-cfg.p_discharge_max_kw, cfg.p_charge_max_kw, 41)
    for s in np.linspace(0.0, cap, 21):
        Ac = _feasible_actions(
            grid, np.array(s), dt, cap, 0.0, 0.0, cfg.eta_c, cfg.eta_d
        )
        eff = np.where(Ac > 0, Ac * cfg.eta_c, Ac / cfg.eta_d)
        nxt = s + eff * dt
        assert nxt.min() >= -1e-9, f"s={s}: discharges to {nxt.min()}, below empty"
        assert nxt.max() <= cap + 1e-9, f"s={s}: charges to {nxt.max()}, above full"


def test_a_full_battery_is_reachable(horizon, cfg):
    """The un-adjusted charge bound stops at cap - (cap-s)(1-eta_c), so the
    store creeps toward capacity without arriving."""
    from home_energy_optimizer.dp_battery import _feasible_actions

    s = cfg.capacity_kwh - 1.0
    Ac = _feasible_actions(
        np.linspace(-5, 5, 101), np.array(s), horizon.dt, cfg.capacity_kwh,
        0.0, 0.0, cfg.eta_c, cfg.eta_d,
    )
    reached = s + Ac.max() * cfg.eta_c * horizon.dt
    assert reached == pytest.approx(cfg.capacity_kwh, abs=1e-9)


def test_an_empty_battery_delivers_only_what_it_holds(horizon, cfg):
    """The failure this guards: energy created from nothing at the low bound."""
    from home_energy_optimizer.dp_battery import _feasible_actions

    s = 1.0
    Ac = _feasible_actions(
        np.linspace(-5, 5, 101), np.array(s), horizon.dt, cfg.capacity_kwh,
        0.0, 0.0, cfg.eta_c, cfg.eta_d,
    )
    drained = -Ac.min() / cfg.eta_d * horizon.dt
    assert drained == pytest.approx(s, abs=1e-9), "must not drain more than is stored"


def test_arbitrage_buys_low_and_sells_high(horizon, cfg):
    """With a cheap first half and an expensive second half, charge then discharge."""
    n = horizon.steps
    buy = np.concatenate([np.full(n // 2, 0.05), np.full(n - n // 2, 0.60)])
    sell = buy * 0.9  # high export price so discharging is actually worth it
    sol = solve_battery(cfg, horizon, buy, sell)

    first_half = sol.power[: n // 2].sum()
    second_half = sol.power[n // 2 :].sum()
    assert first_half > 0, "should charge during the cheap window"
    assert second_half < 0, "should discharge during the expensive window"


def test_flat_prices_produce_no_cycling(horizon, cfg):
    """No price spread and a lossy battery means cycling only burns energy."""
    n = horizon.steps
    buy = np.full(n, 0.30)
    sell = np.full(n, 0.30)
    sol = solve_battery(cfg, horizon, buy, sell)
    throughput = np.abs(sol.power).sum() * horizon.dt
    assert throughput < 0.5 * cfg.capacity_kwh


def test_value_function_increases_with_stored_energy(horizon, cfg):
    """More stored energy is never worth less. This is what makes dV/ds a price."""
    n = horizon.steps
    buy = np.linspace(0.10, 0.50, n)
    sol = solve_battery(cfg, horizon, buy, buy * 0.9)
    mid = n // 2
    diffs = np.diff(sol.value[mid])
    assert (diffs >= -1e-6).mean() > 0.9, "V should be essentially monotone in SoE"


def test_soc_gate_is_respected(horizon, cfg):
    """A gate demanding 90% at hour 8 should pull the trajectory up to meet it."""
    n = horizon.steps
    buy = np.full(n, 0.30)
    sell = np.full(n, 0.05)
    gate = SocGate(hour=8.0, soc_frac=0.9)

    without = solve_battery(cfg, horizon, buy, sell)
    with_gate = solve_battery(cfg, horizon, buy, sell, soc_gates=(gate,))

    step = int(8.0 / horizon.dt)
    assert with_gate.trajectory[step] > without.trajectory[step]
    assert with_gate.trajectory[step] >= 0.85 * cfg.capacity_kwh


def test_dp_load_shifts_the_plan():
    """A large concurrent load changes the marginal cost of charging."""
    horizon = Horizon(dt=0.25, hours=12.0)
    cfg = BatteryConfig(capacity_kwh=10.0)
    n = horizon.steps
    buy = np.full(n, 0.30)
    sell = np.full(n, 0.25)

    quiet = solve_battery(cfg, horizon, buy, sell, dp_load=np.zeros(n))
    busy = solve_battery(cfg, horizon, buy, sell, dp_load=np.full(n, -6.0))  # big export

    assert not np.allclose(quiet.power, busy.power)


def test_admm_target_pulls_the_solution(horizon, cfg):
    n = horizon.steps
    buy = np.full(n, 0.30)
    sell = np.full(n, 0.05)
    target = np.full(n, 2.0)

    free = solve_battery(cfg, horizon, buy, sell)
    pulled = solve_battery(cfg, horizon, buy, sell, admm_target=target, admm_rho=50.0)

    assert np.abs(pulled.power - target).mean() < np.abs(free.power - target).mean()


def test_min_soe_is_a_hard_floor(horizon, cfg):
    """soe_min_frac holds in the DP's plan and in the fast tier's decisions,
    and a battery that starts below it is simply not discharged further."""
    from dataclasses import replace

    from home_energy_optimizer import demo_forecasts

    fc = demo_forecasts(horizon, tariff="dynamic")
    b = replace(cfg, soe_min_frac=0.3)
    sol = solve_battery(b, horizon, fc.buy, fc.sell)
    assert sol.trajectory.min() >= 0.3 * b.capacity_kwh - 1e-9
    assert sol.states[0] == pytest.approx(0.3 * b.capacity_kwh)
    free = solve_battery(cfg, horizon, fc.buy, fc.sell)
    assert free.trajectory.min() < 0.3 * cfg.capacity_kwh     # the floor binds here

    low = replace(cfg, soe_min_frac=0.5, soc_initial_frac=0.2)
    assert low.soe_floor_kwh == pytest.approx(0.2 * cfg.capacity_kwh)
    assert solve_battery(low, horizon, fc.buy, fc.sell).trajectory.min() >= 0.2 * cfg.capacity_kwh - 1e-9


def test_priced_limits_leave_only_the_headroom(horizon, cfg):
    """With `limits`, a battery charging into a cheap slot takes what is left
    under the import limit given the others' load - and no more, while the
    breach price is above what the stored energy is worth."""
    from home_energy_optimizer.meter import Limits

    n = horizon.steps
    buy = np.where(np.arange(n) < n // 2, 0.05, 0.40)
    sell = np.full(n, 0.02)
    load = np.full(n, 2.0)
    free = solve_battery(cfg, horizon, buy, sell, dp_load=load)
    capped = solve_battery(cfg, horizon, buy, sell, dp_load=load, limits=Limits(3.0, None, 5.0, True))
    assert (free.power + load).max() > 3.0 + 1e-6            # unconstrained, it would breach
    assert (capped.power + load).max() <= 3.0 + 1e-6         # priced, it takes the 1 kW left
    assert capped.power[: n // 2].sum() > 0                   # and still charges when cheap
