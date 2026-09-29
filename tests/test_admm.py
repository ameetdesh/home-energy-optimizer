"""Textbook ADMM - proximal message passing with the DPs as device steps
(admm.coordinator)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dw.coordinator import Column, DWCoordinator  # noqa: E402
from dw.webapi import _site_fc, solve  # noqa: E402
from hemspolicy.coordinate import coordinate  # noqa: E402
from admm.coordinator import _Grid  # noqa: E402
from hemspolicy.types import CoordinationConfig  # noqa: E402


@pytest.mark.parametrize("delta", [0.0, 0.25])
@pytest.mark.parametrize("limits", [(None, None), (7.0, 3.0)])
def test_grid_step_is_the_exact_prox(delta, limits):
    """The grid connection's closed-form step against brute force: for every
    slot, min over z of cost(z) + (rho/2)(z + v)^2 on a fine grid."""
    rng = np.random.default_rng(1)
    n, dt = 24, 0.25
    buy = rng.uniform(0.1, 0.4, n)
    sell = buy * rng.uniform(0.2, 0.9, n)
    grid = _Grid(buy, sell, dt, limits[0], limits[1], 2.0, delta)
    v = rng.uniform(-12, 12, n)
    for rho in (0.01, 0.1, 1.0):
        x = grid.prox(v, rho)                       # power delivered to the grid (= -import)
        zs = np.linspace(-20, 20, 200001)
        for t in range(n):
            obj = grid.cost_t(zs, t) + 0.5 * rho * (zs + v[t]) ** 2
            assert -x[t] == pytest.approx(zs[np.argmin(obj)], abs=5e-4)


def test_the_rounded_kink_is_the_bill_outside_its_band():
    buy, sell = np.array([0.3]), np.array([0.1])
    exact, smooth = _Grid(buy, sell, 1.0, None, None, 0.0, 0.0), _Grid(buy, sell, 1.0, None, None, 0.0, 0.25)
    for z in (-3.0, -0.25, 0.25, 2.0):
        assert smooth.cost(np.array([z]))[0] == pytest.approx(exact.cost(np.array([z]))[0])
    assert smooth.cost(np.array([0.0]))[0] > exact.cost(np.array([0.0]))[0]   # above the kink, within the band


def _score(site, fc, res):
    co = DWCoordinator(site, fc)
    r = co.run(max_iter=40)
    val = co.parts({k: Column(d.power, d.trajectory, 0.0, "x") for k, d in res.devices.items()})["total"]
    return val - r.lower


def test_batteries_alone_come_close_to_the_bound():
    """Batteries only - convex apart from the DP grid - under a limit: textbook
    ADMM lands within the grid's reach of the Dantzig-Wolfe bound."""
    site, fc, _ = _site_fc({"tariff": "dynamic", "n_batteries": 2, "grid": 50, "hours": 24,
                            "enable_wh": False, "enable_hvac": False, "max_import_kw": 7})
    res = coordinate(replace(site, coordination=CoordinationConfig(exchange_rounds=300)), fc)
    assert _score(site, fc, res) < 0.1
    assert res.stop_reason in ("converged", "no improvement", "iteration cap")
    assert 0 <= res.selected_round < len(res.rounds) == res.rounds_run


def test_the_plan_is_runnable_by_the_real_devices():
    """The iterations use fractional copies of the on/off tank and three-way
    HVAC; the returned plan is theirs: the tank on or off in every slot but the
    last, the HVAC's power one of its three actions (duty-capped)."""
    site, fc, _ = _site_fc({"tariff": "day_night", "n_batteries": 1, "grid": 50, "hours": 24})
    seen = []
    res = coordinate(replace(site, coordination=CoordinationConfig(exchange_rounds=60)), fc,
                     progress=lambda *a: seen.append(a))
    assert len(res.devices["water_heater"].actions) == 2          # the real on/off element
    assert len(res.devices["hvac"].actions) == 3                  # the real off / cool / heat unit
    # every iteration: the fractional plan's cost, and a runnable plan recovered from it
    assert all(r.relaxed_objective is not None for r in res.rounds)
    # the page's live figure: after each iteration, the best runnable cost so far - never rising
    best = [a[4] for a in seen if len(a) == 5]
    assert len(best) == res.rounds_run and all(b <= a + 1e-12 for a, b in zip(best, best[1:]))
    assert best[-1] == pytest.approx(min(r.total_objective for r in res.rounds))


def test_the_page_plans_with_admm():
    r = solve({"method": "admm", "tariff": "dynamic", "n_batteries": 1, "grid": 50, "hours": 24, "max_iter": 40})
    s = r["summary"]
    assert s["relaxed_rounds"] is True and s["battery_step"] == "dp"
    assert len(r["iterations"]) == s["iterations"] and s["stop_reason"]
    # every iteration's numbers; full plans for about ten of them, the best and the last
    its = r["iterations"]
    kept = [i for i, it in enumerate(its) if it["snapshot"]]
    assert all("plan_value" in it and "best_plan" in it for it in its)
    assert all("devices" in its[i] for i in kept) and all("devices" not in it for it in its if not it["snapshot"])
    assert len(kept) <= 12 and s["selected_round"] - 1 in kept and len(its) - 1 in kept


def test_a_paused_run_resumes_exactly():
    """Run in chunks, looking at the paused plan between them, and it ends where
    one uninterrupted run ends - pausing changes nothing."""
    from admm.coordinator import ExchangeRun

    site, fc, _ = _site_fc({"tariff": "dynamic", "n_batteries": 1, "grid": 50, "hours": 24})
    site = replace(site, coordination=CoordinationConfig(exchange_rounds=25))
    whole = ExchangeRun(site, fc)
    whole.step()
    chunked = ExchangeRun(site, fc)
    paused = []
    while not chunked.step(4):
        snap = chunked.result()                        # what the page shows while paused
        paused.append((chunked.k, snap.stop_reason, snap.total_objective))
    assert paused and all(r == "paused" for _, r, _ in paused)
    a, b = whole.result(), chunked.result()
    assert chunked.k == whole.k and b.stop_reason == a.stop_reason != "paused"
    assert b.total_objective == pytest.approx(a.total_objective, abs=1e-12)
    assert [r.total_objective for r in b.rounds] == [r.total_objective for r in a.rounds]


def test_the_page_can_pause_and_resume_a_solve():
    """The page's route: a few iterations per call, a pause that returns the
    best plan so far, and a resume that ends where an unpaused solve ends."""
    from dw import webapi

    p = {"method": "admm", "tariff": "dynamic", "n_batteries": 1, "grid": 50, "hours": 24, "max_iter": 20}
    whole = webapi.call("solve", {**p, "_progress_id": "w"})
    r = webapi.call("solve", {**p, "_progress_id": "c", "chunk": 3})
    assert r["running"] and r["progress"][0] == 3 and r["progress"][4] is not None
    paused = webapi.call("solve", {**p, "_progress_id": "c", "resume": True, "pause": True})
    assert paused["summary"]["stop_reason"] == "paused" and paused["summary"]["iterations"] == 3
    while True:
        r = webapi.call("solve", {**p, "_progress_id": "c", "resume": True, "chunk": 5})
        if not r.get("running"):
            break
    assert r["summary"]["iterations"] == whole["summary"]["iterations"]
    assert r["summary"]["upper"] == pytest.approx(whole["summary"]["upper"], abs=1e-9)


def test_the_lp_battery_step_is_the_exact_prox():
    """A battery's LP step (admm.battery_qp) against OSQP on the same QP,
    over random batteries and rho across four decades."""
    pytest.importorskip("osqp")
    sys.path.insert(0, str(ROOT / "bench"))
    from prior_art import Battery

    from admm.battery_qp import battery_prox
    from hemspolicy.types import BatteryConfig

    rng = np.random.default_rng(3)
    for _ in range(20):
        b = BatteryConfig(capacity_kwh=float(rng.uniform(5, 30)), p_charge_max_kw=float(rng.uniform(2, 7)),
                          p_discharge_max_kw=float(rng.uniform(2, 7)), soc_initial_frac=float(rng.uniform(0.1, 0.9)),
                          soe_min_frac=float(rng.choice([0.0, 0.2])), terminal_price=float(rng.uniform(0, 0.4)))
        v, rho = rng.normal(0, 4, 96), float(10 ** rng.uniform(-3, 1))
        mine = battery_prox(b, 0.25, v, rho, b.terminal_price)
        ref = Battery(b, 96, 0.25)
        p_ref = ref.prox(v, rho)
        assert mine.converged
        obj = lambda p, s: 0.5 * rho * np.sum((p - v) ** 2) - b.terminal_price * s[-1]  # noqa: E731
        assert obj(mine.power, mine.trajectory) <= obj(p_ref, ref.trajectory()) + 1e-6 * (1 + rho)
        if rho >= 0.01:        # the plan is unique, but only as sharply as rho makes it
            assert np.abs(mine.power - p_ref).max() < 1e-3


def test_lp_batteries_reach_the_optimum_of_a_convex_site():
    """Batteries alone are convex: with each battery's step its exact LP and the
    bill left unrounded, textbook ADMM converges to the optimum - here the DW
    master's, which holds the batteries exactly (gap 0) - to within what the
    stopping test allows. With DP steps the same site does not settle."""
    site, fc, _ = _site_fc({"tariff": "dynamic", "n_batteries": 3, "grid": 50, "hours": 24,
                            "enable_wh": False, "enable_hvac": False, "max_import_kw": 7})
    co = DWCoordinator(site, fc)
    r = co.run(max_iter=40)
    assert r.upper - r.lower < 1e-6
    cc = CoordinationConfig(exchange_battery_step="lp", kink_smoothing=0.0,
                            exchange_eps=1e-4, exchange_rounds=600, exchange_patience=600)
    res = coordinate(replace(site, coordination=cc), fc)
    assert res.stop_reason == "converged"
    assert _score(site, fc, res) < 0.01
    dp = coordinate(replace(site, coordination=CoordinationConfig()), fc)
    assert dp.stop_reason == "iteration cap"


def test_the_page_offers_lp_battery_steps():
    r = solve({"method": "admm", "battery_step": "lp", "tariff": "dynamic", "n_batteries": 2, "grid": 50,
               "hours": 24, "max_iter": 40})
    assert r["summary"]["battery_step"] == "lp"
    assert solve({"method": "admm", "tariff": "dynamic", "n_batteries": 1, "grid": 50, "hours": 24,
                  "max_iter": 10})["summary"]["battery_step"] == "dp"


def test_a_warm_started_battery_step_gives_the_same_answer_sooner():
    from admm.battery_qp import battery_prox
    from hemspolicy.types import BatteryConfig

    b = BatteryConfig(capacity_kwh=12.0, terminal_price=0.2)
    rng = np.random.default_rng(5)
    v1 = rng.normal(0, 3, 96)
    v2 = v1 + rng.normal(0, 0.2, 96)                   # the next iteration's target: a little different
    first = battery_prox(b, 0.25, v1, 0.3, 0.2)
    cold = battery_prox(b, 0.25, v2, 0.3, 0.2)
    warm = battery_prox(b, 0.25, v2, 0.3, 0.2, start=first.state)
    assert np.abs(warm.power - cold.power).max() < 1e-4
    assert warm.iterations < cold.iterations


def test_a_solve_can_start_where_the_last_one_stood():
    """After a setting changes, a solve started from the last one's best state
    reaches a good plan in far fewer iterations; a state for other devices is
    ignored."""
    from admm.coordinator import ExchangeRun

    base = {"tariff": "dynamic", "n_batteries": 1, "grid": 50, "hours": 24, "max_import_kw": 7}
    cc = CoordinationConfig(exchange_rounds=40)
    site_a, fc_a, _ = _site_fc(base)
    first = ExchangeRun(replace(site_a, coordination=cc), fc_a)
    first.step()
    site_b, fc_b, _ = _site_fc({**base, "solar_peak": 6})
    cold = ExchangeRun(replace(site_b, coordination=cc), fc_b)
    warm = ExchangeRun(replace(site_b, coordination=cc), fc_b, warm=first.warm_state())
    cold.step()
    warm.step()
    assert warm.warm_used and not cold.warm_used
    within = lambda run: next(i for i, r in enumerate(run.records) if r.total_objective <= run.best_obj + 0.05)  # noqa: E731
    assert within(warm) < within(cold)
    site_c, fc_c, _ = _site_fc({**base, "n_batteries": 2})
    assert not ExchangeRun(replace(site_c, coordination=cc), fc_c, warm=first.warm_state()).warm_used


def test_a_warm_state_moves_with_the_horizon():
    from admm.coordinator import WarmStart

    p = np.arange(12.0).reshape(3, 4)
    w = WarmStart(("battery",), p, np.array([1.0, 2.0, 3.0, 4.0]), 0.1, {}).shift(1)
    assert w.p.tolist() == [[1, 2, 3, 3], [5, 6, 7, 7], [9, 10, 11, 11]]
    assert w.u.tolist() == [2, 3, 4, 4] and w.rho == 0.1


def test_the_page_starts_warm_only_when_asked():
    from dw import webapi

    p = {"method": "admm", "tariff": "dynamic", "n_batteries": 1, "grid": 50, "hours": 24, "max_iter": 15}
    assert webapi.call("solve", {**p, "_progress_id": "w1"})["summary"]["warm_started"] is False
    warm = webapi.call("solve", {**p, "solar_peak": 6, "warm_start": True, "_progress_id": "w2"})
    assert warm["summary"]["warm_started"] is True
    assert webapi.call("solve", {**p, "_progress_id": "w3"})["summary"]["warm_started"] is False
