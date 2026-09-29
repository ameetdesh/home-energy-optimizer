"""Is lambda a real shadow price?

`marginal_value` is the load-bearing claim of this whole package
(optimsurvey_notes/poc-integration-analysis.md section 2.2), so it is validated
against two independent ground truths rather than only self-consistency:

* the **LP dual from HiGHS** - the multiplier on the SoE transition equality is
  the discrete costate, i.e. exactly d(optimal cost)/d(stored energy);
* the **exact joint DP costate** - dV/ds of a value function that optimises the
  battery and the tank together, which is what the decomposed lambda claims to
  approximate.

Requires scipy (dev/bench dependency only), so these skip if it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="LP dual comparison needs scipy/HiGHS")

from bench.duals import agreement, joint_costate, lp_battery_with_duals  # noqa: E402
from bench.reference import joint_dp_battery_water_heater  # noqa: E402
from home_energy_optimizer import (  # noqa: E402
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    PolicySnapshot,
    SiteConfig,
    WaterHeaterConfig,
    coordinate,
    demo_forecasts,
    marginal_value,
)

# Absolute tolerance in currency/kWh. Measured worst cases: 0.031 against the
# LP dual (flat, where the true dual is nearly constant so there is no signal
# to track) and 0.039 against the joint costate (day_night). 0.045 is ~15% of a
# 0.30 tariff - loose, deliberately: this is a signal for gating devices, and
# the shipped caveat is already that decisions within +-0.04/kWh of the margin
# are unreliable. Tightening this without tightening that claim would be
# self-deception.
LAMBDA_TOL = 0.045


def _band(cfg: BatteryConfig, buy, sell) -> tuple[float, float]:
    """[sell/eta, buy*eta] - the no-arbitrage band for a lossy store.

    Acquiring a stored kWh from forgone export costs sell/eta; delivering one
    to the house saves buy*eta. Valid for a battery-ONLY problem: once a
    comfort-penalised device is coupled in, stored energy can legitimately be
    worth more than the tariff because it averts a penalty (see
    test_joint_costate_may_exceed_the_battery_only_band).
    """
    return float(np.min(sell) / cfg.eta_c), float(np.max(buy) * cfg.eta_d)


@pytest.fixture(scope="module")
def horizon() -> Horizon:
    return Horizon(dt=0.25, hours=24.0)


# --------------------------------------------------------------------------
# A. Against the LP dual
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tariff", ["flat", "day_night", "dynamic"])
def test_lp_dual_lies_in_the_no_arbitrage_band(horizon, tariff):
    """Validates the reference before trusting it as ground truth."""
    cfg = BatteryConfig(capacity_kwh=10.0)
    fc = demo_forecasts(horizon, tariff=tariff)
    lp = lp_battery_with_duals(cfg, horizon, fc.buy, fc.sell, fc.net_fixed_demand)

    lo, hi = _band(cfg, fc.buy, fc.sell)
    lam = lp["lambda"]
    assert np.all(lam >= lo - 1e-6), f"{lam.min()} below {lo}"
    assert np.all(lam <= hi + 1e-6), f"{lam.max()} above {hi}"


def test_lp_dual_matches_the_analytic_bounds_exactly(horizon):
    """On a flat tariff the band edges are hit exactly: sell/eta and buy*eta.

    This is the sharpest available check that the multiplier really is the
    costate and not, say, an off-by-dt scaling of something else.
    """
    cfg = BatteryConfig(capacity_kwh=10.0)
    fc = demo_forecasts(horizon, tariff="flat")
    lam = lp_battery_with_duals(cfg, horizon, fc.buy, fc.sell, fc.net_fixed_demand)["lambda"]

    assert lam.min() == pytest.approx(float(fc.sell[0]) / cfg.eta_c, abs=1e-6)
    assert lam.max() == pytest.approx(float(fc.buy[0]) * cfg.eta_d, abs=1e-6)


def test_two_lp_formulations_agree(horizon):
    """The explicit-SoE LP must reproduce the compact one, or the duals are
    duals of a different problem."""
    from bench.reference import lp_battery

    cfg = BatteryConfig(capacity_kwh=10.0)
    fc = demo_forecasts(horizon, tariff="day_night")
    a = lp_battery_with_duals(cfg, horizon, fc.buy, fc.sell, fc.net_fixed_demand)
    b = lp_battery(cfg, horizon, fc.buy, fc.sell, fc.net_fixed_demand)
    assert a["objective"] == pytest.approx(b["objective"], abs=1e-6)


@pytest.mark.parametrize("tariff", ["flat", "day_night", "dynamic"])
def test_dp_lambda_tracks_the_lp_dual(horizon, tariff):
    """The headline validation: dV/ds IS the shadow price.

    Priced at the LP's own optimal states, because a dual is a local object.
    """
    cfg = BatteryConfig(capacity_kwh=10.0)
    fc = demo_forecasts(horizon, tariff=tariff)
    lp = lp_battery_with_duals(cfg, horizon, fc.buy, fc.sell, fc.net_fixed_demand)

    site = SiteConfig(
        horizon=horizon, battery=cfg, water_heater=None, hvac=None,
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
    lam_dp = np.array(
        [marginal_value(snap, t, float(lp["soe"][t])) for t in range(horizon.steps)]
    )

    stats = agreement(lam_dp, lp["lambda"])
    assert stats["mae"] < LAMBDA_TOL, f"{tariff}: MAE {stats['mae']:.4f}"
    assert stats["mean_a"] == pytest.approx(stats["mean_b"], abs=LAMBDA_TOL)


def test_dp_lambda_converges_to_the_lp_dual_with_grid_refinement(horizon):
    """Residual disagreement is discretisation, so it must shrink with the grid.

    If it did not, the gap would be a modelling error rather than a resolution
    one, and no amount of compute would fix it.
    """
    fc = demo_forecasts(horizon, tariff="day_night")
    maes = []
    for ns, na in [(50, 21), (200, 81)]:
        cfg = BatteryConfig(capacity_kwh=10.0, n_states=ns, n_actions=na)
        lp = lp_battery_with_duals(cfg, horizon, fc.buy, fc.sell, fc.net_fixed_demand)
        site = SiteConfig(
            horizon=horizon, battery=cfg, water_heater=None, hvac=None,
            coordination=CoordinationConfig(exchange_rounds=20),
        )
        snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
        lam = np.array(
            [marginal_value(snap, t, float(lp["soe"][t])) for t in range(horizon.steps)]
        )
        maes.append(agreement(lam, lp["lambda"])["mae"])

    assert maes[1] < maes[0], f"refinement did not help: {maes}"


# --------------------------------------------------------------------------
# B. Against the exact joint costate  (the section 8.4 question)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def joint_case(horizon):
    """Battery + water heater, decomposed and jointly, on a coarse grid.

    Coarse because the joint DP is O(n_s * n_t * n_a) per step. Module-scoped
    because it takes ~0.5 s per tariff.
    """
    batt = BatteryConfig(capacity_kwh=10.0, n_states=50, n_actions=21)
    wh = WaterHeaterConfig()
    site = SiteConfig(
        horizon=horizon, battery=batt, water_heater=wh, hvac=None,
        coordination=CoordinationConfig(exchange_rounds=20),
    )
    out = {}
    for tariff in ("flat", "day_night", "dynamic"):
        fc = demo_forecasts(horizon, tariff=tariff)
        snap = PolicySnapshot.from_result(site, fc, coordinate(site, fc))
        joint = joint_dp_battery_water_heater(
            batt, wh, horizon, fc.buy, fc.sell, fc.hot_water_demand, fc.net_fixed_demand
        )
        lam_joint = joint_costate(joint, horizon)
        lam_dec = np.array(
            [marginal_value(snap, t, float(joint["soe"][t])) for t in range(horizon.steps)]
        )
        out[tariff] = (batt, fc, lam_dec, lam_joint)
    return out


@pytest.mark.parametrize("tariff", ["flat", "day_night", "dynamic"])
def test_decomposed_lambda_tracks_the_joint_costate(joint_case, tariff):
    """Does the price survive decomposition? Measured: yes, to ~0.03/kWh.

    The decomposed battery solves with the tank as exogenous load, so its
    dV/ds is a marginal value CONDITIONAL on the tank's plan. This asserts
    that conditioning costs about as much as the state grid already does -
    i.e. decomposition does not additionally break the price.
    """
    _, _, lam_dec, lam_joint = joint_case[tariff]
    stats = agreement(lam_dec, lam_joint)
    assert stats["mae"] < LAMBDA_TOL, f"{tariff}: MAE {stats['mae']:.4f}"


@pytest.mark.parametrize("tariff", ["flat", "day_night", "dynamic"])
def test_decomposed_lambda_is_directionally_right(joint_case, tariff):
    """Correlation matters more than level for a signal used to gate devices."""
    _, _, lam_dec, lam_joint = joint_case[tariff]
    assert agreement(lam_dec, lam_joint)["corr"] > 0.8


def test_joint_costate_may_exceed_the_battery_only_band(joint_case):
    """Documents why the no-arbitrage band is NOT a valid check once devices
    are coupled.

    With a comfort-penalised tank in the problem, a stored kWh can be worth
    more than buy*eta, because it can avert a comfort penalty rather than
    merely displace an import. Asserting the band here would be wrong, and
    noticing that is the point.
    """
    batt, fc, _, lam_joint = joint_case["flat"]
    lo, hi = _band(batt, fc.buy, fc.sell)
    assert lam_joint.max() > hi, "expected coupling to lift lambda above the tariff bound"
