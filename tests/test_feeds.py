"""Forecast ingestion and measured-value blending.

Network tests are opt-in (`HEMS_NETWORK_TESTS=1`) so the suite stays offline
by default; everything else here is pure.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hemspolicy import Horizon
from hemspolicy.feeds import (
    blend_measured,
    from_csv,
    from_series,
    open_meteo_pv,
    open_meteo_temperature,
    resample_to_horizon,
)


@pytest.fixture
def horizon() -> Horizon:
    return Horizon(dt=0.25, hours=24.0)


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def test_hourly_series_upsamples_to_the_horizon(horizon):
    hourly = np.arange(24, dtype=float)
    out = resample_to_horizon(hourly, 1.0, horizon)
    assert out.shape == (horizon.steps,)
    assert out[0] == pytest.approx(0.0)
    assert out[4] == pytest.approx(1.0)  # 1 h in = second hourly sample


def test_short_series_holds_its_last_value(horizon):
    """A day-ahead tariff publishes 24 h; the horizon may want 48.

    EMHASS's AGENTS.md flags silent misalignment as a way to get a plan that
    reports Optimal with every timestep offset, so padding must be explicit.
    """
    long_h = Horizon(dt=0.25, hours=48.0)
    day = np.full(24, 0.3)
    out = resample_to_horizon(day, 1.0, long_h)
    assert out.shape == (long_h.steps,)
    assert out[-1] == pytest.approx(0.3)


def test_zero_fill_for_pv_beyond_the_forecast(horizon):
    long_h = Horizon(dt=0.25, hours=48.0)
    out = resample_to_horizon(np.full(24, 2.0), 1.0, long_h, fill="zero")
    assert out[-1] == pytest.approx(0.0)


def test_from_series_assumes_full_horizon_span(horizon):
    out = from_series([0.0, 1.0], horizon)
    assert out.shape == (horizon.steps,)
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0, abs=0.02)


def test_empty_series_is_rejected(horizon):
    with pytest.raises(ValueError, match="empty"):
        resample_to_horizon(np.array([]), 1.0, horizon)


def test_from_csv_roundtrip(tmp_path, horizon):
    p = tmp_path / "tariff.csv"
    p.write_text(
        "\n".join(f"2026-01-01T{h:02d}:00:00,{0.1 + 0.01 * h:.3f}" for h in range(24))
    )
    out = from_csv(str(p), horizon)
    assert out.shape == (horizon.steps,)
    assert out[0] == pytest.approx(0.10)
    assert out[-1] == pytest.approx(0.33, abs=0.01)


def test_from_csv_needs_two_rows(tmp_path, horizon):
    p = tmp_path / "x.csv"
    p.write_text("2026-01-01T00:00:00,1.0")
    with pytest.raises(ValueError, match="at least 2 rows"):
        from_csv(str(p), horizon)


# --------------------------------------------------------------------------
# Measured-value blending  (evcc core/optimizer.md)
# --------------------------------------------------------------------------


def test_replace_anchors_slot_zero_exactly():
    fc = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = blend_measured(fc, measured=2.0, decay_slots=4, mode="replace")
    assert out[0] == pytest.approx(2.0)


def test_replace_decays_back_to_the_forecast():
    fc = np.full(8, 1.0)
    out = blend_measured(fc, measured=2.0, decay_slots=4, mode="replace")
    assert out[4] == pytest.approx(1.0), "should be back on forecast after decay_slots"
    assert out[1] > out[2] > out[3], "monotone decay"
    assert np.allclose(out[4:], 1.0)


def test_scale_applies_a_ratio_not_an_offset():
    """Solar: the shape is trusted, the level is not."""
    fc = np.array([4.0, 8.0, 8.0, 8.0, 8.0])
    out = blend_measured(fc, measured=2.0, decay_slots=4, mode="scale")
    assert out[0] == pytest.approx(2.0)  # 0.5x
    assert out[1] == pytest.approx(8.0 * (1 + (0.5 - 1) * 0.75))
    assert out[4] == pytest.approx(8.0)


def test_scale_is_a_noop_at_night():
    """A zero forecast makes the ratio meaningless; leave it alone rather than
    dividing by ~0 and producing a garbage plan."""
    fc = np.array([0.0, 0.0, 1.0, 3.0])
    out = blend_measured(fc, measured=0.4, decay_slots=4, mode="scale")
    assert np.allclose(out, fc)


def test_blending_never_changes_length_or_touches_the_tail():
    fc = np.linspace(1.0, 5.0, 20)
    for mode, m in (("replace", 3.0), ("scale", 3.0)):
        out = blend_measured(fc, m, decay_slots=4, mode=mode)
        assert out.shape == fc.shape
        assert np.allclose(out[4:], fc[4:])


def test_zero_decay_is_a_noop():
    fc = np.full(5, 1.0)
    assert np.allclose(blend_measured(fc, 9.0, decay_slots=0), fc)


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError, match="replace"):
        blend_measured(np.ones(4), 1.0, mode="nonsense")


def test_blend_does_not_mutate_its_input():
    fc = np.ones(6)
    blend_measured(fc, 5.0)
    assert np.allclose(fc, 1.0)


# --------------------------------------------------------------------------
# Network (opt-in)
# --------------------------------------------------------------------------

network = pytest.mark.skipif(
    os.environ.get("HEMS_NETWORK_TESTS") != "1",
    reason="set HEMS_NETWORK_TESTS=1 to hit Open-Meteo",
)


@network
def test_open_meteo_pv_shape_and_physics(horizon):
    pv = open_meteo_pv(52.52, 13.41, horizon, peak_kw=5.0)
    assert pv.shape == (horizon.steps,)
    assert pv.min() >= 0.0
    assert pv.max() <= 5.0 * 1.2, "production should not exceed peak by much"
    assert (pv == 0).sum() > horizon.steps * 0.3, "expect a night"


@network
def test_open_meteo_temperature_is_plausible(horizon):
    t = open_meteo_temperature(52.52, 13.41, horizon)
    assert t.shape == (horizon.steps,)
    assert -50.0 < t.min() and t.max() < 60.0
