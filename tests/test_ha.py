"""Home Assistant publishing.

No live HA needed: the client is stubbed, so these check the payloads rather
than the transport. The transport is exercised for real by
`tools/ha-lambda-demo/run.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemspolicy import (
    BatteryConfig,
    CoordinationConfig,
    Horizon,
    PolicySnapshot,
    SiteConfig,
    coordinate,
    demo_forecasts,
)
from hemspolicy.ha import (
    LAMBDA_UNCERTAINTY,
    HomeAssistant,
    _jsonable,
    lambda_forecast,
    publish_policy,
)


class FakeHA(HomeAssistant):
    """Captures what would have been posted."""

    def __init__(self):
        super().__init__(base_url="http://stub", token="stub")
        self.states: dict[str, dict] = {}

    def _post(self, path: str, payload: dict) -> dict:
        entity = path.rsplit("/", 1)[-1]
        self.states[entity] = payload
        return payload


@pytest.fixture(scope="module")
def snap() -> PolicySnapshot:
    site = SiteConfig(
        horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=10.0, n_states=100, n_actions=51),
        water_heater=None,
        hvac=None,
        coordination=CoordinationConfig(max_rounds=6),
    )
    fc = demo_forecasts(site.horizon, tariff="day_night")
    return PolicySnapshot.from_result(site, fc, coordinate(site, fc))


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_numpy_is_converted():
    """HA's REST API rejects numpy scalars, and they leak out of any solver."""
    out = _jsonable(
        {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, 2.0]), "d": [np.float32(0.5)]}
    )
    assert out == {"a": 1.5, "b": 3, "c": [1.0, 2.0], "d": [0.5]}
    import json

    json.dumps(out)  # must not raise


# --------------------------------------------------------------------------
# The lambda sensor
# --------------------------------------------------------------------------


def test_publishes_the_expected_entities(snap):
    ha = FakeHA()
    publish_policy(ha, snap, 40, 5.0)
    assert set(ha.states) == {
        "sensor.hems_lambda",
        "sensor.hems_worth_running",
        "sensor.hems_battery_action",
        "sensor.hems_battery_soe",
        "sensor.hems_import_price",
        # Reservation prices: the band a storage owner trades around.
        "sensor.hems_import_below",
        "sensor.hems_export_above",
    }


def test_reservation_prices_bracket_lambda(snap):
    """The band must straddle lambda: you would pay less than it is worth to
    buy, and demand more than it is worth to sell. The gap is the round trip."""
    ha = FakeHA()
    publish_policy(ha, snap, 40, 5.0)
    lam = float(ha.states["sensor.hems_lambda"]["state"])
    bid = float(ha.states["sensor.hems_import_below"]["state"])
    ask = float(ha.states["sensor.hems_export_above"]["state"])
    assert bid <= lam + 1e-4 <= ask + 2e-4
    assert bid < ask


def test_lambda_ships_with_its_uncertainty_band(snap):
    """The validation put lambda within ~0.04/kWh of two ground truths, so a
    decision inside that band is not reliable. Publishing a bare number would
    invite exactly the over-trust the measurement warns against."""
    ha = FakeHA()
    publish_policy(ha, snap, 40, 5.0)
    attrs = ha.states["sensor.hems_lambda"]["attributes"]

    lam = float(ha.states["sensor.hems_lambda"]["state"])
    assert attrs["uncertainty"] == LAMBDA_UNCERTAINTY
    assert attrs["confident_above"] == pytest.approx(lam + LAMBDA_UNCERTAINTY, abs=1e-4)
    assert attrs["confident_below"] == pytest.approx(lam - LAMBDA_UNCERTAINTY, abs=1e-4)


def test_lambda_is_in_tariff_units(snap):
    ha = FakeHA()
    publish_policy(ha, snap, 40, 5.0, currency="EUR")
    s = ha.states["sensor.hems_lambda"]
    assert s["attributes"]["unit_of_measurement"] == "EUR/kWh"
    assert s["attributes"]["device_class"] == "monetary"
    assert 0.0 < float(s["state"]) < 1.0


def test_forecast_attribute_follows_emhass_shape(snap):
    """EMHASS publishes a `forecast` attribute array; matching it means the
    same ApexCharts card configs work."""
    fcst = lambda_forecast(snap, 5.0, ahead=96)
    assert len(fcst) == 96
    for p in fcst[:3]:
        assert set(p) == {"hours_ahead", "lambda", "import_price", "export_price"}
    assert fcst[0]["hours_ahead"] == 0.0
    assert fcst[4]["hours_ahead"] == pytest.approx(1.0)


def test_forecast_is_priced_at_a_fixed_state(snap):
    """The curve answers "what would a kWh be worth at each future time", not
    "where will the battery be" - otherwise it depends on the plan and stops
    being a price."""
    a = lambda_forecast(snap, 3.0, ahead=8)
    b = lambda_forecast(snap, 8.0, ahead=8)
    assert [p["lambda"] for p in a] != [p["lambda"] for p in b]
    # ...and each is internally consistent with marginal_value at that state.
    from hemspolicy import marginal_value

    assert a[5]["lambda"] == pytest.approx(marginal_value(snap, 5, 3.0), abs=1e-5)


def test_lambda_tracks_the_tariff_over_the_day(snap):
    """The signal must carry information, not just be well-formed.

    Theory: a stored kWh is worth about the export price when PV is spilling,
    and about the import price at the evening peak.
    """
    fcst = lambda_forecast(snap, 5.0, ahead=96)
    lam = np.array([p["lambda"] for p in fcst])
    hours = np.array([p["hours_ahead"] for p in fcst])

    midday = lam[(hours >= 10) & (hours <= 14)].mean()
    evening = lam[(hours >= 17) & (hours <= 20)].mean()
    assert evening > midday, "energy must be worth more at the peak than under surplus"
    assert midday == pytest.approx(float(snap.sell[0]) / snap.battery.eta_c, abs=0.05)


def test_battery_action_carries_a_readable_mode(snap):
    ha = FakeHA()
    publish_policy(ha, snap, 40, 5.0)
    s = ha.states["sensor.hems_battery_action"]
    assert s["attributes"]["mode"] in ("charge", "discharge", "idle")
    act = float(s["state"])
    if s["attributes"]["mode"] == "charge":
        assert act > 0
    elif s["attributes"]["mode"] == "discharge":
        assert act < 0


def test_soe_reports_percent_too(snap):
    ha = FakeHA()
    publish_policy(ha, snap, 40, 2.5)
    attrs = ha.states["sensor.hems_battery_soe"]["attributes"]
    assert attrs["soc_percent"] == pytest.approx(25.0)
    assert attrs["capacity_kwh"] == pytest.approx(10.0)


def test_prefix_is_honoured(snap):
    ha = FakeHA()
    publish_policy(ha, snap, 10, 5.0, prefix="house")
    assert "sensor.house_lambda" in ha.states
    assert "sensor.hems_lambda" not in ha.states


def test_everything_published_is_json_serialisable(snap):
    import json

    ha = FakeHA()
    publish_policy(ha, snap, 10, 5.0)
    for payload in ha.states.values():
        json.dumps(payload)  # numpy would raise here


# --------------------------------------------------------------------------
# real-time clamping of the published setpoint
# --------------------------------------------------------------------------


def test_publish_policy_is_unclamped_by_default(snap):
    """Default behaviour must not change: no limits, no clamping."""
    from hemspolicy.ha import publish_policy
    from hemspolicy.policy import action

    ha = FakeHA()
    out = publish_policy(ha, snap, t=0, soe=5.0)
    assert out["action_kw"] == pytest.approx(action(snap, 0, 5.0))
    assert out["clamped_by"] == []


def test_publish_policy_clamps_against_a_measured_load_spike(snap):
    """A load spike the forecast never saw must bound the published setpoint.

    This is the gap the value function cannot close on its own: dp_load is
    frozen at plan time, so only this clamp reacts within a planning cycle.
    """
    from hemspolicy.ha import publish_policy
    from hemspolicy.policy import HardLimits, action

    raw = action(snap, 0, 5.0)
    ha = FakeHA()
    out = publish_policy(
        ha, snap, t=0, soe=5.0,
        limits=HardLimits(max_import_kw=2.0),
        measured_other_load_kw=2.0,      # limit already fully consumed by the house
    )
    assert out["action_kw"] <= 0.0 + 1e-9
    if raw > 0:
        assert "max_import_kw" in out["clamped_by"]
        assert out["action_kw"] < raw


def test_published_setpoint_sensor_records_why_it_was_clamped(snap):
    from hemspolicy.ha import publish_policy
    from hemspolicy.policy import HardLimits

    ha = FakeHA()
    publish_policy(
        ha, snap, t=0, soe=5.0,
        limits=HardLimits(max_import_kw=0.0), measured_other_load_kw=5.0,
    )
    payload = ha.states["sensor.hems_battery_action"]
    assert payload["attributes"]["clamped_by"] != "none"
