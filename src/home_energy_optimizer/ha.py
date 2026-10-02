"""Publish the plan and its prices to Home Assistant.

Every signal becomes an ordinary sensor, so any automation - for a device the
optimiser has never heard of - can compare its own value per kWh with one.
`publish_site` publishes the coordinated plan: power flows, device states and,
from Dantzig-Wolfe, the meter price and the plan's gap. `publish_policy` adds,
between plans, what the battery's own solver gives at its measured state when
that solver is a dynamic programme: the setpoint, lambda (the marginal value of
a stored kWh) and the prices derived from it.

Entity naming and the `forecast` attribute follow EMHASS's convention
(`utils.py` `custom_*_id`), so the same ApexCharts card configs work.

**lambda ships with an uncertainty band, deliberately.** Validation against a
HiGHS LP dual and an exact joint DP put it within ~0.04/kWh worst case
(docs/NOTES.md section 3), so a decision within that of the margin is not
reliable. `sensor.hems_lambda` therefore carries `uncertainty` and a
`confident_above` attribute rather than pretending to a precision it does not
have. Publishing a bare number would invite exactly the over-trust the
measurement warns against.

Uses the REST API and the standard long-lived access token, so it needs no
custom component - a plain `requests`-free stdlib client is enough.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .policy import (
    HardLimits,
    PolicySnapshot,
    action,
    clamp,
    marginal_value,
    price_signal,
    reservation_prices,
)
from .types import CoordinationResult, Forecasts, SiteConfig

# Worst-case disagreement with both ground truths, in currency/kWh.
# See docs/NOTES.md section 3.
LAMBDA_UNCERTAINTY = 0.04


@dataclass
class HomeAssistant:
    """Minimal Home Assistant REST client - just enough to publish states."""

    base_url: str = "http://127.0.0.1:8123"
    token: str = ""
    timeout: float = 10.0

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read() or b"{}")

    def ping(self) -> bool:
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                return resp.status == 200
        except urllib.error.URLError:
            return False

    def set_state(
        self,
        entity_id: str,
        state: Any,
        unit: str | None = None,
        device_class: str | None = None,
        friendly_name: str | None = None,
        **attributes: Any,
    ) -> dict:
        attrs = {k: v for k, v in attributes.items() if v is not None}
        if unit:
            attrs["unit_of_measurement"] = unit
        if device_class:
            attrs["device_class"] = device_class
        if friendly_name:
            attrs["friendly_name"] = friendly_name
        # HA rejects non-JSON-native types; numpy leaks in easily from a solver.
        return self._post(
            f"/api/states/{entity_id}",
            {"state": _jsonable(state), "attributes": _jsonable(attrs)},
        )


def _jsonable(value: Any) -> Any:
    """numpy -> plain Python, recursively. HA's API rejects numpy scalars."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value


def lambda_forecast(snap: PolicySnapshot, soe: float, ahead: int = 96) -> list[dict]:
    """lambda over the coming slots, in EMHASS's `forecast` attribute shape.

    Evaluated at a FIXED state of energy on purpose: this is "what would a kWh
    be worth at each future time", the price curve, not a projection of where
    the battery will actually be. Mixing the two would make the series depend
    on the plan and stop being a price.
    """
    n = min(ahead, snap.horizon.steps)
    dt = snap.horizon.dt
    return [
        {
            "hours_ahead": round(t * dt, 3),
            "lambda": round(marginal_value(snap, t, soe), 5),
            "import_price": round(float(snap.buy[t]), 5),
            "export_price": round(float(snap.sell[t]), 5),
        }
        for t in range(n)
    ]


def publish_policy(
    ha: HomeAssistant,
    snap: PolicySnapshot,
    t: int,
    soe: float,
    prefix: str = "hems",
    currency: str = "EUR",
    forecast_slots: int = 96,
    limits: HardLimits | None = None,
    measured_other_load_kw: float | None = None,
) -> dict[str, float]:
    """Publish the current policy state. Returns what was published.

    Five sensors, mirroring how EMHASS publishes a plan:

    - `sensor.<prefix>_lambda`            marginal value of a stored kWh
    - `sensor.<prefix>_worth_running`     the threshold a flexible load compares against
    - `sensor.<prefix>_battery_action`    optimal battery power right now (kW)
    - `sensor.<prefix>_battery_soe`       state of energy (kWh)
    - `sensor.<prefix>_import_price`      the prevailing tariff, for context
    - `sensor.<prefix>_import_below`      grid price below which to IMPORT
    - `sensor.<prefix>_export_above`      grid price above which to EXPORT

    The last two are the reservation prices - the bid/ask band a storage owner
    faces. Compare them against the prevailing meter price: `import_price`
    while importing, the export price while exporting.

    If `limits` is given, the published setpoint is clamped against them using
    `measured_other_load_kw` - the meter reading MINUS this battery. That is
    the only part of this path that reacts to real time: the value function was
    solved against a forecast and cannot know that a load just switched on
    (docs/theory.tex, "What is guaranteed, and what is not"). Without `limits` the raw policy action is
    published unchanged, which is the previous behaviour.

    For more than one storage unit sharing the limit, do NOT call this per unit
    with the same `measured_other_load_kw` - each would be granted the whole
    headroom. Use `policy.clamp_fleet()` to allocate it in sequence and pass
    each unit its own result.
    """
    sig = price_signal(snap, t, soe)
    lam = sig["lambda_per_kwh"]
    act = action(snap, t, soe)

    bound_by: list[str] = []
    if limits is not None:
        other = (
            float(measured_other_load_kw)
            if measured_other_load_kw is not None
            else float(snap.dp_load[t])  # fall back to the forecast, and say so
        )
        act, bound_by = clamp(act, other_load_kw=other, limits=limits)

    ha.set_state(
        f"sensor.{prefix}_lambda",
        round(lam, 4),
        unit=f"{currency}/kWh",
        device_class="monetary",
        friendly_name="Marginal value of stored energy",
        # The band is the point: a device whose own value per kWh falls inside
        # [lambda - u, lambda + u] should not switch on the strength of this.
        uncertainty=LAMBDA_UNCERTAINTY,
        confident_above=round(lam + LAMBDA_UNCERTAINTY, 4),
        confident_below=round(lam - LAMBDA_UNCERTAINTY, 4),
        step=t,
        hours_ahead=round(t * snap.horizon.dt, 3),
        forecast=lambda_forecast(snap, soe, forecast_slots),
        import_below=round(sig["import_below"], 4),
        export_above=round(sig["export_above"], 4),
    )
    ha.set_state(
        f"sensor.{prefix}_worth_running",
        round(sig["worth_running_above"], 4),
        unit=f"{currency}/kWh",
        device_class="monetary",
        friendly_name="Run a flexible load below this value",
        # NOT the tariff and NOT lambda: what one more kWh at the meter really
        # costs, given the battery will serve it and refill later. On a
        # day/night tariff the meter reads 0.40 at 21:00 while this says 0.167.
        lambda_per_kwh=round(lam, 4),
        import_price=round(float(snap.buy[t]), 4),
    )
    ha.set_state(
        f"sensor.{prefix}_import_below",
        round(sig["import_below"], 4),
        unit=f"{currency}/kWh",
        device_class="monetary",
        friendly_name="Import while the grid price is below this",
        explanation="lambda * charge efficiency: buying 1 kWh only stores eta_c of it",
    )
    ha.set_state(
        f"sensor.{prefix}_export_above",
        round(sig["export_above"], 4),
        unit=f"{currency}/kWh",
        device_class="monetary",
        friendly_name="Export while the grid price is above this",
        explanation="lambda / discharge efficiency: selling 1 kWh drains 1/eta_d from store",
        spread=round(sig["spread"], 4),
    )
    ha.set_state(
        f"sensor.{prefix}_battery_action",
        round(act, 3),
        unit="kW",
        device_class="power",
        friendly_name="Battery setpoint (+ charge / - discharge)",
        mode="charge" if act > 0 else ("discharge" if act < 0 else "idle"),
        clamped_by=",".join(bound_by) if bound_by else "none",
    )
    ha.set_state(
        f"sensor.{prefix}_battery_soe",
        round(float(soe), 3),
        unit="kWh",
        device_class="energy_storage",
        friendly_name="Battery state of energy",
        capacity_kwh=snap.battery.capacity_kwh,
        soc_percent=round(100.0 * soe / snap.battery.capacity_kwh, 1),
    )
    ha.set_state(
        f"sensor.{prefix}_import_price",
        round(float(snap.buy[t]), 4),
        unit=f"{currency}/kWh",
        device_class="monetary",
        friendly_name="Grid import price",
        export_price=round(float(snap.sell[t]), 4),
    )

    return {
        "lambda": lam,
        "import_below": sig["import_below"],
        "export_above": sig["export_above"],
        "action_kw": act,
        "clamped_by": bound_by,
        "soe_kwh": float(soe),
        "import_price": float(snap.buy[t]),
        "worth_running_above": sig["worth_running_above"],
    }


def _slot_forecast(times: Sequence[float] | np.ndarray,
                   values: Sequence[float] | np.ndarray, key: str, n: int) -> list[dict]:
    return [
        {"hours_ahead": round(float(times[i]), 3), key: round(float(values[i]), 4)}
        for i in range(min(n, len(values)))
    ]


def publish_site(
    ha: HomeAssistant,
    site: SiteConfig,
    fc: Forecasts,
    res: CoordinationResult,
    t: int,
    prefix: str = "hems",
    forecast_slots: int = 96,
    currency: str = "EUR",
) -> dict[str, float | None]:
    """Publish every signal the local GUI plots, not just the battery.

    The GUI shows prices + lambda, battery, power flows and thermal state; this
    puts the same set into Home Assistant so the two tell the same story. Each
    sensor carries a `forecast` attribute over the remaining horizon, in the
    shape EMHASS uses, so charts work without a custom component.

    Thermal and PV values come from the plan rather than a simulation - these
    devices have no feedback loop in the demo, so their planned trajectory IS
    what would happen absent disturbance.

    A Dantzig-Wolfe result adds `sensor.<prefix>_meter_price` (the master's
    price pi, with a forecast). Every result publishes
    `sensor.<prefix>_plan_gap`: the certified distance from the best possible
    plan for DW, "unknown" for ADMM, which has no certificate.
    """
    times = site.horizon.times()
    ahead = min(forecast_slots, site.horizon.steps - t)
    out: dict[str, float | None] = {}

    def pub(name: str, value: float, unit: str | None, device_class: str | None,
            friendly: str | None, series: np.ndarray | None = None,
            key: str = "value", **attrs: Any) -> None:
        ha.set_state(
            f"sensor.{prefix}_{name}",
            round(float(value), 4),
            unit=unit,
            device_class=device_class,
            friendly_name=friendly,
            forecast=(
                _slot_forecast(times[t:], series[t:], key, ahead)
                if series is not None
                else None
            ),
            **attrs,
        )
        out[name] = float(value)

    pub("pv", fc.solar[t], "kW", "power", "PV production", fc.solar, "pv")
    pub("load", fc.load[t], "kW", "power", "Household load", fc.load, "load")
    pub(
        "net_grid", res.net_grid[t], "kW", "power",
        "Net grid exchange (+ import / - export)", res.net_grid, "net_grid",
        mode="import" if res.net_grid[t] >= 0 else "export",
    )
    pub("outdoor_temp", fc.outdoor_temp[t], "°C", "temperature",
        "Outdoor temperature", fc.outdoor_temp, "outdoor_temp")

    if "water_heater" in res.devices:
        sol = res.devices["water_heater"]
        assert site.water_heater is not None        # it has a plan, so it is configured
        pub("water_heater_temp", sol.trajectory[t], "°C", "temperature",
            "Hot water tank", sol.trajectory, "temp",
            setpoint=site.water_heater.t_comfort)
        pub("water_heater_power", sol.power[t], "kW", "power",
            "Hot water heater", sol.power, "power")

    if "hvac" in res.devices:
        sol = res.devices["hvac"]
        assert site.hvac is not None
        pub("hvac_temp", sol.trajectory[t], "°C", "temperature",
            "Room temperature", sol.trajectory, "temp",
            comfort_low=site.hvac.t_comfort_low, comfort_high=site.hvac.t_comfort_high)
        pub("hvac_power", sol.power[t], "kW", "power", "HVAC", sol.power, "power")

    # Dantzig-Wolfe plans carry two more things worth showing.
    if res.meter_price is not None:
        # pi: what one more kWh at the meter costs the whole house in this
        # slot, per the plan - grid limits and every device included. The
        # plan-time answer to worth_running's question; worth_running re-asks
        # it from the battery's measured state between plans.
        pub("meter_price", res.meter_price[t], f"{currency}/kWh", "monetary",
            "Meter price: cost of one more kWh (plan)", res.meter_price, "price",
            import_price=round(float(fc.buy[t]), 4))
    ha.set_state(
        f"sensor.{prefix}_plan_gap",
        round(float(res.gap), 4) if res.gap is not None else "unknown",
        unit=currency,
        friendly_name="Plan: at most this far from the best possible",
        method=res.method,
        lower_bound=None if res.lower_bound is None else round(res.lower_bound, 4),
        plan_objective=None if res.plan_objective is None else round(res.plan_objective, 4),
        note=res.note or None,
    )
    out["plan_gap"] = res.gap

    return out
