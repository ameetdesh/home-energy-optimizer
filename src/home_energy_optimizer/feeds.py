"""Real forecast inputs, replacing the synthetic shapes in `profiles.py`.

Three sources, in the order a real deployment would reach for them:

* `open_meteo_pv` - a keyless PV forecast from Open-Meteo's 15-minute
  shortwave-radiation field. No API key, no account, no rate limit worth
  worrying about, which makes it the right default. EMHASS's `forecast.py`
  does this far more thoroughly (pvlib ModelChain over a described array);
  this is the simple version, and the seam is here if you want the thorough one.
* `from_series` - accept a caller-supplied list. This is how a real integration
  feeds in Solcast, a tariff provider, or an ML load forecast; every published
  EMHASS/evcc integration ultimately reduces to this.
* `from_csv` - timestamped CSV, matching EMHASS's convention.

`blend_measured` is lifted in spirit from evcc's optimizer (`core/optimizer.md`
"Measured value blending"): anchor the first forecast slot to what was actually
just measured and decay back to the forecast over a few slots. ~20 lines for a
real accuracy win, and it is the cheapest correction available to anyone
running a forecast against reality.

Only `open_meteo_pv` touches the network, and only when called.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from datetime import datetime

import numpy as np
import numpy.typing as npt

from .types import Horizon

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# --------------------------------------------------------------------------
# Resampling helpers
# --------------------------------------------------------------------------


def resample_to_horizon(
    values: np.ndarray, source_dt_hours: float, horizon: Horizon, fill: str = "hold"
) -> np.ndarray:
    """Put an arbitrary-cadence series onto the horizon's grid.

    `fill="hold"` repeats the last value when the source runs short, which is
    what you want for a tariff (a day-ahead feed publishes 24 h, the horizon
    may want 48). `fill="zero"` pads with zeros, for PV after sunset.

    Length mismatches are normal, not exceptional - EMHASS's `AGENTS.md` calls
    out silent misalignment as a way to get a plan that looks Optimal while
    every timestep is offset. So this pads loudly rather than quietly: callers
    get the padded length back and can compare.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("cannot resample an empty series")

    src_t = np.arange(values.size) * source_dt_hours
    dst_t = horizon.times()

    out = np.interp(dst_t, src_t, values, left=values[0], right=values[-1])
    beyond = dst_t > src_t[-1]
    if fill == "zero":
        out[beyond] = 0.0
    return out


def from_series(
    values: npt.ArrayLike, horizon: Horizon, source_dt_hours: float | None = None,
    fill: str = "hold"
) -> np.ndarray:
    """Accept a caller-supplied forecast of any length.

    With `source_dt_hours=None` the series is assumed to span the whole horizon,
    which is the common case when someone hands you "the next 24 hours".
    """
    values = np.asarray(values, dtype=float).ravel()
    if source_dt_hours is None:
        source_dt_hours = horizon.hours / max(values.size, 1)
    return resample_to_horizon(values, source_dt_hours, horizon, fill=fill)


def from_csv(
    path: str, horizon: Horizon, value_column: int = 1, fill: str = "hold"
) -> np.ndarray:
    """Read `timestamp,value` rows (no header), EMHASS's CSV convention."""
    stamps: list[datetime] = []
    vals: list[float] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            stamps.append(datetime.fromisoformat(row[0]))
            vals.append(float(row[value_column]))
    if len(vals) < 2:
        raise ValueError(f"{path}: need at least 2 rows, got {len(vals)}")
    step_h = (stamps[1] - stamps[0]).total_seconds() / 3600.0
    return resample_to_horizon(np.array(vals), step_h, horizon, fill=fill)


# --------------------------------------------------------------------------
# Open-Meteo PV
# --------------------------------------------------------------------------


def open_meteo_pv(
    latitude: float,
    longitude: float,
    horizon: Horizon,
    peak_kw: float,
    system_efficiency: float = 0.85,
    timeout: float = 15.0,
) -> np.ndarray:
    """PV production forecast in kW, from Open-Meteo shortwave radiation.

    Deliberately the simple model: production scales with global horizontal
    irradiance against 1000 W/m^2 standard test conditions, times a lumped
    system efficiency. It ignores tilt, azimuth, shading, temperature
    derating and the inverter curve - all of which EMHASS models properly via
    pvlib. Good enough to drive a plan, and honest about what it is not.

    Raises on network failure rather than silently substituting a synthetic
    profile: a plan built on a forecast that is quietly wrong is worse than
    no plan.
    """
    days = max(1, int(np.ceil(horizon.hours / 24.0)) + 1)
    url = (
        f"{OPEN_METEO_URL}?latitude={latitude}&longitude={longitude}"
        f"&minutely_15=shortwave_radiation&forecast_days={min(days, 16)}"
        f"&timezone=UTC"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed host
        payload = json.loads(resp.read())

    ghi = payload.get("minutely_15", {}).get("shortwave_radiation")
    if not ghi:
        raise ValueError("Open-Meteo returned no shortwave_radiation series")

    ghi_arr = np.array([0.0 if v is None else float(v) for v in ghi])
    kw = peak_kw * (ghi_arr / 1000.0) * system_efficiency
    return resample_to_horizon(kw, 0.25, horizon, fill="zero")


def open_meteo_temperature(
    latitude: float, longitude: float, horizon: Horizon, timeout: float = 15.0
) -> np.ndarray:
    """Outdoor temperature in degC, for the thermal models."""
    days = max(1, int(np.ceil(horizon.hours / 24.0)) + 1)
    url = (
        f"{OPEN_METEO_URL}?latitude={latitude}&longitude={longitude}"
        f"&minutely_15=temperature_2m&forecast_days={min(days, 16)}&timezone=UTC"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    temps = payload.get("minutely_15", {}).get("temperature_2m")
    if not temps:
        raise ValueError("Open-Meteo returned no temperature_2m series")
    arr = np.array([np.nan if v is None else float(v) for v in temps])
    if np.isnan(arr).any():  # short gaps happen; carry the last good value
        idx = np.where(~np.isnan(arr))[0]
        arr = np.interp(np.arange(arr.size), idx, arr[idx])
    return resample_to_horizon(arr, 0.25, horizon, fill="hold")


# --------------------------------------------------------------------------
# Measured-value blending  (evcc, core/optimizer.md)
# --------------------------------------------------------------------------


def blend_measured(
    forecast: np.ndarray,
    measured: float,
    decay_slots: int = 4,
    mode: str = "replace",
) -> np.ndarray:
    """Anchor a forecast to the last completed measurement, decaying back.

    evcc applies two variants, and both are worth having:

    * ``mode="replace"`` (base load): the measured value replaces slot 0 and
      the forecast is blended back linearly over `decay_slots`. Use when the
      measurement and the forecast are the same quantity in the same units.
    * ``mode="scale"`` (solar): the ratio measured/forecast scales slot 0 and
      the factor decays toward 1. Use when the forecast has a shape you trust
      but a level you do not - a 20% overcast bias persists for a while.

    Cheap, and it is the single highest-value correction available to anyone
    running a day-ahead forecast against live telemetry.
    """
    out = np.array(forecast, dtype=float, copy=True)
    if decay_slots <= 0 or out.size == 0:
        return out
    n = min(decay_slots, out.size)

    if mode == "replace":
        offset = measured - out[0]
        for i in range(n):
            out[i] += offset * (1.0 - i / n)
        return out

    if mode == "scale":
        # Guard a near-zero denominator: at night the forecast is 0 and the
        # ratio is meaningless, so leave the series alone.
        if abs(out[0]) < 1e-6:
            return out
        factor = measured / out[0]
        for i in range(n):
            f = 1.0 + (factor - 1.0) * (1.0 - i / n)
            out[i] *= f
        return out

    raise ValueError(f"mode must be 'replace' or 'scale', got {mode!r}")
