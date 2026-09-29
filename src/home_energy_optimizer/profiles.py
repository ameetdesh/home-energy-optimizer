"""Synthetic forecast profiles.

These are the POC's hardcoded shapes, kept only so tests and demos have
something deterministic to run against. Phase 2 in docs/PLAN.md replaces them
with real feeds (Open-Meteo/Solcast PV, a tariff provider, a measured load
profile). Nothing in the solver core depends on this module.
"""

from __future__ import annotations

import numpy as np

from .types import Forecasts, Horizon


def solar_profile(horizon: Horizon, peak_kw: float = 5.0) -> np.ndarray:
    """Gaussian bell centred on 12:30, zero outside 06:30-18:00, repeating daily."""
    h = horizon.times() % 24.0
    solar = peak_kw * np.exp(-0.5 * ((h - 12.5) / 2.8) ** 2)
    solar[(h < 6.5) | (h > 18.0)] = 0.0
    return solar


def hot_water_demand_profile(horizon: Horizon) -> np.ndarray:
    """Morning shower, midday tap, evening shower."""
    h = horizon.times() % 24.0
    demand = np.zeros(horizon.steps)
    demand[(h >= 7.0) & (h < 8.0)] = 4.0
    demand[(h >= 12.0) & (h < 12.5)] = 1.0
    demand[(h >= 18.5) & (h < 19.5)] = 3.0
    return demand


def flat_tariff(horizon: Horizon, buy: float = 0.30, sell: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    return np.full(horizon.steps, buy), np.full(horizon.steps, sell)


def day_night_tariff(
    horizon: Horizon, peak: float = 0.40, offpeak: float = 0.15, sell: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    """Cheap 00:00-07:00, expensive otherwise. The classic economy-7 shape."""
    h = horizon.times() % 24.0
    buy = np.where(h < 7.0, offpeak, peak)
    return buy, np.full(horizon.steps, sell)


def dynamic_tariff(
    horizon: Horizon, base: float = 0.25, amplitude: float = 0.20, sell_frac: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Agile-style shape: overnight trough, midday dip, evening peak."""
    h = horizon.times() % 24.0
    shape = (
        np.sin((h - 8.0) / 24.0 * 2 * np.pi)
        + 0.6 * np.sin((h - 6.0) / 12.0 * 2 * np.pi)
    )
    buy = base + amplitude * shape / 1.6
    buy = np.maximum(buy, 0.02)
    return buy, buy * sell_frac


def outdoor_temp_profile(
    horizon: Horizon, mean_c: float = 24.0, swing_c: float = 6.0
) -> np.ndarray:
    """Daily sinusoid peaking mid-afternoon."""
    h = horizon.times() % 24.0
    return mean_c + swing_c * np.sin((h - 9.0) / 24.0 * 2 * np.pi)


def base_load_profile(horizon: Horizon, base_kw: float = 0.4, peak_kw: float = 1.6) -> np.ndarray:
    """Inflexible household demand: low overnight, morning and evening bumps."""
    h = horizon.times() % 24.0
    load = np.full(horizon.steps, base_kw)
    load += (peak_kw - base_kw) * np.exp(-0.5 * ((h - 7.5) / 1.2) ** 2)
    load += (peak_kw - base_kw) * np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2)
    return load


def demo_forecasts(
    horizon: Horizon | None = None,
    tariff: str = "dynamic",
    solar_peak_kw: float = 5.0,
) -> Forecasts:
    """A complete, deterministic input set for tests and demos."""
    horizon = horizon or Horizon()
    if tariff == "flat":
        buy, sell = flat_tariff(horizon)
    elif tariff == "day_night":
        buy, sell = day_night_tariff(horizon)
    elif tariff == "dynamic":
        buy, sell = dynamic_tariff(horizon)
    else:
        raise ValueError(f"unknown tariff profile {tariff!r}")

    return Forecasts(
        buy=buy,
        sell=sell,
        load=base_load_profile(horizon),
        solar=solar_profile(horizon, solar_peak_kw),
        outdoor_temp=outdoor_temp_profile(horizon),
        hot_water_demand=hot_water_demand_profile(horizon),
    )
