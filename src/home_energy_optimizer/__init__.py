"""home-energy-optimizer: makes a home's energy devices cooperate.

Each device (battery, EV, water heater, heat pump) plans itself with its own
solver behind one interface, and a coordinator makes the plans agree at the
meter: Dantzig-Wolfe (`plan()`, the default) or ADMM (`coordinate()`). The
saving can then be split fairly among the devices (`dw.attribution`). A battery
solved by dynamic programming also gives an action and a marginal value of
stored energy at any state, for acting between plans (`policy`).
"""

from .coordinate import baseline_solution, coordinate, net_cost, total_objective
from .dp_battery import rollout_battery, solve_battery
from .dp_thermal import (
    baseline_hvac,
    baseline_water_heater,
    rollout_hvac,
    rollout_water_heater,
    solve_hvac,
    solve_water_heater,
)
from .feeds import (
    blend_measured,
    from_csv,
    from_series,
    open_meteo_pv,
    open_meteo_temperature,
    resample_to_horizon,
)
from .policy import (
    Counterfactual,
    HardLimits,
    PolicySnapshot,
    action,
    clamp,
    evaluate,
    marginal_value,
    marginal_value_curve,
    meter_price,
    price_signal,
    q_values,
    reservation_prices,
    rollout,
)
from .planner import plan
from .profiles import demo_forecasts
from .types import (
    BatteryConfig,
    CoordinationConfig,
    CoordinationResult,
    DeviceSolution,
    Forecasts,
    GridLimits,
    Horizon,
    HvacConfig,
    RoundRecord,
    SiteConfig,
    SocGate,
    SubMeter,
    WaterHeaterConfig,
    group_limit,
    hybrid_inverter,
)

__version__ = "0.2.9"

__all__ = [
    "BatteryConfig", "CoordinationConfig", "CoordinationResult", "Counterfactual",
    "DeviceSolution", "Forecasts", "GridLimits", "HardLimits", "Horizon", "HvacConfig",
    "PolicySnapshot", "RoundRecord", "SiteConfig", "SocGate", "SubMeter", "WaterHeaterConfig",
    "group_limit", "hybrid_inverter",
    "action", "baseline_hvac", "blend_measured", "from_csv", "from_series",
    "open_meteo_pv", "open_meteo_temperature", "resample_to_horizon", "baseline_solution", "baseline_water_heater",
    "clamp", "coordinate", "demo_forecasts", "evaluate", "marginal_value",
    "marginal_value_curve", "meter_price", "net_cost", "plan", "price_signal", "q_values", "reservation_prices", "rollout",
    "rollout_battery", "rollout_hvac", "rollout_water_heater", "solve_battery",
    "solve_hvac", "solve_water_heater", "total_objective",
]
