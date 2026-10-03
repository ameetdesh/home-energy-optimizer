"""Configuration and result types for the headless solver core.

Everything the solvers need is passed in explicitly. There are no module-level
mutable globals: the POC this was extracted from kept DT/STEPS/ETA and every
device parameter at module scope, which made the solvers untestable and
un-reusable outside the single browser page they were written for.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # admm.battery_qp imports this module
    from .admm.battery_qp import BatteryStep
    from .meter import Bus

# --------------------------------------------------------------------------
# Horizon
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Horizon:
    """Discretisation of the planning window.

    dt is in HOURS (0.25 = 15 min slots), matching the units every price and
    power quantity in this package uses: power in kW, energy in kWh, price in
    currency/kWh.
    """

    dt: float = 0.25
    hours: float = 48.0

    @property
    def steps(self) -> int:
        return int(round(self.hours / self.dt))

    def times(self) -> np.ndarray:
        """Slot start times in hours from the horizon start."""
        return np.arange(self.steps) * self.dt

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.hours <= 0:
            raise ValueError(f"hours must be positive, got {self.hours}")
        if self.steps < 2:
            raise ValueError(f"horizon needs >= 2 steps, got {self.steps}")


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryConfig:
    """Stationary battery (or an EV modelled as one).

    Efficiency can be given as a single one-way `eta` (the POC's convention) or
    split into `eta_charge` / `eta_discharge`. Split is what EMHASS uses
    (battery_charge_efficiency / battery_discharge_efficiency) and what evcc's
    optimizer wire contract uses (eta_c / eta_d), so it is needed for either
    integration; `eta` remains the shorthand that sets both.

    Convention: charging power `a` stores `eta_charge * a`; delivering `d` to
    the meter drains `d / eta_discharge` from the store.
    """

    capacity_kwh: float = 10.0
    p_charge_max_kw: float = 5.0
    p_discharge_max_kw: float = 5.0
    eta: float = 0.90
    eta_charge: float | None = None
    eta_discharge: float | None = None
    # Phase 1 measured the discretisation cost directly (docs/NOTES.md section 4):
    # the POC's 50x21 grid gives up 3-6% of available savings, and refining to
    # 200x81 takes capture past 99% for 36 ms instead of 7 ms - free against a
    # 15-minute re-solve cadence. Drop back to 50x21 where the product state
    # space matters (e.g. the joint-DP reference in bench/).
    n_states: int = 200
    n_actions: int = 81
    soc_initial_frac: float = 0.5
    # Reserve the plan never draws below (backup, battery health), as a
    # fraction of capacity. A hard floor on the store, like capacity itself.
    # If the battery starts below it, the floor is where it starts: it may
    # charge, but not discharge further (see soe_floor_kwh).
    soe_min_frac: float = 0.0

    # -- End-of-horizon valuation ------------------------------------------
    # "linear" (default): V[N] = terminal_price * S. Energy left in the
    #   battery is worth what it would cost to buy, so dV/ds stays a PRICE all
    #   the way to the horizon edge. terminal_price=None means "cheapest
    #   import price on the horizon", which is what evcc's optimizer uses.
    #
    # "quadratic": the POC's original V[N] = -w * (S - target)^2 / cap.
    #   Retained for parity, but it is a trap at realistic battery sizes: at
    #   w=5, cap=10 the penalty is 12.5 currency at either extreme, roughly 6x
    #   a whole day's arbitrage opportunity, so the battery gets pinned at the
    #   target and does nothing. It also forces dV/ds = 0 at the target, which
    #   corrupts the marginal-value signal. See docs/NOTES.md.
    terminal_mode: str = "linear"
    terminal_price: float | None = None
    terminal_weight: float = 5.0
    terminal_target_frac: float = 0.5

    # -- Per-slot demands (evcc's p_demand / s_goal) ------------------------
    # evcc models a LOADPOINT as a battery that only charges (d_max = 0), and
    # carries the actual charging requirement in p_demand. Without honouring
    # it, an EV would simply never charge - so this is not optional for any
    # evcc user with a charger.
    #
    # min_charge_kw: floor on charge power per slot (evcc p_demand, Wh/slot).
    # soc_goal_kwh:  desired stored energy per slot (evcc s_goal). Soft, priced
    #   at soc_goal_penalty per kWh of shortfall, so an unreachable goal
    #   degrades to "as close as possible" instead of returning infeasible -
    #   the same treatment EMHASS gives soc_final.
    min_charge_kw: tuple[float, ...] | None = None
    soc_goal_kwh: tuple[float, ...] | None = None
    soc_goal_penalty: float = 100.0

    # Semi-continuous charge floor (evcc c_min = Voltage x minCurrent x
    # minPhases). A charger is OFF or at >= this power; it cannot modulate
    # below. Planning 2 kW into a wallbox with a 4.14 kW floor yields a
    # schedule the hardware cannot execute.
    #
    # A DP enforces this by deleting actions from the grid - free. An LP needs
    # a binary per timestep, which is exactly the semi-continuous machinery
    # EMHASS carries as `treat_deferrable_load_as_semi_cont`.
    charge_deadband_kw: float = 0.0

    @property
    def eta_c(self) -> float:
        """One-way charge efficiency; falls back to `eta`."""
        return self.eta if self.eta_charge is None else self.eta_charge

    @property
    def eta_d(self) -> float:
        """One-way discharge efficiency; falls back to `eta`."""
        return self.eta if self.eta_discharge is None else self.eta_discharge

    @property
    def soe_floor_kwh(self) -> float:
        """The lowest stored energy a plan may reach, kWh."""
        return self.capacity_kwh * min(self.soe_min_frac, self.soc_initial_frac)

    @property
    def round_trip(self) -> float:
        return self.eta_c * self.eta_d

    def validate(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        for name, val in (("eta", self.eta), ("eta_charge", self.eta_c), ("eta_discharge", self.eta_d)):
            if not 0 < val <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {val}")
        if self.n_states < 2 or self.n_actions < 2:
            raise ValueError("n_states and n_actions must be >= 2")
        if not 0 <= self.soc_initial_frac <= 1:
            raise ValueError("soc_initial_frac must be in [0, 1]")
        if not 0 <= self.soe_min_frac < 1:
            raise ValueError("soe_min_frac must be in [0, 1)")
        if self.terminal_mode not in ("linear", "quadratic"):
            raise ValueError(
                f"terminal_mode must be 'linear' or 'quadratic', got {self.terminal_mode!r}"
            )


@dataclass(frozen=True)
class WaterHeaterConfig:
    """Resistive hot-water tank. Binary on/off per slot by default; set
    `n_duty_levels` above 2 to let the element run a fraction of each slot."""

    power_kw: float = 3.0
    liters: float = 150.0
    # `t_min` is the FLOOR OF THE STATE GRID, not a comfort bound and not a
    # temperature the tank is somehow prevented from passing. Nothing stops a
    # tank cooling: unlike `t_max`, which a thermostat genuinely enforces by
    # cutting out, there is no physical mechanism at the bottom.
    #
    # It therefore has to sit at or below the equilibrium of the dynamics. The
    # tank relaxes toward a flow-weighted mixture of `t_ambient` (through the
    # insulation) and `t_inlet` (through the draw), so the lowest temperature
    # any slot can reach is min(t_ambient, t_inlet). See docs/theory.tex.
    t_min: float = 15.0
    t_max: float = 75.0
    t_comfort: float = 55.0
    t_ambient: float = 25.0
    # Cold-water mains temperature. This is what replaces every litre drawn,
    # so it - not the air around the tank - is the reference for the draw
    # term: a tank drained flat sits at mains temperature, not at room
    # temperature. Typical UK/EU mains is 10-15 C, seasonal.
    t_inlet: float = 15.0
    r_thermal: float = 80.0
    # 60 K over 68 points is 0.896 K/state, matching the resolution the
    # narrower grid had.
    n_states: int = 68
    # Element duty per slot: `n_duty_levels` evenly spaced values in [0, 1].
    # 2 is the on/off element (off, or on for the whole slot) - the default,
    # and what the compiled kernel and the bench references model. More levels
    # let the element run for part of a slot, which a relay or thermostat can
    # do within 15 minutes: 5 gives 0, 1/4, 1/2, 3/4, 1. The DP cost grows
    # linearly with the count.
    #
    # The temperature grid must resolve one duty step, or the DP plans moves
    # its own rollout cannot follow: at 13 levels one step is 0.36 K against
    # the default 0.9 K spacing, and the rolled-out plan cost 0.18 more than
    # its value function claimed (0.004 at 200 states). See
    # `states_for_duty_levels`.
    n_duty_levels: int = 2

    # -- Discomfort pricing -------------------------------------------------
    # The POC multiplied degC^2 by an arbitrary constant and added the result
    # to a bill in currency. The exchange rate was undocumented and huge - the
    # thermostat baseline scored ~608 "comfort" against a ~6 bill - which made
    # every savings number partly a statement about a made-up number.
    #
    # "linear" (default) prices discomfort in currency per kelvin-hour, and
    # derives that price from physics rather than taste: restoring 1 K costs
    # heat_capacity * energy_price, so being 1 K short for an hour is worth
    # `discomfort_multiplier` times that. The multiplier is the only taste
    # parameter left, and it has a plain reading: "how many times the cost of
    # fixing it am I willing to pay to avoid the drift in the first place".
    #
    # "quadratic" reproduces the POC exactly, for parity checks.
    comfort_mode: str = "linear"
    discomfort_multiplier: float = 3.0
    discomfort_price: float | None = None  # currency/(K.h); None = derive
    comfort_weight: float = 10.0  # quadratic mode only
    terminal_weight: float = 15.0

    @property
    def heat_capacity_kwh_per_k(self) -> float:
        """Thermal capacity in kWh/K.  liters * 4.186 kJ/(kg.K) / 3600 s/h."""
        return self.liters * 4.186 / 3600.0

    def discomfort_price_per_kelvin_hour(self, reference_price: float) -> float:
        """Currency per K of shortfall per hour.

        Derived as (energy needed to restore 1 K) * (price of that energy) *
        multiplier. A resistive tank has COP 1, so restoring 1 K costs exactly
        heat_capacity_kwh_per_k * reference_price.
        """
        if self.discomfort_price is not None:
            return float(self.discomfort_price)
        return self.heat_capacity_kwh_per_k * float(reference_price) * self.discomfort_multiplier

    def validate(self) -> None:
        if not self.t_min < self.t_comfort < self.t_max:
            raise ValueError("require t_min < t_comfort < t_max")
        if self.t_inlet >= self.t_comfort:
            raise ValueError(
                f"t_inlet ({self.t_inlet}) must be below t_comfort ({self.t_comfort}); "
                "the draw factor is normalised on (t_comfort - t_inlet)"
            )
        # The grid floor must sit at or below the equilibrium of the dynamics.
        # Above it, the state would have to be clamped, which holds the tank at
        # a temperature it never reached: that fabricates energy and truncates
        # the discomfort the objective is supposed to price.
        floor = min(self.t_ambient, self.t_inlet)
        if self.t_min > floor:
            raise ValueError(
                f"t_min ({self.t_min}) is above min(t_ambient, t_inlet) ({floor}); the "
                "state grid floor must be at or below the thermal equilibrium, or the "
                "transition has to clamp a temperature the tank never held"
            )
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if self.n_duty_levels < 2:
            raise ValueError("n_duty_levels must be >= 2 (2 = on/off)")
        if self.comfort_mode not in ("linear", "quadratic"):
            raise ValueError(f"comfort_mode must be 'linear' or 'quadratic', got {self.comfort_mode!r}")

    @property
    def duty_actions(self) -> np.ndarray:
        """The element's admissible duty fractions per slot."""
        return np.linspace(0.0, 1.0, self.n_duty_levels)

    def states_for_duty_levels(self, dt: float, per_step: float = 1.2) -> int:
        """Smallest temperature grid with `per_step` states per duty step.

        A duty step heats the tank by power/(levels-1)*dt/C kelvin; the grid
        spacing has to be finer than that for the DP's plan to be one its
        rollout can execute. Never below the configured n_states.
        """
        step_k = self.power_kw / (self.n_duty_levels - 1) * dt / self.heat_capacity_kwh_per_k
        need = int(np.ceil((self.t_max - self.t_min) * per_step / step_k)) + 1
        return max(self.n_states, need)


@dataclass(frozen=True)
class HvacConfig:
    """Air conditioner / reversible heat pump: off, cool or heat, each slot.
    Set `n_duty_levels` above 2 to let it run a fraction of a slot."""

    power_kw: float = 1.5
    cop: float = 3.0
    t_min: float = 18.0
    t_max: float = 38.0
    t_comfort_low: float = 22.0
    t_comfort_high: float = 26.0
    c_room_kwh_per_k: float = 2.0
    r_wall_k_per_kw: float = 5.0
    n_states: int = 40
    # Duty per slot in each direction: `n_duty_levels` evenly spaced values in
    # [0, 1], as for the tank. 2 is off, or full heat or cool for the slot (the
    # default, and what the compiled kernel models); 5 gives quarters, 13
    # twelfths. Pair more levels with a finer grid (states_for_duty_levels).
    n_duty_levels: int = 2

    # See WaterHeaterConfig for why this exists. A heat pump moves COP kWh of
    # heat per kWh of electricity, so restoring 1 K of room temperature costs
    # c_room / cop kWh rather than c_room kWh.
    comfort_mode: str = "linear"
    discomfort_multiplier: float = 3.0
    discomfort_price: float | None = None  # currency/(K.h); None = derive
    comfort_weight: float = 8.0  # quadratic mode only
    terminal_weight: float = 10.0
    # A comfort band that changes over the horizon, optional: one bound per
    # point of the room trajectory, so n + 1 for n slots (point i is hour
    # i * dt). The room starts mid-band at point 0 and the DP prices points
    # 1..n against their own band (comfort_band). None: the flat band
    # (t_comfort_low, t_comfort_high) throughout.
    comfort_low_profile: tuple[float, ...] | None = None
    comfort_high_profile: tuple[float, ...] | None = None

    @property
    def duty_actions(self) -> np.ndarray:
        """Admissible actions: 0 (off), then cooling -1/(L-1) .. -1, then
        heating 1/(L-1) .. 1. For L = 2 that is off, cool, heat."""
        steps = np.linspace(0.0, 1.0, self.n_duty_levels)[1:]
        return np.concatenate([[0.0], -steps, steps])

    def states_for_duty_levels(self, dt: float, per_step: float = 1.2) -> int:
        """Smallest temperature grid with `per_step` states per heating duty
        step, so the DP's plan is one its rollout can follow."""
        step_k = self.power_kw * (self.cop + 1.0) / (self.n_duty_levels - 1) * dt / self.c_room_kwh_per_k
        need = int(np.ceil((self.t_max - self.t_min) * per_step / step_k)) + 1
        return max(self.n_states, need)

    def band_at(self, i: int) -> tuple[float, float]:
        """(low, high) at point i of the room trajectory; -1 is the horizon's end."""
        low, high = self.comfort_low_profile, self.comfort_high_profile
        if low is None:
            return self.t_comfort_low, self.t_comfort_high
        assert high is not None, "validate() requires both profiles or neither"
        return float(low[i]), float(high[i])

    def comfort_band(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) at the n + 1 points of an n-slot room trajectory."""
        if self.comfort_low_profile is None:
            return np.full(n + 1, self.t_comfort_low), np.full(n + 1, self.t_comfort_high)
        if len(self.comfort_low_profile) != n + 1:
            raise ValueError(f"the comfort profile has {len(self.comfort_low_profile)} points; "
                             f"{n} slots need {n + 1}")
        return (np.array(self.comfort_low_profile, dtype=float),
                np.array(self.comfort_high_profile, dtype=float))

    @property
    def t_comfort_mid(self) -> float:
        """Mid-band at the start: where the room starts."""
        low, high = self.band_at(0)
        return (low + high) / 2.0

    def discomfort_price_per_kelvin_hour(self, reference_price: float) -> float:
        """Currency per K outside the comfort band per hour."""
        if self.discomfort_price is not None:
            return float(self.discomfort_price)
        return (
            self.c_room_kwh_per_k / max(self.cop, 1e-6)
            * float(reference_price)
            * self.discomfort_multiplier
        )

    def validate(self) -> None:
        if not self.t_min < self.t_comfort_low <= self.t_comfort_high < self.t_max:
            raise ValueError("require t_min < t_comfort_low <= t_comfort_high < t_max")
        if (self.comfort_low_profile is None) != (self.comfort_high_profile is None):
            raise ValueError("give comfort_low_profile and comfort_high_profile together")
        if self.comfort_low_profile is not None:
            low = np.asarray(self.comfort_low_profile, dtype=float)
            high = np.asarray(self.comfort_high_profile, dtype=float)
            if low.shape != high.shape or low.size < 2:
                raise ValueError("the comfort profiles need the same length, one value per trajectory point")
            if not (np.all(self.t_min < low) and np.all(low <= high) and np.all(high < self.t_max)):
                raise ValueError("require t_min < low <= high < t_max at every point of the comfort profile")
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if self.n_duty_levels < 2:
            raise ValueError("n_duty_levels must be >= 2 (2 = off or full)")
        if self.comfort_mode not in ("linear", "quadratic"):
            raise ValueError(f"comfort_mode must be 'linear' or 'quadratic', got {self.comfort_mode!r}")


@dataclass(frozen=True)
class SocGate:
    """A soft requirement that SoC be at least `soc_frac` at `hour`.

    Enforced as a large penalty in the DP, not a hard constraint. Anything
    safety-critical must be clamped outside the solver - see policy.clamp().
    """

    hour: float
    soc_frac: float
    penalty: float = 1e4


@dataclass(frozen=True)
class CoordinationConfig:
    """ADMM parameters: proximal message passing (admm.coordinator).

    Every device takes its cheapest plan near a target, responding only to the
    house's imbalance and a shared price; the tank and HVAC iterate as relaxed
    copies, and every iteration recovers a runnable plan. See docs/theory.tex,
    "The second coordinator: ADMM".
    """

    # Relative improvement below which an iteration does not count as a better
    # plan (for exchange_patience).
    converge_tol: float = 0.002
    exchange_rounds: int = 100           # iteration budget (on 12 test sites, 300 came 0.02 closer to the DW bound, at twice the time)
    exchange_rho: float = 0.1            # starting rho (currency per kW^2-slot)
    exchange_rho_gain: float = 0.01      # the paper's rho rule: lambda = mu
    exchange_rho_freeze: int = 200       # hold rho after this many iterations
    exchange_eps: float = 1e-3           # the paper's stopping test, kW per terminal-slot
    exchange_patience: int = 100         # or stop after this many iterations without a better plan
    relax_levels: int = 13               # duty steps the on/off tank and HVAC may take while iterating
    # Each battery's step, each battery on its own: "dp", its DP on its state
    # grid; "lp", its LP (the Dantzig-Wolfe master's battery model) solved
    # exactly (admm.battery_qp). Plain batteries only - an EV's charger
    # minimum, goals and SoC gates stay with the DP.
    exchange_battery_step: str = "dp"
    # Start each LP battery step from that battery's previous solution (about a
    # third fewer interior point iterations; the same answer to tolerance).
    exchange_warm_battery: bool = True
    kink_smoothing: float = 0.25         # kW: round the bill's kink at zero flow over |z| < this (12 test sites: 3x as many runs converge, plans 0.02 further from the DW bound)
    exchange_momentum: bool = False      # Nesterov extrapolation with restart (fast ADMM)
    enable_baseline_fallback: bool = True
    # After the iterations: each device in turn re-plans against the others'
    # CURRENT plans; a change is kept only if the objective falls, so it can
    # never make the plan worse. On 12 test sites (bench/exchange_tune.py) the
    # best iteration's plan lies 1.99 above the DW bound, 0.42 after polishing.
    polish: bool = True
    polish_sweeps: int = 3


@dataclass(frozen=True)
class GridLimits:
    """Hard limits on the grid connection (fuse, or an operator dimming order).

    These are a COUPLING constraint: they bind the sum of every device's power
    plus the inflexible load, so no per-device plan can enforce one on its own.
    The coordinators price them instead, at `breach_price` per kWh beyond the
    limit: Dantzig-Wolfe in its master LP, ADMM in the grid connection's step
    (and each device where it re-plans against the others).

    A plan that exceeds a limit is not cut back afterwards: the exceedance
    stays in the plan, is reported, and is charged at `breach_price`, so its
    impact can be seen. Clipping a setpoint to the measured headroom is
    opt-in (policy.clamp, /api/setpoint). `enforce` is not used.
    """

    max_import_kw: float | None = None
    max_export_kw: float | None = None
    # Throwing PV away is the only way to respect an export cap once storage is
    # full - without it a cap is unenforceable, because surplus has nowhere to
    # go and no price can make it disappear (docs/NOTES.md section 1).
    # Curtailment is also the right answer to a negative export price: being
    # paid to stop generating beats paying to generate.
    allow_curtailment: bool = True
    enforce: bool = True
    # What a kWh drawn (or pushed) beyond the limit costs, in currency per kWh,
    # on top of the energy's own price. ONE constant for the whole horizon: a
    # fuse or a regulatory cap is not less binding when energy is cheap. (It
    # used to be a multiple of each slot's import tariff, which made a breach
    # free at a zero price and paid for at a negative one - exactly when a
    # battery wants to import hardest.)
    #
    # `breach_price` sets it directly. Left as None, it is
    # `breach_price_multiplier` times the largest |price| on the horizon, so it
    # outweighs anything a breach could save in any currency. It is still a
    # finite price rather than a hard constraint, so an unavoidable breach is
    # planned, reported and charged instead of the plan failing. Nothing clips
    # the plan to the limit afterwards. 0 disables the term.
    breach_price: float | None = None
    breach_price_multiplier: float = 10.0

    def validate(self) -> None:
        for name in ("max_import_kw", "max_export_kw"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive or None, got {v}")
        if self.breach_price_multiplier < 0:
            raise ValueError("breach_price_multiplier must be >= 0")
        if self.breach_price is not None and self.breach_price < 0:
            raise ValueError("breach_price must be >= 0")

    @property
    def active(self) -> bool:
        return self.max_import_kw is not None or self.max_export_kw is not None


@dataclass(frozen=True)
class SubMeter:
    """Devices (and optionally the PV) behind one connection to the house's
    AC bus, with that connection's own limits: a hybrid inverter, or a
    sub-panel whose breaker several devices share.

    Like GridLimits these are COUPLING constraints, but on a subset of the
    devices. Dantzig-Wolfe models each sub-meter in its master LP: a balance
    row at the sub-meter (its "bus") and the connection's flows to the house,
    so every member device is priced at the sub-meter's own price (that row's
    dual) instead of the meter's. Where the connection is not at a limit the
    two differ only by the conversion efficiency; where it is, they separate
    (docs/theory.tex, "Sub-meters and local prices").

    Signs follow the devices: a member's power is positive when it draws. The
    connection's limits are AC-side, in kW: `max_export_kw` from the bus to
    the house (an inverter's AC output rating), `max_import_kw` from the house
    to the bus (its AC input rating). None means no limit. `eta_export` and
    `eta_import` are the conversion efficiencies each way (1.0 for a
    sub-panel). Power beyond a limit is planned only where nothing else fits,
    and charged at the grid's breach price, as a grid-limit breach is.

    `pv`: the PV is on this bus (a hybrid inverter's DC side). Its output then
    reaches the house through the connection, and is clipped there - at no
    cost - when the connection cannot pass it.
    """

    name: str
    members: tuple[str, ...] = ()
    pv: bool = False
    max_export_kw: float | None = None
    max_import_kw: float | None = None
    eta_export: float = 1.0
    eta_import: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("a sub-meter needs a name")
        for name in ("max_export_kw", "max_import_kw"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{self.name}: {name} must be >= 0 or None, got {v}")
        for name in ("eta_export", "eta_import"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{self.name}: {name} must be in (0, 1], got {v}")
        if not self.members and not self.pv:
            raise ValueError(f"{self.name}: a sub-meter needs members or the PV")
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"{self.name}: a member is listed twice")

    @property
    def export_cap_dc(self) -> float:
        """The most the bus can send to the house, bus-side (kW); inf if unlimited."""
        return np.inf if self.max_export_kw is None else self.max_export_kw / self.eta_export

    @property
    def import_cap_dc(self) -> float:
        """The most the bus can take from the house, bus-side (kW); inf if unlimited."""
        return np.inf if self.max_import_kw is None else self.max_import_kw * self.eta_import


def hybrid_inverter(batteries: tuple[str, ...] = ("battery",), max_output_kw: float | None = None,
                    max_input_kw: float | None = None, eta_dc_ac: float = 1.0,
                    eta_ac_dc: float = 1.0) -> SubMeter:
    """A hybrid inverter: the PV and `batteries` on its DC bus, the house on
    its AC side. `max_output_kw` is its AC output rating (None: no limit),
    `max_input_kw` its AC input rating (None: the same as the output).
    EMHASS's inverter_ac_output_max, inverter_ac_input_max,
    inverter_efficiency_dc_ac and inverter_efficiency_ac_dc."""
    return SubMeter("inverter", tuple(batteries), pv=True, max_export_kw=max_output_kw,
                    max_import_kw=max_output_kw if max_input_kw is None else max_input_kw,
                    eta_export=eta_dc_ac, eta_import=eta_ac_dc)


def group_limit(name: str, members: tuple[str, ...], min_kw: float | None = None,
                max_kw: float | None = None) -> SubMeter:
    """A limit on what `members` draw together, kW: at most `max_kw` (a shared
    breaker), at least `min_kw` (negative: at most -min_kw of export). None:
    no limit that way."""
    if min_kw is not None and min_kw > 0:
        raise ValueError("min_kw > 0 would force the members to draw; use <= 0 (an export limit)")
    return SubMeter(name, tuple(members), max_import_kw=max_kw,
                    max_export_kw=None if min_kw is None else -min_kw)


@dataclass(frozen=True)
class SiteConfig:
    """Everything the coordinator needs. Devices are optional (None = absent)."""

    horizon: Horizon = field(default_factory=Horizon)
    battery: BatteryConfig | None = field(default_factory=BatteryConfig)
    # Additional batteries beyond `battery`. evcc sends one BatteryConfig per
    # stationary battery AND one per loadpoint (core/site_optimizer.go:532,564),
    # so anyone with a charger has at least two. `battery` stays the first one
    # rather than being folded into a list, so single-battery call sites and
    # `result.devices["battery"]` keep working unchanged.
    batteries: tuple[BatteryConfig, ...] = ()
    water_heater: WaterHeaterConfig | None = field(default_factory=WaterHeaterConfig)
    hvac: HvacConfig | None = field(default_factory=HvacConfig)
    soc_gates: tuple[SocGate, ...] = ()
    grid: GridLimits = field(default_factory=GridLimits)
    coordination: CoordinationConfig = field(default_factory=CoordinationConfig)
    # Devices behind their own connection to the house (a hybrid inverter, a
    # shared breaker): see SubMeter. Each device and the PV in at most one.
    submeters: tuple[SubMeter, ...] = ()

    @property
    def battery_list(self) -> tuple[BatteryConfig, ...]:
        """Every battery, `battery` first. The canonical ordering."""
        head = (self.battery,) if self.battery is not None else ()
        return head + self.batteries

    @staticmethod
    def battery_key(index: int) -> str:
        """Device key for battery `index`. Index 0 is plain "battery" so that
        every existing single-battery call site and test keeps working."""
        return "battery" if index == 0 else f"battery{index}"

    def validate(self) -> None:
        self.horizon.validate()
        self.grid.validate()
        if self.batteries and self.battery is None:
            raise ValueError("cannot set `batteries` without `battery`; battery is index 0")
        for dev in (*self.battery_list, self.water_heater, self.hvac):
            if dev is not None:
                dev.validate()
        if self.hvac is not None:
            self.hvac.comfort_band(self.horizon.steps)   # a profile must fit the horizon
        seen: set[str] = set()
        names: set[str] = set()
        for sm in self.submeters:
            sm.validate()
            if sm.name in names:
                raise ValueError(f"two sub-meters are named {sm.name!r}")
            names.add(sm.name)
            if seen & set(sm.members):
                raise ValueError(f"{sm.name}: a device is behind two sub-meters")
            seen |= set(sm.members)
        if sum(sm.pv for sm in self.submeters) > 1:
            raise ValueError("the PV can be behind one sub-meter only")

    @property
    def pv_submeter(self) -> SubMeter | None:
        """The sub-meter the PV is behind, or None (the PV is on the house's AC bus)."""
        return next((sm for sm in self.submeters if sm.pv), None)

    def submeter_of(self, key: str) -> int | None:
        """Index of the sub-meter device `key` is behind, or None."""
        return next((i for i, sm in enumerate(self.submeters) if key in sm.members), None)

    def without(self, *names: str) -> SiteConfig:
        """Return a copy with the named devices disabled. Test convenience."""
        off: dict[str, Any] = {n: None for n in names}
        return replace(self, **off)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecasts:
    """Exogenous time series, all length `horizon.steps`.

    buy/sell in currency per kWh; every power series in kW.
    `load` is inflexible household demand; `solar` is PV production.
    """

    buy: np.ndarray
    sell: np.ndarray
    load: np.ndarray
    solar: np.ndarray
    outdoor_temp: np.ndarray
    hot_water_demand: np.ndarray

    def validate(self, horizon: Horizon) -> None:
        n = horizon.steps
        for name in (
            "buy",
            "sell",
            "load",
            "solar",
            "outdoor_temp",
            "hot_water_demand",
        ):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")

    @property
    def net_fixed_demand(self) -> np.ndarray:
        """Inflexible load minus PV. The baseline the devices act on top of."""
        return self.load - self.solar


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class DeviceSolution:
    """One device's DP output.

    `value` is V[t, s] with shape (steps + 1, n_states) and `policy` is
    POL[t, s] with shape (steps, n_states). These are the whole point of using
    DP rather than an LP: they define an optimal action from EVERY state, not
    just the one the forecast predicted. See policy.py.
    """

    trajectory: np.ndarray  # state over time, length steps + 1
    power: np.ndarray  # electrical power drawn (+) / supplied (-), length steps
    value: np.ndarray  # V[t, s]
    policy: np.ndarray  # POL[t, s] - action index or value, per device
    states: np.ndarray  # the state grid, length n_states
    actions: np.ndarray  # the action grid
    solve_ms: float = 0.0


@dataclass
class RoundRecord:
    """One ADMM iteration, kept so it can be inspected after the fact.

    The coordinator returns the BEST iteration's runnable plan, polished, not
    the last one, and which iteration that was is not obvious from the outside -
    ADMM is not a descent method, so the cost rises as well as falls. These
    records are what make that visible: every iteration's runnable plan, what it
    cost, and how far the house still was from balance when it was scored.

    Only 1-D series are kept. The value functions and policies are not: they
    are the largest object in a solve by two orders of magnitude, and anything
    that needs one for a non-retained round can re-solve from `dp_load`.
    """

    index: int  # 0-based
    powers: dict[str, np.ndarray]  # per-device electrical power, kW
    trajectories: dict[str, np.ndarray]  # per-device state over time
    net_grid: np.ndarray  # kW at the meter, after curtailment
    curtailment: np.ndarray  # PV thrown away, kW

    import_cost: float
    export_revenue: float
    net_cost: float
    total_objective: float

    # Total kW of grid-limit breach summed over slots, for reporting. Not a
    # ranking key: `total_objective` already prices the breach.
    violation: float
    import_excess: float  # worst single-slot import breach, kW
    export_excess: float  # worst single-slot export breach, kW

    # ADMM residuals as they stood when this iteration was scored: primal is
    # the house's imbalance, sqrt(N) |pbar|; dual is how far the plans moved,
    # rho |(p - pbar) - (p - pbar)_prev|. Convergence needs both small.
    primal_res: float
    dual_res: float
    rho: float

    round_ms: float
    device_ms: dict[str, float]

    # The residual load the battery's runnable plan sits in: the rest of the
    # house. Enough to re-derive this iteration's value function and lambda at
    # the tariff without keeping either.
    battery_dp_load: np.ndarray | None = None

    selected: bool = False

    # The objective this iteration was ranked on (the same as total_objective).
    score_objective: float = 0.0
    fallback_applied: bool = False
    # The cost of this iteration's relaxed plan (fractional tank / HVAC),
    # beside total_objective, the runnable plan recovered from it.
    relaxed_objective: float | None = None


@dataclass
class CoordinationResult:
    """Output of a full coordinated solve."""

    # A DP solution per device - or, for a battery ADMM ran as an exact LP
    # step, that step's plan, which has no value function (battery_pricing has).
    devices: dict[str, DeviceSolution | BatteryStep]
    net_grid: np.ndarray
    import_cost: float
    export_revenue: float
    net_cost: float
    total_objective: float
    rounds_run: int
    # Every round, in order. `selected_round` indexes the one returned.
    rounds: list[RoundRecord] = field(default_factory=list)
    selected_round: int = 0
    # Why the loop ended: "converged" (small residuals), "no improvement"
    # (CoordinationConfig.exchange_patience iterations without a better plan),
    # "iteration cap", or "paused" (an ExchangeRun stopped part-way).
    stop_reason: str = ""
    # Context needed to replay the battery value function later (policy.py).
    # Economics-only battery solve (no ADMM term) used by the pricing/policy
    # tier. See coordinate._pricing_resolve for why this is separate.
    battery_pricing: "DeviceSolution | None" = None
    # Worst residual breach of a grid limit, in kW. Dual methods on a
    # non-convex problem may not close the gap entirely, so this is reported
    # rather than assumed zero.
    grid_import_excess: float = 0.0
    grid_export_excess: float = 0.0
    # PV thrown away per slot (kW) and its total (kWh).
    curtailment: np.ndarray | None = None
    curtailed_kwh: float = 0.0
    # battery 0's bus (meter.Bus) when it sits behind a sub-meter; None on
    # the AC bus. Its pricing solve and the policy snapshot use it.
    battery_bus: Bus | None = None
    battery_dp_load: np.ndarray | None = None
    baseline_cost: float = 0.0
    # Which coordinator produced this: "admm" (coordinate) or "dw" (dw/).
    method: str = "admm"
    # Why the requested method was not the one that ran, if it was not.
    note: str = ""
    # ADMM only: where the solve stood at its best plan (exchange.WarmStart),
    # to start the next solve from - e.g. the next planning cycle, shifted.
    warm_start: object = None
    # Dantzig-Wolfe only. `plan_objective` is the returned plan's objective on
    # the same basis as `lower_bound` (total_objective plus the horizon-edge
    # and SoC-goal terms the DPs optimise), so `gap` is a certificate: no plan
    # is better than the returned one by more than `gap`, up to the DP grids.
    lower_bound: float | None = None
    plan_objective: float | None = None
    # The master's meter price pi_t (currency/kWh): what one more kWh drawn at
    # the meter in slot t would cost the whole house, grid limits included.
    meter_price: np.ndarray | None = None

    @property
    def gap(self) -> float | None:
        if self.lower_bound is None or self.plan_objective is None:
            return None
        return self.plan_objective - self.lower_bound

    @property
    def savings(self) -> float:
        return self.baseline_cost - self.net_cost

    # Kept as views over `rounds` so existing call sites do not have to know
    # about RoundRecord.
    @property
    def round_objectives(self) -> list[float]:
        return [r.total_objective for r in self.rounds]

    @property
    def round_costs(self) -> list[float]:
        return [r.net_cost for r in self.rounds]

    @property
    def timings(self) -> list[dict]:
        return [
            {"round_ms": r.round_ms, "primal_res": r.primal_res,
             "dual_res": r.dual_res, **{f"{k}_ms": v for k, v in r.device_ms.items()}}
            for r in self.rounds
        ]
