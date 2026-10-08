"""Dantzig-Wolfe / column-generation coordinator - the default planner
(home_energy_optimizer.plan); bench/dw_compare.py runs it beside ADMM.

Design notes: dw/README.md. Reference: G. B. Dantzig and P. Wolfe, "Decomposition
Principle for Linear Programs", Operations Research 8(1):101-111, 1960.

The split is the one Dantzig and Wolfe describe: the meter is the only thing
the devices share, so it becomes the (small, linear) master programme; each
device is a sub-programme that only ever has to answer one question -

    "at these per-slot prices, what would you do, and what would it cost you
     privately (comfort, terminal energy)?"

- which is exactly what the existing 1-D DPs already compute when handed a
linear price (buy == sell, no dp_load). Every answer becomes a column; old
columns are never forgotten; the master mixes them to satisfy the meter
balance at least cost and emits new prices (its duals). Unlike the ADMM loop:

* the meter bill, grid limits (soft, priced at the same breach multiplier as
  `total_objective`) and curtailment are modelled EXACTLY in the master LP;
* every iteration yields a Lagrangian LOWER bound, so the result carries a
  certificate: `gap = upper - lower`;
* binary devices need no special treatment - pricing only needs an oracle.

The fractional master solution is a convex mix of device plans. For the
battery that mix is physically realisable (conservatively); for binary devices
it is a duty-cycle schedule. `recover()` turns it into one plan per device.

The master is solved by src/home_energy_optimizer/dw/lpsolver.py (numpy only) by default; pass
solver="highs" to use scipy's HiGHS instead, e.g. to cross-check. The recovery
MILP (one plan per on/off device) uses HiGHS whenever scipy is installed:
the numpy branch and bound stops at a node limit, and with several on/off
devices sharing a rating it can stop on a plan that breaches it while a
feasible one is in the pool.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # 3.11+; only ever an annotation here
    from typing import Unpack

from home_energy_optimizer.dw.lpsolver import Triplets, choose_one
from home_energy_optimizer.dw.lpsolver import linprog as np_linprog

from home_energy_optimizer.coordinate import (
    breach_price,
    battery_terminal_penalty,
    breach_energy,
    comfort_penalty,
    grid_penalty,
    total_objective,
)
from home_energy_optimizer.dp_battery import _gate_thresholds, solve_battery, terminal_price
from home_energy_optimizer.dp_thermal import (
    _max_duty,
    _relaxation,
    _usable_outflow,
    baseline_hvac,
    baseline_water_heater,
    hvac_discomfort,
    solve_hvac,
    solve_water_heater,
    wh_discomfort,
)
from home_energy_optimizer.interface import Participant, Query
from home_energy_optimizer.meter import Bus, bus_cost
from home_energy_optimizer.submeter import (
    MAIN,
    beyond_main,
    device_bus,
    extra_bills,
    fixed_demand,
    pv_on,
    root_grid,
    site_meter,
    tariff,
    topology,
)
from home_energy_optimizer.types import (
    BatteryConfig,
    DeviceSolution,
    Forecasts,
    Horizon,
    HvacConfig,
    SiteConfig,
    SocGate,
    WaterHeaterConfig,
)


def thermal_terminal(cfg: WaterHeaterConfig | HvacConfig, kind: str, t_end: float,
                     ref: float) -> float:
    """The horizon-edge term the thermal DPs put in V[n] (linear comfort mode).

    `total_objective` omits it, but every DP - and the joint-DP and MILP
    references - optimise WITH it. Leaving it out of a column's cost would let
    the master buy a cheaper day by handing tomorrow a cold tank, and would
    make the pricing DP optimise a different objective from the master, which
    voids the lower bound.
    """
    if kind == "water_heater":
        assert isinstance(cfg, WaterHeaterConfig)
        return cfg.heat_capacity_kwh_per_k * ref * max(0.0, cfg.t_comfort - t_end)
    assert isinstance(cfg, HvacConfig)
    return float(hvac_discomfort(cfg, np.array([t_end]), ref, 1.0, *cfg.band_at(-1)).item())


def _pinned_price(b: BatteryConfig) -> float:
    """The terminal price of a battery the coordinator holds. Each has it
    pinned to the tariff's when it is added (DWCoordinator.__init__)."""
    assert b.terminal_price is not None, "a coordinator battery without a pinned terminal price"
    return float(b.terminal_price)


def is_plain_battery(b: BatteryConfig, gates: tuple[SocGate, ...] = ()) -> bool:
    """Can this battery sit in the master as plain LP variables?

    Only if it is nothing more than capacity, power and efficiency. An evcc
    loadpoint is not: it carries a per-slot charge floor (p_demand), a SoC goal
    (s_goal) and often a charger minimum (c_min), which is off-or-at-least - a
    binary per slot. SoC gates are penalties the LP block does not carry either.
    Such a battery bids plans instead, from its DP, which honours all of them.
    """
    return (
        not (b.min_charge_kw is not None and np.any(np.asarray(b.min_charge_kw) > 0))
        and b.soc_goal_kwh is None
        and b.charge_deadband_kw <= 0.0
        and not gates
    )


def battery_extras(b: BatteryConfig, traj: np.ndarray, horizon: Horizon,
                   gates: tuple[SocGate, ...] = ()) -> float:
    """The battery DP's state penalties, as a cost on a trajectory.

    `solve_battery` subtracts a SoC-goal shortfall and a SoC-gate shortfall
    from V[t]. A plan's cost must carry the same terms, or the master would see
    an EV that skips its goal as free - and never charge it.
    """
    total = 0.0
    n = horizon.steps
    if b.soc_goal_kwh is not None:
        goal = np.asarray(b.soc_goal_kwh, dtype=float)
        total += b.soc_goal_penalty * float(np.sum(np.maximum(goal[:n] - traj[:n], 0.0)))
    if gates:
        min_soe, pen = _gate_thresholds(gates, horizon, b.capacity_kwh)
        on = min_soe >= 0
        total += float(np.sum(pen[on] * np.maximum(min_soe[on] - traj[on], 0.0) / b.capacity_kwh))
    return total


class _PriceVector(np.ndarray):
    """A per-slot price whose np.mean() reports a FIXED reference price.

    The thermal DPs derive their comfort price from `np.mean(buy)`. A pricing
    oracle hands them the master's duals as `buy`, and the comfort price must
    not move with those duals - comfort is valued at the tariff in the true
    objective. The clean fix is a `ref_price` argument on the solvers; this
    keeps the library untouched for the prototype.
    """

    _ref: float | None

    def __new__(cls, values: npt.ArrayLike, ref: float) -> _PriceVector:
        obj = np.asarray(values, dtype=float).view(cls)
        obj._ref = ref
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        self._ref = getattr(obj, "_ref", None)

    def mean(self, *a: Any, **k: Any) -> float | None:  # type: ignore[override]  # noqa: D401
        return self._ref


@dataclass
class Column:
    power: np.ndarray        # kW per slot, + = consumption
    trajectory: np.ndarray   # state over time
    cost: float              # PRIVATE cost only: comfort / terminal energy
    source: str              # which oracle proposed it
    mix: list | None = None  # for a kept blend: [(weight, source, born), ...]
    born: int = 0            # iteration that proposed it (0 = seed, -1 = polish)
    runnable: bool = True    # False for an aggregate of an on/off device (a blend it cannot run)
    detail: Any = None       # a participant's own result for this plan (interface.Answer.detail)
    exact: bool = True       # False: a participant's stand-in, not the optimum of the query asked


@dataclass
class Device:
    key: str
    kind: str                # battery | water_heater | hvac | participant (interface.Participant)
    cfg: BatteryConfig | WaterHeaterConfig | HvacConfig | Participant
    columns: list[Column] = field(default_factory=list)
    stamp: int = 0           # iteration number given to columns added now
    on_add: Callable[[Device, Column], None] | None = None  # coordinator hook: keeps the
                                                            # bound's per-price minima current
    duty_cycle_ok: bool = False  # an EV whose charger may be switched within a slot

    @property
    def modulating(self) -> bool:
        """Can a blend of this device's plans be executed as-is?

        A tank whose element runs a fraction of each slot can: its dynamics are
        linear in duty, so a weighted mix of feasible duty schedules is itself
        a feasible duty schedule (re-simulated in `blend_plan` to be exact at
        the cut-out). So can a battery: run the blended power and the meter
        sees exactly the blend, while the store gains at least the blended
        energy (losses make stored energy concave in power) - re-simulated in
        `blend_plan`, clipped where that would overfill it. An on/off tank
        cannot, nor can HVAC, whose blend would mix heating and cooling; nor an
        EV whose charger is off-or-at-least-c_min, since a blend can ask for
        less than c_min - unless the charger may be duty-cycled within a slot
        (`duty_cycle_ok`), which delivers the same energy at allowed setpoints.
        """
        if self.kind == "water_heater":
            return self.tank_cfg.n_duty_levels > 2
        if self.kind == "battery":
            return self.battery_cfg.charge_deadband_kw <= 0.0 or self.duty_cycle_ok
        if self.kind == "participant":
            return bool(self.participant.modulating)
        return False

    # `cfg` as the type its `kind` says it is. Each view checks that, so a
    # device reached through the wrong one fails here, by name.
    @property
    def battery_cfg(self) -> BatteryConfig:
        assert isinstance(self.cfg, BatteryConfig), f"{self.key} is a {self.kind}, not a battery"
        return self.cfg

    @property
    def tank_cfg(self) -> WaterHeaterConfig:
        assert isinstance(self.cfg, WaterHeaterConfig), f"{self.key} is a {self.kind}, not a tank"
        return self.cfg

    @property
    def hvac_cfg(self) -> HvacConfig:
        assert isinstance(self.cfg, HvacConfig), f"{self.key} is a {self.kind}, not an HVAC"
        return self.cfg

    @property
    def participant(self) -> Participant:
        cfg = self.cfg
        assert not isinstance(cfg, (BatteryConfig, WaterHeaterConfig, HvacConfig)), \
            f"{self.key} is a {self.kind}, not a participant"
        return cfg

    def add(self, col: Column, tol: float = 1e-9) -> bool:
        """Add unless an identical plan is already in the pool."""
        for c in self.columns:
            if np.max(np.abs(c.power - col.power)) < tol:
                return False
        col.born = self.stamp
        self.columns.append(col)
        if self.on_add is not None:
            self.on_add(self, col)
        return True


class RunOptions(TypedDict, total=False):
    """`DWCoordinator.run`'s keyword options, for passing them through
    `**kwargs` (`dw_plan`, `dw_coordinate`, `plan`). Each means what it means on
    `run`, and an absent one takes `run`'s default."""

    max_iter: int
    gap_tol: float
    smoothing: float | str
    heuristic_columns: bool
    integer: str
    polish: bool
    verbose: bool
    record: bool
    anytime: bool
    pool: str
    seed_admm: bool
    response: str
    sens_keep: int
    stall: int
    dive: int
    progress: Callable[..., None] | None


@dataclass
class DWResult:
    upper: float                 # objective of the recovered (implementable) plan
    lower: float                 # best Lagrangian bound
    relaxed: float               # master LP value (convexified problem)
    iterations: int
    plan: dict[str, Column]
    weights: dict[str, np.ndarray]
    prices: np.ndarray           # final meter price, currency/kWh
    history: list[dict]
    ms: float
    n_columns: int
    fractional_devices: int
    pricing_error: float = 0.0   # worst amount a pool column undercut the DP oracle
    relaxed_view: dict | None = None  # snapshot of the final master (record=True)
    stop_reason: str = ""             # converged | no new proposals | iteration cap
    plan_parts: dict | None = None    # objective breakdown of the returned plan
    incumbent_used: bool = False      # the returned plan came from an earlier iteration
    admm_value: float | None = None   # ADMM's objective, when the pool was seeded from it
    columns: dict | None = None       # device key -> list of Column (the pool)
    # node or further grid connection -> its bus's price, currency/kWh (the
    # dual of its balance row): what the devices on it were priced at. Empty
    # with no tree.
    local_prices: dict[str, np.ndarray] = field(default_factory=dict)
    # set limit -> its premium, currency/kWh (its row's dual): what its
    # members paid on top of their bus's price while it bound.
    limit_prices: dict[str, np.ndarray] = field(default_factory=dict)


class DWCoordinator:
    def __init__(self, cfg: SiteConfig, fc: Forecasts, battery_in_master: bool = True,
                 tank_in_master: bool = False, solver: str = "numpy", ev_duty_cycle: bool = False,
                 participants: Sequence[Participant] = (),
                 export_ceiling: np.ndarray | None = None,
                 soe_targets: dict | None = None) -> None:
        """`participants`: devices answered through the interface (interface.
        Participant) rather than modelled here, priced in the master beside the
        site's own. `export_ceiling`: a hard per-slot cap on export, kW (for
        example "export no more than the PV surplus"). `soe_targets`: battery
        key -> the stored energy (kWh) a battery held in the master must end at."""
        cfg.validate()
        fc.validate(cfg.horizon)
        self.cfg, self.fc = cfg, fc
        if solver not in ("numpy", "highs"):
            raise ValueError("solver must be 'numpy' or 'highs'")
        self.solver = solver
        self.n, self.dt = cfg.horizon.steps, cfg.horizon.dt
        # The main meter's inflexible part. PV behind a node (a hybrid
        # inverter) is not in it: it reaches the meter through its bus.
        self.d = fixed_demand(cfg, fc).astype(float)
        self.topo = topology(cfg, fc)
        self.ref = float(np.mean(fc.buy))
        for root in self.topo.roots:
            buy, sell = tariff(cfg, fc, root)
            if np.any(sell > buy + 1e-12):
                # The meter cost is then non-convex (buy low, sell high in the same
                # slot) and the master LP would import and export at once.
                raise ValueError("DW master needs sell <= buy in every slot")

        self.devices: list[Device] = []
        # Batteries the master models directly as LP variables (the "hybrid
        # master"): a battery IS a linear programme, so convexifying it into
        # columns only makes the master rediscover its polytope one extreme
        # point at a time. Only the non-linear / binary devices stay as DW
        # blocks - and so does a battery with more than an LP's worth of
        # model (an EV: see is_plain_battery).
        self.lp_batts: list[tuple[str, BatteryConfig]] = []
        for i, b in enumerate(cfg.battery_list):
            if b.capacity_kwh > 0:
                # Pin the terminal price to the TARIFF's: by default the DP
                # derives it from min(buy), and buy will be the master's duals.
                pinned = replace(b, terminal_price=terminal_price(b, fc.buy))
                key = SiteConfig.battery_key(i)
                gates = cfg.soc_gates if i == 0 else ()
                if battery_in_master and is_plain_battery(b, gates):
                    self.lp_batts.append((key, pinned))
                else:
                    # ev_duty_cycle: an EV's charger may be switched within a
                    # slot, so a blend below c_min is run as on/off at >= c_min
                    self.devices.append(Device(key, "battery", pinned, duty_cycle_ok=ev_duty_cycle))
        # The tank in the master (variant "send the model, not proposals"):
        # with the element free to run any fraction of a slot, the tank's
        # dynamics, cut-out and comfort penalty are all linear, so it is an LP
        # exactly - continuous duty, no DP grid, no columns, no memory growth.
        self.lp_tank: WaterHeaterConfig | None = None
        if cfg.water_heater is not None:
            if tank_in_master:
                if cfg.water_heater.comfort_mode != "linear":
                    raise ValueError("tank_in_master needs linear comfort pricing")
                self.lp_tank = cfg.water_heater
            else:
                self.devices.append(Device("water_heater", "water_heater", cfg.water_heater))
        if cfg.hvac is not None:
            self.devices.append(Device("hvac", "hvac", cfg.hvac))
        for p in participants:
            self.devices.append(Device(p.key, "participant", p))
        self.export_ceiling = None if export_ceiling is None else np.asarray(export_ceiling, dtype=float)
        self.soe_targets = dict(soe_targets or {})
        keys = {d.key for d in self.devices} | {k for k, _ in self.lp_batts}
        if self.lp_tank is not None:
            keys.add("water_heater")
        pv_keys = set(fc.pv_keys)
        for name, members in ([(sm.name, sm.members) for sm in cfg.submeters]
                              + [(c.name, c.members) for c in cfg.connections]
                              + [(lim.name, lim.members) for lim in cfg.set_limits]):
            unknown = set(members) - keys - pv_keys
            if unknown:
                raise ValueError(f"{name!r}: no device {sorted(unknown)}")
        # The master's coupling rows, n each: every bus (the roots, then the
        # nodes), then every set limit. A device is in its bus's row and in
        # each of its set limits' rows; its price is the sum of their duals.
        self.rows = list(self.topo.buses) + [f"set:{lim.name}" for lim in cfg.set_limits]
        self._row = {name: i for i, name in enumerate(self.rows)}

        self._build_meter_block()
        # Running minimum of (cost + price . power) per device at every price
        # point priced so far, over EVERY plan ever proposed - including ones
        # the pool later forgets. That is what keeps the bound valid when the
        # pool is pruned or aggregated.
        self._priced: list[np.ndarray] = []
        self._seen_min: list[dict[str, float]] = []
        for dev in self.devices:
            dev.on_add = self._note

    # ----------------------------------------------------------- local prices
    def _rows_of(self, key: str) -> list[int]:
        """The coupling rows device `key` is in: its bus's, then each of its set limits'."""
        rows = [self._row[self.cfg.bus_of(key)]]
        rows += [self._row[f"set:{lim.name}"] for lim in self.cfg.set_limits if key in lim.members]
        return rows

    def _bus(self, key: str) -> int:
        """The row of the bus device `key` is on (0: the main meter's)."""
        return self._row[self.cfg.bus_of(key)]

    def price_of(self, y: np.ndarray, dev: Device | str) -> np.ndarray:
        """The price device `dev` sees in the dual vector `y` (n per coupling
        row, `self.rows`): its bus's, plus each of its set limits'. Per
        kW-slot, like `y`."""
        n = self.n
        rows = self._rows_of(dev if isinstance(dev, str) else dev.key)
        if len(rows) == 1:
            return y[rows[0] * n : (rows[0] + 1) * n]
        return sum((y[r * n : (r + 1) * n] for r in rows), np.zeros(n))

    def _shown(self, pi: np.ndarray) -> np.ndarray:
        """The duals to report for the master's last solve, `pi`: the ones
        nearest the pass-through prices, if they are of the same solve."""
        raw, centered = getattr(self, "_centered", (None, pi))
        return centered if raw is pi else pi

    def local_prices(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Each bus's price in `y` but the main meter's - a node's, a further
        grid connection's - currency/kWh."""
        n = self.n
        return {b: y[self._row[b] * n : (self._row[b] + 1) * n] / self.dt for b in self.topo.buses if b != MAIN}

    def limit_prices(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Each set limit's premium in `y` (its dual), currency/kWh: what its
        members pay on top of their bus's price while it binds."""
        n = self.n
        return {lim.name: y[self._row[f"set:{lim.name}"] * n : (self._row[f"set:{lim.name}"] + 1) * n] / self.dt
                for lim in self.cfg.set_limits}

    # ------------------------------------------------------------------ costs
    def private_cost(self, dev: Device, traj: np.ndarray) -> float:
        """Must match the non-bill terms of `total_objective` exactly."""
        if dev.kind == "battery":
            gates = self.cfg.soc_gates if dev.key == "battery" else ()
            b = dev.battery_cfg
            return (float(traj[0] - traj[-1]) * _pinned_price(b)
                    + battery_extras(b, traj, self.cfg.horizon, gates))
        if dev.kind == "water_heater":
            wh = dev.tank_cfg
            return (float(np.sum(wh_discomfort(wh, traj[1:], self.ref, self.dt)))
                    + thermal_terminal(wh, "water_heater", traj[-1], self.ref))
        hv = dev.hvac_cfg
        low, high = hv.comfort_band(len(traj) - 1)
        return (float(np.sum(hvac_discomfort(hv, traj[1:], self.ref, self.dt, low[1:], high[1:])))
                + thermal_terminal(hv, "hvac", traj[-1], self.ref))

    # --------------------------------------------------------------- oracles
    def _solve(self, dev: Device, buy: np.ndarray, sell: np.ndarray,
               dp_load: np.ndarray | None = None, bus: Bus | None = None) -> DeviceSolution:
        h, fc = self.cfg.horizon, self.fc
        if dev.kind == "battery":
            gates = self.cfg.soc_gates if dev.key == "battery" else ()
            return solve_battery(dev.battery_cfg, h, buy, sell, dp_load=dp_load, soc_gates=gates, bus=bus)
        buy = _PriceVector(buy, self.ref)
        sell = _PriceVector(sell, self.ref)
        if dev.kind == "water_heater":
            return solve_water_heater(dev.tank_cfg, h, buy, sell, fc.hot_water_demand, dp_load=dp_load, bus=bus)
        return solve_hvac(dev.hvac_cfg, h, buy, sell, fc.outdoor_temp, dp_load=dp_load, bus=bus)

    def _note(self, dev: Device, col: Column) -> None:
        for p, mins in zip(self._priced, self._seen_min):
            v = col.cost + float(self.price_of(p, dev) @ col.power)
            if v < mins.get(dev.key, np.inf):
                mins[dev.key] = v

    def _ask(self, dev: Device, query: Query, source: str) -> Column:
        """Ask participant `dev` the interface query `query`; return its answer
        as a pool column tagged `source` (how it was proposed: "price",
        "load_aware"). The column is marked inexact when the participant's
        solve failed and it answered with a stand-in plan."""
        a = dev.participant.respond(query)
        return Column(np.asarray(a.plan_kw, dtype=float).copy(), np.asarray(a.trajectory, dtype=float).copy(),
                      float(a.private_cost), source, detail=a.detail, exact=a.status == "ok")

    def price_solve(self, dev: Device,
                    price_kwh: np.ndarray) -> tuple[Column, DeviceSolution | None]:
        """The pricing DP, returning its solution too (value function and
        policy), which is what the sensitivity variants replay. A participant
        returns no solution, so it offers no sensitivity variants."""
        if dev.kind == "participant":
            return self._ask(dev, Query("price_response", price_kwh, price_kwh), "price"), None
        sol = self._solve(dev, price_kwh, price_kwh, dp_load=None)
        return Column(sol.power.copy(), sol.trajectory.copy(),
                      self.private_cost(dev, sol.trajectory), "price"), sol

    def price_oracle(self, dev: Device, price_kwh: np.ndarray) -> Column:
        """The Dantzig-Wolfe pricing problem: min private cost + price . power.

        buy == sell and no dp_load makes the DP's bill term exactly linear, so
        its argmax is the column of most negative reduced cost (up to the DP's
        own discretisation - the same caveat every lambda in this package has).
        """
        if dev.kind == "participant":
            return self._ask(dev, Query("price_response", price_kwh, price_kwh), "price")
        sol = self._solve(dev, price_kwh, price_kwh, dp_load=None)
        return Column(sol.power.copy(), sol.trajectory.copy(),
                      self.private_cost(dev, sol.trajectory), "price")

    def load_aware_oracle(self, dev: Device, others: Mapping[str, np.ndarray], price_kwh: np.ndarray) -> Column:
        """A HEURISTIC column: today's load-aware DP against the others' mix
        (`others`: every other device's power, by key).

        Any feasible plan is a legal column, so the master can absorb the
        ADMM-style best response too. The kink makes this column good at
        self-consumption plans that a linear price reaches only by mixing
        many extreme points. Limits reach it the way mu does in coordinate():
        as a premium where the master price exceeds the tariff.

        With sub-meters the rest of the meter is what `site_meter` makes of
        the others (a battery's bus-side power reaches the meter through its
        inverter), and a device behind one meets its bus as well (meter.Bus).
        """
        view = device_bus(self.cfg, self.fc, dev.key, others)
        bus, residual = view.bus, view.dp_load
        buy = np.maximum(view.buy, price_kwh)          # its grid connection's tariff
        sell = np.minimum(view.sell, price_kwh)
        if dev.kind == "participant":
            if bus is not None or view.root != MAIN or not getattr(dev.participant, "takes_residual", True):
                # Told no household load, it is asked at its connection's
                # marginal prices given everyone else's plans: what one more kWh
                # drawn there costs, what one more supplied is worth - the
                # meter's tariff and every rating on its way in them. Behind a
                # panel that cannot export, supplying is then worth the breach
                # price wherever the panel is full, and it plans accordingly.
                draw, supply = self._marginal_prices(bus, residual, buy, sell)
                return self._ask(dev, Query("price_response", draw, supply), "load_aware")
            return self._ask(dev, Query("best_response", buy, sell, residual_kw=residual), "load_aware")
        sol = self._solve(dev, buy, sell, dp_load=residual, bus=bus)
        return Column(sol.power.copy(), sol.trajectory.copy(),
                      self.private_cost(dev, sol.trajectory), "load_aware")

    def _capped(self, key: str) -> bool:
        """Whether device `key` sits behind a rated node or counts in a set limit."""
        chain = self.topo.chain(self.cfg.bus_of(key))
        return (any(self.topo.sm[n].max_export_kw is not None or self.topo.sm[n].max_import_kw is not None
                    for n in chain)
                or any(key in lim.members for lim in self.cfg.set_limits))

    def _repair(self, plan: dict[str, Column], integer: str, rounds: int = 3) -> dict[str, Column]:
        """A recovered plan that still breaches a rating, repaired: each device
        behind a rated node is asked again against the plans the others will
        actually run (not a mix of them) - a participant at the secant prices
        of its whole power, so an EV of 7 kW sees that a slot with 3 kW of
        headroom costs it the breach - and the plan is recovered again from
        the larger pool, kept only if it scores better. Up to `rounds` times."""
        for _ in range(rounds):
            if self.parts(plan)["breach_kwh"] <= 1e-6:
                break
            added = 0
            for dev in self.devices:
                if not self._capped(dev.key):
                    continue
                others = {k: c.power for k, c in plan.items() if k != dev.key}
                added += dev.add(self.load_aware_oracle(dev, others, self.fc.buy))
                if dev.kind == "participant":
                    view = device_bus(self.cfg, self.fc, dev.key, others)
                    q = max(float(dev.participant.max_power_kw), 1e-3)
                    draw, supply = self._marginal_prices(view.bus, view.dp_load, view.buy, view.sell, eps=q)
                    added += dev.add(self._ask(dev, Query("price_response", draw, supply), "repair"))
            if not added:
                break
            _, lam, _, _, bp = self.solve_master()
            new = self.recover(integer, self.split_weights(lam), bp)
            if self.score(new) >= self.score(plan) - 1e-9:
                break
            plan = new
        return plan

    def _add_contained(self, dev: Device, price_kwh: np.ndarray) -> int:
        """For a participant behind a node that caps what it may send up: its
        plan at `price_kwh` when supplying is worth the breach price less than
        drawing - one that never sends power up. A plan whose discharge fits
        the others' current mix may not fit the plans recovery picks for them
        (two on/off devices behind a panel that cannot feed back); this one
        fits any. Returns how many columns were added (0 or 1)."""
        if dev.kind != "participant":
            return 0
        capped = any(self.topo.sm[node].max_export_kw is not None
                     for node in self.topo.chain(self.cfg.bus_of(dev.key)))
        if not capped:
            return 0
        penalty = breach_price(self.cfg.grid, self.fc.buy, self.fc.sell)
        return dev.add(self._ask(dev, Query("price_response", price_kwh, price_kwh - penalty), "contained"))

    def _marginal_prices(self, bus: Bus | None, residual: np.ndarray, buy: np.ndarray, sell: np.ndarray,
                         eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
        """Per slot, currency/kWh: what one more kWh drawn by a device on
        `bus` (None: at the meter) costs, and what one more kWh it supplies
        is worth, with `residual` the rest of its meter and `buy` / `sell`
        that meter's prices; breach of any rating on the way included."""
        draw, supply = np.empty(self.n), np.empty(self.n)
        for t in range(self.n):
            q = np.array([0.0, eps, -eps])
            if bus is None:
                at_meter, over = q, np.zeros(3)
            else:
                at_meter, over = bus_cost(q, t, bus)
            imp = at_meter + residual[t]
            cost = buy[t] * np.maximum(imp, 0.0) - sell[t] * np.maximum(-imp, 0.0) + over
            draw[t] = (cost[1] - cost[0]) / eps
            supply[t] = (cost[0] - cost[2]) / eps
        return draw, supply

    def _add_variants(self, dev: Device, sol: DeviceSolution, col: Column,
                      price_kwh: np.ndarray, keep: int, flex_out: dict | None) -> int:
        """Add the device's sensitivity variants around its pricing plan.

        Each is "force one step more (or less) at hour t, then re-plan with
        the policy". The `keep` cheapest per direction (by cost at the price)
        go into the pool, so the master sees the directions in which the
        device is cheapest to move.
        """
        from home_energy_optimizer.dw.sensitivity import flexibility, variants
        found, _ = variants(self, dev, sol)
        if flex_out is not None:
            flex_out[dev.key] = flexibility(self, dev, sol, price_kwh, col.cost, found)
        scored = []
        for t, d, pw, tr in found:
            cost = self.private_cost(dev, tr)
            scored.append((cost + float(price_kwh @ pw) * self.dt, d, pw, tr, cost))
        added = 0
        for direction in (+1, -1, 0):
            group = sorted((z for z in scored if z[1] == direction), key=lambda z: z[0])
            for _, _, pw, tr, cost in group[: (keep if direction else keep // 4)]:
                added += dev.add(Column(pw.copy(), tr.copy(), cost, "sens"))
        return added

    def seed_from_admm(self) -> dict[str, Column]:
        """Variant: ADMM as a column generator, DW as referee.

        Every ADMM round proposes a plan per device, and any feasible plan is a
        legal column, so all of them go into the pool. Returns ADMM's returned
        plan as runnable Columns, which the caller keeps as the incumbent -
        the DW result can then never be worse than ADMM's, and the DW bound
        certifies ADMM's plan too.
        """
        from home_energy_optimizer.coordinate import coordinate
        res = coordinate(self.cfg, self.fc)
        self.admm_value = extended_objective(self.cfg, self.fc, res.net_grid,
                                             {k: sol.trajectory for k, sol in res.devices.items()})
        by_key = {d.key: d for d in self.devices}
        for rec in res.rounds:
            for k, pw in rec.powers.items():
                if k in by_key:
                    tr = rec.trajectories[k]
                    by_key[k].add(Column(pw.copy(), tr.copy(), self.private_cost(by_key[k], tr), "admm"))
        # Scored by parts() from the trajectories, so no per-column cost needed.
        return {k: Column(sol.power.copy(), sol.trajectory.copy(), 0.0, "admm")
                for k, sol in res.devices.items()}

    def seed(self) -> None:
        """Starting columns. Any feasible plan per device keeps the master
        feasible (the meter variables absorb any mismatch), so Phase I is
        free: no artificial columns, no Farkas pricing."""
        h, fc = self.cfg.horizon, self.fc
        for dev in self.devices:
            if dev.kind == "battery":
                # Idle is feasible - unless the device must charge somewhere
                # (an EV's p_demand floor); the price-oracle seeds below are
                # feasible either way. Its cost carries any missed SoC goal.
                b = dev.battery_cfg
                mc = b.min_charge_kw
                if mc is None or not np.any(np.asarray(mc) > 0):
                    s0 = b.capacity_kwh * b.soc_initial_frac
                    traj = np.full(self.n + 1, s0)
                    dev.add(Column(np.zeros(self.n), traj, self.private_cost(dev, traj), "idle"))
            elif dev.kind == "water_heater":
                temp, pw = baseline_water_heater(dev.tank_cfg, h, fc.hot_water_demand)
                dev.add(Column(pw, temp, self.private_cost(dev, temp), "thermostat"))
            elif dev.kind == "participant":
                a = dev.participant.baseline()
                dev.add(Column(np.asarray(a.plan_kw, dtype=float), np.asarray(a.trajectory, dtype=float),
                               float(a.private_cost), "baseline", detail=a.detail))
            else:
                temp, pw = baseline_hvac(dev.hvac_cfg, h, fc.outdoor_temp)
                dev.add(Column(pw, temp, self.private_cost(dev, temp), "thermostat"))
        # The two corners of the dual box: "everything imports" and
        # "everything exports". Cheap and spans the extremes.
        for p in (fc.buy, np.maximum(fc.sell, 0.0)):
            for dev in self.devices:
                dev.add(self.price_oracle(dev, p))

    # ---------------------------------------------------------------- master
    def _build_meter_block(self) -> None:
        """The master's own variables, as families of n (one per slot), each
        with its coefficient in one or two coupling rows, its cost per slot
        and its upper bound: `self.blocks`, a list of ([(row, coef)], cost,
        upper). And the coupling rows' right-hand sides, `self.rhs_bal`.

        At each grid connection (a root): import within / over its limit,
        export within / over it, and curtailment of the PV on it. Breach is
        priced exactly as grid_penalty prices it, so the master minimises
        total_objective. At each node: its link to its parent - sending x
        (bus-side) puts eta_export * x in the parent's row, taking x draws
        x / eta_import from it - within its ratings and, beyond them, at the
        breach price; and the PV clipped on it. At each set limit: its
        members' total within the limit, and beyond it at the breach price.
        """
        g, fc, dt = self.cfg.grid, self.fc, self.dt
        cfg, topo, n = self.cfg, self.topo, self.n
        # No flow can exceed everything the house could draw or push at
        # once. Using that instead of +inf changes no solution, but it matters
        # when export pays exactly the import price: "import one more kWh and
        # export it again" then costs nothing, the optimal set is unbounded,
        # and an interior-point method drifts along it until it stalls.
        reach = (np.abs(fc.load) + np.abs(fc.total_solar)
                 + sum(max(b.p_charge_max_kw, b.p_discharge_max_kw) for b in cfg.battery_list)
                 + (cfg.water_heater.power_kw if cfg.water_heater is not None else 0.0)
                 + (cfg.hvac.power_kw if cfg.hvac is not None else 0.0)
                 + sum(float(d.participant.max_power_kw) for d in self.devices if d.kind == "participant"))
        inf = 2.0 * reach + 1.0
        full = lambda v: np.broadcast_to(v, (n,)).astype(float)      # noqa: E731
        zero = np.zeros(n)
        blocks: list[tuple[list[tuple[int, float]], np.ndarray, np.ndarray]] = []
        rhs = np.zeros(len(self.rows) * n)

        for root in topo.roots:
            r = self._row[root]
            rg = root_grid(cfg, root)
            buy, sell = tariff(cfg, fc, root)
            # breach: a constant price per kWh beyond a limit, as grid_penalty
            B = breach_price(rg, buy, sell) if rg.active else 0.0
            imp_cap = np.minimum(rg.max_import_kw, inf) if rg.max_import_kw is not None else inf
            exp_cap = np.minimum(rg.max_export_kw, inf) if rg.max_export_kw is not None else inf
            if root == MAIN and self.export_ceiling is not None:
                # a hard cap: no over-cap block below lets the master buy past it
                exp_cap = np.minimum(exp_cap, self.export_ceiling)
            blocks.append(([(r, +1.0)], buy * dt, full(imp_cap)))                     # import
            if rg.max_import_kw is not None:
                blocks.append(([(r, +1.0)], (buy + B) * dt, full(inf)))               # over import cap
            blocks.append(([(r, -1.0)], -sell * dt, full(exp_cap)))                   # export
            if rg.max_export_kw is not None:
                blocks.append(([(r, -1.0)], (B - sell) * dt, full(inf)))              # over export cap
            pv = pv_on(fc, topo, root)
            if rg.allow_curtailment and np.any(pv > 0):
                blocks.append(([(r, -1.0)], zero, pv.copy()))                        # curtail
            rhs[r * n : (r + 1) * n] = self.d if root == MAIN else -pv

        Bn = breach_price(g, fc.buy, fc.sell)    # a rating is a limit even with no grid limit
        # node -> the block indices of its link: (send within / past, take within / past)
        self.link_blocks: dict[str, tuple[list[int], list[int]]] = {}
        for node in topo.nodes:
            sm = topo.sm[node]
            r, up = self._row[node], self._row[topo.parent[node]]
            out_cap, in_cap = sm.export_cap_dc, sm.import_cap_dc
            sends, takes = [len(blocks)], []
            blocks.append(([(up, sm.eta_export), (r, -1.0)], zero, np.minimum(full(out_cap), inf)))     # send
            if np.isfinite(out_cap):
                sends.append(len(blocks))
                blocks.append(([(up, sm.eta_export), (r, -1.0)], full(Bn * dt), full(inf)))           # past it
            takes.append(len(blocks))
            blocks.append(([(up, -1.0 / sm.eta_import), (r, 1.0)], zero, np.minimum(full(in_cap), inf)))  # take
            if np.isfinite(in_cap):
                takes.append(len(blocks))
                blocks.append(([(up, -1.0 / sm.eta_import), (r, 1.0)], full(Bn * dt), full(inf)))    # past it
            self.link_blocks[node] = (sends, takes)
            pv = pv_on(fc, topo, node)
            if np.any(pv > 0):
                # PV clipped on the bus: free; only what the rating forces,
                # unless curtailment is allowed (then any of it)
                clip = pv.copy() if g.allow_curtailment else np.maximum(pv - out_cap, 0.0)
                blocks.append(([(r, -1.0)], zero, clip))
            rhs[r * n : (r + 1) * n] = -pv

        for lim in cfg.set_limits:
            # its row: imp - exp (+ over imp - over exp) = its members' total
            r = self._row[f"set:{lim.name}"]
            imp_cap = full(inf if lim.max_import_kw is None else np.minimum(lim.max_import_kw, inf))
            exp_cap = full(inf if lim.max_export_kw is None else np.minimum(lim.max_export_kw, inf))
            blocks.append(([(r, +1.0)], zero, imp_cap))
            if lim.max_import_kw is not None:
                blocks.append(([(r, +1.0)], full(Bn * dt), full(inf)))
            blocks.append(([(r, -1.0)], zero, exp_cap))
            if lim.max_export_kw is not None:
                blocks.append(([(r, -1.0)], full(Bn * dt), full(inf)))
        self.blocks = blocks
        self.rhs_bal = rhs
        # the main meter's own, as earlier releases exposed them: (coef, cost, upper)
        self.meter_blocks = [(rc[0][1], cost, up) for rc, cost, up in blocks if rc[0][0] == 0 and len(rc) == 1]

    # ---------------------------------------------------------------- solvers
    def _lp(self, c: np.ndarray, A: Triplets, b: np.ndarray, lb: np.ndarray,
            ub: np.ndarray, border: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
        """min c.x s.t. A x = b, lb <= x <= ub -> (x, fun, equality duals)."""
        if self.solver == "highs":
            from scipy import sparse
            from scipy.optimize import linprog
            r, k, v = A.coo()
            M = sparse.csr_matrix((v, (r, k)), shape=A.shape)
            res = linprog(c, A_eq=M, b_eq=b, bounds=list(zip(lb, ub)), method="highs")
            if res.status != 0:
                raise RuntimeError(f"LP failed: {res.message}")
            return res.x, res.fun, res.eqlin.marginals
        res = np_linprog(c, A, b, lb, ub, border=border)
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        return res.x, res.fun, res.y

    def _choose(self, c: np.ndarray, A: Triplets, b: np.ndarray, lb: np.ndarray,
                ub: np.ndarray, groups: list[np.ndarray], time_limit: float,
                node_limit: int = 150, border: int = 0) -> tuple[np.ndarray, float]:
        """The recovery MILP: one plan per group. -> (x, fun). HiGHS, exact,
        when scipy is there; else the numpy branch and bound, to `node_limit`."""
        if self.solver == "highs" or _have_scipy():
            from scipy import sparse
            from scipy.optimize import Bounds, LinearConstraint, milp
            r, k, v = A.coo()
            M = sparse.csr_matrix((v, (r, k)), shape=A.shape)
            integrality = np.zeros(len(c))
            for g in groups:
                integrality[g] = 1
            res = milp(c, constraints=LinearConstraint(M, b, b), integrality=integrality,
                       bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
            if res.x is None:
                raise RuntimeError(f"integer master failed: {res.message}")
            return res.x, res.fun
        res = choose_one(c, A, b, lb, ub, groups, time_limit=time_limit, node_limit=node_limit,
                         border=border)
        if res.x is None:
            raise RuntimeError("integer master: no plan found within the limits")
        return res.x, res.fun

    def _tank_block(self, A: Triplets, c: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                    rhs: np.ndarray, j0: int, row0: int, meter: bool = True, bal_rows: Sequence[int] = (0,)
                    ) -> tuple[int, int, Callable[[np.ndarray], Column]]:
        """Write the tank's LP into (A, c, bounds, rhs) at column j0, row row0.

        Variables, each length n unless noted: duty D in [0, 1], temperature
        T[1..n] <= t_max (the cut-out), shortfall S >= 0, a slack U >= 0 per
        comfort row, then terminal shortfall ST and its slack UT (1 each).
        Rows: n dynamics, n comfort, 1 terminal. The dynamics are the DP's,
        exactly: outflow C*r*(T - t_inf) with r capped at 1/dt, so
            T[t+1] - (1 - r dt) T[t] - (P dt / C) D[t] = r dt t_inf
        Returns (n_vars, n_rows, decode) where decode(x) -> Column.
        """
        cfg, n, dt = self.lp_tank, self.n, self.dt
        assert cfg is not None, "_tank_block runs only with the tank in the master"
        C, P = cfg.heat_capacity_kwh_per_k, cfg.power_kw
        price_k = cfg.discomfort_price_per_kelvin_hour(self.ref)
        r = np.empty(n); tinf = np.empty(n)
        for i in range(n):
            rate, ti = _relaxation(cfg, float(self.fc.hot_water_demand[i]))
            r[i], tinf[i] = min(rate, 1.0 / dt), ti
        D, T, S, U = j0, j0 + n, j0 + 2 * n, j0 + 3 * n
        ST, UT = j0 + 4 * n, j0 + 4 * n + 1
        t = np.arange(n)
        ub[D:D + n] = 1.0
        # A finite floor that can never bind (the tank cannot cool below
        # min(t_ambient, t_inlet) >= t_min): the interior-point solver needs
        # finite lower bounds.
        lb[T:T + n] = cfg.t_min - 100.0
        ub[T:T + n] = cfg.t_max
        c[S:S + n] = price_k * dt
        c[ST] = C * self.ref
        if meter:
            for bal_row in bal_rows:             # its coupling rows: minus consumption
                A[bal_row + t, D + t] = -P
        dyn = row0 + t
        A[dyn, T + t] = 1.0
        A[dyn[1:], T + t[:-1]] = -(1.0 - r[1:] * dt)
        A[dyn, D + t] = -P * dt / C
        rhs[dyn] = r * dt * tinf
        rhs[dyn[0]] += (1.0 - r[0] * dt) * cfg.t_comfort
        com = row0 + n + t                       # S[t] + T[t+1] - U[t] = t_comfort
        A[com, S + t] = 1.0
        A[com, T + t] = 1.0
        A[com, U + t] = -1.0
        rhs[com] = cfg.t_comfort
        term = row0 + 2 * n                      # ST + T[n] - UT = t_comfort
        A[term, ST] = 1.0
        A[term, T + n - 1] = 1.0
        A[term, UT] = -1.0
        rhs[term] = cfg.t_comfort

        def decode(x: np.ndarray) -> Column:
            traj = np.concatenate([[cfg.t_comfort], x[T:T + n]])
            return Column(P * x[D:D + n], traj, self.private_cost(Device("water_heater", "water_heater", cfg), traj), "lp")
        return 4 * n + 2, 2 * n + 1, decode

    @overload
    def solve_master(self, integer: Literal[False] = ..., time_limit: float = ...,
                     node_limit: int = ...
                     ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, Column]]: ...
    @overload
    def solve_master(self, integer: Literal[True], time_limit: float = ...,
                     node_limit: int = ...
                     ) -> tuple[float, np.ndarray, None, None, dict[str, Column]]: ...
    def solve_master(self, integer: bool = False, time_limit: float = 30.0,
                     node_limit: int = 150) -> tuple[
            float, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Column]]:
        """Returns (value, column weights, meter duals, convexity duals, battery plans)."""
        n, devs, dt = self.n, self.devices, self.dt
        # Rows: the coupling rows (every bus, then every set limit; n each),
        # the convexity rows, then the batteries' and the tank's own rows.
        # The first two are the border that couples the rest.
        n_bal = n * len(self.rows)
        n_meter = len(self.blocks) * n
        n_cols = sum(len(d.columns) for d in devs)
        nb = len(self.lp_batts)
        # per LP battery: charge c[n], discharge e[n], soe s[1..n]
        tank_nv = 4 * n + 2 if self.lp_tank is not None else 0
        tank_nr = 2 * n + 1 if self.lp_tank is not None else 0
        nv = n_meter + n_cols + 3 * n * nb + tank_nv
        n_rows = n_bal + len(devs) + n * nb + tank_nr

        c = np.zeros(nv)
        ub = np.full(nv, np.inf)
        lb = np.zeros(nv)
        # Coordinate triplets: a 10-battery, 48 h site is ~2k rows x ~7k
        # columns, which dense would put at ~100 MB.
        A = Triplets((n_rows, nv))
        rhs = np.zeros(n_rows)
        rhs[:n_bal] = self.rhs_bal
        rhs[n_bal : n_bal + len(devs)] = 1.0
        t_all = np.arange(n)
        j = 0
        for coefs, cost, upper in self.blocks:
            c[j : j + n] = cost
            ub[j : j + n] = upper
            for row, coef in coefs:
                A[row * n + t_all, j + t_all] = coef
            j += n
        for b, shut in getattr(self, "link_fix", {}).items():
            ub[b * n : (b + 1) * n][shut] = 0.0            # a link's direction, fixed for recovery
        for di, dev in enumerate(devs):
            rows = self._rows_of(dev.key)
            for col in dev.columns:
                c[j] = col.cost
                for row in rows:
                    A[row * n : (row + 1) * n, j] = -col.power.reshape(-1, 1)
                A[n_bal + di, j] = 1.0
                # An aggregate an on/off device cannot run is fine for the LP
                # (it is a point of the convex hull) but not for recovery.
                ub[j] = 1.0 if (not integer or col.runnable or dev.modulating) else 0.0
                j += 1
        base_row = n_bal + len(devs)
        batt_cols: list[tuple[str, BatteryConfig, int, int, int]] = []
        for bi, (key, b) in enumerate(self.lp_batts):
            C0, E0, S0 = j, j + n, j + 2 * n
            j += 3 * n
            batt_cols.append((key, b, C0, E0, S0))
            ub[C0 : C0 + n] = b.p_charge_max_kw
            ub[E0 : E0 + n] = b.p_discharge_max_kw
            ub[S0 : S0 + n] = b.capacity_kwh
            lb[S0 : S0 + n] = b.soe_floor_kwh
            if key in self.soe_targets:
                lb[S0 + n - 1] = ub[S0 + n - 1] = self.soe_targets[key]
            t = np.arange(n)
            for row in self._rows_of(key):  # its bus's row, and its set limits'
                A[row * n + t, C0 + t] = -1.0     # minus consumption
                A[row * n + t, E0 + t] = +1.0
            r = base_row + bi * n + t    # s[t+1] - s[t] - dt*eta_c*c + dt*e/eta_d = 0
            A[r, S0 + t] = 1.0
            A[r[1:], S0 + t[:-1]] = -1.0
            A[r, C0 + t] = -dt * b.eta_c
            A[r, E0 + t] = dt / b.eta_d
            s0 = b.capacity_kwh * b.soc_initial_frac
            rhs[r[0]] = s0
            # terminal: bill the net depletion at the terminal price
            c[S0 + n - 1] = -_pinned_price(b)

        tank_decode = None
        if self.lp_tank is not None:
            _, _, tank_decode = self._tank_block(A, c, lb, ub, rhs, j, base_row + n * nb,
                                                 bal_rows=[r * n for r in self._rows_of("water_heater")])
        const = sum(_pinned_price(b) * b.capacity_kwh * b.soc_initial_frac
                    for _, b in self.lp_batts)
        lam_sl = slice(n_meter, n_meter + n_cols)

        def batt_plans(x: np.ndarray) -> dict[str, Column]:
            out = {}
            for key, b, C0, E0, S0 in batt_cols:
                s0 = b.capacity_kwh * b.soc_initial_frac
                traj = np.concatenate([[s0], x[S0 : S0 + n]])
                out[key] = Column(x[C0 : C0 + n] - x[E0 : E0 + n], traj,
                                  (s0 - traj[-1]) * _pinned_price(b), "lp")
            if tank_decode is not None:
                out["water_heater"] = tank_decode(x)
            return out

        if integer:
            groups, j = [], n_meter
            for dev in devs:
                if not dev.modulating:
                    groups.append(np.arange(j, j + len(dev.columns)))
                j += len(dev.columns)
            # the meter + convexity rows are the border that couples the
            # batteries' and the tank's otherwise independent row blocks
            x, fun = self._choose(c, A, rhs, lb, ub, groups, time_limit, node_limit, base_row)
            self._note_links(x)
            return fun + const, x[lam_sl], None, None, batt_plans(x)

        x, fun, duals = self._lp(c, A, rhs, lb, ub, border=base_row)
        self._note_links(x)
        # d(cost)/d(d_t), currency per kW-slot: the meter's, then each bus's.
        # Every price this class hands around is this whole vector; price_of
        # picks a device's own.
        pi = duals[:n_bal]
        # the same optimum's duals nearest the pass-through prices: a second
        # point to price at, and the local prices shown (_shown)
        self._centered = (pi, self._center_duals(c, A, lb, ub, x, duals)[:n_bal]
                          if self.topo.nodes or self.cfg.set_limits else pi)
        sigma = duals[n_bal : n_bal + len(devs)]
        self.batt_lambda = {     # SoE-balance duals: the exact LP costate
            key: duals[base_row + bi * n : base_row + (bi + 1) * n]
            for bi, (key, *_rest) in enumerate(batt_cols)
        }
        return fun + const, x[lam_sl], pi, sigma, batt_plans(x)

    def _center_duals(self, c: np.ndarray, A: Triplets, lb: np.ndarray, ub: np.ndarray,
                      x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Among the master's optimal duals, the one nearest the pass-through
        prices: each node's price within its connection's band of its
        parent's, [eta_export, 1/eta_import] times it, and each set limit's
        premium at 0 - as far as staying optimal allows.

        A node's price is not unique while its connection carries nothing:
        behind a panel that cannot export, with nothing there drawing, any
        price from the parent's minus the breach price up to the parent's is
        optimal. A solver picks one end, the devices there are then priced
        far below the meter, and the loop chases that price for dozens of
        iterations without closing the gap. The loop prices at this point as
        well as at the solver's (the solver's own, an interior point's near
        the centre of the optimal set, is often the better guide elsewhere),
        and reports it as the local prices.

        Parents before children, a slot at a time, a node moves with every
        node below it whose link is tight (a link variable at zero reduced
        cost - flowing, or about to), each scaled by its link's efficiency
        so those links stay tight; otherwise raising a panel would make its
        breaker's idle link worth using. The step keeps every reduced cost's
        sign (a variable strictly between its bounds pins it), so the vector
        stays dual-feasible and complementary to x.
        """
        n, topo = self.n, self.topo
        rows, cols, vals = A.coo()
        y = y.copy()
        red = c - np.bincount(cols, weights=vals * y[rows], minlength=len(c))
        order = np.argsort(rows, kind="stable")
        rows, cols, vals = rows[order], cols[order], vals[order]
        start = np.searchsorted(rows, np.arange(A.shape[0] + 1))
        tol_r = 1e-7
        tol_x = 1e-6 * (1.0 + np.minimum(np.abs(np.where(np.isfinite(lb), lb, 0.0)),
                                          np.abs(np.where(np.isfinite(ub), ub, 0.0))))
        free = ub - lb > tol_x
        # +1: holds at its lower bound (keep red >= 0), -1: at its upper
        # (keep red <= 0), 0: strictly between (pins every row it is in)
        side = np.where(red > tol_r, 1, np.where(red < -tol_r, -1,
                        np.where(x - lb <= tol_x, 1, np.where(ub - x <= tol_x, -1, 0))))

        def tight(node: str, t: int) -> float:
            """The factor a tight link scales its node's step by, or 0."""
            sends, takes = self.link_blocks[node]
            sm = topo.sm[node]
            if any(abs(red[b * n + t]) <= tol_r for b in sends):
                return sm.eta_export
            if any(abs(red[b * n + t]) <= tol_r for b in takes):
                return 1.0 / sm.eta_import
            return 0.0

        def move(at: list[tuple[int, float]], target: float) -> None:
            """Shift y[i] by f * d for each (i, f): d toward target - y[at[0]]."""
            k = np.concatenate([np.arange(start[i], start[i + 1]) for i, _ in at])
            f = np.concatenate([np.full(start[i + 1] - start[i], fi) for i, fi in at])
            J, inv = np.unique(cols[k], return_inverse=True)
            a = np.bincount(inv, weights=vals[k] * f, minlength=len(J))
            on = free[J] & (np.abs(a) > 1e-12)
            J, a = J[on], a[on]
            s = side[J]
            if np.any(s == 0):
                return
            r = red[J]
            # red_j - a_j * d keeps its sign: a bound on d from each
            bound = np.where(s > 0, np.maximum(r, 0.0), np.minimum(r, 0.0)) / a
            upper = (s > 0) == (a > 0)
            hi = float(np.min(bound[upper], initial=np.inf))
            lo = float(np.max(bound[~upper], initial=-np.inf))
            d = min(max(target - y[at[0][0]], min(lo, 0.0)), max(hi, 0.0))
            if d != 0.0:
                for i, fi in at:
                    y[i] += fi * d
                red[J] -= a * d

        for node in reversed(topo.order):                 # parents first
            sm = topo.sm[node]
            me, up = self._row[node] * n, self._row[topo.parent[node]] * n
            for t in range(n):
                at, stack = [(me + t, 1.0)], [(node, 1.0)]
                while stack:
                    b, fb = stack.pop()
                    for ch in topo.children[b]:
                        f_ch = tight(ch, t) * fb
                        if f_ch:
                            at.append((self._row[ch] * n + t, f_ch))
                            stack.append((ch, f_ch))
                p = y[up + t]
                band = sorted((sm.eta_export * p, p / sm.eta_import))
                move(at, min(max(y[me + t], band[0]), band[1]))
        for lim in self.cfg.set_limits:
            me = self._row[f"set:{lim.name}"] * n
            for t in range(n):
                move([(me + t, 1.0)], 0.0)
        return y

    def split_weights(self, lam: np.ndarray) -> dict[str, np.ndarray]:
        out, j = {}, 0
        for dev in self.devices:
            out[dev.key] = lam[j : j + len(dev.columns)]
            j += len(dev.columns)
        return out

    def snapshot(self, weights: dict[str, np.ndarray], pi: np.ndarray,
                 bplans: dict[str, Column]) -> dict:
        """What one master solve looks like, for plotting: the convex mix of
        each DW device's plans (a mix of temperature trajectories is shown as
        the same mix - an approximation for a non-linear device), the battery
        LP plans, the meter price and the battery costates."""
        powers, trajs, active = {}, {}, {}
        for dev in self.devices:
            w = weights[dev.key]
            powers[dev.key] = sum(wi * c.power for wi, c in zip(w, dev.columns))
            trajs[dev.key] = sum(wi * c.trajectory for wi, c in zip(w, dev.columns))
            active[dev.key] = [(float(wi), dev.columns[i].source, dev.columns[i].born)
                               for i, wi in enumerate(w) if wi > 1e-6]
        for key, col in bplans.items():
            powers[key], trajs[key] = col.power, col.trajectory
        # lambda on s[t+1]: an extra kWh on the right of SoE row t lowers cost.
        lam = {k: -v for k, v in getattr(self, "batt_lambda", {}).items()}
        return {"price": pi[: self.n] / self.dt, "local_prices": self.local_prices(self._shown(pi)),
                "limit_prices": self.limit_prices(self._shown(pi)),
                "powers": powers, "trajectories": trajs, "active": active, "lambda": lam}

    def blend_plan(self, dev: Device, w: np.ndarray) -> Column:
        """Execute a weighted mix of a modulating device's plans.

        The mixed power is re-run through the device's physics, so the
        trajectory and cost are what it actually produces - exact even where a
        mix would carry a tank past its cut-out or a battery past full.
        """
        dt = self.dt
        power = sum(wi * c.power for wi, c in zip(w, dev.columns))
        mix = [(float(wi), c.source, c.born) for wi, c in zip(w, dev.columns) if wi > 1e-6]
        if dev.kind == "participant":
            # A participant that says it modulates has linear dynamics and a
            # convex cost: the mix of its plans runs as it is (within each slot,
            # as a share of the slot per plan), at no more than the mixed cost.
            used = [(wi, c) for wi, c in zip(w, dev.columns) if wi > 1e-12]
            blend = getattr(dev.cfg, "blend", None)    # optional (interface.Participant)
            detail = blend([wi for wi, _ in used], [c.detail for _, c in used]) if blend is not None else None
            return Column(power, sum(wi * c.trajectory for wi, c in used), float(sum(wi * c.cost for wi, c in used)),
                          "blend", mix, detail=detail)
        if dev.kind == "battery":
            b = dev.battery_cfg
            cap, s = b.capacity_kwh, b.capacity_kwh * b.soc_initial_frac
            floor = b.soe_floor_kwh
            soe, real = np.empty(self.n + 1), np.empty(self.n)
            soe[0] = s
            for t in range(self.n):
                a = float(power[t])
                if a > 0:                          # charge, but never past full
                    a = min(a, (cap - s) / (b.eta_c * dt))
                    s += a * b.eta_c * dt
                else:                              # discharge, but never below the reserve
                    a = max(a, -(s - floor) * b.eta_d / dt)
                    s += a / b.eta_d * dt
                s = min(max(s, floor), cap)
                real[t], soe[t + 1] = a, s
            return Column(real, soe, self.private_cost(dev, soe), "blend", mix)
        wh = dev.tank_cfg                          # HVAC never blends: it is not modulating
        C = wh.heat_capacity_kwh_per_k
        temp = np.empty(self.n + 1)
        temp[0] = wh.t_comfort
        real = np.empty(self.n)
        for t in range(self.n):
            q_out = float(_usable_outflow(temp[t], wh, self.fc.hot_water_demand[t], dt))
            duty = min(max(power[t] / wh.power_kw, 0.0), float(_max_duty(temp[t], wh, q_out, dt)))
            real[t] = duty * wh.power_kw
            temp[t + 1] = temp[t] + (real[t] - q_out) / C * dt
        return Column(real, temp, self.private_cost(dev, temp), "blend", mix)

    def mixed_power(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            dev.key: sum(w * col.power for w, col in zip(weights[dev.key], dev.columns))
            for dev in self.devices
        }

    # --------------------------------------------------------------- bounds
    def lagrangian(self, pi: np.ndarray, best: dict[str, Column]) -> float:
        """L(pi) = pi.d + sum_k min_x (f_k + pi.p_k) + sum_t min_meter(...).

        Valid at ANY pi, not only the master's, which is what lets the
        smoothed price produce a bound too. -inf outside the dual box.
        """
        n = self.n
        total = float(pi[: len(self.rows) * n] @ self.rhs_bal)
        for dev in self.devices:
            col = best[dev.key]
            total += col.cost + float(self.price_of(pi, dev) @ col.power)
        boxes = []
        for coefs, cost, upper in self.blocks:
            red = cost - sum((pi[r * n : (r + 1) * n] * coef for r, coef in coefs), np.zeros(n))
            boxes.append((red, upper))
        for red, upper in boxes:
            if np.any((red < -1e-12) & ~np.isfinite(upper)):
                return -np.inf
            total += float(np.sum(np.minimum(0.0, red * np.where(np.isfinite(upper), upper, 0.0))))
        return total

    def lagrangian_lp(self, pi: np.ndarray, best: dict[str, Column]) -> float:
        """L(pi) when batteries live in the master: each battery contributes
        min over its own polytope of (terminal cost + pi . p), a tiny LP."""
        total = self.lagrangian(pi, best)
        if not np.isfinite(total):
            return total
        n, dt = self.n, self.dt
        for key, b in self.lp_batts:
            # vars c[n], e[n], s[n]
            nv = 3 * n
            cc = np.zeros(nv)
            price = self.price_of(pi, key)
            cc[:n] = price
            cc[n : 2 * n] = -price
            cc[3 * n - 1] = -_pinned_price(b)
            A = Triplets((n, nv))
            t = np.arange(n)
            A[t, 2 * n + t] = 1.0
            A[t[1:], 2 * n + t[:-1]] = -1.0
            A[t, t] = -dt * b.eta_c
            A[t, n + t] = dt / b.eta_d
            s0 = b.capacity_kwh * b.soc_initial_frac
            rhs = np.zeros(n)
            rhs[0] = s0
            ub = np.concatenate([np.full(n, b.p_charge_max_kw), np.full(n, b.p_discharge_max_kw),
                                 np.full(n, b.capacity_kwh)])
            lbb = np.zeros(nv)
            lbb[2 * n :] = b.soe_floor_kwh
            if key in self.soe_targets:
                lbb[3 * n - 1] = ub[3 * n - 1] = self.soe_targets[key]
            _, fun, _ = self._lp(cc, A, rhs, lbb, ub)
            total += fun + _pinned_price(b) * s0
        if self.lp_tank is not None:
            nv, nr = 4 * n + 2, 2 * n + 1
            A = Triplets((nr, nv))
            cc, lbt, ubt, rhs = np.zeros(nv), np.zeros(nv), np.full(nv, np.inf), np.zeros(nr)
            self._tank_block(A, cc, lbt, ubt, rhs, 0, 0, meter=False)
            cc[:n] += self.price_of(pi, "water_heater") * self.lp_tank.power_kw   # its bus's price on its consumption
            _, fun, _ = self._lp(cc, A, rhs, lbt, ubt)
            total += fun
        return total

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, powers: dict[str, np.ndarray], trajs: dict[str, np.ndarray]) -> float:
        """Score a plan exactly as coordinate() scores its rounds."""
        flows = site_meter(self.cfg, self.fc, powers)
        net, curtail = flows.net, flows.curtail
        if self.export_ceiling is not None:
            # A hard cap: what is exported past it must be curtailed PV, as the
            # master would; a plan that cannot is not runnable.
            over = np.maximum(-net - self.export_ceiling, 0.0)
            if np.any(over > 1e-6):
                if not (self.cfg.grid.allow_curtailment and np.all(over <= self.fc.solar - curtail + 1e-6)):
                    return np.inf
                net = net + over
        return extended_objective(self.cfg, self.fc, net, trajs) + beyond_main(self.cfg, self.fc, flows)

    def participants_cost(self, plan: dict[str, Column]) -> float:
        """The sum of the participants' private costs (currency) in `plan`
        (device key -> chosen column), as their own models priced them. The
        site's own devices are not included: `evaluate` prices those."""
        return float(sum(plan[d.key].cost for d in self.devices if d.kind == "participant" and d.key in plan))

    def score(self, plan: dict[str, Column]) -> float:
        """The objective (currency) of `plan` (device key -> chosen column):
        the bill and the site's own devices' costs (`evaluate`), plus the
        participants' private costs. Infinite if the plan breaks the export
        ceiling and cannot curtail its way under it."""
        return (self.evaluate({k: c.power for k, c in plan.items()}, {k: c.trajectory for k, c in plan.items()})
                + self.participants_cost(plan))

    def _note_links(self, x: np.ndarray) -> None:
        """Record each node's link flows in master solution `x`: node ->
        (sent, taken), bus-side kW per slot."""
        n = self.n
        self.link_flows = {
            node: (sum((x[b * n : (b + 1) * n] for b in sends), np.zeros(n)),
                   sum((x[b * n : (b + 1) * n] for b in takes), np.zeros(n)))
            for node, (sends, takes) in self.link_blocks.items()}

    def _fix_directions(self) -> bool:
        """Where the last master solve both sent and took on a lossy link in
        one slot - burning energy in conversion losses, which the LP allows
        (theory, "One relaxation") but no converter can do - fix that slot's
        direction to the net flow, for recovery. Returns whether any was fixed."""
        fixed = False
        fix: dict[int, np.ndarray] = dict(getattr(self, "link_fix", {}))
        for node, (sent, taken) in getattr(self, "link_flows", {}).items():
            loop = np.minimum(sent, taken) > 1e-6
            if not np.any(loop):
                continue
            sends, takes = self.link_blocks[node]
            shut_take, shut_send = loop & (sent >= taken), loop & (sent < taken)
            for b in takes:
                fix[b] = fix.get(b, np.zeros(self.n, dtype=bool)) | shut_take
            for b in sends:
                fix[b] = fix.get(b, np.zeros(self.n, dtype=bool)) | shut_send
            fixed = True
        self.link_fix = fix
        return fixed

    def recover(self, integer: str, weights: dict[str, np.ndarray] | None = None,
                bplans: dict[str, Column] | None = None, time_limit: float = 30.0,
                node_limit: int = 150) -> dict[str, Column]:
        """One implementable plan per device from the CURRENT pool.

        milp: choose jointly - binary weights for devices that cannot run a
        blend, continuous for modulating ones, batteries re-optimised in the
        same solve. maxweight: each device's heaviest column in `weights`
        (the relaxed master's), batteries as the master left them.
        """
        def pick(dev: Device, w: np.ndarray) -> Column:
            if dev.modulating and np.sum(w > 1e-6) > 1:
                return self.blend_plan(dev, w)
            runnable = np.array([wi if c.runnable else -1.0 for wi, c in zip(w, dev.columns)])
            k = int(np.argmax(runnable))
            if dev.modulating and dev.columns[k].source == "aggregate":
                # an aggregate carries an averaged trajectory: re-run its power
                # through the physics so the plan is scored as it would run
                one = np.zeros(len(dev.columns)); one[k] = 1.0
                return self.blend_plan(dev, one)
            return dev.columns[k]

        if integer == "milp":
            try:
                _, lam_int, _, _, bplans_int = self.solve_master(integer=True, time_limit=time_limit,
                                                                 node_limit=node_limit)
            except RuntimeError:
                # No integer solution within the time limit (large pools):
                # fall back to the heaviest runnable column per device.
                integer, lam_int = "maxweight", None
        if integer == "milp" and lam_int is not None:
            w_int = self.split_weights(lam_int)
            plan = {d.key: pick(d, w_int[d.key]) for d in self.devices}
            plan.update(bplans_int)
        else:
            if weights is None or bplans is None:
                _, lam, _, _, bplans = self.solve_master()
                weights = self.split_weights(lam)
            plan = {d.key: pick(d, weights[d.key]) for d in self.devices}
            plan.update(bplans)
        return plan

    def parts(self, plan: dict[str, Column]) -> dict:
        """The objective of a plan, term by term - what `evaluate` sums."""
        flows = site_meter(self.cfg, self.fc, {k: c.power for k, c in plan.items()})
        net = flows.net
        trajs = {k: c.trajectory for k, c in plan.items()}
        fc, dt = self.fc, self.dt
        bill = float(np.sum(np.maximum(net, 0) * fc.buy * dt) - np.sum(np.maximum(-net, 0) * fc.sell * dt))
        bill += extra_bills(self.cfg, fc, flows) if self.cfg.connections else 0.0
        comfort = comfort_penalty(self.cfg, trajs.get("water_heater"), trajs.get("hvac"), self.ref)
        breach = grid_penalty(self.cfg, net, fc.buy, fc.sell)
        if self.cfg.has_tree:
            breach += beyond_main(self.cfg, fc, flows) - (extra_bills(self.cfg, fc, flows) if self.cfg.connections else 0.0)
        batt = battery_terminal_penalty(
            self.cfg, {k: v for k, v in trajs.items() if k.startswith("battery")} or None, fc.buy)
        goal = site_battery_extras(self.cfg, trajs)   # SoC goals / gates (EVs)
        edge = 0.0
        if self.cfg.water_heater is not None and "water_heater" in trajs:
            edge += thermal_terminal(self.cfg.water_heater, "water_heater", trajs["water_heater"][-1], self.ref)
        if self.cfg.hvac is not None and "hvac" in trajs:
            edge += thermal_terminal(self.cfg.hvac, "hvac", trajs["hvac"][-1], self.ref)
        part = self.participants_cost(plan)
        return {"bill": bill, "comfort": comfort, "breach": breach, "battery_terminal": batt,
                "thermal_edge": edge, "battery_goal": goal, "participants": part,
                "total": bill + comfort + breach + batt + edge + goal + part,
                "breach_kwh": float(np.sum(breach_energy(self.cfg, net)) + np.sum(flows.breach_kwh)),
                "peak_import": float(net.max()), "peak_export": float(-net.min())}

    def polish(self, plan: dict[str, Column], sweeps: int = 3) -> tuple[dict[str, Column], float]:
        """Gauss-Seidel load-aware best response from the recovered plan.

        Accept a device's new plan only when the TRUE objective falls, so this
        is monotone - it cannot undo what the master found.
        """
        plan = dict(plan)
        cur = self.score(plan)
        for _ in range(sweeps):
            improved = False
            for dev in self.devices:
                others = {k: c.power for k, c in plan.items() if k != dev.key}
                cand = self.load_aware_oracle(dev, others, self.fc.buy)
                trial = dict(plan)
                trial[dev.key] = cand
                val = self.score(trial)
                if val < cur - 1e-9:
                    plan, cur, improved = trial, val, True
                    dev.add(cand)
            if not improved:
                break
        return plan, cur

    def _dive(self, iters: int, heuristic_columns: bool, history: list, t0: float) -> None:
        """Fix every on/off device to the plan the integer master picks, then
        price the others for up to `iters` more iterations, so their plans answer
        the on/off plans that will run rather than a fractional mix of them. A
        participant says whether it is on/off (`onoff`); a device of the site's
        own is when it cannot run a blend. The bound is unaffected: it is
        re-stated from every plan ever proposed (`_seen_min`).

        `iters`: the most pricing rounds after fixing. `heuristic_columns`: also
        ask each free device its best response to the rest (as `run` does).
        `history`: `run`'s per-iteration log, appended to ("dive 1", ...). `t0`:
        `run`'s start time (perf_counter), for the log's elapsed ms. Changes the
        device pools in place; returns nothing. Does nothing if the integer
        master fails, or if every device or none is on/off."""
        try:
            _, lam_int, _, _, _ = self.solve_master(integer=True)
        except RuntimeError:
            return
        w_int = self.split_weights(lam_int)
        onoff = lambda d: (bool(getattr(d.cfg, "onoff", not d.modulating))       # noqa: E731
                           if d.kind == "participant" else not d.modulating)
        fixed = {d.key for d in self.devices if onoff(d)}
        if not fixed or len(fixed) == len(self.devices):
            return
        for d in self.devices:
            if d.key in fixed:
                runnable = np.array([wi if c.runnable else -1.0 for wi, c in zip(w_int[d.key], d.columns)])
                d.columns = [d.columns[int(np.argmax(runnable))]]
        dt = self.dt
        for k in range(iters):
            rmp, lam, pi, _, bplans = self.solve_master()
            weights = self.split_weights(lam)
            mix = self.mixed_power(weights)
            powers = {**mix, **{k: c.power for k, c in bplans.items()}}
            added = 0
            for d in self.devices:
                if d.key not in fixed:
                    added += d.add(self.price_oracle(d, self.price_of(pi, d) / dt))
                    if heuristic_columns:
                        others = {k: v for k, v in powers.items() if k != d.key}
                        added += d.add(self.load_aware_oracle(d, others, self.price_of(pi, d) / dt))
                        added += self._add_contained(d, self.price_of(pi, d) / dt)
            history.append({"iter": f"dive {k + 1}", "rmp": rmp, "lb": history[-1]["lb"] if history else -np.inf,
                            "gap": np.nan, "columns": sum(len(d.columns) for d in self.devices), "added": added,
                            "ms": (time.perf_counter() - t0) * 1000})
            if added == 0:
                break

    def extra_price_points(self, pi: np.ndarray, center: np.ndarray | None,
                           it: int) -> list[np.ndarray]:
        """Further meter prices to ask every device at, this iteration.

        None by default. Any price is a valid question - the answer is a
        feasible plan, and its Lagrangian value is a valid bound - so a
        subclass can add, e.g., momentum points (bench/dw_accel.py).
        """
        return []

    # ------------------------------------------------------------------ run
    def run(
        self,
        max_iter: int = 40,
        gap_tol: float = 1e-3,
        smoothing: float | str = "auto",  # "auto": 0.8 if a plain battery bids plans, else 0.5
        heuristic_columns: bool = True,
        integer: str = "milp",        # milp | maxweight
        polish: bool = True,
        verbose: bool = False,
        record: bool = False,         # keep a per-iteration snapshot (for the GUI)
        anytime: bool = False,        # also recover a plan at every iteration
        pool: str = "auto",           # active: what the master uses + the newest | full: keep all | auto (below)
        seed_admm: bool = False,      # start the pool from every ADMM round's plans
        response: str = "proposals",  # proposals | both | sensitivity (see src/home_energy_optimizer/dw/sensitivity.py)
        sens_keep: int = 24,          # sensitivity variants kept per device, per direction
        stall: int = 0,               # stop once the master's value has not fallen for this many
                                      # iterations (0: never). For participants whose answers give a
                                      # weak bound, where the gap test alone runs to max_iter.
        dive: int = 0,                # then fix each on/off device to the plan the integer master
                                      # picks and re-price the modulating ones this many more
                                      # iterations (0: no dive). Their pools were built against a
                                      # fractional mix of the on/off plans, which cannot run.
        progress: Callable[..., None] | None = None,
                                      # progress(it, max_iter) as each iteration starts, then
                                      # progress(it, max_iter, blend, bound, best runnable) as it ends
                                      # (None where not known yet); (-1, max_iter) when finishing
    ) -> DWResult:
        t0 = time.perf_counter()
        # "auto": a plain battery that bids plans (batteries off the LP)
        # converges slowly - its optimum lies inside its polytope, which the
        # master rebuilds one plan at a time. Keeping every plan and pricing
        # nearer the best-bound prices closed the gap 0.12 -> 0.015 and
        # 0.25 -> 0.029 in 45 iterations (bench/dw_accel.py). Otherwise the
        # bounded pool and the lighter smoothing are as good and leaner.
        bidding_battery = any(d.kind == "battery" and is_plain_battery(d.battery_cfg) for d in self.devices)
        if pool == "auto":
            pool = "full" if bidding_battery else "active"
        smooth = (0.8 if bidding_battery else 0.5) if smoothing == "auto" else float(smoothing)
        self.pool_used, self.smoothing_used = pool, smooth
        self.seed()
        admm_plan = None
        if seed_admm:
            admm_plan = self.seed_from_admm()
        dt = self.dt
        best_lb, center = -np.inf, None
        priced: list[np.ndarray] = []
        history: list[dict] = []
        it = 0
        stop_reason = "iteration cap"
        best_plan_val = np.inf
        best_rmp, flat = np.inf, 0
        incumbent = None            # best runnable plan seen at any iteration
        if admm_plan is not None:
            # ADMM's own plan is runnable, so DW can never return worse.
            incumbent, best_plan_val = admm_plan, self.parts(admm_plan)["total"]

        for it in range(1, max_iter + 1):
            if progress is not None:
                progress(it, max_iter)
            rmp, lam, pi, _sigma, bplans = self.solve_master()
            weights = self.split_weights(lam)
            mix = self.mixed_power(weights)
            snap = self.snapshot(weights, pi, bplans) if record else None
            if pool == "active":
                # Bounded memory: a column the master gives zero weight is not
                # needed to represent the current solution, only (maybe) a
                # later one. Keep the support (at most rows-many, by LP basis
                # theory) plus last iteration's proposals, drop the rest.
                for dev in self.devices:
                    w = weights[dev.key]
                    keep = [c for c, wi in zip(dev.columns, w) if wi > 1e-9 or c.born >= it - 1]
                    weights[dev.key] = np.array([wi for c, wi in zip(dev.columns, w) if wi > 1e-9 or c.born >= it - 1])
                    dev.columns = keep
                self.peak_pool = max(getattr(self, "peak_pool", 0), sum(len(d.columns) for d in self.devices))
            for dev in self.devices:
                dev.stamp = it

            # "If we stopped now": recover from the pool as it stands. Costs a
            # small MILP per iteration, so only when asked (the GUI asks).
            anytime_rec = {}
            if anytime:
                # quick: a few nodes only - the incumbent keeps the best anyway
                snap_plan = self.recover(integer, weights, bplans, time_limit=0.5, node_limit=6)
                pp = self.parts(snap_plan)
                if pp["total"] < best_plan_val:
                    best_plan_val, incumbent = pp["total"], dict(snap_plan)
                anytime_rec = {"plan_value": pp["total"], "plan_parts": pp, "best_plan": best_plan_val}

            if response == "sensitivity":
                # Bounded memory: fold everything the master is using into one
                # aggregate proposal (a point of the device's convex hull, with
                # its cost), keep the heaviest runnable plan for recovery, and
                # forget the rest. The newest proposal and its sensitivity
                # variants are added below. Memory per device stays ~2 + 2k.
                for dev in self.devices:
                    w = weights[dev.key]
                    agg = Column(sum(wi * c.power for wi, c in zip(w, dev.columns)),
                                 sum(wi * c.trajectory for wi, c in zip(w, dev.columns)),
                                 float(sum(wi * c.cost for wi, c in zip(w, dev.columns))),
                                 "aggregate",
                                 # runnable only if it IS one runnable plan (an aggregate of
                                 # an aggregate is still a blend), or the device modulates
                                 runnable=dev.modulating or (
                                     int(np.sum(w > 1e-6)) <= 1
                                     and all(c.runnable for wi, c in zip(w, dev.columns) if wi > 1e-6)))
                    agg.born = it - 1
                    keep = [agg]
                    run_w = [(wi, c) for wi, c in zip(w, dev.columns) if c.runnable]
                    if run_w:
                        keep.append(max(run_w, key=lambda z: z[0])[1])
                    dev.columns = keep

            # Price at the master's dual, and (Wentges) at a point pulled toward
            # the best-bound centre. Both give valid columns AND valid bounds.
            points = [pi]
            if np.max(np.abs(self._shown(pi) - pi), initial=0.0) > 1e-9:
                points.append(self._shown(pi))
            if center is not None and smooth > 0:
                points.append(smooth * center + (1 - smooth) * pi)
            points.extend(self.extra_price_points(pi, center, it))

            added = 0
            flex: dict[str, dict] = {}
            for pk, p in enumerate(points):
                best = {}
                exact = True
                for dev in self.devices:
                    q = self.price_of(p, dev)
                    col, sol = self.price_solve(dev, q / dt)
                    if pk == 0 and response in ("both", "sensitivity") and sol is not None:
                        added += self._add_variants(dev, sol, col, q / dt, sens_keep, flex if record else None)
                    # The DP is an exact pricing oracle only up to its grid:
                    # a pool column can undercut it (measured: HVAC, 0.0026 at
                    # a binding import limit). Take the min over both, so the
                    # bound never exceeds the master; record the shortfall,
                    # because it is how far the certificate itself can be off.
                    pool_best = min(dev.columns, key=lambda c: c.cost + q @ c.power)
                    shortfall = (col.cost + q @ col.power) - (pool_best.cost + q @ pool_best.power)
                    self.pricing_error = max(getattr(self, "pricing_error", 0.0), float(shortfall))
                    best[dev.key] = col if shortfall <= 0 else pool_best
                    exact = exact and col.exact

                    added += dev.add(col)
                if not exact:
                    # a participant's solve failed here: its stand-in is a plan,
                    # not its optimum, so this price proves no bound
                    continue
                lb = self.lagrangian_lp(p, best) if (self.lp_batts or self.lp_tank is not None) else self.lagrangian(p, best)
                priced.append(p.copy())
                self._priced.append(p.copy())
                self._seen_min.append({d.key: min(c.cost + float(self.price_of(p, d) @ c.power) for c in d.columns)
                                       for d in self.devices})
                if lb > best_lb:
                    best_lb, center = lb, p.copy()

            if heuristic_columns:
                powers = {**mix, **{k: c.power for k, c in bplans.items()}}
                for dev in self.devices:
                    others = {k: v for k, v in powers.items() if k != dev.key}
                    added += dev.add(self.load_aware_oracle(dev, others, self.price_of(pi, dev) / dt))
                    added += self._add_contained(dev, self.price_of(pi, dev) / dt)

            gap = rmp - best_lb
            if snap is not None and flex:
                snap["flex"] = flex
            history.append({"iter": it, "rmp": rmp, "lb": best_lb, "gap": gap,
                            "columns": sum(len(d.columns) for d in self.devices),
                            "added": added,
                            "ms": (time.perf_counter() - t0) * 1000,
                            **anytime_rec,
                            **({"snapshot": snap} if record else {})})
            if verbose:
                print(f"  it {it:2d}  rmp {rmp:9.4f}  lb {best_lb:9.4f}  gap {gap:8.4f}  cols {history[-1]['columns']}")
            if progress is not None:
                fin = lambda v: float(v) if np.isfinite(v) else None  # noqa: E731
                progress(it, max_iter, fin(rmp), fin(best_lb), fin(best_plan_val))
            if gap <= gap_tol * max(1.0, abs(rmp)):
                stop_reason = "converged"
                break
            if rmp < best_rmp - 1e-6 * max(1.0, abs(rmp)):
                best_rmp, flat = rmp, 0
            else:
                flat += 1
            if stall and flat >= stall:
                stop_reason = "stalled"
                break
            if added == 0:
                stop_reason = "no new proposals"
                break

        if dive:
            self._dive(dive, heuristic_columns, history, t0)
        if progress is not None:
            progress(-1, max_iter)
        relaxed, lam, pi, _, bplans = self.solve_master()
        shown = self._shown(pi)          # before recovery solves the master again

        # Re-state the bound against the FINAL pool. Columns added after a
        # point was priced (heuristic ones especially) can undercut the DP's
        # answer there; with the min taken over the whole pool, L(pi) <= the
        # final master value by weak duality, so the reported ordering holds.
        # What remains unguarded is a plan NO oracle proposed that beats the
        # DP - the DP's own discretisation error (see pricing_error).
        best_lb = -np.inf
        for i, p in enumerate(priced):
            best = {}
            for d in self.devices:
                # a stand-in column carrying the minimum value at this price
                v = min(self._seen_min[i][d.key], min(c.cost + float(self.price_of(p, d) @ c.power) for c in d.columns))
                best[d.key] = Column(np.zeros(self.n), np.zeros(self.n + 1), v, "min")
            lb = self.lagrangian_lp(p, best) if (self.lp_batts or self.lp_tank is not None) else self.lagrangian(p, best)
            best_lb = max(best_lb, lb)
        weights = self.split_weights(lam)
        frac = sum(int(np.sum(w > 1e-6) > 1) for w in weights.values())
        relaxed_view = self.snapshot(weights, pi, bplans) if record else None

        # ---- primal recovery -------------------------------------------------
        # A modulating device keeps its blend (continuous weights in the MILP
        # too); everything else is rounded to one plan.
        for dev in self.devices:
            dev.stamp = -1          # anything added from here on is polish's
        plan = self.recover(integer, weights, bplans)
        # A lossy link used both ways in a slot is no plan a converter can run:
        # fix those directions and recover again (the bound is unaffected).
        for _ in range(5):
            if not self._fix_directions():
                break
            _, lam_f, _, _, bplans_f = self.solve_master()
            plan = self.recover(integer, self.split_weights(lam_f), bplans_f)
        self.link_fix = {}
        plan = self._repair(plan, integer)
        upper = self.score(plan)
        if polish:
            plan, upper = self.polish(plan)
        # Keep the incumbent. Recovery from the final pool is a heuristic
        # (exactly so for "maxweight"), and can return a worse plan than one
        # already seen: measured 1.168 returned against 0.637 at iteration 2.
        incumbent_used = False
        if incumbent is not None and best_plan_val < upper - 1e-9:
            plan, upper, incumbent_used = incumbent, best_plan_val, True

        return DWResult(
            upper=upper, lower=best_lb, relaxed=relaxed, iterations=it, plan=plan,
            weights=weights, prices=pi[: self.n] / dt, history=history,
            ms=(time.perf_counter() - t0) * 1000,
            n_columns=sum(len(d.columns) for d in self.devices),
            fractional_devices=frac,
            pricing_error=getattr(self, "pricing_error", 0.0),
            relaxed_view=relaxed_view,
            stop_reason=stop_reason,
            plan_parts=self.parts(plan),
            incumbent_used=incumbent_used,
            admm_value=getattr(self, "admm_value", None),
            columns={d.key: d.columns for d in self.devices},
            local_prices=self.local_prices(shown),
            limit_prices=self.limit_prices(shown),
        )


def _have_scipy() -> bool:
    """Whether scipy (HiGHS) can be imported: the recovery MILP's exact solver."""
    import importlib.util
    return importlib.util.find_spec("scipy") is not None


def site_battery_extras(cfg: SiteConfig, trajs: dict[str, np.ndarray]) -> float:
    """battery_extras over every battery with a trajectory; 0 for plain ones."""
    total = 0.0
    for i, b in enumerate(cfg.battery_list):
        traj = trajs.get(SiteConfig.battery_key(i))
        if traj is not None:
            total += battery_extras(b, traj, cfg.horizon, cfg.soc_gates if i == 0 else ())
    return total


def extended_objective(cfg: SiteConfig, fc: Forecasts, net: np.ndarray,
                       trajs: dict[str, np.ndarray]) -> float:
    """total_objective + the thermal horizon-edge terms. The basis every method
    in bench/run_dw.py is compared on (net must already be curtailed)."""
    ref = float(np.mean(fc.buy))
    soe = {k: v for k, v in trajs.items() if k.startswith("battery")} or None
    val = total_objective(cfg, net, fc, trajs.get("water_heater"), trajs.get("hvac"), soe)
    val += site_battery_extras(cfg, trajs)
    if cfg.water_heater is not None and "water_heater" in trajs:
        val += thermal_terminal(cfg.water_heater, "water_heater", trajs["water_heater"][-1], ref)
    if cfg.hvac is not None and "hvac" in trajs:
        val += thermal_terminal(cfg.hvac, "hvac", trajs["hvac"][-1], ref)
    return val


def baseline_objective(cfg: SiteConfig, fc: Forecasts) -> float:
    """Thermostats, idle batteries, same curtailment - on extended_objective."""
    h = cfg.horizon
    trajs: dict[str, np.ndarray] = {}
    powers: dict[str, np.ndarray] = {}
    for i, b in enumerate(cfg.battery_list):
        trajs[SiteConfig.battery_key(i)] = np.full(h.steps + 1, b.capacity_kwh * b.soc_initial_frac)
    if cfg.water_heater is not None:
        trajs["water_heater"], powers["water_heater"] = baseline_water_heater(cfg.water_heater, h, fc.hot_water_demand)
    if cfg.hvac is not None:
        trajs["hvac"], powers["hvac"] = baseline_hvac(cfg.hvac, h, fc.outdoor_temp)
    flows = site_meter(cfg, fc, powers)
    return extended_objective(cfg, fc, flows.net, trajs) + beyond_main(cfg, fc, flows)


def dw_coordinate(cfg: SiteConfig, fc: Forecasts, battery_in_master: bool = True,
                  tank_in_master: bool = False, solver: str = "numpy",
                  ev_duty_cycle: bool = False, **kw: Unpack[RunOptions]) -> DWResult:
    return DWCoordinator(cfg, fc, battery_in_master=battery_in_master,
                         tank_in_master=tank_in_master, solver=solver,
                         ev_duty_cycle=ev_duty_cycle).run(**kw)
