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
solver="highs" to use scipy's HiGHS instead, e.g. to cross-check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from home_energy_optimizer.dw.lpsolver import Triplets, choose_one
from home_energy_optimizer.dw.lpsolver import linprog as np_linprog

from home_energy_optimizer.coordinate import (
    apply_curtailment,
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
from home_energy_optimizer.types import Forecasts, SiteConfig


def thermal_terminal(cfg, kind: str, t_end: float, ref: float) -> float:
    """The horizon-edge term the thermal DPs put in V[n] (linear comfort mode).

    `total_objective` omits it, but every DP - and the joint-DP and MILP
    references - optimise WITH it. Leaving it out of a column's cost would let
    the master buy a cheaper day by handing tomorrow a cold tank, and would
    make the pricing DP optimise a different objective from the master, which
    voids the lower bound.
    """
    if kind == "water_heater":
        return cfg.heat_capacity_kwh_per_k * ref * max(0.0, cfg.t_comfort - t_end)
    return float(hvac_discomfort(cfg, np.array([t_end]), ref, 1.0)[0])


def is_plain_battery(b, gates=()) -> bool:
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


def battery_extras(b, traj: np.ndarray, horizon, gates=()) -> float:
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

    def __new__(cls, values, ref: float):
        obj = np.asarray(values, dtype=float).view(cls)
        obj._ref = ref
        return obj

    def __array_finalize__(self, obj):
        self._ref = getattr(obj, "_ref", None)

    def mean(self, *a, **k):  # noqa: D401 - numpy protocol
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


@dataclass
class Device:
    key: str
    kind: str                # battery | water_heater | hvac
    cfg: object
    columns: list[Column] = field(default_factory=list)
    stamp: int = 0           # iteration number given to columns added now
    on_add: object = None    # coordinator hook: keeps the bound's per-price minima current
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
            return self.cfg.n_duty_levels > 2
        if self.kind == "battery":
            return self.cfg.charge_deadband_kw <= 0.0 or self.duty_cycle_ok
        return False

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


class DWCoordinator:
    def __init__(self, cfg: SiteConfig, fc: Forecasts, battery_in_master: bool = True,
                 tank_in_master: bool = False, solver: str = "numpy", ev_duty_cycle: bool = False):
        cfg.validate()
        fc.validate(cfg.horizon)
        self.cfg, self.fc = cfg, fc
        if solver not in ("numpy", "highs"):
            raise ValueError("solver must be 'numpy' or 'highs'")
        self.solver = solver
        self.n, self.dt = cfg.horizon.steps, cfg.horizon.dt
        self.d = fc.net_fixed_demand.copy()
        self.ref = float(np.mean(fc.buy))
        if np.any(fc.sell > fc.buy + 1e-12):
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
        self.lp_batts: list[tuple[str, object]] = []
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
        self.lp_tank = None
        if cfg.water_heater is not None:
            if tank_in_master:
                if cfg.water_heater.comfort_mode != "linear":
                    raise ValueError("tank_in_master needs linear comfort pricing")
                self.lp_tank = cfg.water_heater
            else:
                self.devices.append(Device("water_heater", "water_heater", cfg.water_heater))
        if cfg.hvac is not None:
            self.devices.append(Device("hvac", "hvac", cfg.hvac))

        self._build_meter_block()
        # Running minimum of (cost + price . power) per device at every price
        # point priced so far, over EVERY plan ever proposed - including ones
        # the pool later forgets. That is what keeps the bound valid when the
        # pool is pruned or aggregated.
        self._priced: list[np.ndarray] = []
        self._seen_min: list[dict[str, float]] = []
        for dev in self.devices:
            dev.on_add = self._note

    # ------------------------------------------------------------------ costs
    def private_cost(self, dev: Device, traj: np.ndarray) -> float:
        """Must match the non-bill terms of `total_objective` exactly."""
        if dev.kind == "battery":
            gates = self.cfg.soc_gates if dev.key == "battery" else ()
            return (float(traj[0] - traj[-1]) * float(dev.cfg.terminal_price)
                    + battery_extras(dev.cfg, traj, self.cfg.horizon, gates))
        if dev.kind == "water_heater":
            return (float(np.sum(wh_discomfort(dev.cfg, traj[1:], self.ref, self.dt)))
                    + thermal_terminal(dev.cfg, "water_heater", traj[-1], self.ref))
        return (float(np.sum(hvac_discomfort(dev.cfg, traj[1:], self.ref, self.dt)))
                + thermal_terminal(dev.cfg, "hvac", traj[-1], self.ref))

    # --------------------------------------------------------------- oracles
    def _solve(self, dev: Device, buy, sell, dp_load=None):
        h, fc = self.cfg.horizon, self.fc
        if dev.kind == "battery":
            gates = self.cfg.soc_gates if dev.key == "battery" else ()
            return solve_battery(dev.cfg, h, buy, sell, dp_load=dp_load, soc_gates=gates)
        buy = _PriceVector(buy, self.ref)
        sell = _PriceVector(sell, self.ref)
        if dev.kind == "water_heater":
            return solve_water_heater(dev.cfg, h, buy, sell, fc.hot_water_demand, dp_load=dp_load)
        return solve_hvac(dev.cfg, h, buy, sell, fc.outdoor_temp, dp_load=dp_load)

    def _note(self, dev: Device, col: Column) -> None:
        for p, mins in zip(self._priced, self._seen_min):
            v = col.cost + float(p @ col.power)
            if v < mins.get(dev.key, np.inf):
                mins[dev.key] = v

    def price_solve(self, dev: Device, price_kwh: np.ndarray):
        """The pricing DP, returning its solution too (value function and
        policy), which is what the sensitivity variants replay."""
        sol = self._solve(dev, price_kwh, price_kwh, dp_load=None)
        return Column(sol.power.copy(), sol.trajectory.copy(),
                      self.private_cost(dev, sol.trajectory), "price"), sol

    def price_oracle(self, dev: Device, price_kwh: np.ndarray) -> Column:
        """The Dantzig-Wolfe pricing problem: min private cost + price . power.

        buy == sell and no dp_load makes the DP's bill term exactly linear, so
        its argmax is the column of most negative reduced cost (up to the DP's
        own discretisation - the same caveat every lambda in this package has).
        """
        sol = self._solve(dev, price_kwh, price_kwh, dp_load=None)
        return Column(sol.power.copy(), sol.trajectory.copy(),
                      self.private_cost(dev, sol.trajectory), "price")

    def load_aware_oracle(self, dev: Device, others: np.ndarray, price_kwh: np.ndarray) -> Column:
        """A HEURISTIC column: today's load-aware DP against the others' mix.

        Any feasible plan is a legal column, so the master can absorb the
        ADMM-style best response too. The kink makes this column good at
        self-consumption plans that a linear price reaches only by mixing
        many extreme points. Limits reach it the way mu does in coordinate():
        as a premium where the master price exceeds the tariff.
        """
        buy = np.maximum(self.fc.buy, price_kwh)
        sell = np.minimum(self.fc.sell, price_kwh)
        sol = self._solve(dev, buy, sell, dp_load=self.d + others)
        return Column(sol.power.copy(), sol.trajectory.copy(),
                      self.private_cost(dev, sol.trajectory), "load_aware")

    def _add_variants(self, dev, sol, col, price_kwh, keep: int, flex_out) -> int:
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
                mc = dev.cfg.min_charge_kw
                if mc is None or not np.any(np.asarray(mc) > 0):
                    s0 = dev.cfg.capacity_kwh * dev.cfg.soc_initial_frac
                    traj = np.full(self.n + 1, s0)
                    dev.add(Column(np.zeros(self.n), traj, self.private_cost(dev, traj), "idle"))
            elif dev.kind == "water_heater":
                temp, pw = baseline_water_heater(dev.cfg, h, fc.hot_water_demand)
                dev.add(Column(pw, temp, self.private_cost(dev, temp), "thermostat"))
            else:
                temp, pw = baseline_hvac(dev.cfg, h, fc.outdoor_temp)
                dev.add(Column(pw, temp, self.private_cost(dev, temp), "thermostat"))
        # The two corners of the dual box: "everything imports" and
        # "everything exports". Cheap and spans the extremes.
        for p in (fc.buy, np.maximum(fc.sell, 0.0)):
            for dev in self.devices:
                dev.add(self.price_oracle(dev, p))

    # ---------------------------------------------------------------- master
    def _build_meter_block(self) -> None:
        """Per-slot meter variables, in the order the balance row uses them.

        z = d + sum(p) + curtail, split into import within/over the limit and
        export within/over the limit. Breach is priced exactly as
        grid_penalty prices it, so the master minimises total_objective.
        """
        g, fc, dt = self.cfg.grid, self.fc, self.dt
        # breach: a constant price per kWh beyond a limit, as grid_penalty
        B = breach_price(g, fc.buy, fc.sell) if g.active else 0.0
        # No meter flow can exceed everything the house could draw or push at
        # once. Using that instead of +inf changes no solution, but it matters
        # when export pays exactly the import price: "import one more kWh and
        # export it again" then costs nothing, the optimal set is unbounded,
        # and an interior-point method drifts along it until it stalls.
        cfg = self.cfg
        reach = (np.abs(fc.load) + np.abs(fc.solar)
                 + sum(max(b.p_charge_max_kw, b.p_discharge_max_kw) for b in cfg.battery_list)
                 + (cfg.water_heater.power_kw if cfg.water_heater is not None else 0.0)
                 + (cfg.hvac.power_kw if cfg.hvac is not None else 0.0))
        inf = 2.0 * reach + 1.0
        imp_cap = np.minimum(g.max_import_kw, inf) if g.max_import_kw is not None else inf
        exp_cap = np.minimum(g.max_export_kw, inf) if g.max_export_kw is not None else inf
        curtail_ok = g is not None and g.allow_curtailment

        blocks = []  # (coef in balance row, cost per slot, upper bound per slot)
        full = lambda v: np.broadcast_to(v, (self.n,)).astype(float)      # noqa: E731
        blocks.append((+1.0, fc.buy * dt, full(imp_cap)))                       # import
        if g.max_import_kw is not None:
            blocks.append((+1.0, (fc.buy + B) * dt, full(inf)))                 # over import cap
        blocks.append((-1.0, -fc.sell * dt, full(exp_cap)))                     # export
        if g.max_export_kw is not None:
            blocks.append((-1.0, (B - fc.sell) * dt, full(inf)))                # over export cap
        if curtail_ok:
            blocks.append((-1.0, np.zeros(self.n), fc.solar.copy()))            # curtail
        self.meter_blocks = blocks

    # ---------------------------------------------------------------- solvers
    def _lp(self, c, A: Triplets, b, lb, ub, border: int = 0):
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

    def _choose(self, c, A: Triplets, b, lb, ub, groups, time_limit: float, node_limit: int = 150,
                border: int = 0):
        """The recovery MILP: one plan per group. -> (x, fun)."""
        if self.solver == "highs":
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

    def _tank_block(self, A, c, lb, ub, rhs, j0: int, row0: int, meter: bool = True):
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
        C, P = cfg.heat_capacity_kwh_per_k, cfg.power_kw
        price_k = cfg.discomfort_price_per_kelvin_hour(self.ref)
        r = np.empty(n); tinf = np.empty(n)
        for t in range(n):
            rate, ti = _relaxation(cfg, float(self.fc.hot_water_demand[t]))
            r[t], tinf[t] = min(rate, 1.0 / dt), ti
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
            A[t, D + t] = -P                     # meter balance: minus consumption
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

        def decode(x):
            traj = np.concatenate([[cfg.t_comfort], x[T:T + n]])
            return Column(P * x[D:D + n], traj, self.private_cost(Device("water_heater", "water_heater", cfg), traj), "lp")
        return 4 * n + 2, 2 * n + 1, decode

    def solve_master(self, integer: bool = False, time_limit: float = 30.0, node_limit: int = 150):
        """Returns (value, column weights, meter duals, convexity duals, battery plans)."""
        n, devs, dt = self.n, self.devices, self.dt
        n_meter = len(self.meter_blocks) * n
        n_cols = sum(len(d.columns) for d in devs)
        nb = len(self.lp_batts)
        # per LP battery: charge c[n], discharge e[n], soe s[1..n]
        tank_nv = 4 * n + 2 if self.lp_tank is not None else 0
        tank_nr = 2 * n + 1 if self.lp_tank is not None else 0
        nv = n_meter + n_cols + 3 * n * nb + tank_nv
        n_rows = n + len(devs) + n * nb + tank_nr

        c = np.zeros(nv)
        ub = np.full(nv, np.inf)
        lb = np.zeros(nv)
        # Coordinate triplets: a 10-battery, 48 h site is ~2k rows x ~7k
        # columns, which dense would put at ~100 MB.
        A = Triplets((n_rows, nv))
        rhs = np.zeros(n_rows)
        rhs[:n] = self.d
        rhs[n : n + len(devs)] = 1.0
        for b, (coef, cost, upper) in enumerate(self.meter_blocks):
            c[b * n : (b + 1) * n] = cost
            ub[b * n : (b + 1) * n] = upper
            A[np.arange(n), np.arange(b * n, (b + 1) * n)] = coef
        j = n_meter
        for di, dev in enumerate(devs):
            for col in dev.columns:
                c[j] = col.cost
                A[:n, j] = -col.power.reshape(-1, 1)
                A[n + di, j] = 1.0
                # An aggregate an on/off device cannot run is fine for the LP
                # (it is a point of the convex hull) but not for recovery.
                ub[j] = 1.0 if (not integer or col.runnable or dev.modulating) else 0.0
                j += 1
        base_row = n + len(devs)
        batt_cols = []
        for bi, (key, b) in enumerate(self.lp_batts):
            C0, E0, S0 = j, j + n, j + 2 * n
            j += 3 * n
            batt_cols.append((key, b, C0, E0, S0))
            ub[C0 : C0 + n] = b.p_charge_max_kw
            ub[E0 : E0 + n] = b.p_discharge_max_kw
            ub[S0 : S0 + n] = b.capacity_kwh
            lb[S0 : S0 + n] = b.soe_floor_kwh
            t = np.arange(n)
            A[t, C0 + t] = -1.0          # meter balance: minus consumption
            A[t, E0 + t] = +1.0
            r = base_row + bi * n + t    # s[t+1] - s[t] - dt*eta_c*c + dt*e/eta_d = 0
            A[r, S0 + t] = 1.0
            A[r[1:], S0 + t[:-1]] = -1.0
            A[r, C0 + t] = -dt * b.eta_c
            A[r, E0 + t] = dt / b.eta_d
            s0 = b.capacity_kwh * b.soc_initial_frac
            rhs[r[0]] = s0
            # terminal: bill the net depletion at the terminal price
            c[S0 + n - 1] = -float(b.terminal_price)

        tank_decode = None
        if self.lp_tank is not None:
            _, _, tank_decode = self._tank_block(A, c, lb, ub, rhs, j, base_row + n * nb)
        const = sum(float(b.terminal_price) * b.capacity_kwh * b.soc_initial_frac
                    for _, b in self.lp_batts)
        lam_sl = slice(n_meter, n_meter + n_cols)

        def batt_plans(x):
            out = {}
            for key, b, C0, E0, S0 in batt_cols:
                s0 = b.capacity_kwh * b.soc_initial_frac
                traj = np.concatenate([[s0], x[S0 : S0 + n]])
                out[key] = Column(x[C0 : C0 + n] - x[E0 : E0 + n], traj,
                                  (s0 - traj[-1]) * float(b.terminal_price), "lp")
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
            return fun + const, x[lam_sl], None, None, batt_plans(x)

        x, fun, duals = self._lp(c, A, rhs, lb, ub, border=base_row)
        pi = duals[:n]           # d(cost)/d(d_t): currency per kW-slot
        sigma = duals[n : n + len(devs)]
        self.batt_lambda = {     # SoE-balance duals: the exact LP costate
            key: duals[base_row + bi * n : base_row + (bi + 1) * n]
            for bi, (key, *_rest) in enumerate(batt_cols)
        }
        return fun + const, x[lam_sl], pi, sigma, batt_plans(x)

    def split_weights(self, lam: np.ndarray) -> dict[str, np.ndarray]:
        out, j = {}, 0
        for dev in self.devices:
            out[dev.key] = lam[j : j + len(dev.columns)]
            j += len(dev.columns)
        return out

    def snapshot(self, weights, pi, bplans) -> dict:
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
        return {"price": pi / self.dt, "powers": powers, "trajectories": trajs,
                "active": active, "lambda": lam}

    def blend_plan(self, dev: Device, w: np.ndarray) -> Column:
        """Execute a weighted mix of a modulating device's plans.

        The mixed power is re-run through the device's physics, so the
        trajectory and cost are what it actually produces - exact even where a
        mix would carry a tank past its cut-out or a battery past full.
        """
        cfg, dt = dev.cfg, self.dt
        power = sum(wi * c.power for wi, c in zip(w, dev.columns))
        mix = [(float(wi), c.source, c.born) for wi, c in zip(w, dev.columns) if wi > 1e-6]
        if dev.kind == "battery":
            cap, s = cfg.capacity_kwh, cfg.capacity_kwh * cfg.soc_initial_frac
            floor = cfg.soe_floor_kwh
            soe, real = np.empty(self.n + 1), np.empty(self.n)
            soe[0] = s
            for t in range(self.n):
                a = float(power[t])
                if a > 0:                          # charge, but never past full
                    a = min(a, (cap - s) / (cfg.eta_c * dt))
                    s += a * cfg.eta_c * dt
                else:                              # discharge, but never below the reserve
                    a = max(a, -(s - floor) * cfg.eta_d / dt)
                    s += a / cfg.eta_d * dt
                s = min(max(s, floor), cap)
                real[t], soe[t + 1] = a, s
            return Column(real, soe, self.private_cost(dev, soe), "blend", mix)
        C = cfg.heat_capacity_kwh_per_k
        temp = np.empty(self.n + 1)
        temp[0] = cfg.t_comfort
        real = np.empty(self.n)
        for t in range(self.n):
            q_out = float(_usable_outflow(temp[t], cfg, self.fc.hot_water_demand[t], dt))
            duty = min(max(power[t] / cfg.power_kw, 0.0), float(_max_duty(temp[t], cfg, q_out, dt)))
            real[t] = duty * cfg.power_kw
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
        total = float(pi @ self.d)
        for dev in self.devices:
            col = best[dev.key]
            total += col.cost + float(pi @ col.power)
        for coef, cost, upper in self.meter_blocks:
            red = cost - pi * coef
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
            cc[:n] = pi
            cc[n : 2 * n] = -pi
            cc[3 * n - 1] = -float(b.terminal_price)
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
            _, fun, _ = self._lp(cc, A, rhs, lbb, ub)
            total += fun + float(b.terminal_price) * s0
        if self.lp_tank is not None:
            nv, nr = 4 * n + 2, 2 * n + 1
            A = Triplets((nr, nv))
            cc, lbt, ubt, rhs = np.zeros(nv), np.zeros(nv), np.full(nv, np.inf), np.zeros(nr)
            self._tank_block(A, cc, lbt, ubt, rhs, 0, 0, meter=False)
            cc[:n] += pi * self.lp_tank.power_kw          # the meter price on its consumption
            _, fun, _ = self._lp(cc, A, rhs, lbt, ubt)
            total += fun
        return total

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, powers: dict[str, np.ndarray], trajs: dict[str, np.ndarray]) -> float:
        """Score a plan exactly as coordinate() scores its rounds."""
        net = self.d.copy()
        for p in powers.values():
            net = net + p
        net, _ = apply_curtailment(net, self.fc.solar, self.fc.sell, self.cfg.grid)
        return extended_objective(self.cfg, self.fc, net, trajs)

    def recover(self, integer: str, weights=None, bplans=None, time_limit: float = 30.0,
                node_limit: int = 150) -> dict[str, Column]:
        """One implementable plan per device from the CURRENT pool.

        milp: choose jointly - binary weights for devices that cannot run a
        blend, continuous for modulating ones, batteries re-optimised in the
        same solve. maxweight: each device's heaviest column in `weights`
        (the relaxed master's), batteries as the master left them.
        """
        def pick(dev, w):
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
        if integer == "milp":
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
        net = self.d + sum(c.power for c in plan.values())
        net, curtail = apply_curtailment(net, self.fc.solar, self.fc.sell, self.cfg.grid)
        trajs = {k: c.trajectory for k, c in plan.items()}
        fc, dt = self.fc, self.dt
        bill = float(np.sum(np.maximum(net, 0) * fc.buy * dt) - np.sum(np.maximum(-net, 0) * fc.sell * dt))
        comfort = comfort_penalty(self.cfg, trajs.get("water_heater"), trajs.get("hvac"), self.ref)
        breach = grid_penalty(self.cfg, net, fc.buy, fc.sell)
        batt = battery_terminal_penalty(
            self.cfg, {k: v for k, v in trajs.items() if k.startswith("battery")} or None, fc.buy)
        goal = site_battery_extras(self.cfg, trajs)   # SoC goals / gates (EVs)
        edge = 0.0
        if self.cfg.water_heater is not None and "water_heater" in trajs:
            edge += thermal_terminal(self.cfg.water_heater, "water_heater", trajs["water_heater"][-1], self.ref)
        if self.cfg.hvac is not None and "hvac" in trajs:
            edge += thermal_terminal(self.cfg.hvac, "hvac", trajs["hvac"][-1], self.ref)
        return {"bill": bill, "comfort": comfort, "breach": breach, "battery_terminal": batt,
                "thermal_edge": edge, "battery_goal": goal,
                "total": bill + comfort + breach + batt + edge + goal,
                "breach_kwh": float(np.sum(breach_energy(self.cfg, net))),
                "peak_import": float(net.max()), "peak_export": float(-net.min())}

    def polish(self, plan: dict[str, Column], sweeps: int = 3) -> tuple[dict[str, Column], float]:
        """Gauss-Seidel load-aware best response from the recovered plan.

        Accept a device's new plan only when the TRUE objective falls, so this
        is monotone - it cannot undo what the master found.
        """
        plan = dict(plan)
        cur = self.evaluate({k: c.power for k, c in plan.items()},
                            {k: c.trajectory for k, c in plan.items()})
        for _ in range(sweeps):
            improved = False
            for dev in self.devices:
                others = sum(c.power for k, c in plan.items() if k != dev.key)
                cand = self.load_aware_oracle(dev, others, self.fc.buy)
                trial = dict(plan)
                trial[dev.key] = cand
                val = self.evaluate({k: c.power for k, c in trial.items()},
                                    {k: c.trajectory for k, c in trial.items()})
                if val < cur - 1e-9:
                    plan, cur, improved = trial, val, True
                    dev.add(cand)
            if not improved:
                break
        return plan, cur

    def extra_price_points(self, pi: np.ndarray, center, it: int) -> list[np.ndarray]:
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
        progress=None,                # progress(it, max_iter) as each iteration starts, then
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
        bidding_battery = any(d.kind == "battery" and is_plain_battery(d.cfg) for d in self.devices)
        if pool == "auto":
            pool = "full" if bidding_battery else "active"
        if smoothing == "auto":
            smoothing = 0.8 if bidding_battery else 0.5
        self.pool_used, self.smoothing_used = pool, smoothing
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
            if center is not None and smoothing > 0:
                points.append(smoothing * center + (1 - smoothing) * pi)
            points.extend(self.extra_price_points(pi, center, it))

            added = 0
            flex = {}
            for pk, p in enumerate(points):
                best = {}
                for dev in self.devices:
                    col, sol = self.price_solve(dev, p / dt)
                    if pk == 0 and response in ("both", "sensitivity"):
                        added += self._add_variants(dev, sol, col, p / dt, sens_keep, flex if record else None)
                    # The DP is an exact pricing oracle only up to its grid:
                    # a pool column can undercut it (measured: HVAC, 0.0026 at
                    # a binding import limit). Take the min over both, so the
                    # bound never exceeds the master; record the shortfall,
                    # because it is how far the certificate itself can be off.
                    pool_best = min(dev.columns, key=lambda c: c.cost + p @ c.power)
                    shortfall = (col.cost + p @ col.power) - (pool_best.cost + p @ pool_best.power)
                    self.pricing_error = max(getattr(self, "pricing_error", 0.0), float(shortfall))
                    best[dev.key] = col if shortfall <= 0 else pool_best

                    added += dev.add(col)
                lb = self.lagrangian_lp(p, best) if (self.lp_batts or self.lp_tank is not None) else self.lagrangian(p, best)
                priced.append(p.copy())
                self._priced.append(p.copy())
                self._seen_min.append({d.key: min(c.cost + float(p @ c.power) for c in d.columns)
                                       for d in self.devices})
                if lb > best_lb:
                    best_lb, center = lb, p.copy()

            if heuristic_columns:
                total_mix = sum(mix.values()) + sum(c.power for c in bplans.values())
                for dev in self.devices:
                    added += dev.add(self.load_aware_oracle(dev, total_mix - mix[dev.key], pi / dt))

            gap = rmp - best_lb
            if record and flex:
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
            if added == 0:
                stop_reason = "no new proposals"
                break

        if progress is not None:
            progress(-1, max_iter)
        relaxed, lam, pi, _, bplans = self.solve_master()

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
                v = min(self._seen_min[i][d.key], min(c.cost + float(p @ c.power) for c in d.columns))
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
        upper = self.evaluate({k: c.power for k, c in plan.items()},
                              {k: c.trajectory for k, c in plan.items()})
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
            weights=weights, prices=pi / dt, history=history,
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
        )


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
    net = fc.net_fixed_demand.copy()
    trajs = {}
    for i, b in enumerate(cfg.battery_list):
        trajs[SiteConfig.battery_key(i)] = np.full(h.steps + 1, b.capacity_kwh * b.soc_initial_frac)
    if cfg.water_heater is not None:
        t, p = baseline_water_heater(cfg.water_heater, h, fc.hot_water_demand)
        net, trajs["water_heater"] = net + p, t
    if cfg.hvac is not None:
        t, p = baseline_hvac(cfg.hvac, h, fc.outdoor_temp)
        net, trajs["hvac"] = net + p, t
    net, _ = apply_curtailment(net, fc.solar, fc.sell, cfg.grid)
    return extended_objective(cfg, fc, net, trajs)


def dw_coordinate(cfg: SiteConfig, fc: Forecasts, battery_in_master: bool = True,
                  tank_in_master: bool = False, solver: str = "numpy",
                  ev_duty_cycle: bool = False, **kw) -> DWResult:
    return DWCoordinator(cfg, fc, battery_in_master=battery_in_master,
                         tank_in_master=tank_in_master, solver=solver,
                         ev_duty_cycle=ev_duty_cycle).run(**kw)
