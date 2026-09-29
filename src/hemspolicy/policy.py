"""Policy layer: the part that does not exist in either EMHASS or evcc.

A MILP returns one optimal trajectory. A DP returns a value function V[t, s]
and a policy over the WHOLE state space. That difference buys three things,
and this module is where all three are exposed:

1. ``action()``    - optimal action from any state, as an array lookup. No
                     re-solve, so it can run at sensor cadence rather than at
                     solver cadence.
2. ``marginal_value()`` - dV/ds, the money value of one more stored kWh. This
                     is a price signal that ANY device can consume without
                     being modelled inside the optimizer.
3. ``evaluate()`` / ``rollout()`` - counterfactual evaluation at microsecond
                     cost: "what does forcing this action now cost me over the
                     rest of the horizon?"

Everything here reads a `PolicySnapshot` and mutates nothing.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .dp_battery import rollout_battery
from .interp import interp_grid
from .types import BatteryConfig, CoordinationResult, Forecasts, Horizon, SiteConfig


@dataclass(frozen=True)
class PolicySnapshot:
    """Everything needed to evaluate the battery policy without re-solving.

    Deliberately self-contained and array-only so it can be persisted (.npz)
    and served by a process that never runs the solver.
    """

    horizon: Horizon
    battery: BatteryConfig
    value: np.ndarray  # V[t, s], shape (steps + 1, n_states)
    policy: np.ndarray  # POL[t, s], shape (steps, n_states)
    states: np.ndarray  # SoE grid, kWh
    actions: np.ndarray  # power action grid, kW
    buy: np.ndarray
    sell: np.ndarray
    dp_load: np.ndarray  # every other flow on the meter
    admm_target: np.ndarray | None = None
    admm_rho: float = 0.0
    generated_at: float = 0.0

    @classmethod
    def from_result(
        cls, cfg: SiteConfig, fc: Forecasts, res: CoordinationResult
    ) -> PolicySnapshot:
        if "battery" not in res.devices:
            raise ValueError("result has no battery solution to snapshot")
        if cfg.battery is None:
            raise ValueError("site config has no battery")
        # Prefer the economics-only solve; fall back to the coordinated one
        # only if pricing was not run (e.g. a bare solve_battery result).
        sol = res.battery_pricing or res.devices["battery"]
        clean = res.battery_pricing is not None
        n = cfg.horizon.steps
        return cls(
            horizon=cfg.horizon,
            battery=cfg.battery,
            value=sol.value,
            policy=sol.policy,
            states=sol.states,
            actions=sol.actions,
            buy=fc.buy.copy(),
            sell=fc.sell.copy(),
            dp_load=(
                res.battery_dp_load.copy()
                if res.battery_dp_load is not None
                else np.zeros(n)
            ),
            admm_target=(
                None
                if clean
                else (
                    res.battery_admm_target.copy()
                    if res.battery_admm_target is not None
                    else None
                )
            ),
            admm_rho=0.0 if clean else res.battery_rho,
            generated_at=time.time(),
        )

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            value=self.value,
            policy=self.policy,
            states=self.states,
            actions=self.actions,
            buy=self.buy,
            sell=self.sell,
            dp_load=self.dp_load,
            admm_target=(
                self.admm_target if self.admm_target is not None else np.array([])
            ),
            admm_rho=self.admm_rho,
            generated_at=self.generated_at,
            dt=self.horizon.dt,
            hours=self.horizon.hours,
            capacity_kwh=self.battery.capacity_kwh,
            p_charge_max_kw=self.battery.p_charge_max_kw,
            p_discharge_max_kw=self.battery.p_discharge_max_kw,
            eta=self.battery.eta,
            eta_charge=self.battery.eta_c,
            eta_discharge=self.battery.eta_d,
            soc_initial_frac=self.battery.soc_initial_frac,
            soe_min_frac=self.battery.soe_min_frac,
        )

    @classmethod
    def load(cls, path: str) -> PolicySnapshot:
        z = np.load(path)
        target = z["admm_target"]
        return cls(
            horizon=Horizon(dt=float(z["dt"]), hours=float(z["hours"])),
            battery=BatteryConfig(
                capacity_kwh=float(z["capacity_kwh"]),
                p_charge_max_kw=float(z["p_charge_max_kw"]),
                p_discharge_max_kw=float(z["p_discharge_max_kw"]),
                eta=float(z["eta"]),
                eta_charge=float(z['eta_charge']),
                eta_discharge=float(z['eta_discharge']),
                # older snapshots predate the reserve
                soc_initial_frac=float(z["soc_initial_frac"]) if "soc_initial_frac" in z else 0.5,
                soe_min_frac=float(z["soe_min_frac"]) if "soe_min_frac" in z else 0.0,
                n_states=len(z["states"]),
                n_actions=len(z["actions"]),
            ),
            value=z["value"],
            policy=z["policy"],
            states=z["states"],
            actions=z["actions"],
            buy=z["buy"],
            sell=z["sell"],
            dp_load=z["dp_load"],
            admm_target=(target if target.size else None),
            admm_rho=float(z["admm_rho"]),
            generated_at=float(z["generated_at"]),
        )

    # -- helpers ------------------------------------------------------------

    def step_for(self, hours_from_start: float) -> int:
        """Map wall-clock offset to a horizon step, clamped to the horizon."""
        return int(np.clip(round(hours_from_start / self.horizon.dt), 0, self.horizon.steps - 1))

    def _feasible_actions(self, soe: float) -> np.ndarray:
        from .dp_battery import _feasible_actions as feasible

        return feasible(
            self.actions,
            np.array(float(soe)),
            self.horizon.dt,
            self.battery.capacity_kwh,
            0.0,
            self.battery.charge_deadband_kw,
            self.battery.eta_c,
            self.battery.eta_d,
            self.battery.soe_floor_kwh,
        )


# --------------------------------------------------------------------------
# 1. Action lookup
# --------------------------------------------------------------------------


def q_values(
    snap: PolicySnapshot,
    t: int,
    soe: float,
    dp_load_kw: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Q(t, soe, a) for every feasible action. Returns (actions, Q).

    Recomputed from V rather than read from POL so it is exact for states
    between grid points - the whole reason a policy beats a stored trajectory
    is that reality lands off-grid.

    `dp_load_kw` overrides the residual load for THIS step only. Pass the
    measured value - everything on the meter except this device - and the stage
    reward becomes exact for the step about to be taken, while the tail V[t+1]
    remains the forecast-conditional continuation. That is one-step lookahead
    with a value-function terminal cost, and it is the cheapest way to react to
    something the forecast never saw.

    It is an approximation in one specific way: V[t+1] still assumes the
    forecast holds from t+1 onward. For a transient deviation that is exactly
    right. For a persistent one the tail is wrong and the answer is to re-plan,
    not to keep overriding the stage cost - see `deviation_is_persistent()`.

    WARNING for multiple storage units: do not call this per unit with the same
    measured value. Each would see the whole deviation and respond to all of
    it, and the fleet would over-correct by roughly its own size. Use
    `fleet_action()`.
    """
    dt = snap.horizon.dt
    cap = snap.battery.capacity_kwh
    eta_c, eta_d = snap.battery.eta_c, snap.battery.eta_d
    floor = snap.battery.soe_floor_kwh
    soe = float(np.clip(soe, floor, cap))

    Ac = snap._feasible_actions(soe)
    effA = np.where(Ac > 0, Ac * eta_c, Ac / eta_d)
    s_next = np.clip(soe + effA * dt, floor, cap)
    v_next = interp_grid(s_next.reshape(1, -1), snap.states, snap.value[t + 1]).ravel()

    delta = snap.dp_load[t] if dp_load_kw is None else float(dp_load_kw)
    imp = Ac + delta
    reward = (
        -snap.buy[t] * np.maximum(imp, 0.0) + snap.sell[t] * np.maximum(-imp, 0.0)
    ) * dt
    if snap.admm_target is not None and snap.admm_rho > 0:
        reward = reward - (snap.admm_rho / 2.0) * (Ac - snap.admm_target[t]) ** 2 * dt

    return Ac, reward + v_next


def action(
    snap: PolicySnapshot,
    t: int,
    soe: float,
    dp_load_kw: float | None = None,
) -> float:
    """Optimal battery power (kW, + = charge) from state (t, soe).

    This is the fast path. It is O(n_actions) - a few dozen float ops - versus
    a MILP re-solve or a 15-minute wait for the next cloud plan.

    `dp_load_kw` supplies a measured residual load for this step; see
    `q_values()`. With several storage units on one meter, use `fleet_action()`
    instead of calling this per unit.
    """
    acts, q = q_values(snap, t, soe, dp_load_kw=dp_load_kw)
    return float(acts[int(np.argmax(q))])


# --------------------------------------------------------------------------
# 2. Marginal value  (the headline result)
# --------------------------------------------------------------------------


def marginal_value(snap: PolicySnapshot, t: int, soe: float) -> float:
    """dV/ds at (t, soe): the value of one more stored kWh, in currency/kWh.

    This is the shadow price of energy in the house. Any device - including
    one the optimizer has never heard of - can compare its own benefit per kWh
    against this number and make a globally consistent decision.

    It replaces evcc's hand-tuned `bufferSoc`/`prioritySoc` thresholds with an
    actual number, and it lets EMHASS-style setups price a load without adding
    binaries to a MILP.

    Caveat that must not be lost: V was solved with the other devices' power as
    exogenous load, so this is a marginal value CONDITIONAL on their current
    plans, not a full system-wide dual. Quantifying that gap is an open
    question - see docs/NOTES.md.
    """
    grad = np.gradient(snap.value[t], snap.states)
    return float(interp_grid(np.array([np.clip(soe, snap.battery.soe_floor_kwh, snap.battery.capacity_kwh)]), snap.states, grad)[0])


def marginal_value_curve(snap: PolicySnapshot, t: int) -> np.ndarray:
    """dV/ds across the whole SoE grid at step t. Useful for plotting/debug."""
    return np.gradient(snap.value[t], snap.states)


def meter_price(snap: PolicySnapshot, t: int, soe: float) -> float:
    """Marginal cost of one more kWh drawn AT THE METER, in currency/kWh.

    This - not the tariff, and not lambda itself - is what a flexible load
    should compare against. With storage present, extra demand is served by
    whatever is cheapest at the margin, so the price is a min over the three
    ways to find another kWh:

        meter_price = min( buy[t],                 # buy it
                           lambda(t, s) / eta_d,   # discharge the battery
                           sell[t] if exporting )  # keep surplus at home

    The battery route costs lambda/eta_d because delivering 1 kWh to a load
    drains 1/eta_d kWh of store. The export route only exists when the site is
    actually spilling: then the surplus is already there and the only cost is
    the export revenue forgone.

    That third branch is easy to miss and matters a lot. Omitting it put the
    price at 0.165 through a sunny afternoon when the true figure was 0.080 -
    a load would have sat idle through the cheapest hours of the day.

    Why this matters: on a day/night tariff at 21:00 the meter reads 0.40/kWh
    while the true marginal cost is 0.1667 - the battery serves the load and
    refills overnight at 0.15. A load gating on `buy[t]` would wrongly refuse
    to run. Validated against an LP finite difference in
    tests/test_meter_price.py.

    Caveat: this is a marginal quantity. It holds for a load small enough not
    to move the plan; a 7 kW EV on a 5 kW battery will move it.
    """
    lam = marginal_value(snap, t, soe)
    options = [float(snap.buy[t]), lam / max(snap.battery.eta_d, 1e-9)]

    # Is the site exporting at this state? Ask the policy what it would do.
    #
    # The export route exists when there is genuinely surplus generation, which
    # is a property of everything OTHER than the battery: dp_load < 0 means the
    # site is producing more than it consumes before the battery acts.
    #
    # Deliberately NOT keyed on the battery's own net position. Two attempts
    # that failed: `net < 0` flips on the discrete action grid (+0.01 where the
    # continuous optimum is 0.00), and widening that to a tolerance band let
    # the branch fire at 21:00 when there is no PV at all and the battery
    # merely happened to balance the load.
    if float(snap.dp_load[t]) < 0.0:
        options.append(float(snap.sell[t]))

    return float(min(options))


def reservation_prices(snap: PolicySnapshot, t: int, soe: float) -> dict:
    """The grid prices at which it becomes worth importing or exporting.

    This is the bid/ask spread a storage owner faces, and it is what most
    people actually mean by "the price I would trade at":

        import_below = lambda * eta_c    pay at most this to put a kWh IN
        export_above = lambda / eta_d    accept at least this to take one OUT

    Buying 1 kWh only stores eta_c of it, so the most it is worth paying is
    discounted. Selling 1 kWh drains 1/eta_d from store, so the least worth
    accepting is inflated. The gap between them is exactly the round-trip loss
    - a genuine no-trade band, not a modelling artefact. Inside it, holding
    beats both directions.

    Compare against the **prevailing meter price**: `buy[t]` while the site is
    importing, `sell[t]` while it is exporting. A discharge that displaces
    household load realises `buy`, not `sell` - getting that wrong makes the
    rule look broken on any site with meaningful self-consumption.

    Measured agreement with what the optimiser actually does, per slot:
    93.8% (high export price), 89.6% (day/night, low export), 94.8%
    (dynamic tariff with PV). The residual is mostly slots where the battery
    is at a bound, or where both directions are simultaneously profitable
    because the tariff itself contains an arbitrage.
    """
    lam = marginal_value(snap, t, soe)
    bid = lam * snap.battery.eta_c
    ask = lam / max(snap.battery.eta_d, 1e-9)
    return {
        "lambda_per_kwh": lam,
        "import_below": bid,
        "export_above": ask,
        "spread": ask - bid,
    }


def price_signal(snap: PolicySnapshot, t: int, soe: float) -> dict:
    """A device-facing summary: what is a kWh worth to me right now?

    `lambda_eur_per_kwh` is the marginal value of STORED energy.
    `import_price` / `export_price` are the prevailing tariff for reference.
    A flexible load should run when its own value per kWh exceeds
    `worth_running_above`, which is the cheaper of buying and displacing.
    """
    lam = marginal_value(snap, t, soe)
    buy = float(snap.buy[t])
    sell = float(snap.sell[t])
    return {
        "step": t,
        # Value of energy INSIDE the battery.
        "lambda_per_kwh": lam,
        "import_price": buy,
        "export_price": sell,
        # Cost of one more kWh AT THE METER - the number a load should use.
        # These are different objects: lambda is what stored energy is worth,
        # this is what consuming costs, and they differ by the discharge
        # efficiency whenever the battery is the marginal source.
        "meter_price": meter_price(snap, t, soe),
        "worth_running_above": meter_price(snap, t, soe),
        # Grid trading thresholds: import while the meter price is below the
        # first, export while it is above the second.
        **{k: v for k, v in reservation_prices(snap, t, soe).items()
           if k != "lambda_per_kwh"},
    }


# --------------------------------------------------------------------------
# 3. Counterfactual evaluation
# --------------------------------------------------------------------------


@dataclass
class Counterfactual:
    """Result of forcing a non-optimal action for one step."""

    forced_action: float
    optimal_action: float
    q_forced: float
    q_optimal: float
    elapsed_ms: float

    @property
    def opportunity_cost(self) -> float:
        """Currency lost over the remaining horizon by taking the forced action.

        Always >= 0 (up to grid resolution). This is the number that answers
        "what did letting the car charge now actually cost me?".
        """
        return self.q_optimal - self.q_forced


def evaluate(snap: PolicySnapshot, t: int, soe: float, forced_action: float) -> Counterfactual:
    """Cost of forcing `forced_action` at (t, soe) versus acting optimally.

    Microseconds, because V already encodes the entire optimal future. The
    MILP equivalent is a re-solve with an added equality constraint.
    """
    t0 = time.perf_counter()
    acts, q = q_values(snap, t, soe)
    best = int(np.argmax(q))

    lo, hi = float(acts.min()), float(acts.max())
    forced = float(np.clip(forced_action, lo, hi))
    q_forced = float(interp_grid(np.array([forced]), acts, q)[0]) if acts[0] < acts[-1] else float(q[0])

    return Counterfactual(
        forced_action=forced,
        optimal_action=float(acts[best]),
        q_forced=q_forced,
        q_optimal=float(q[best]),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def rollout(snap: PolicySnapshot, t: int, soe: float) -> tuple[np.ndarray, np.ndarray]:
    """Replay the optimal trajectory forward from ANY hypothetical (t, soe).

    Answers "the forecast was wrong and I'm at 20% instead of 60% - what now?"
    without a re-solve. A MILP trajectory has nothing to say about a state it
    did not predict.
    """
    return rollout_battery(
        snap.battery,
        snap.horizon,
        snap.value,
        snap.states,
        snap.actions,
        snap.buy,
        snap.sell,
        snap.dp_load,
        snap.admm_target,
        snap.admm_rho,
        start_step=t,
        start_soe=soe,
    )


# --------------------------------------------------------------------------
# 4. Hard-constraint clamp  (must live OUTSIDE the DP)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HardLimits:
    """Constraints that must never be violated, whatever the policy says.

    The DP enforces SoC gates as big-M penalties, which is fine for
    preferences and wrong for safety. Fuse ratings and grid-operator dimming
    (evcc's `hems/`, German 14a EnWG) are clamped here instead, so a badly
    tuned penalty can never produce an unsafe setpoint.
    """

    max_import_kw: float | None = None
    max_export_kw: float | None = None
    max_charge_kw: float | None = None
    max_discharge_kw: float | None = None
    max_ramp_kw_per_step: float | None = None


def clamp(
    proposed_kw: float,
    other_load_kw: float,
    limits: HardLimits,
    previous_kw: float | None = None,
) -> tuple[float, list[str]]:
    """Clip a proposed battery power to the hard limits.

    Returns the safe setpoint and the list of limits that bound it, so the
    caller can log or surface why the policy was overridden.
    """
    a = float(proposed_kw)
    hit: list[str] = []

    if limits.max_charge_kw is not None and a > limits.max_charge_kw:
        a, _ = limits.max_charge_kw, hit.append("max_charge_kw")
    if limits.max_discharge_kw is not None and a < -limits.max_discharge_kw:
        a, _ = -limits.max_discharge_kw, hit.append("max_discharge_kw")

    if limits.max_import_kw is not None:
        # net import = battery draw + everything else
        allowed = limits.max_import_kw - other_load_kw
        if a > allowed:
            a = allowed
            hit.append("max_import_kw")
    if limits.max_export_kw is not None:
        allowed = -limits.max_export_kw - other_load_kw
        if a < allowed:
            a = allowed
            hit.append("max_export_kw")

    if limits.max_ramp_kw_per_step is not None and previous_kw is not None:
        lo = previous_kw - limits.max_ramp_kw_per_step
        hi = previous_kw + limits.max_ramp_kw_per_step
        if not lo <= a <= hi:
            a = float(np.clip(a, lo, hi))
            hit.append("max_ramp_kw_per_step")

    return a, hit


def clamp_fleet(
    proposals: "Sequence[tuple[str, float]]",
    other_load_kw: float,
    limits: HardLimits,
    unit_limits: "Mapping[str, HardLimits] | None" = None,
    previous_kw: "Mapping[str, float] | None" = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Clamp several storage units against ONE shared grid limit, in sequence.

    Calling `clamp()` per unit with the same `other_load_kw` is wrong whenever
    more than one unit can move: each is told the full headroom is available,
    each takes it, and together they overshoot by a factor of the fleet size.
    That is the same double-counting a simultaneous best response produces
    (docs/theory.tex, "Why iterating the local solve is not enough"), except
    that here there are no coordination rounds to damp it - this runs once,
    between plans.

    So the headroom is consumed as it is handed out: unit i is clamped against
    the measured load PLUS the setpoints already granted to units 0..i-1. The
    result is feasible by construction for any fleet size.

    `proposals` is ordered, and the order is the priority: earlier units get
    the headroom first. The caller owns that decision - a sensible default is
    descending marginal value (`marginal_value()`), so the unit with most to
    gain moves first - but it must be deterministic, or the fleet will chatter
    between equally good allocations.

    `limits` supplies the SHARED grid limits (max_import_kw / max_export_kw).
    Per-unit limits (charge, discharge, ramp) come from `unit_limits[name]`,
    since they differ per device and are not a shared resource.

    Returns the safe setpoint per unit and, per unit, which limits bound it.
    """
    granted: dict[str, float] = {}
    bound_by: dict[str, list[str]] = {}
    running = float(other_load_kw)

    for name, proposed in proposals:
        own = unit_limits.get(name, HardLimits()) if unit_limits else HardLimits()
        effective = HardLimits(
            # shared, consumed cumulatively
            max_import_kw=limits.max_import_kw,
            max_export_kw=limits.max_export_kw,
            # per unit
            max_charge_kw=own.max_charge_kw,
            max_discharge_kw=own.max_discharge_kw,
            max_ramp_kw_per_step=own.max_ramp_kw_per_step,
        )
        safe, hit = clamp(
            proposed,
            other_load_kw=running,
            limits=effective,
            previous_kw=previous_kw.get(name) if previous_kw else None,
        )
        granted[name] = safe
        bound_by[name] = hit
        running += safe  # this unit's draw is load as far as the next one is concerned

    return granted, bound_by


def fleet_action(
    units: "Sequence[tuple[str, PolicySnapshot, float]]",
    t: int,
    measured_load_kw: float,
    rounds: int = 4,
    damping: float = 0.5,
    tol: float = 1e-3,
) -> tuple[dict[str, float], int]:
    """Real-time actions for several storage units sharing one meter.

    `units` is (name, snapshot, soe) in a deterministic priority order.
    `measured_load_kw` is the metered flow with EVERY listed unit's own draw
    removed - i.e. house load minus PV, the part none of them controls.

    Why this is not just `action()` in a loop
    -----------------------------------------
    Each unit's optimal reply depends on what the others do, through the kink
    in the bill. Handing all of them the same measurement and letting each
    reply to the whole deviation is simultaneous best response: every unit
    covers the spike in full, so a fleet of m covers it m times, and the next
    tick sees the over-correction and reverses it. That is the oscillation of
    docs/theory.tex 3.2, now with no proximal term and no rho to damp it,
    because at inference there are no coordination rounds.

    So the replies are computed in sequence, each unit seeing the deviation net
    of what the units before it have already committed to - a Gauss-Seidel
    sweep - and the sweep is repeated a few times to approach the fixed point
    where every reply is optimal against the others' final replies. Each
    evaluation is an array lookup, so several sweeps still cost microseconds.

    Convergence is not guaranteed (same non-convexity as ever), hence
    `damping`: a unit moves only part of the way to its new reply each sweep,
    which turns most oscillations into decay. `rounds` bounds the work
    regardless. The last iterate is returned with the sweep count, so a caller
    that cares can tell whether it settled.

    A physical meter gives this for free if the units are polled in sequence
    rather than together: each reads a meter that already contains the
    previous unit's response. Simultaneous polling is what creates the problem.
    """
    names = [u[0] for u in units]
    reply = {n: 0.0 for n in names}
    settled = 0

    for sweep in range(max(1, rounds)):
        largest_move = 0.0
        for name, snap, soe in units:
            # everything on the meter except me: the uncontrolled load plus the
            # other units' current replies
            others = measured_load_kw + sum(reply[n] for n in names if n != name)
            target = action(snap, t, soe, dp_load_kw=others)
            moved = target - reply[name]
            reply[name] = reply[name] + damping * moved
            largest_move = max(largest_move, abs(moved))
        settled = sweep + 1
        if largest_move < tol:
            break

    return reply, settled


def deviation_is_persistent(
    snap: PolicySnapshot,
    t: int,
    measured_load_kw: float,
    tolerance_kw: float = 1.0,
) -> bool:
    """True when the measured residual load has left the forecast far enough
    that the stored value function should no longer be trusted as a tail.

    `q_values(dp_load_kw=...)` corrects the stage cost but not V[t+1]. Once the
    deviation is large the continuation is being valued under a forecast that
    is no longer describing this site, and the right response is to re-plan
    rather than to keep patching the current step.
    """
    return abs(float(measured_load_kw) - float(snap.dp_load[t])) > tolerance_kw
