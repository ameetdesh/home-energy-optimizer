"""ADMM / price-based coordination of the per-device DPs.

Each device solves its own 1-D DP seeing every other flow on the meter as
exogenous load, so it internalises its own marginal effect on the grid
exchange. The battery additionally carries an ADMM proximity term; the binary
thermal devices deliberately do not (non-convex, no convergence guarantee), so
they are coordinated by price alone.

This is a heuristic. It has no optimality certificate, unlike the MILPs in
EMHASS and evcc. Measuring the gap against EMHASS's MILP is Phase 1 in
docs/PLAN.md and gates everything downstream.
"""

from __future__ import annotations

import time

import numpy as np

from .dp_battery import solve_battery, terminal_price
from .meter import Limits
from .dp_thermal import (
    baseline_hvac,
    baseline_water_heater,
    hvac_discomfort,
    solve_hvac,
    solve_water_heater,
    wh_discomfort,
)
from .types import (
    CoordinationResult,
    DeviceSolution,
    Forecasts,
    RoundRecord,
    SiteConfig,
)


def apply_curtailment(
    net_raw: np.ndarray, solar: np.ndarray, sell: np.ndarray, grid
) -> tuple[np.ndarray, np.ndarray]:
    """Throw away the PV that cannot usefully leave the site.

    Returns (net_grid_after, curtailment). Curtailment reduces export 1:1 and
    is bounded by the PV actually available, so this is exact rather than an
    approximation - there is no interaction with the devices to iterate over.

    Two reasons to curtail, and only two:

    * **an export cap** that storage cannot absorb. Once the battery is full
      the surplus has nowhere to go, and no price can make it vanish - this is
      the only lever left.
    * **a negative export price**, where being paid to stop generating beats
      paying someone to take the energy.

    Deliberately applied AFTER the device solves rather than inside them: it
    only ever discards energy that was already going to be exported, so it
    cannot change what a device should have done. That keeps it out of the
    coordination loop entirely.
    """
    curtail = np.zeros_like(net_raw)
    if grid is None or not grid.allow_curtailment:
        return net_raw, curtail

    export = np.maximum(-net_raw, 0.0)
    if grid.max_export_kw is not None:
        curtail = np.minimum(solar, np.maximum(export - grid.max_export_kw, 0.0))
    # Never pay to export.
    negative_price = sell < 0
    if negative_price.any():
        curtail = np.where(
            negative_price, np.minimum(solar, export), curtail
        )
    return net_raw + curtail, curtail


def device_sell_price(sell: np.ndarray, grid) -> np.ndarray:
    """The export price a DEVICE should be optimising against.

    Curtailment is free and always available up to the PV on the roof, so a
    negative export price is never actually paid on PV-driven surplus - you
    stop generating instead. A device priced at the raw negative tariff sees a
    phantom benefit for absorbing that surplus, and will spend round-trip
    losses to capture something curtailment delivers for nothing.

    Measured on a day/night fixture with a -0.10 midday export price: pricing
    devices at the raw tariff costs 0.31 currency/day and 7.3 kWh of extra
    battery throughput against pricing them at max(sell, 0). The gap scales
    linearly with how negative the price is, and is exactly zero for sell >= 0,
    so this is a no-op on every non-negative tariff.

    The BILL is still computed on the true tariff - this changes what devices
    optimise, not what the site is charged.
    """
    if grid is None or not grid.allow_curtailment:
        return sell
    return np.maximum(sell, 0.0)


def net_cost(net_grid: np.ndarray, buy: np.ndarray, sell: np.ndarray, dt: float) -> float:
    """Bill for a net grid exchange series. Positive net_grid = import."""
    imp = float(np.sum(np.maximum(net_grid, 0.0) * buy * dt))
    exp = float(np.sum(np.maximum(-net_grid, 0.0) * sell * dt))
    return imp - exp


def comfort_penalty(
    cfg: SiteConfig,
    wh_temp: np.ndarray | None,
    hvac_temp: np.ndarray | None,
    reference_price: float = 0.0,
) -> float:
    """Total discomfort cost. In linear mode this is genuinely in currency, so
    adding it to a bill is meaningful rather than an arbitrary exchange rate.

    Must stay in lockstep with the penalties inside the DPs - they are the same
    two functions, deliberately shared rather than reimplemented.
    """
    dt = cfg.horizon.dt
    total = 0.0
    if cfg.water_heater is not None and wh_temp is not None:
        total += float(np.sum(wh_discomfort(cfg.water_heater, wh_temp[1:], reference_price, dt)))
    if cfg.hvac is not None and hvac_temp is not None:
        total += float(np.sum(hvac_discomfort(cfg.hvac, hvac_temp[1:], reference_price, dt)))
    return total


def breach_energy(cfg: SiteConfig, net_grid: np.ndarray) -> np.ndarray:
    """Energy drawn or pushed beyond the grid limits, per slot, in kWh.

    Zero everywhere when no limit is configured, which is why every benchmark
    objective is untouched by the penalty built on it.
    """
    over = np.zeros_like(net_grid)
    if cfg.grid.max_import_kw is not None:
        over = over + np.maximum(net_grid - cfg.grid.max_import_kw, 0.0)
    if cfg.grid.max_export_kw is not None:
        over = over + np.maximum(-net_grid - cfg.grid.max_export_kw, 0.0)
    return over * cfg.horizon.dt


def breach_price(grid, buy: np.ndarray, sell: np.ndarray | None = None) -> float:
    """Currency per kWh beyond a grid limit: one constant for the horizon.

    `grid.breach_price` if set, else `breach_price_multiplier` times the
    largest |price| on the horizon (1.0 if every price is zero).
    """
    if grid.breach_price is not None:
        return float(grid.breach_price)
    prices = np.abs(np.asarray(buy, dtype=float))
    if sell is not None:
        prices = np.concatenate([prices, np.abs(np.asarray(sell, dtype=float))])
    scale = float(prices.max()) if prices.size and prices.max() > 0 else 1.0
    return grid.breach_price_multiplier * scale


def grid_penalty(cfg: SiteConfig, net_grid: np.ndarray, buy: np.ndarray,
                 sell: np.ndarray | None = None) -> float:
    """Cost of breaching the grid limits, in currency.

    Over-limit energy is priced at `breach_price` - a constant, not the slot's
    tariff. A grid limit is usually a fuse or a regulatory cap; it does not
    matter less at a cheap hour, and a tariff-scaled penalty vanishes at a zero
    import price and turns into a reward at a negative one.

    Putting the breach in the objective is what lets round selection be a
    single comparison on money. The alternative - rank on violation first and
    cost second - never lets the two trade at all, which sounds safer but in
    practice decides on the eighth decimal place of a breach that is
    infeasible either way.

    This does NOT enforce anything. The multiplier that steers the device
    solves is mu in `dual ascent` below, and anything safety-critical is
    clamped outside the solver entirely (policy.clamp). This term scores a
    finished plan; it does not produce one.
    """
    if not cfg.grid.active:
        return 0.0
    price = breach_price(cfg.grid, buy, sell)
    if price <= 0:
        return 0.0
    return float(np.sum(breach_energy(cfg, net_grid)) * price)


def battery_terminal_penalty(
    cfg: SiteConfig,
    soe: np.ndarray | dict[str, np.ndarray] | None,
    buy: np.ndarray | None = None,
) -> float:
    """Cost of ending the horizon with less energy than you started with.

    In linear mode this bills the net depletion at the terminal price, which is
    what stops a plan from looking cheap simply by emptying the battery. In
    quadratic mode it reproduces the POC's pull toward a target SoC.

    Accepts either a single trajectory (battery 0, the common case) or a
    key -> trajectory mapping, in which case every battery is charged.
    """
    if soe is None or not cfg.battery_list:
        return 0.0

    if not isinstance(soe, dict):
        soe = {SiteConfig.battery_key(0): soe}

    total = 0.0
    for i, b in enumerate(cfg.battery_list):
        traj = soe.get(SiteConfig.battery_key(i))
        if traj is None:
            continue
        if b.terminal_mode == "linear":
            if buy is None:
                continue
            total += float(traj[0] - traj[-1]) * terminal_price(b, buy)
        else:
            target = b.capacity_kwh * b.terminal_target_frac
            total += float(
                b.terminal_weight * (traj[-1] - target) ** 2 / max(b.capacity_kwh, 0.01)
            )
    return total


def total_objective(
    cfg: SiteConfig,
    net_grid: np.ndarray,
    fc: Forecasts,
    wh_temp: np.ndarray | None,
    hvac_temp: np.ndarray | None,
    soe: np.ndarray | dict[str, np.ndarray] | None,
) -> float:
    """Bill + comfort + grid-breach penalties + battery terminal penalty.

    Lower is better, and every term is in currency, so the sum is a money
    number that can be compared against another solver's objective. Used for
    round selection, which is therefore a single comparison: the coordinator
    cannot trade a comfort violation or a grid breach for a smaller bill,
    because both already cost money here.

    The grid term is identically zero with no limit configured, so the
    benchmark objectives are exactly the bill-plus-comfort they always were.
    """
    return (
        net_cost(net_grid, fc.buy, fc.sell, cfg.horizon.dt)
        + comfort_penalty(cfg, wh_temp, hvac_temp, float(np.mean(fc.buy)))
        + grid_penalty(cfg, net_grid, fc.buy, fc.sell)
        + battery_terminal_penalty(cfg, soe, fc.buy)
    )


def baseline_solution(cfg: SiteConfig, fc: Forecasts) -> tuple[np.ndarray, float]:
    """What the house does with no optimization: thermostats, idle battery.

    Curtailed on the same terms as the optimised plan. A baseline that keeps
    paying to export when curtailment is available is not the right comparison -
    it would flatter the optimiser by an amount that has nothing to do with
    scheduling.
    """
    net = fc.net_fixed_demand.copy()
    if cfg.water_heater is not None:
        net = net + baseline_water_heater(cfg.water_heater, cfg.horizon, fc.hot_water_demand)[1]
    if cfg.hvac is not None:
        net = net + baseline_hvac(cfg.hvac, cfg.horizon, fc.outdoor_temp)[1]
    net, _ = apply_curtailment(net, fc.solar, fc.sell, cfg.grid)
    return net, net_cost(net, fc.buy, fc.sell, cfg.horizon.dt)


def coordinate(cfg: SiteConfig, fc: Forecasts, progress=None, warm=None) -> CoordinationResult:
    """Run the coordination loop and return the best round found.

    `progress`, if given, is called as progress(round, max_rounds) when a
    round starts, progress(round, max_rounds, objective, None, best) when it
    has been scored (None: ADMM has no bound), and progress(-1, max_rounds)
    before the fallback and polish - the same shape the DW app reads.
    """
    if cfg.coordination.algorithm == "exchange":          # textbook ADMM
        from .exchange import coordinate_exchange
        return coordinate_exchange(cfg, fc, progress, warm)
    cfg.validate()
    fc.validate(cfg.horizon)

    n = cfg.horizon.steps
    dt = cfg.horizon.dt
    cc = cfg.coordination
    rho = cc.rho

    # A grid limit needs materially more rounds than pure cost coordination:
    # the multiplier has to climb from zero until the plan fits. Measured 14-20
    # rounds for a hard-binding limit, against 2 with no limit at all, so the
    # 15-round default would silently ship an infeasible plan.
    max_rounds = cc.max_rounds
    if cfg.grid.active:
        max_rounds = max(max_rounds, 40)

    # Fixed demand: inflexible load minus PV, plus the baseline behaviour of
    # any device that is switched off (it still consumes, just not smartly).
    fixed = fc.net_fixed_demand.copy()

    # One device per battery, keyed "battery", "battery1", ... (SiteConfig
    # .battery_key). evcc sends one BatteryConfig per stationary battery AND
    # one per loadpoint, so N > 1 is the normal case for anyone with a charger.
    batt_cfgs: dict[str, "object"] = {}
    dev_keys: list[str] = []
    for i, b in enumerate(cfg.battery_list):
        if b.capacity_kwh > 0:
            key = SiteConfig.battery_key(i)
            batt_cfgs[key] = b
            dev_keys.append(key)
    if cfg.water_heater is not None:
        dev_keys.append("water_heater")
    if cfg.hvac is not None:
        dev_keys.append("hvac")
    n_dev = max(len(dev_keys), 1)

    targets = {k: np.zeros(n) for k in dev_keys}
    duals = {k: np.zeros(n) for k in dev_keys}

    # Warm-start the coordination targets from each device's uncoordinated
    # behaviour. For the thermal devices that is their thermostat baseline.
    if "water_heater" in targets:
        targets["water_heater"] = baseline_water_heater(
            cfg.water_heater, cfg.horizon, fc.hot_water_demand
        )[1].copy()
    if "hvac" in targets:
        targets["hvac"] = baseline_hvac(cfg.hvac, cfg.horizon, fc.outdoor_temp)[1].copy()

    # The battery MUST be warm-started the same way, and the POC's omission of
    # this silently disabled it: with target = 0 and dual = 0, round 0 applies
    # a proximity penalty of (rho/2) * a^2 * dt about ZERO. At the default
    # rho = 5 that is ~0.6 * a^2, i.e. ~15 currency at 5 kW, dwarfing the
    # ~0.2/kWh available from arbitrage - so the battery is pinned off. The
    # z-update then only moves the target by g'/rho (~0.008) per round, which
    # cannot escape within max_rounds. Seeding from a free (rho = 0) solve
    # starts the proximity term centred on something the battery actually
    # wants to do. See docs/NOTES.md.
    # Each battery is seeded from a free (rho = 0) solve, and each sees the
    # batteries seeded before it, so two identical batteries do not both plan
    # the same charge and then fight over it for several rounds.
    # Export price the devices optimise against (see device_sell_price).
    sell_dev = device_sell_price(fc.sell, cfg.grid)
    # "exact" grid mode: the devices price the limits themselves (see
    # CoordinationConfig.grid_mode); the limit prices mu then stay at zero.
    limits = (Limits(cfg.grid.max_import_kw, cfg.grid.max_export_kw,
                     breach_price(cfg.grid, fc.buy, fc.sell), cfg.grid.allow_curtailment)
              if cfg.grid.active and cc.grid_mode == "exact" else None)

    seed_load = fixed.copy()
    for k in ("water_heater", "hvac"):
        if k in dev_keys:
            seed_load = seed_load + targets[k]
    for key in [k for k in dev_keys if k in batt_cfgs]:
        targets[key] = solve_battery(
            batt_cfgs[key],
            cfg.horizon,
            fc.buy,
            sell_dev,
            dp_load=seed_load,
            admm_rho=0.0,
            soc_gates=cfg.soc_gates if key == "battery" else (),
            limits=limits,
        ).power.copy()
        seed_load = seed_load + targets[key]

    prev_power = {k: targets.get(k, np.zeros(n)).copy() for k in dev_keys}
    # Momentum: the device and z steps use extrapolated targets and duals.
    z_hat = {k: targets[k].copy() for k in dev_keys}
    u_hat = {k: duals[k].copy() for k in dev_keys}
    z_prev = {k: targets[k].copy() for k in dev_keys}
    u_prev = {k: duals[k].copy() for k in dev_keys}
    alpha, c_prev = 1.0, np.inf
    damping = 1.0               # share of a device's new plan the others see
    recent: list[np.ndarray] = []   # the last few rounds' meter flows

    # Multipliers on the grid coupling constraint. A grid limit binds the SUM
    # of every device's power plus the inflexible load, so no per-device DP can
    # enforce it alone. Instead we price it: raise the effective import price
    # (or cut the export price) until the plan fits. mu is a shadow price on
    # the constraint - the same kind of object lambda is for stored energy.
    mu_imp = np.zeros(n)
    mu_exp = np.zeros(n)
    # |buy|: a mean tariff near zero or below must not stall or reverse the step
    grid_step = cfg.grid.price_step * max(float(np.mean(np.abs(fc.buy))), 1e-6)
    grid_step_min = grid_step / 16.0
    last_improve = 0

    best_obj = np.inf
    best: CoordinationResult | None = None
    best_round = 0
    records: list[RoundRecord] = []
    rounds_run = 0
    stop_reason = "iteration cap"

    for r in range(max_rounds):
        t_round = time.perf_counter()
        if progress is not None:
            progress(r + 1, max_rounds)
        rounds_run = r + 1
        sols: dict[str, DeviceSolution] = {}

        # Each device sees everything EXCEPT itself as exogenous load.
        def other_load(me: str) -> np.ndarray:
            out = fixed.copy()
            for k in dev_keys:
                if k != me:
                    out = out + prev_power[k]
            return out

        # Prices the devices actually see this round: the tariff plus the
        # current grid-constraint multipliers.
        buy_eff = fc.buy + mu_imp
        sell_eff = sell_dev - mu_exp

        # ---- Step 1: device DP updates -------------------------------------
        batt_dp_load = other_load("battery")
        batt_shifted = None
        for key, bcfg in batt_cfgs.items():
            shifted = z_hat[key] - u_hat[key]
            sols[key] = solve_battery(
                bcfg,
                cfg.horizon,
                buy_eff,
                sell_eff,
                dp_load=other_load(key),
                admm_target=shifted,
                admm_rho=rho,
                soc_gates=cfg.soc_gates if key == "battery" else (),
                limits=limits,
            )
            if key == "battery":
                batt_shifted = shifted
        if "water_heater" in dev_keys:
            sols["water_heater"] = solve_water_heater(
                cfg.water_heater,
                cfg.horizon,
                buy_eff,
                sell_eff,
                fc.hot_water_demand,
                dp_load=other_load("water_heater"),
                limits=limits,
            )
        if "hvac" in dev_keys:
            sols["hvac"] = solve_hvac(
                cfg.hvac,
                cfg.horizon,
                buy_eff,
                sell_eff,
                fc.outdoor_temp,
                dp_load=other_load("hvac"),
                limits=limits,
            )

        powers = {k: sols[k].power for k in dev_keys}
        rho_snapshot = rho

        # ---- Step 2: z-update (closed-form per-timestep bill minimisation) --
        old_targets = {k: targets[k].copy() for k in dev_keys}
        for t in range(n):
            s_vals = {k: powers[k][t] + u_hat[k][t] for k in dev_keys}
            S = sum(s_vals.values())
            d_t = fixed[t]

            v_import = S - n_dev * fc.buy[t] * dt / rho
            v_export = S + n_dev * fc.sell[t] * dt / rho
            if v_import + d_t > 0:
                g_prime = fc.buy[t] * dt
            elif v_export + d_t < 0:
                g_prime = -fc.sell[t] * dt
            else:
                g_prime = 0.0

            for k in dev_keys:
                targets[k][t] = s_vals[k] - g_prime / rho

        # ---- Step 3: dual update -------------------------------------------
        for k in dev_keys:
            duals[k] = u_hat[k] + powers[k] - targets[k]

        # ---- Momentum, restarted when the combined residual grows ----------
        # (Goldstein et al. 2014, fast ADMM with restart; here a restart just
        # drops the extrapolation for a round.)
        c_k = rho * sum(float(np.sum((duals[k] - u_hat[k]) ** 2) + np.sum((targets[k] - z_hat[k]) ** 2))
                        for k in dev_keys)
        if cc.momentum and c_k < cc.restart_eta * c_prev:
            a_next = (1.0 + np.sqrt(1.0 + 4.0 * alpha * alpha)) / 2.0
            w = (alpha - 1.0) / a_next
            for k in dev_keys:
                z_hat[k] = targets[k] + w * (targets[k] - z_prev[k])
                u_hat[k] = duals[k] + w * (duals[k] - u_prev[k])
            alpha, c_prev = a_next, c_k
        else:                    # restart: no extrapolation this round
            alpha = 1.0
            c_prev = c_prev / cc.restart_eta if cc.momentum else c_k
            for k in dev_keys:
                z_hat[k], u_hat[k] = targets[k].copy(), duals[k].copy()
        for k in dev_keys:
            z_prev[k], u_prev[k] = targets[k].copy(), duals[k].copy()

        # ---- Evaluate -------------------------------------------------------
        net_grid = fixed.copy()
        for k in dev_keys:
            net_grid = net_grid + powers[k]
        net_grid, curtail = apply_curtailment(net_grid, fc.solar, fc.sell, cfg.grid)

        wh_temp = sols["water_heater"].trajectory if "water_heater" in sols else None
        hvac_temp = sols["hvac"].trajectory if "hvac" in sols else None
        soe = {k: sols[k].trajectory for k in batt_cfgs if k in sols} or None

        # ---- Dual ascent on the grid limits --------------------------------
        # Raise the price wherever the plan still exceeds a limit, relax it
        # where there is headroom. Projected onto mu >= 0: a limit that is not
        # binding must not subsidise.
        if cfg.grid.active and limits is None:
            if cfg.grid.max_import_kw is not None:
                mu_imp = np.maximum(
                    0.0, mu_imp + grid_step * (net_grid - cfg.grid.max_import_kw)
                )
            if cfg.grid.max_export_kw is not None:
                mu_exp = np.maximum(
                    0.0, mu_exp + grid_step * (-net_grid - cfg.grid.max_export_kw)
                )

        cost = net_cost(net_grid, fc.buy, fc.sell, dt)
        obj = total_objective(cfg, net_grid, fc, wh_temp, hvac_temp, soe)

        primal = sum(float(np.linalg.norm(powers[k] - targets[k])) for k in dev_keys)
        dual = sum(float(rho * np.linalg.norm(targets[k] - old_targets[k])) for k in dev_keys)

        # Physical breach, for reporting. It carries no weight in the ranking:
        # `obj` already prices it (grid_penalty), so a cheaper-but-breaching
        # round is not cheaper on the only number that decides.
        violation = float(np.sum(breach_energy(cfg, net_grid)) / dt)
        import_excess = (
            float(np.maximum(net_grid - cfg.grid.max_import_kw, 0.0).max())
            if cfg.grid.max_import_kw is not None else 0.0
        )
        export_excess = (
            float(np.maximum(-net_grid - cfg.grid.max_export_kw, 0.0).max())
            if cfg.grid.max_export_kw is not None else 0.0
        )
        import_cost = float(np.sum(np.maximum(net_grid, 0.0) * fc.buy * dt))
        export_revenue = float(np.sum(np.maximum(-net_grid, 0.0) * fc.sell * dt))

        records.append(
            RoundRecord(
                index=len(records),
                powers={k: sols[k].power.copy() for k in dev_keys},
                trajectories={k: sols[k].trajectory.copy() for k in dev_keys},
                net_grid=net_grid.copy(),
                curtailment=curtail.copy(),
                import_cost=import_cost,
                export_revenue=export_revenue,
                net_cost=cost,
                total_objective=obj,
                violation=violation,
                import_excess=import_excess,
                export_excess=export_excess,
                score_objective=obj,
                primal_res=primal,
                dual_res=dual,
                rho=rho_snapshot,
                round_ms=(time.perf_counter() - t_round) * 1000.0,
                device_ms={k: sols[k].solve_ms for k in dev_keys},
                battery_dp_load=batt_dp_load.copy(),
            )
        )

        if obj < best_obj - cc.converge_tol * max(abs(best_obj), 1e-6) or not np.isfinite(best_obj):
            last_improve = r
        if obj < best_obj:
            best_obj = obj
            best_round = records[-1].index
            best = CoordinationResult(
                devices=dict(sols),
                net_grid=net_grid,
                import_cost=import_cost,
                export_revenue=export_revenue,
                net_cost=cost,
                total_objective=obj,
                rounds_run=rounds_run,
                grid_import_excess=import_excess,
                grid_export_excess=export_excess,
                curtailment=curtail.copy(),
                curtailed_kwh=float(curtail.sum() * dt),
                battery_dp_load=batt_dp_load.copy(),
                battery_admm_target=(batt_shifted.copy() if batt_shifted is not None else None),
                battery_rho=rho_snapshot if "battery" in dev_keys else 0.0,
            )

        if progress is not None:
            progress(r + 1, max_rounds, float(obj), None, float(best_obj))

        # A limit cycle: this round's meter flow repeats one from 2-4 rounds
        # ago (repeating the LAST round is a fixed point, not a cycle). Damp
        # what the devices see of each other and the grid-limit price step,
        # and tighten the tether (rho only ever rises here, up to rho_max, so
        # it still settles).
        cycle = (bool(recent) and np.max(np.abs(net_grid - recent[-1])) > 1e-6
                 and any(np.max(np.abs(net_grid - g)) < 1e-6 for g in recent[:-1]))
        if cycle:
            damping = max(damping / 2.0, cc.damping_min)
            grid_step = max(grid_step / 2.0, grid_step_min)
        recent = (recent + [net_grid.copy()])[-4:]
        for k in dev_keys:
            prev_power[k] = (1.0 - damping) * prev_power[k] + damping * powers[k]

        # Adaptive rho (Boyd et al. 3.4.1): keep the residuals balanced, then
        # hold it. The scaled duals are y / rho, so they are rescaled with it,
        # or every change of rho would also kick the prices.
        rho_old = rho
        if cycle and cc.rho_adapt_factor > 1.0:    # "adapt rho" off: rho stays put
            rho = min(rho * 2.0, cc.rho_max)
        elif r < cc.rho_freeze_round:
            if primal > cc.rho_adapt_ratio * max(dual, 1e-6):
                rho = min(rho * cc.rho_adapt_factor, cc.rho_max)
            elif dual > cc.rho_adapt_ratio * max(primal, 1e-6):
                rho = max(rho / cc.rho_adapt_factor, cc.rho_min)
        if rho != rho_old:
            for k in dev_keys:
                for u in (duals, u_hat, u_prev):
                    u[k] = u[k] * (rho_old / rho)
            alpha, c_prev = 1.0, np.inf     # momentum restarts with a new rho
            for k in dev_keys:
                z_hat[k], u_hat[k] = targets[k].copy(), duals[k].copy()

        # Converging on cost is not enough while the plan is still infeasible.
        # A stall ends the loop either way: the best round is chosen on an
        # objective that already prices any breach.
        feasible = not cfg.grid.active or violation <= 1e-6
        if feasible and r > 0 and primal < cc.residual_tol and dual < cc.residual_tol:
            stop_reason = "converged"
            break
        if r - last_improve >= cc.patience:
            stop_reason = "no improvement"
            break

    if progress is not None:
        progress(-1, max_rounds)
    assert best is not None, "coordination produced no rounds"
    best.rounds_run = rounds_run
    best.stop_reason = stop_reason
    records[best_round].selected = True
    best.rounds = records
    best.selected_round = best_round

    # The fallback can swap a device for its thermostat baseline AFTER the
    # round was scored, so when it does, the retained record has to be
    # refreshed or it would describe a plan the caller never receives.
    if cc.enable_baseline_fallback and _apply_baseline_fallback(cfg, fc, best, dev_keys):
        _refresh_selected_record(records[best_round], best, cfg)

    if cc.polish and _polish(cfg, fc, best, dev_keys, fc.buy + mu_imp, sell_dev - mu_exp,
                             cc.polish_sweeps, limits):
        _refresh_selected_record(records[best_round], best, cfg)

    if "battery" in best.devices:
        best.battery_pricing = _pricing_resolve(cfg, fc, best, limits)

    best.baseline_cost = baseline_solution(cfg, fc)[1]
    return best


def _polish(cfg: SiteConfig, fc: Forecasts, res: CoordinationResult, dev_keys: list[str],
            buy: np.ndarray, sell: np.ndarray, sweeps: int, limits: Limits | None = None) -> bool:
    """Gauss-Seidel best response from the returned plan; keep only improvements.

    ADMM settles where each device is a best response to the others' LAST
    plans - consistent, but often not where the house would be best off. Here
    the devices re-plan one at a time, each against the others' CURRENT plans
    (at the tariff plus the final grid-limit prices), and a re-plan is kept
    only if `total_objective` falls. So it is monotone: it cannot undo what
    the rounds found. Updates `res` in place; returns whether anything changed.
    """
    h, dt = cfg.horizon, cfg.horizon.dt
    batt = {SiteConfig.battery_key(i): b for i, b in enumerate(cfg.battery_list)}

    def objective(devices) -> tuple[float, np.ndarray, np.ndarray]:
        net = fc.net_fixed_demand + sum(devices[k].power for k in dev_keys)
        net, curtail = apply_curtailment(net, fc.solar, fc.sell, cfg.grid)
        soe = {k: devices[k].trajectory for k in dev_keys if k in batt} or None
        obj = total_objective(cfg, net, fc,
                              devices["water_heater"].trajectory if "water_heater" in devices else None,
                              devices["hvac"].trajectory if "hvac" in devices else None, soe)
        return obj, net, curtail

    cur, net, curtail = objective(res.devices)
    changed = False
    for _ in range(max(sweeps, 0)):
        improved = False
        for k in dev_keys:
            others = fc.net_fixed_demand + sum(res.devices[j].power for j in dev_keys if j != k)
            if k in batt:
                sol = solve_battery(batt[k], h, buy, sell, dp_load=others, admm_rho=0.0,
                                    soc_gates=cfg.soc_gates if k == "battery" else (), limits=limits)
            elif k == "water_heater":
                sol = solve_water_heater(cfg.water_heater, h, buy, sell, fc.hot_water_demand,
                                         dp_load=others, limits=limits)
            else:
                sol = solve_hvac(cfg.hvac, h, buy, sell, fc.outdoor_temp, dp_load=others,
                                 limits=limits)
            trial = dict(res.devices)
            trial[k] = sol
            obj, t_net, t_curtail = objective(trial)
            if obj < cur - 1e-9:
                res.devices, cur, net, curtail = trial, obj, t_net, t_curtail
                improved = changed = True
        if not improved:
            break
    if not changed:
        return False
    res.net_grid, res.curtailment = net, curtail
    res.curtailed_kwh = float(curtail.sum() * dt)
    res.import_cost = float(np.sum(np.maximum(net, 0.0) * fc.buy * dt))
    res.export_revenue = float(np.sum(np.maximum(-net, 0.0) * fc.sell * dt))
    res.net_cost = res.import_cost - res.export_revenue
    res.total_objective = cur
    if cfg.grid.max_import_kw is not None:
        res.grid_import_excess = float(np.maximum(net - cfg.grid.max_import_kw, 0.0).max())
    if cfg.grid.max_export_kw is not None:
        res.grid_export_excess = float(np.maximum(-net - cfg.grid.max_export_kw, 0.0).max())
    if "battery" in res.devices:
        # the pricing re-solve is conditioned on the others' FINAL plans
        res.battery_dp_load = fc.net_fixed_demand + sum(
            res.devices[j].power for j in dev_keys if j != "battery")
    return True


def _refresh_selected_record(
    rec: RoundRecord, res: CoordinationResult, cfg: SiteConfig
) -> None:
    """Re-point a round record at the plan that is actually being returned.

    Only used for the retained round, and only after the baseline fallback has
    had its say. The rejected rounds are left exactly as they were scored -
    they are a record of what the loop tried, not of what it would have
    returned.
    """
    violation = 0.0
    if cfg.grid.active:
        if cfg.grid.max_import_kw is not None:
            violation += float(
                np.maximum(res.net_grid - cfg.grid.max_import_kw, 0.0).sum()
            )
        if cfg.grid.max_export_kw is not None:
            violation += float(
                np.maximum(-res.net_grid - cfg.grid.max_export_kw, 0.0).sum()
            )
    rec.powers = {k: sol.power.copy() for k, sol in res.devices.items()}
    rec.trajectories = {k: sol.trajectory.copy() for k, sol in res.devices.items()}
    rec.net_grid = res.net_grid.copy()
    if res.curtailment is not None:
        rec.curtailment = res.curtailment.copy()
    rec.import_cost = res.import_cost
    rec.export_revenue = res.export_revenue
    rec.net_cost = res.net_cost
    rec.total_objective = res.total_objective
    rec.import_excess = res.grid_import_excess
    rec.export_excess = res.grid_export_excess
    rec.violation = violation
    rec.fallback_applied = True


def _pricing_resolve(cfg: SiteConfig, fc: Forecasts, res: CoordinationResult,
                     limits: Limits | None = None) -> DeviceSolution:
    """Re-solve the battery on pure economics, holding the other devices fixed.

    The value function produced INSIDE the coordination loop carries the ADMM
    proximity term -(rho/2)(a - target)^2, which is internal machinery, not
    money. Differentiating it gives a "price" polluted by rho: at rho = 5 and a
    10 kW action span that term reaches ~60 currency, swamping a 0.30/kWh
    tariff and producing nonsense marginal values.

    So the pricing/execution tier gets its own solve: same dp_load as the
    winning round (i.e. the other devices' agreed plans), but rho = 0. dV/ds is
    then an honest marginal value of stored energy, CONDITIONAL on those plans
    - which is exactly what the price signal claims to be, and no more.
    """
    dp_load = (
        res.battery_dp_load
        if res.battery_dp_load is not None
        else fc.net_fixed_demand
    )
    return solve_battery(
        cfg.battery,
        cfg.horizon,
        fc.buy,
        device_sell_price(fc.sell, cfg.grid),
        dp_load=dp_load,
        admm_rho=0.0,
        soc_gates=cfg.soc_gates,
        limits=limits,
    )


def _apply_baseline_fallback(
    cfg: SiteConfig, fc: Forecasts, res: CoordinationResult, dev_keys: list[str]
) -> bool:
    """Safety net: if a device's thermostat baseline beats its optimised plan
    on TOTAL objective, keep the baseline.

    Compares the objective rather than the bill so we never buy a smaller bill
    with a comfort violation. Iterates because swapping one device changes the
    net grid the others are scored against.

    Returns whether any device was actually swapped, so the caller can tell a
    retained plan that survived the check from one that was rewritten by it.
    """
    n = cfg.horizon.steps
    dt = cfg.horizon.dt

    batt_keys = [k for k in dev_keys if k.startswith("battery")]

    def rebuild(overrides: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Net grid for a candidate swap, curtailed the same way a round is.

        Skipping curtailment here silently produced a result whose net_grid
        breached the export cap while grid_export_excess read 0.000, because the
        two were computed from different arrays.
        """
        net = fc.net_fixed_demand.copy()
        for k in dev_keys:
            net = net + overrides.get(k, res.devices[k].power)
        return apply_curtailment(net, fc.solar, fc.sell, cfg.grid)

    swapped = False
    changed = True
    while changed:
        changed = False
        soes = {k: res.devices[k].trajectory for k in batt_keys if k in res.devices}
        cur_obj = total_objective(
            cfg,
            res.net_grid,
            fc,
            res.devices["water_heater"].trajectory if "water_heater" in res.devices else None,
            res.devices["hvac"].trajectory if "hvac" in res.devices else None,
            soes or None,
        )

        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        for idx, key in enumerate(batt_keys):
            b = cfg.battery_list[idx]
            # An idle battery is only a valid fallback when nothing obliges it
            # to move: SoC gates and p_demand are enforced inside the DP, and
            # doing nothing would violate them.
            if (key == "battery" and cfg.soc_gates) or b.min_charge_kw or b.soc_goal_kwh:
                continue
            idle_soe = np.full(n + 1, b.capacity_kwh * b.soc_initial_frac)
            candidates.append((key, np.zeros(n), idle_soe))
        if "water_heater" in dev_keys:
            temp, pw = baseline_water_heater(cfg.water_heater, cfg.horizon, fc.hot_water_demand)
            candidates.append(("water_heater", pw, temp))
        if "hvac" in dev_keys:
            temp, pw = baseline_hvac(cfg.hvac, cfg.horizon, fc.outdoor_temp)
            candidates.append(("hvac", pw, temp))

        for key, alt_power, alt_traj in candidates:
            alt_net, alt_curtail = rebuild({key: alt_power})
            trajs = {k: res.devices[k].trajectory for k in res.devices}
            trajs[key] = alt_traj
            alt_obj = total_objective(
                cfg,
                alt_net,
                fc,
                trajs.get("water_heater"),
                trajs.get("hvac"),
                {k: trajs[k] for k in batt_keys if k in trajs} or None,
            )
            if alt_obj < cur_obj - 1e-12:
                res.devices[key].power = alt_power
                res.devices[key].trajectory = alt_traj
                res.net_grid = alt_net
                res.curtailment = alt_curtail
                res.curtailed_kwh = float(alt_curtail.sum() * dt)
                if cfg.grid.max_import_kw is not None:
                    res.grid_import_excess = float(
                        np.maximum(alt_net - cfg.grid.max_import_kw, 0.0).max()
                    )
                if cfg.grid.max_export_kw is not None:
                    res.grid_export_excess = float(
                        np.maximum(-alt_net - cfg.grid.max_export_kw, 0.0).max()
                    )
                res.import_cost = float(np.sum(np.maximum(alt_net, 0.0) * fc.buy * dt))
                res.export_revenue = float(np.sum(np.maximum(-alt_net, 0.0) * fc.sell * dt))
                res.net_cost = res.import_cost - res.export_revenue
                res.total_objective = alt_obj
                changed = swapped = True
                break

    return swapped
