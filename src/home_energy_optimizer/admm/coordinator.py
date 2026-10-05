"""ADMM: proximal message passing on the house, the device solvers as its steps.

This is the ADMM of Kraning, Chu, Lavaei & Boyd ("Dynamic Network Energy
Management via Proximal Message Passing", Foundations and Trends in
Optimization 1(2), 2013), in exchange form, on one net - the house. Its
terminals are the household load (fixed), the PV (curtailable, at no cost),
the grid connection (import at the buy price, export at the sell price, energy
beyond a grid limit at the breach price) and every device. Each iteration

  1. every device, in parallel, takes its cheapest plan near a target,
         p_d <- argmin f_d(p) + (rho/2) |p - (p_d - pbar - u)|^2,
     the grid connection in closed form, the PV by clipping, and the
     batteries, tank and HVAC by their own DPs with a tether and no bill (the
     grid connection carries the bill) - or, as an option, each plain battery
     by its LP, solved exactly (admm.battery_qp);
  2. the net averages the imbalance and moves the price: u <- u + pbar;
  3. rho adapts by the paper's proportional-derivative rule, then is held.

Optionally (`exchange_momentum`) the targets and prices are extrapolated with
Nesterov weights, dropped for an iteration whenever the combined residual grows
- fast ADMM with restart (Goldstein, O'Donoghue, Setzer & Baraniuk, 2014).

No device sees the others' plans: each responds only to the shared price and
its own last plan, which is what lets several devices move together. For
convex devices with exact steps the cost and the prices converge to optimal
ones.

The on/off tank and three-way HVAC are not convex. As the paper prescribes,
the iterations use relaxed copies that may run a fraction of each slot
(`CoordinationConfig.relax_levels`). Every iteration also recovers a runnable
plan - each relaxed device re-plans as itself against the others' current
plans, two quick DP solves - and the best runnable plan seen is kept, reported
as it improves, and polished at the end (each device in turn re-plans, kept
only if the objective falls).

Dynamic programmes and closed-form steps only, unless the batteries take LP
steps (a small interior point method in numpy).

Sub-meters (types.SubMeter: a hybrid inverter, a shared breaker) make the
house a network, as in the paper: each sub-meter's bus is a net of its own,
with its own price, holding its member devices (and the PV, for a hybrid
inverter); its connection to the house is a two-terminal device, one terminal
on each net, that sends x from its bus (eta_export x reaching the house) or
takes y from the house (y / eta_import leaving it), within its ratings and at
the breach price beyond them. Its proximal step is closed form (`_Link`).
Every plan is scored by submeter.site_meter, as Dantzig-Wolfe's are.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from home_energy_optimizer.coordinate import (
    _apply_baseline_fallback,
    _polish,
    _pricing_resolve,
    baseline_solution,
    device_terms,
    set_battery_view,
    breach_energy,
    breach_price,
    device_sell_price,
    net_cost,
    total_objective,
)
from home_energy_optimizer.dp_battery import solve_battery, terminal_price
from home_energy_optimizer.dp_thermal import hvac_discomfort, solve_hvac, solve_water_heater
from home_energy_optimizer.meter import Limits
from home_energy_optimizer.submeter import MAIN, beyond_main, device_bus, pv_on, root_grid, site_meter, tariff, topology
from home_energy_optimizer.types import (
    BatteryConfig,
    CoordinationResult,
    DeviceSolution,
    Forecasts,
    HvacConfig,
    RoundRecord,
    SiteConfig,
    SocGate,
    SubMeter,
    WaterHeaterConfig,
)

from .battery_qp import BatteryStep, battery_prox, lp_step_applies

_ThermalT = TypeVar("_ThermalT", WaterHeaterConfig, HvacConfig)


class _Grid:
    """The grid connection as a terminal. p = power delivered TO the grid
    (-import). Its cost, in the import z = -p, per slot:
        dt (buy z+ - sell z- + c_br (z - L_imp)+ + c_br (-z - L_exp)+),
    with the kink at z = 0 optionally rounded off over |z| < delta
    (`kink_smoothing`): there the bill is sell z + (buy - sell)(z + delta)^2 / (4 delta),
    which equals the exact bill outside the band and makes the price at a
    balanced meter unique. The prox is closed form: on each piece the
    stationary point, clipped to the piece; the cheapest of those."""

    def __init__(self, buy: npt.ArrayLike, sell: npt.ArrayLike, dt: float,
                 l_imp: float | None, l_exp: float | None, c_br: float,
                 delta: float) -> None:
        self.buy, self.sell, self.dt = np.asarray(buy, float), np.asarray(sell, float), dt
        self.l_imp, self.l_exp, self.c_br = l_imp, l_exp, c_br
        lims = [x for x in (l_imp, l_exp) if x is not None]
        self.delta = min(delta, 0.5 * min(lims)) if lims and delta > 0 else max(delta, 0.0)

    def cost(self, z: np.ndarray) -> np.ndarray:
        return self._cost(z, self.buy, self.sell)

    def cost_t(self, z: npt.ArrayLike, t: int) -> np.ndarray:
        """The cost of slot t at every import in the array z."""
        return self._cost(np.asarray(z, float), self.buy[t], self.sell[t])

    def _cost(self, z: np.ndarray, buy: np.ndarray | float,
              sell: np.ndarray | float) -> np.ndarray:
        d = self.delta
        if d > 0:
            plus = np.where(z >= d, z, np.where(z <= -d, 0.0, (z + d) ** 2 / (4 * d)))
        else:
            plus = np.maximum(z, 0.0)
        c = sell * z + (buy - sell) * plus
        if self.l_imp is not None:
            c = c + self.c_br * np.maximum(z - self.l_imp, 0.0)
        if self.l_exp is not None:
            c = c + self.c_br * np.maximum(-z - self.l_exp, 0.0)
        return c * self.dt

    def prox(self, v: np.ndarray, rho: float) -> np.ndarray:
        # in the import z = -p the target is -v: min cost(z) + (rho/2)(z + v)^2
        d, dt = self.delta, self.dt
        edges = sorted({-d, d} | ({-self.l_exp} if self.l_exp is not None else set())
                       | ({self.l_imp} if self.l_imp is not None else set()))
        edges = [-np.inf] + edges + [np.inf]
        best, best_val = None, None
        for a, b in zip(edges[:-1], edges[1:]):
            if a == b:
                continue
            if d > 0 and a == -d and b == d:                       # the rounded kink
                k = dt * (self.buy - self.sell) / (2 * d)
                z = -(dt * self.sell + k * d + rho * v) / (k + rho)
            else:
                mid = b - 1.0 if a == -np.inf else (a + 1.0 if b == np.inf else 0.5 * (a + b))
                slope = (self.cost(np.full_like(v, mid + 1e-6)) - self.cost(np.full_like(v, mid - 1e-6))) / 2e-6
                z = -v - slope / rho
            z = np.clip(z, a, b)
            val = self.cost(z) + 0.5 * rho * (z + v) ** 2
            if best is None:
                best, best_val = z, val
            else:
                m = val < best_val
                best, best_val = np.where(m, z, best), np.where(m, val, best_val)
        assert best is not None    # the pieces cover the line, so one was cheapest
        return -best


class _Link:
    """A sub-meter's connection as a two-terminal device: terminal 0 on its
    bus, terminal 1 on the house's AC bus, each the power it withdraws from
    its net. Sending s >= 0 from the bus: (s, -eta_export s); taking -s > 0:
    (s, -s / eta_import). Beyond a rating, at the breach price. The prox is
    closed form: on each of the four pieces of s the stationary point,
    clipped to the piece; the cheapest of those."""

    def __init__(self, sm: SubMeter, dt: float, c_br: float) -> None:
        self.e, self.i = sm.eta_export, sm.eta_import
        self.cap_e, self.cap_i = sm.export_cap_dc, sm.import_cap_dc
        self.c, self.dt = c_br, dt

    def terminals(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return s, np.where(s >= 0, -self.e * s, -s / self.i)

    def cost(self, s: np.ndarray) -> np.ndarray:
        return self.c * self.dt * (np.maximum(s - self.cap_e, 0.0) + np.maximum(-s - self.cap_i, 0.0))

    def prox(self, vb: np.ndarray, va: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
        pieces = [(-np.inf, -self.cap_i, -1.0 / self.i, -self.c * self.dt), (-self.cap_i, 0.0, -1.0 / self.i, 0.0),
                  (0.0, self.cap_e, -self.e, 0.0), (self.cap_e, np.inf, -self.e, self.c * self.dt)]
        best = best_val = None
        for a, b, k, lin in pieces:
            if not a < b:
                continue
            s = np.clip((vb + k * va - lin / rho) / (1.0 + k * k), a, b)
            pb, pa = self.terminals(s)
            val = self.cost(s) + 0.5 * rho * ((pb - vb) ** 2 + (pa - va) ** 2)
            if best is None or best_val is None:
                best, best_val = s, val
            else:
                m = val < best_val
                best, best_val = np.where(m, s, best), np.where(m, val, best_val)
        assert best is not None
        return self.terminals(best)


def _objective(cfg: SiteConfig, fc: Forecasts, net: np.ndarray,
               sols: Mapping[str, BatteryStep | DeviceSolution], batteries: Iterable[str],
               ref: float) -> float:
    """A plan's cost as the testbed reports it: total_objective plus the
    thermal devices' horizon-edge terms their DPs optimise (as
    dw.coordinator.thermal_terminal and DWCoordinator.parts have it)."""
    wh = sols["water_heater"].trajectory if "water_heater" in sols else None
    hv = sols["hvac"].trajectory if "hvac" in sols else None
    soe = {k: sols[k].trajectory for k in batteries if k in sols} or None
    obj = total_objective(cfg, net, fc, wh, hv, soe)
    wcfg, hcfg = cfg.water_heater, cfg.hvac      # set whenever their plans are
    if wh is not None and wcfg is not None and wcfg.comfort_mode == "linear":
        obj += wcfg.heat_capacity_kwh_per_k * ref * max(0.0, wcfg.t_comfort - float(wh[-1]))
    if hv is not None and hcfg is not None and hcfg.comfort_mode == "linear":
        obj += float(hvac_discomfort(hcfg, np.array([float(hv[-1])]), ref, 1.0, *hcfg.band_at(-1)).item())
    return float(obj)


def _relaxed(cfg: _ThermalT | None, levels: int, dt: float) -> _ThermalT | None:
    """A copy of an on/off device allowed `levels` duty steps, its grid
    refined to match - or the device itself if it is already that fine."""
    if cfg is None or levels <= cfg.n_duty_levels:
        return cfg
    r = replace(cfg, n_duty_levels=levels)
    return replace(r, n_states=r.states_for_duty_levels(dt))


@dataclass
class WarmStart:
    """Where an ADMM solve stood at its best runnable plan: every terminal's
    plan, the scaled price and rho, and each LP battery step's solver state.
    A later solve given it (`ExchangeRun(..., warm=...)`) starts there instead
    of from zero, if its terminals and horizon are the same."""

    keys: tuple                # the devices, in terminal order after load, PV and grid
    p: np.ndarray              # (terminals, slots)
    u: np.ndarray              # the scaled price, per slot
    rho: float
    qp: dict                   # battery key -> battery_qp solver state

    def shift(self, k: int) -> "WarmStart":
        """The same state k slots later, for a rolling horizon: the first k
        slots dropped, the last one repeated."""
        if k <= 0:
            return self

        def roll(a: np.ndarray) -> np.ndarray:
            return np.concatenate([a[..., k:], np.repeat(a[..., -1:], k, axis=-1)], axis=-1)

        qp = {}
        for key, (x, y, zl, zu) in self.qp.items():
            n = y.size
            blocks = lambda z: np.concatenate([roll(z[i * n:(i + 1) * n]) for i in range(3)])  # noqa: E731
            qp[key] = (blocks(x), roll(y), blocks(zl), blocks(zu))
        return WarmStart(self.keys, roll(self.p), roll(self.u), self.rho, qp)


def coordinate_exchange(cfg: SiteConfig, fc: Forecasts,
                        progress: Callable[..., None] | None = None,
                        warm: WarmStart | None = None) -> CoordinationResult:
    """Plan the site by ADMM (see the module docstring)."""
    run = ExchangeRun(cfg, fc, progress, warm)
    run.step()
    return run.result()


class ExchangeRun:
    """An ADMM solve that can stop after any iteration and carry on.

    `step(n)` runs up to n more iterations (all that remain by default) and
    says whether the run has finished; `result()` is the best runnable plan so
    far, polished - the answer if finished, a snapshot to look at if paused.
    Nothing `result()` does changes the iteration state, so a paused run
    resumes exactly where it stopped.
    """

    def __init__(self, cfg: SiteConfig, fc: Forecasts,
                 progress: Callable[..., None] | None = None,
                 warm: WarmStart | None = None) -> None:
        cfg.validate()
        fc.validate(cfg.horizon)
        self.cfg, self.fc, self.progress = cfg, fc, progress
        cc, h = cfg.coordination, cfg.horizon
        self.cc, self.h = cc, h
        n, dt = h.steps, h.dt
        self.n, self.dt = n, dt
        self.ref = float(np.mean(fc.buy))
        zero = np.zeros(n)
        self.zero = zero
        g = cfg.grid
        self.g = g
        c_br = breach_price(g, fc.buy, fc.sell) if g.active else 0.0
        self.limits = Limits(g.max_import_kw, g.max_export_kw, c_br, g.allow_curtailment) if g.active else None
        self.sell_dev = device_sell_price(fc.sell, g)
        self.max_rounds = cc.exchange_rounds

        # ---- the terminals --------------------------------------------------
        self.grid = _Grid(fc.buy, fc.sell, dt, g.max_import_kw, g.max_export_kw, c_br, cc.kink_smoothing)
        def dp_step(b: BatteryConfig, gates: tuple[SocGate, ...]
                    ) -> Callable[[np.ndarray, float], DeviceSolution]:
            def step(v: np.ndarray, r: float) -> DeviceSolution:
                return solve_battery(b, h, zero, zero, dp_load=zero, admm_target=v, admm_rho=r / dt,
                                     soc_gates=gates)
            return step

        keys: list[str] = []
        steps: list[Callable[[np.ndarray, float], BatteryStep | DeviceSolution]] = []
        real: dict[str, BatteryConfig] = {}
        self._qp: dict = {}                                 # LP battery steps' solver states, for warm starts
        for i, b in enumerate(cfg.battery_list):
            if b.capacity_kwh > 0:
                key = SiteConfig.battery_key(i)
                pinned = replace(b, terminal_price=terminal_price(b, fc.buy))  # the prox sees no prices
                gates = cfg.soc_gates if key == "battery" else ()
                keys.append(key)
                if cc.exchange_battery_step == "lp" and lp_step_applies(b) and not gates:
                    steps.append(partial(self._battery_lp, key, pinned))
                else:
                    steps.append(dp_step(pinned, gates))
                real[key] = b
        ref = self.ref
        wh = _relaxed(cfg.water_heater, cc.relax_levels, dt)
        if wh is not None:
            keys.append("water_heater")
            steps.append(lambda v, r: solve_water_heater(wh, h, zero, zero, fc.hot_water_demand, dp_load=zero,
                                                         admm_target=v, admm_rho=r / dt, ref_price=ref))
        hv = _relaxed(cfg.hvac, cc.relax_levels, dt)
        if hv is not None:
            keys.append("hvac")
            steps.append(lambda v, r: solve_hvac(hv, h, zero, zero, fc.outdoor_temp, dp_load=zero,
                                                 admm_target=v, admm_rho=r / dt, ref_price=ref))
        self.keys, self.steps, self.real = keys, steps, real
        self.relaxed_cfg = replace(cfg, water_heater=wh, hvac=hv)
        self.relaxed_keys = [k for k in ("water_heater", "hvac")
                             if k in keys and getattr(cfg, k) is not getattr(self.relaxed_cfg, k)]
        # The nets: every bus of the site's tree - the main meter's (0), each
        # further grid connection's, each node's. Rows of p: the load, one
        # per PV array, one grid terminal per connection, the devices, then
        # each node's link: two terminals, its own net and its parent's.
        topo = topology(cfg, fc)
        net = {b: i for i, b in enumerate(topo.buses)}
        self.pv_rows = [(1 + i, key) for i, key in enumerate(fc.pv_keys)]
        g0 = 1 + len(self.pv_rows)
        self.grid_rows: list[tuple[int, _Grid]] = []
        for i, root in enumerate(topo.roots):
            if root == MAIN:
                self.grid_rows.append((g0 + i, self.grid))
            else:
                rg = root_grid(cfg, root)
                buy_r, sell_r = tariff(cfg, fc, root)
                self.grid_rows.append((g0 + i, _Grid(buy_r, sell_r, dt, rg.max_import_kw, rg.max_export_kw,
                                                     breach_price(rg, buy_r, sell_r) if rg.active else 0.0,
                                                     cc.kink_smoothing)))
        self.dev0 = g0 + len(topo.roots)
        self.links = [_Link(topo.sm[x], dt, breach_price(g, fc.buy, fc.sell)) for x in topo.nodes]
        net_of = [net[MAIN]] + [net[cfg.bus_of(key)] for _, key in self.pv_rows] + [net[r] for r in topo.roots]
        net_of += [net[cfg.bus_of(kk)] for kk in keys]
        for x in topo.nodes:
            net_of += [net[x], net[topo.parent[x]]]
        # Each set limit is a net of its own: its limit is a terminal there
        # (free within it, the breach price beyond), and each member device
        # has a second terminal on it, tied to its first - so its step pulls
        # toward the mean of its terminals' targets, at rho per terminal.
        nb = len(topo.buses)
        self.limit_rows: list[tuple[int, _Grid]] = []
        for i, lim in enumerate(cfg.set_limits):
            self.limit_rows.append((len(net_of), _Grid(zero, zero, dt, lim.max_import_kw, lim.max_export_kw,
                                                       breach_price(g, fc.buy, fc.sell), 0.0)))
            net_of.append(nb + i)
        self.member_rows: dict[int, list[int]] = {}
        for j, kk in enumerate(keys):
            for i, lim in enumerate(cfg.set_limits):
                if kk in lim.members:
                    self.member_rows.setdefault(j, []).append(len(net_of))
                    net_of.append(nb + i)
        self.net_of = np.array(net_of)
        self.n_nets = nb + len(cfg.set_limits)
        self.count = np.bincount(self.net_of, minlength=self.n_nets).astype(float)
        # PV each PV terminal may leave unused: any of it if curtailment is
        # allowed; on a node, otherwise, what its rating forces.
        self.pv_spare = {}
        for _, key in self.pv_rows:
            where = cfg.bus_of(key)
            pv = fc.pv(key)
            if g.allow_curtailment:
                self.pv_spare[key] = pv.copy()
            elif where in topo.sm:
                share = pv / np.maximum(pv_on(fc, topo, where), 1e-12)      # its share of what the rating forces
                self.pv_spare[key] = share * np.maximum(pv_on(fc, topo, where) - topo.sm[where].export_cap_dc, 0.0)
            else:
                self.pv_spare[key] = np.zeros(n)
        self.N = len(net_of)

        # ---- iteration state ------------------------------------------------
        self.p = np.zeros((self.N, n))                      # rows: load, PV, grid, the devices, the links
        self.p[0] = fc.load
        self.u = np.zeros((self.n_nets, n))                 # the scaled price, per net
        self.rho, self.w_prev = cc.exchange_rho, 0.0
        self.pbar = self._means(self.p)
        self.prev_dev = self.p - self.pbar[self.net_of]
        self.z_hat, self.u_hat = self.prev_dev.copy(), self.u.copy()     # momentum
        self.alpha, self.c_prev = 1.0, np.inf
        self.eps = cc.exchange_eps * np.sqrt(self.N * n)
        self.records: list[RoundRecord] = []
        self.best_sols: dict[str, BatteryStep | DeviceSolution] | None = None
        self.best_obj, self.best_round, self.last_improve = np.inf, 0, 0
        self.stop_reason = "iteration cap"
        self.k = 0                                          # iterations run
        self.done = False
        self.seconds = 0.0                                  # time spent iterating, across steps
        self._best_warm: WarmStart | None = None
        self.warm_used = False
        if (warm is not None and tuple(warm.keys) == tuple(keys) and warm.p.shape == self.p.shape
                and warm.u.size == self.u.size):
            self.p = warm.p.copy()
            self.p[0] = fc.load                             # the load is data, not a plan
            self.u, self.rho = warm.u.copy().reshape(self.n_nets, n), float(warm.rho)
            self.pbar = self._means(self.p)
            self.prev_dev = self.p - self.pbar[self.net_of]
            self.z_hat, self.u_hat = self.prev_dev.copy(), self.u.copy()
            if cc.exchange_warm_battery:
                self._qp = {kk: tuple(a.copy() for a in st) for kk, st in warm.qp.items()}
            self.warm_used = True

    def _means(self, p: np.ndarray) -> np.ndarray:
        """Each net's average terminal power, per slot: (nets, slots)."""
        out = np.zeros((self.n_nets, p.shape[1]))
        np.add.at(out, self.net_of, p)
        return out / self.count[:, None]

    def _meter(self, sols: Mapping[str, BatteryStep | DeviceSolution]) -> tuple[np.ndarray, np.ndarray, float]:
        """(meter flow, curtailment, sub-meter breach cost) of the devices' plans."""
        flows = site_meter(self.cfg, self.fc, {kk: sols[kk].power for kk in self.keys if kk in sols})
        return flows.net, flows.curtail, beyond_main(self.cfg, self.fc, flows)

    def _battery_lp(self, key: str, b: BatteryConfig, v: np.ndarray, rho: float) -> BatteryStep:
        """A battery's exact LP step, started from its previous solution."""
        assert b.terminal_price is not None    # pinned when the step was built
        sol = battery_prox(b, self.dt, v, rho, float(b.terminal_price),
                           start=self._qp.get(key) if self.cc.exchange_warm_battery else None)
        self._qp[key] = sol.state
        return sol

    def warm_state(self) -> WarmStart | None:
        """Where this run stood at its best runnable plan, to start a later
        solve from (`WarmStart.shift` for a horizon that has moved on)."""
        return self._best_warm

    def _as_itself(self, key: str, powers: Mapping[str, np.ndarray]) -> DeviceSolution:
        """The real (on/off, three-way) device's plan against the others' `powers`."""
        cfg, h, fc = self.cfg, self.h, self.fc
        view = device_bus(cfg, fc, key, powers)
        bus, others = view.bus, view.dp_load
        buy, sell, lim = ((fc.buy, self.sell_dev, self.limits) if view.root == MAIN
                          else device_terms(view.buy, view.sell, view.grid))
        if key == "water_heater":
            assert cfg.water_heater is not None
            return solve_water_heater(cfg.water_heater, h, buy, sell, fc.hot_water_demand,
                                      dp_load=others, limits=lim, bus=bus)
        assert cfg.hvac is not None
        return solve_hvac(cfg.hvac, h, buy, sell, fc.outdoor_temp, dp_load=others, limits=lim, bus=bus)

    def step(self, n_iter: int | None = None) -> bool:
        """Run up to `n_iter` more iterations (all that remain if None)."""
        cfg, cc, fc, g, dt = self.cfg, self.cc, self.fc, self.g, self.dt
        keys, N, progress = self.keys, self.N, self.progress
        todo = self.max_rounds - self.k if n_iter is None else min(n_iter, self.max_rounds - self.k)
        t_step = time.perf_counter()
        sols = {}
        for _ in range(max(todo, 0)):
            if self.done:
                break
            k = self.k
            t_round = time.perf_counter()
            if progress is not None:
                progress(k + 1, self.max_rounds)
            p, pbar, u, at = self.p, self.pbar, self.u, self.net_of
            v = (self.z_hat - self.u_hat[at]) if cc.exchange_momentum else (p - pbar[at] - u[at])
            new = np.empty_like(p)
            new[0] = fc.load
            for row, key in self.pv_rows:
                pv = fc.pv(key)
                new[row] = np.clip(v[row], -pv, -pv + self.pv_spare[key])
            for row, grid in self.grid_rows:
                new[row] = grid.prox(v[row], self.rho)
            dev_ms = {}
            for j, (key, stepf) in enumerate(zip(keys, self.steps)):
                rows = [self.dev0 + j] + self.member_rows.get(j, [])
                # one plan for all its terminals: the mean target, rho per terminal
                sols[key] = stepf(v[rows].mean(axis=0) if len(rows) > 1 else v[rows[0]], self.rho * len(rows))
                new[rows] = sols[key].power
                dev_ms[key] = sols[key].solve_ms
            for row, limit in self.limit_rows:
                new[row] = limit.prox(v[row], self.rho)
            for li, link in enumerate(self.links):
                r0 = self.dev0 + len(keys) + 2 * li
                new[r0], new[r0 + 1] = link.prox(v[r0], v[r0 + 1], self.rho)
            p = new
            pbar = self._means(p)
            u_old = u
            u = (self.u_hat if cc.exchange_momentum else u) + pbar
            dev = p - pbar[at]
            cnt = self.count[:, None]
            r_res = float(np.sqrt(np.sum(cnt * pbar ** 2)))
            s_res = float(self.rho * np.linalg.norm(dev - self.prev_dev))
            if cc.exchange_momentum:
                c_k = self.rho * (float(np.sum(cnt * (u - self.u_hat) ** 2)) + float(np.sum((dev - self.z_hat) ** 2)))
                if c_k < 0.999 * self.c_prev:
                    a_next = (1.0 + np.sqrt(1.0 + 4.0 * self.alpha * self.alpha)) / 2.0
                    wgt = (self.alpha - 1.0) / a_next
                    self.z_hat, self.u_hat = dev + wgt * (dev - self.prev_dev), u + wgt * (u - u_old)
                    self.alpha, self.c_prev = a_next, c_k
                else:                                   # restart: no extrapolation this time
                    self.alpha, self.c_prev = 1.0, self.c_prev / 0.999
                    self.z_hat, self.u_hat = dev.copy(), u.copy()
            self.prev_dev = dev
            self.p, self.pbar, self.u = p, pbar, u

            # the relaxed plan's cost (the devices' plans, the grid taking the balance)
            net, _, extra = self._meter(sols)
            relaxed_obj = _objective(self.relaxed_cfg, fc, net, sols, self.real, self.ref) + extra
            # a runnable plan from this iteration: each relaxed device as itself
            run = dict(sols)
            for key in self.relaxed_keys:
                run[key] = self._as_itself(key, {jj: run[jj].power for jj in keys})
                dev_ms[key] += run[key].solve_ms
            net, curtail, extra = self._meter(run)
            obj = _objective(cfg, fc, net, run, self.real, self.ref) + extra
            self.records.append(RoundRecord(
                index=k, powers={kk: run[kk].power.copy() for kk in keys},
                trajectories={kk: run[kk].trajectory.copy() for kk in keys},
                net_grid=net.copy(), curtailment=curtail.copy(),
                import_cost=float(np.sum(np.maximum(net, 0.0) * fc.buy * dt)),
                export_revenue=float(np.sum(np.maximum(-net, 0.0) * fc.sell * dt)),
                net_cost=net_cost(net, fc.buy, fc.sell, dt), total_objective=obj, score_objective=obj,
                violation=float(np.sum(breach_energy(cfg, net)) / dt),
                import_excess=(float(np.maximum(net - g.max_import_kw, 0.0).max())
                               if g.max_import_kw is not None else 0.0),
                export_excess=(float(np.maximum(-net - g.max_export_kw, 0.0).max())
                               if g.max_export_kw is not None else 0.0),
                primal_res=r_res, dual_res=s_res, rho=self.rho,
                round_ms=(time.perf_counter() - t_round) * 1000.0, device_ms=dev_ms,
                battery_dp_load=(device_bus(cfg, fc, "battery", {jj: run[jj].power for jj in keys}).dp_load
                                 if "battery" in run else None),
                relaxed_objective=relaxed_obj))
            if not np.isfinite(self.best_obj) or obj < self.best_obj - cc.converge_tol * abs(self.best_obj):
                self.last_improve = k
            if obj < self.best_obj:
                self.best_obj, self.best_round, self.best_sols = obj, k, run
                self._best_warm = WarmStart(tuple(keys), self.p.copy(), self.u.copy(), self.rho,
                                            {kk: tuple(a.copy() for a in st) for kk, st in self._qp.items()})
            if progress is not None:
                progress(k + 1, self.max_rounds, float(obj), None, float(self.best_obj))
            self.k = k + 1

            if r_res <= self.eps and s_res <= self.eps:
                self.stop_reason, self.done = "converged", True
                break
            if k - self.last_improve >= cc.exchange_patience:
                self.stop_reason, self.done = "no improvement", True
                break
            if k < cc.exchange_rho_freeze and s_res > 0:        # the paper's rho rule, then held
                w = self.rho * r_res / s_res - 1.0
                new_rho = self.rho * float(np.exp(cc.exchange_rho_gain * (w + (w - self.w_prev))))
                self.w_prev = w
                self.u = self.u * (self.rho / new_rho)           # the scaled price follows rho
                self.u_hat = self.u.copy()
                self.z_hat, self.alpha, self.c_prev = self.prev_dev.copy(), 1.0, np.inf   # momentum restarts
                self.rho = new_rho
        if self.k >= self.max_rounds:
            self.done = True
        self.seconds += time.perf_counter() - t_step
        return self.done

    def result(self) -> CoordinationResult:
        """The best runnable plan so far, polished."""
        cfg, cc, fc, g, dt, keys = self.cfg, self.cc, self.fc, self.g, self.dt, self.keys
        if self.progress is not None and self.done:
            self.progress(-1, self.max_rounds)
        if self.best_sols is None:
            raise RuntimeError("ADMM: no iteration has run yet")
        devices = dict(self.best_sols)
        net, curtail, extra = self._meter(devices)
        soe = {kk: devices[kk].trajectory for kk in keys if kk in self.real} or None
        obj = total_objective(cfg, net, fc, devices["water_heater"].trajectory if "water_heater" in devices else None,
                              devices["hvac"].trajectory if "hvac" in devices else None, soe) + extra
        records = list(self.records)
        for r in records:
            r.selected = r.index == self.best_round
        res = CoordinationResult(
            devices=devices, net_grid=net,
            import_cost=float(np.sum(np.maximum(net, 0.0) * fc.buy * dt)),
            export_revenue=float(np.sum(np.maximum(-net, 0.0) * fc.sell * dt)),
            net_cost=net_cost(net, fc.buy, fc.sell, dt), total_objective=obj,
            rounds_run=len(records), rounds=records, selected_round=self.best_round,
            stop_reason=self.stop_reason if self.done else "paused",
            grid_import_excess=(float(np.maximum(net - g.max_import_kw, 0.0).max())
                                if g.max_import_kw is not None else 0.0),
            grid_export_excess=(float(np.maximum(-net - g.max_export_kw, 0.0).max())
                                if g.max_export_kw is not None else 0.0),
            curtailment=curtail, curtailed_kwh=float(curtail.sum() * dt), method="admm",
            warm_start=self.warm_state())
        if cc.enable_baseline_fallback:
            _apply_baseline_fallback(cfg, fc, res, keys)
        if cc.polish:
            _polish(cfg, fc, res, keys, fc.buy, self.sell_dev, cc.polish_sweeps, self.limits)
        if "battery" in res.devices:
            set_battery_view(res, device_bus(cfg, fc, "battery", {j: res.devices[j].power for j in keys}))
            res.battery_pricing = _pricing_resolve(cfg, fc, res, self.limits)
        res.baseline_cost = baseline_solution(cfg, fc)[1]
        return res
