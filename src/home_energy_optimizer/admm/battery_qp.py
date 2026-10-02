"""An exact ADMM step for a battery: the prox of its LP model, by a small
interior point method (numpy only).

ADMM's step for a device is prox_{f,rho}(v) = argmin f(p) + (rho/2)|p - v|^2.
For a battery modelled as the Dantzig-Wolfe master models it - charge c and
discharge e per slot, stored energy s after each slot, bounds on all three,
the energy left at the end worth its terminal price - that step is a convex QP:

    min   (rho/2) |c - e - v|^2  -  tau s_n
    s.t.  s_t - s_{t-1} - dt eta_c c_t + (dt / eta_d) e_t = 0      (s_0 given)
          0 <= c <= Pc,   0 <= e <= Pd,   floor <= s <= cap

Solved by a primal-dual interior point method with Mehrotra's
predictor-corrector. The Hessian is block diagonal - a 2x2 block per slot for
(c_t, e_t) - and each dynamics row touches one slot's c and e and two
consecutive s, so the Newton system reduces to a tridiagonal one in the n
dynamics duals: O(n) work per iteration.

As in the master, charging and discharging in the same slot is allowed; it
only wastes energy, so an optimum uses it only where wasting energy costs
nothing.

Warm start: `start` is a previous solve's state (`BatteryStep.state`). An
interior point method needs a point inside the bounds, so the previous plan is
moved 5% of each range off its bounds and its bound multipliers are floored;
between consecutive ADMM iterations that saves about a third of the
iterations, and the answer is the same to solver tolerance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from home_energy_optimizer.types import BatteryConfig


@dataclass
class BatteryStep:
    """What the ADMM loop reads from a device step."""

    power: np.ndarray          # c - e, kW per slot
    trajectory: np.ndarray     # stored energy, length n + 1
    solve_ms: float
    iterations: int
    converged: bool = True
    overlap_kwh: float = 0.0   # energy charged and discharged in the same slots (wasted in losses)
    state: tuple | None = None # (x, y, zl, zu): the solver's final point, to warm-start the next solve


def lp_step_applies(b: BatteryConfig) -> bool:
    """Plain batteries only: a charger minimum is not convex, and goals and
    SoC gates are left to the DP, which prices them."""
    return (b.p_charge_max_kw > 0 and b.p_discharge_max_kw > 0 and b.capacity_kwh > b.soe_floor_kwh
            and b.terminal_mode == "linear"
            and not getattr(b, "min_charge_kw", 0) and not getattr(b, "soc_goal_kwh", None))


def _thomas(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a symmetric tridiagonal system; off[i] couples i and i + 1."""
    n = diag.size
    c, d = np.empty(n), np.empty(n)
    c[0], d[0] = (off[0] / diag[0] if n > 1 else 0.0), rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - off[i - 1] * c[i - 1]
        c[i] = off[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - off[i - 1] * d[i - 1]) / m
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def battery_prox(b: BatteryConfig, dt: float, v: np.ndarray, rho: float, tau: float,
                 tol: float = 1e-9, max_iter: int = 100, start: tuple | None = None) -> BatteryStep:
    """argmin over the battery's LP plans of (rho/2)|p - v|^2 - tau * (energy at the end)."""
    t0 = time.perf_counter()
    v = np.asarray(v, float)
    n = v.size
    ac, ae = -dt * b.eta_c, dt / b.eta_d              # the dynamics row's c and e coefficients
    s_init = b.capacity_kwh * b.soc_initial_frac
    lo = np.concatenate([np.zeros(2 * n), np.full(n, b.soe_floor_kwh)])
    hi = np.concatenate([np.full(n, b.p_charge_max_kw), np.full(n, b.p_discharge_max_kw),
                         np.full(n, b.capacity_kwh)])
    q = np.zeros(3 * n)
    q[:n], q[n:2 * n] = -rho * v, rho * v             # (rho/2)|c - e - v|^2 = (rho/2)|c - e|^2 - rho v.(c - e) + const
    q[3 * n - 1] = -tau
    rhs_b = np.zeros(n)
    rhs_b[0] = s_init

    def A(x: np.ndarray) -> np.ndarray:               # A x
        c, e, s = x[:n], x[n:2 * n], x[2 * n:]
        out = ac * c + ae * e + s
        out[1:] -= s[:-1]
        return out

    def At(y: np.ndarray) -> np.ndarray:              # A' y
        s = y.copy()
        s[:-1] -= y[1:]
        return np.concatenate([ac * y, ae * y, s])

    def Qx(x: np.ndarray) -> np.ndarray:
        d = rho * (x[:n] - x[n:2 * n])
        return np.concatenate([d, -d, np.zeros(n)])

    if start is not None and start[0].shape == lo.shape:
        m = 0.05 * (hi - lo)                          # a previous point, moved off its bounds
        x, y = np.clip(start[0], lo + m, hi - m), start[1].copy()
        zl, zu = np.maximum(start[2], 1e-2), np.maximum(start[3], 1e-2)
    else:
        x = 0.5 * (lo + hi)
        y = np.zeros(n)
        zl, zu = np.ones(3 * n), np.ones(3 * n)
    it = 0
    for it in range(1, max_iter + 1):
        wl, wu = x - lo, hi - x
        grad = Qx(x) + q - At(y)                      # dual residual is grad - zl + zu
        rd = grad - zl + zu
        rp = A(x) - rhs_b
        mu = (wl @ zl + wu @ zu) / (6 * n)
        if (np.abs(rp).max() <= tol * (1 + abs(s_init)) and np.abs(rd).max() <= tol * (1 + np.abs(q).max())
                and mu <= tol):
            break
        sig = zl / wl + zu / wu                       # the barrier's diagonal
        hcc, hee = rho + sig[:n], rho + sig[n:2 * n]
        det = hcc * hee - rho * rho
        icc, iee, ice = hee / det, hcc / det, rho / det
        hs = 1.0 / (sig[2 * n:] + 1e-10)              # s has no curvature of its own: keep the inverse finite
        diag = ac * ac * icc + 2 * ac * ae * ice + ae * ae * iee + hs
        diag[1:] += hs[:-1]
        off = -hs[:-1]

        def hinv(r: np.ndarray) -> np.ndarray:
            rc, re, rs = r[:n], r[n:2 * n], r[2 * n:]
            return np.concatenate([icc * rc + ice * re, ice * rc + iee * re, hs * rs])

        def newton(tl: np.ndarray, tu: np.ndarray) -> tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            r1 = -grad + tl / wl - tu / wu
            dy = _thomas(diag, off, -rp - A(hinv(r1)))
            dx = hinv(r1 + At(dy))
            dzl = (-wl * zl + tl - zl * dx) / wl
            dzu = (-wu * zu + tu + zu * dx) / wu
            return dx, dy, dzl, dzu

        def step(dx: np.ndarray, dzl: np.ndarray, dzu: np.ndarray) -> float:
            a = 1.0
            for w, d in ((wl, dx), (wu, -dx), (zl, dzl), (zu, dzu)):
                neg = d < 0
                if neg.any():
                    a = min(a, float(np.min(-w[neg] / d[neg])))
            return a

        # predictor
        dx, dy, dzl, dzu = newton(np.zeros(3 * n), np.zeros(3 * n))
        a = step(dx, dzl, dzu)
        mu_aff = ((wl + a * dx) @ (zl + a * dzl) + (wu - a * dx) @ (zu + a * dzu)) / (6 * n)
        sigma = (mu_aff / mu) ** 3
        # corrector
        dx, dy, dzl, dzu = newton(sigma * mu - dx * dzl, sigma * mu + dx * dzu)
        a = min(1.0, 0.995 * step(dx, dzl, dzu))
        x, y, zl, zu = x + a * dx, y + a * dy, zl + a * dzl, zu + a * dzu
    c, e, s = x[:n], x[n:2 * n], x[2 * n:]
    traj = np.concatenate([[s_init], np.clip(s, b.soe_floor_kwh, b.capacity_kwh)])
    return BatteryStep(power=c - e, trajectory=traj, solve_ms=(time.perf_counter() - t0) * 1000.0,
                       iterations=it, converged=it < max_iter,
                       overlap_kwh=float(np.minimum(c, e).clip(min=0.0).sum() * dt), state=(x, y, zl, zu))
