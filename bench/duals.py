"""Three independent ways to price a stored kWh, so they can be compared.

The claim under test is `poc-integration-analysis.md` §2.2: that dV/ds from the
DP is a usable marginal value of stored energy. Two things could break it, and
they are different failures:

* **Is the DP's dV/ds the true shadow price at all?** Settled against an LP:
  formulate the same problem with explicit SoE variables and read the dual of
  the state-transition equality. That multiplier is the discrete costate, i.e.
  exactly d(optimal cost)/d(energy at t), computed by HiGHS to solver accuracy.

* **Does it survive decomposition?** Settled against the joint DP: compare the
  decomposed battery's dV/ds (solved with the other devices as exogenous load)
  against dV/ds of the exact joint value function evaluated on the realised
  trajectory.

Sign convention: the DP maximises reward, so V is a value and dV/ds > 0 means
"more stored energy is better". The LP minimises cost, so its costate carries
the opposite sign; `lp_costate` flips it so everything in this module is a
positive currency/kWh price directly comparable to `hemspolicy.marginal_value`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from hemspolicy.dp_battery import terminal_price
from hemspolicy.interp import interp_grid
from hemspolicy.types import BatteryConfig, Horizon


def lp_battery_with_duals(
    cfg: BatteryConfig,
    horizon: Horizon,
    buy: np.ndarray,
    sell: np.ndarray,
    dp_load: np.ndarray,
    soe_initial: float | None = None,
) -> dict:
    """Battery LP with explicit SoE variables, returning the transition duals.

    `lp_battery` folds the SoE trajectory into cumulative sums over the power
    variables, which is compact but leaves no constraint whose dual is the
    marginal value of energy. Here the state is an explicit variable and the
    recursion an explicit equality:

        soe[t+1] - soe[t] - (eta*c[t] - d[t]/eta)*dt = 0        (dual mu[t])

    Perturbing that constraint's RHS by eps is exactly "hand the battery eps
    extra kWh at t+1", so -mu[t] is the marginal value of stored energy - the
    same object the DP computes as dV/ds.

    Variable layout: [c(n), d(n), gp(n), gn(n), soe(n+1)]
    """
    n = horizon.steps
    dt = horizon.dt
    cap = cfg.capacity_kwh
    eta_c, eta_d = cfg.eta_c, cfg.eta_d
    if np.any(sell > buy + 1e-12):
        raise ValueError("lp_battery_with_duals requires sell <= buy")

    soe0 = cap * cfg.soc_initial_frac if soe_initial is None else soe_initial
    tp = terminal_price(cfg, buy)

    C, D, GP, GN, SOE = 0, n, 2 * n, 3 * n, 4 * n
    nvar = 4 * n + (n + 1)

    obj = np.zeros(nvar)
    obj[GP : GP + n] = buy * dt
    obj[GN : GN + n] = -sell * dt
    obj[SOE + n] = -tp  # terminal value of energy left in the battery

    rows_eq, rhs_eq = [], []

    # soe[0] pinned
    r = np.zeros(nvar)
    r[SOE] = 1.0
    rows_eq.append(r)
    rhs_eq.append(soe0)

    transition_rows = []
    for t in range(n):
        r = np.zeros(nvar)
        r[SOE + t + 1] = 1.0
        r[SOE + t] = -1.0
        r[C + t] = -eta_c * dt
        r[D + t] = dt / eta_d
        transition_rows.append(len(rows_eq))
        rows_eq.append(r)
        rhs_eq.append(0.0)

    meter_rows = []
    for t in range(n):
        r = np.zeros(nvar)
        r[GP + t], r[GN + t] = 1.0, -1.0
        r[C + t], r[D + t] = -1.0, 1.0
        meter_rows.append(len(rows_eq))
        rows_eq.append(r)
        rhs_eq.append(dp_load[t])

    bounds = (
        [(0.0, cfg.p_charge_max_kw)] * n
        + [(0.0, cfg.p_discharge_max_kw)] * n
        + [(0.0, None)] * n
        + [(0.0, None)] * n
        + [(0.0, cap)] * (n + 1)
    )

    res = linprog(
        obj,
        A_eq=np.array(rows_eq),
        b_eq=np.array(rhs_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    marg = np.asarray(res.eqlin.marginals)
    mu = marg[np.array(transition_rows)]  # costate on the SoE recursion
    meter_dual = marg[np.array(meter_rows)]  # marginal cost of net demand

    soe = res.x[SOE : SOE + n + 1]
    power = res.x[C : C + n] - res.x[D : D + n]

    return {
        "power": power,
        "soe": soe,
        # -mu is the marginal VALUE of energy, matching the DP's dV/ds sign.
        "lambda": -mu,
        # Marginal cost of one more kWh of demand AT THE METER. Not negated:
        # the sign flip is correct for the SoE costate above (free energy
        # REDUCES cost) and wrong here (extra demand INCREASES it). Verified by
        # finite difference in tests/test_meter_price.py.
        "meter_lambda": meter_dual / dt,
        "objective": float(res.fun) + tp * soe0,
        "raw_mu": mu,
    }


def lp_costate(*args, **kwargs) -> np.ndarray:
    """Convenience: just the marginal value of stored energy, length n."""
    return lp_battery_with_duals(*args, **kwargs)["lambda"]


def joint_costate(
    joint: dict, horizon: Horizon, at_temperature: np.ndarray | None = None
) -> np.ndarray:
    """dV/ds of the exact JOINT value function, along the realised trajectory.

    The joint value function is V[t, s, T]. Its partial derivative in s is the
    true marginal value of stored energy *for the coupled system* - the object
    the decomposed lambda is claiming to approximate. It is evaluated at the
    tank temperature the joint solution actually visits, because that is the
    state the price would be quoted in.
    """
    V = joint["value"]
    S = joint["states"]
    T = joint["temps"]
    soe = joint["soe"]
    temp = at_temperature if at_temperature is not None else joint["temp"]

    n = horizon.steps
    out = np.zeros(n)
    for t in range(n):
        # dV/ds on the full grid at this step, then bilinear-sample it at the
        # realised (soe, tank temperature).
        grad_s = np.gradient(V[t], S, axis=0)  # (ns, nt)
        ti = float(np.clip(temp[t], T[0], T[-1]))
        j = int(np.clip(np.searchsorted(T, ti) - 1, 0, len(T) - 2))
        w = (ti - T[j]) / (T[j + 1] - T[j] + 1e-12)
        col = grad_s[:, j] * (1 - w) + grad_s[:, j + 1] * w
        out[t] = float(interp_grid(np.array([np.clip(soe[t], S[0], S[-1])]), S, col)[0])
    return out


def decomposed_costate(snap, soe_traj: np.ndarray) -> np.ndarray:
    """dV/ds from the decomposed battery solve, along a given trajectory."""
    from hemspolicy import marginal_value

    n = snap.horizon.steps
    return np.array([marginal_value(snap, t, float(soe_traj[t])) for t in range(n)])


def agreement(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Summary statistics for comparing two price series."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    w = w / max(w.sum(), 1e-12)
    denom = np.std(b) if np.std(b) > 1e-12 else 1.0
    corr = (
        float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 1e-12 and np.std(b) > 1e-12
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(d))),
        "weighted_mae": float(np.sum(w * np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "corr": corr,
        "mae_over_sd": float(np.mean(np.abs(d)) / denom),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
    }
