"""Vectorised linear interpolation onto a monotone grid.

Lifted verbatim in behaviour from the POC's `lin_interp_vec`. Kept as its own
module because both the battery DP and the policy replay depend on it being
exactly the same function - any divergence silently changes the value function
between solve time and lookup time.
"""

from __future__ import annotations

import numpy as np


def interp_grid(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Interpolate fp(xp) at points x, preserving x's shape.

    xp must be sorted ascending. Values of x outside [xp[0], xp[-1]] are
    clamped to the endpoints rather than extrapolated, which is what the DP
    needs: a state that falls off the grid is physically clipped anyway.
    """
    shape = x.shape
    x_flat = np.asarray(x, dtype=np.float64).ravel()
    idx = np.searchsorted(xp, x_flat) - 1
    idx = np.clip(idx, 0, len(xp) - 2)
    t = (x_flat - xp[idx]) / (xp[idx + 1] - xp[idx] + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    result = fp[idx] * (1 - t) + fp[idx + 1] * t
    return result.reshape(shape)


def interp_uniform(
    values: np.ndarray, lo: float, hi: float, n: int, at: np.ndarray,
    extrapolate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Index+weight pair for interpolating a uniform grid of n points on [lo, hi].

    Returns (idx, w) such that ``values[idx] * (1 - w) + values[idx + 1] * w``
    is the interpolated result. Split out from the thermal DPs, which computed
    it inline three times each.

    With ``extrapolate=True`` the weight is left unclipped, so a point outside
    [lo, hi] continues along the line through the two nearest grid values
    rather than flattening onto the endpoint. That matters wherever the value
    being interpolated is a COST: clamping makes the cost stop growing once the
    state leaves the grid, so the recursion sees no reason to avoid leaving it
    further - the same saturation that made a projected temperature hide
    discomfort. Extrapolating keeps the marginal penalty alive at the edge.

    It is linear extrapolation off the last two points, so it is trustworthy
    just outside the grid and progressively less so far outside.

    Default OFF, and deliberately unused by the thermal DPs: once the flows are
    bounded (dp_thermal._usable_outflow, _max_duty) the state provably cannot
    leave the grid, so enabling it there changes nothing measurable and would
    only put an unvalidated linear guess on a path nothing takes. It exists for
    replay from an externally MEASURED state, which can genuinely arrive from
    outside the modelled range.
    """
    frac = (at - lo) / (hi - lo) * (n - 1)
    idx = np.clip(frac.astype(int), 0, n - 2)
    w = frac - idx
    return (idx, w) if extrapolate else (idx, np.clip(w, 0.0, 1.0))
