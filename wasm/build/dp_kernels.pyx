# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
dp_kernels.pyx -- compiled backward recursions for hems-policy.

C translations of the inner loops of `solve_battery`, `solve_water_heater` and
`solve_hvac`. The Python in src/hemspolicy remains the reference and the
fallback; these must stay numerically IDENTICAL to it, which
build/verify_kernels.py asserts against the shipped fixtures.

Numerical identity is the whole contract here, and three details carry it:

  * `np.argmax` and `Q > best` both keep the FIRST maximum on a tie. The loops
    below compare with strict `>` while scanning ascending, so ties resolve the
    same way. Relaxing that to `>=` silently changes which action is stored on
    every flat stretch of the value function.
  * `interp_grid` does `searchsorted(S, x) - 1`. `_searchsorted_left` below is
    that function, not an equivalent computed from the grid spacing -- S is a
    linspace, but reconstructing the index arithmetically drifts by one ULP at
    the knots and picks the other cell.
  * `interp_uniform` takes `frac.astype(int)`, which truncates toward zero, not
    floor. A C cast does the same; `floor()` would not, for negative frac.

Arrays cross as typed memoryviews (`double[::1]`) rather than
`cnp.ndarray[double, ndim=1]`: both take a C-contiguous float64 array, but the
buffer syntax pulls in the numpy C-API import machinery and a per-argument
dtype check. The wheel is base64'd into an HTML file, so that is size that
buys nothing.

EVERY buffer here is float64, including the two policy arrays that hold action
INDICES. That is not tidiness: `long` is 64-bit on the machines this is
developed on and 32-bit on wasm32, so a `long[:, ::1]` policy buffer accepted
an int64 array natively and failed in the browser with "Buffer dtype mismatch,
expected 'long' but got 'long long'". Sizing every buffer explicitly removes
the class of bug. The callers convert back to int64 after the call.

That failure is also why verify_kernels.py is not sufficient on its own: it
runs natively, where the type happened to line up. The cross-compiled wheel
has to be exercised in the browser.

The kernels deliberately cover the DEFAULT configuration only. SoC gates,
per-slot charge floors, soc goals and the charger deadband are evcc features
the browser testbed never sets; rather than carry four more branches through
the hot loop, the Python side keeps them and only dispatches here when none is
in play. `battery_backward` returns 0 if it is handed one anyway.
"""

import numpy as np
from libc.math cimport fabs

cdef double INF = float("inf")


cdef inline double _clip(double v, double lo, double hi) noexcept nogil:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline int _searchsorted_left(double[::1] a, int n, double v) noexcept nogil:
    """np.searchsorted(a, v, side='left'): first i with a[i] >= v."""
    cdef int lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline double _interp_grid(double[::1] xp, double[::1] fp,
                                int n, double x) noexcept nogil:
    """One point of interp.interp_grid: clamped, never extrapolated."""
    cdef int idx = _searchsorted_left(xp, n, x) - 1
    cdef double t
    if idx < 0:
        idx = 0
    elif idx > n - 2:
        idx = n - 2
    t = (x - xp[idx]) / (xp[idx + 1] - xp[idx] + 1e-12)
    t = _clip(t, 0.0, 1.0)
    return fp[idx] * (1.0 - t) + fp[idx + 1] * t


cdef inline double _interp_uniform(double[:, ::1] V, int row, int n,
                                   double lo, double hi, double at) noexcept nogil:
    """One point of interp.interp_uniform on row `row` of V, weight clamped."""
    cdef double frac = (at - lo) / (hi - lo) * (n - 1)
    cdef int idx = <int>frac          # truncates toward zero, as astype(int) does
    cdef int nxt
    cdef double w
    if idx < 0:
        idx = 0
    elif idx > n - 2:
        idx = n - 2
    w = _clip(frac - idx, 0.0, 1.0)
    nxt = idx + 1
    if nxt > n - 1:
        nxt = n - 1
    return V[row, idx] * (1.0 - w) + V[row, nxt] * w


def battery_backward(double[::1] buy, double[::1] sell, double[::1] dp_load,
                     double[::1] admm_target, double[::1] S, double[::1] A,
                     double[:, ::1] V, double[:, ::1] POL,
                     int n, int ns, int na,
                     double dt, double cap, double eta_c, double eta_d,
                     double admm_rho, int use_admm):
    """Fill V[0..n-1] and POL from V[n]. Mirrors solve_battery's loop body."""
    cdef int t, s, a, best_a
    cdef double sv, headroom, lo, ac, eff, s_next, v_next
    cdef double imp, reward, q, best_q

    with nogil:
        for t in range(n - 1, -1, -1):
            for s in range(ns):
                sv = S[s]
                headroom = (cap - sv) / (eta_c * dt)
                lo = -sv * eta_d / dt
                best_q = -INF
                best_a = 0
                for a in range(na):
                    ac = _clip(A[a], lo, headroom)
                    if ac > 0.0:
                        eff = ac * eta_c
                    else:
                        eff = ac / eta_d
                    s_next = _clip(sv + eff * dt, 0.0, cap)
                    v_next = _interp_grid(S, V[t + 1], ns, s_next)

                    imp = ac + dp_load[t]
                    if imp > 0.0:
                        reward = -buy[t] * imp * dt
                    else:
                        reward = sell[t] * (-imp) * dt
                    if use_admm:
                        reward = reward - (admm_rho / 2.0) * \
                            (ac - admm_target[t]) * (ac - admm_target[t]) * dt

                    q = reward + v_next
                    if q > best_q:      # strict: first max wins, as np.argmax does
                        best_q = q
                        best_a = a
                V[t, s] = best_q
                # recompute the stored action from the winning index, exactly
                # as the Python takes Ac[rows, best]
                POL[t, s] = _clip(A[best_a], lo, headroom)
    return 1


def water_heater_backward(double[::1] buy, double[::1] sell, double[::1] dp_load,
                          double[::1] rate, double[::1] t_inf, double[::1] T,
                          double[:, ::1] V, double[:, ::1] POL,
                          int n, int ns,
                          double dt, double C, double power_kw,
                          double t_min, double t_max, double t_comfort,
                          double price_k):
    """Fill V[0..n-1] and POL from V[n]. Mirrors solve_water_heater's loop.

    `rate` and `t_inf` are the per-slot relaxation constants the Python
    computes in `_relaxation`; passing them in keeps the mixing-temperature
    physics in one place rather than duplicating it in C.
    """
    cdef int t, s, ai, best_a
    cdef double tv, q_out, duty_cap, duty, q_heat, dT, t_next
    cdef double v_next, imp, cost, comfort, q, best_q, short_

    with nogil:
        for t in range(n - 1, -1, -1):
            for s in range(ns):
                tv = T[s]
                q_out = C * rate[t] * (tv - t_inf[t])
                duty_cap = _clip(((t_max - tv) * C / dt + q_out) / power_kw, 0.0, 1.0)
                best_q = -INF
                best_a = 0
                for ai in range(2):             # off, on
                    duty = 0.0 if ai == 0 else 1.0
                    if duty > duty_cap:
                        duty = duty_cap
                    q_heat = power_kw * duty
                    dT = (q_heat - q_out) / C * dt
                    t_next = tv + dT
                    v_next = _interp_uniform(V, t + 1, ns, t_min, t_max, t_next)

                    imp = q_heat + dp_load[t]
                    if imp > 0.0:
                        cost = -buy[t] * imp * dt
                    else:
                        cost = sell[t] * (-imp) * dt

                    short_ = t_comfort - t_next
                    if short_ < 0.0:
                        short_ = 0.0
                    comfort = -(price_k * short_ * dt)

                    q = cost + comfort + v_next
                    if q > best_q:
                        best_q = q
                        best_a = ai
                V[t, s] = best_q
                POL[t, s] = best_a
    return 1


def hvac_backward(double[::1] buy, double[::1] sell, double[::1] dp_load,
                  double[::1] outdoor, double[::1] T,
                  double[:, ::1] V, double[:, ::1] POL,
                  int n, int ns,
                  double dt, double c_room, double r_wall,
                  double power_kw, double cop,
                  double t_min, double t_max,
                  double t_lo, double t_hi, double price_k):
    """Fill V[0..n-1] and POL from V[n]. Mirrors solve_hvac's loop.

    Actions are (0, -1, +1) in that order -- off, cool, heat -- and the tie
    rule keeps the first, so "off" wins a tie against either. Reordering them
    changes the stored policy on flat stretches even though the value is the
    same.
    """
    cdef int t, s, ai, best_a
    cdef double tv, q_wall, a, q_flow, limit, room_kw, duty, q_ac
    cdef double dT, t_next, v_next, imp, elec, comfort, q, best_q, hot, cold

    with nogil:
        for t in range(n - 1, -1, -1):
            for s in range(ns):
                tv = T[s]
                q_wall = (outdoor[t] - tv) / r_wall
                best_q = -INF
                best_a = 0
                for ai in range(3):
                    if ai == 0:
                        a = 0.0
                    elif ai == 1:
                        a = -1.0
                    else:
                        a = 1.0

                    if a == 0.0:
                        duty = 1.0
                        q_flow = 0.0
                    else:
                        if a < 0.0:
                            q_flow = power_kw * cop * a
                            limit = t_min
                        else:
                            q_flow = power_kw * (cop + 1.0) * a
                            limit = t_max
                        room_kw = (limit - tv) * c_room / dt - q_wall
                        duty = _clip(room_kw / q_flow, 0.0, 1.0)
                    q_ac = q_flow * duty

                    dT = (q_wall + q_ac) / c_room * dt
                    t_next = tv + dT
                    v_next = _interp_uniform(V, t + 1, ns, t_min, t_max, t_next)

                    imp = power_kw * fabs(a) * duty + dp_load[t]
                    if imp > 0.0:
                        elec = -buy[t] * imp * dt
                    else:
                        elec = sell[t] * (-imp) * dt

                    hot = t_next - t_hi
                    if hot < 0.0:
                        hot = 0.0
                    cold = t_lo - t_next
                    if cold < 0.0:
                        cold = 0.0
                    comfort = -(price_k * (hot + cold) * dt)

                    q = elec + comfort + v_next
                    if q > best_q:
                        best_q = q
                        best_a = ai
                V[t, s] = best_q
                POL[t, s] = best_a
    return 1
