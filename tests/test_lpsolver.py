"""src/home_energy_optimizer/dw/lpsolver.py: the numpy-only master solver, checked against HiGHS."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_energy_optimizer.dw.lpsolver import Triplets, choose_one, linprog  # noqa: E402

scipy_opt = pytest.importorskip("scipy.optimize")


def random_lp(seed):
    rng = np.random.default_rng(seed)
    m, n = int(rng.integers(5, 50)), int(rng.integers(40, 160))
    A = rng.normal(size=(m, n)) * (rng.random((m, n)) < 0.3)
    x0 = rng.random(n)
    b = A @ x0
    lb = np.where(rng.random(n) < 0.2, -rng.random(n) * 3, 0.0)
    ub = np.where(rng.random(n) < 0.5, x0 + rng.random(n) * 2, np.inf)
    ub = np.where(rng.random(n) < 0.05, lb, ub)          # some fixed variables
    ub = np.maximum(ub, lb)
    x0 = np.clip(x0, lb, ub); b = A @ x0
    c = rng.normal(size=n)
    c = np.where(np.isfinite(ub), c, np.abs(c) + 0.1)
    T = Triplets((m, n))
    r, k = np.nonzero(A)
    T[r, k] = A[r, k]
    return c, A, T, b, lb, ub


@pytest.mark.parametrize("seed", range(40))
def test_matches_highs_on_random_bounded_lps(seed):
    c, A, T, b, lb, ub = random_lp(seed)
    ref = scipy_opt.linprog(c, A_eq=A, b_eq=b, bounds=list(zip(lb, ub)), method="highs")
    if ref.status != 0:
        pytest.skip("reference not optimal")
    res = linprog(c, T, b, lb, ub)
    assert res.status == 0
    assert abs(res.fun - ref.fun) <= 1e-6 * (1 + abs(ref.fun))
    assert np.max(np.abs(A @ res.x - b)) <= 1e-6 * (1 + np.abs(b).max())
    assert np.all(res.x >= lb - 1e-7) and np.all(res.x <= ub + 1e-7)
    # the duals certify the objective (strong duality, via the dual bound)
    reduced = c - A.T @ res.y
    dual = b @ res.y + np.sum(np.where(reduced > 0, reduced * lb,
                                       np.where(np.isfinite(ub), reduced * ub, 0.0)))
    assert abs(dual - res.fun) <= 1e-5 * (1 + abs(res.fun))


def test_choose_one_finds_the_best_combination():
    """Two groups of three plans, one balance row: enumerate and compare."""
    rng = np.random.default_rng(3)
    n_rows = 4
    plans = [rng.normal(size=(3, n_rows)) for _ in range(2)]
    costs = [rng.random(3) for _ in range(2)]
    # vars: 3 + 3 plan weights, then slack+ and slack- per row (priced)
    nv = 6 + 2 * n_rows
    T = Triplets((n_rows + 2, nv))
    c = np.zeros(nv); c[:3], c[3:6] = costs; c[6:] = 1.0
    for g in range(2):
        for j in range(3):
            T[np.arange(n_rows), 3 * g + j] = plans[g][j]
        T[n_rows + g, np.arange(3 * g, 3 * g + 3)] = 1.0
    T[np.arange(n_rows), 6 + np.arange(n_rows)] = 1.0
    T[np.arange(n_rows), 6 + n_rows + np.arange(n_rows)] = -1.0
    b = np.concatenate([np.full(n_rows, 0.3), [1.0, 1.0]])
    lb, ub = np.zeros(nv), np.concatenate([np.ones(6), np.full(2 * n_rows, np.inf)])
    res = choose_one(c, T, b, lb, ub, [np.arange(3), np.arange(3, 6)])
    best = min(costs[0][i] + costs[1][j] + np.abs(0.3 - plans[0][i] - plans[1][j]).sum()
               for i in range(3) for j in range(3))
    assert res.optimal
    assert abs(res.fun - best) < 1e-6
