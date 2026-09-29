# Dantzig–Wolfe coordination: an alternative to the ADMM loop

This folder holds a column-generation coordinator, kept separate from the ADMM
coordinator in `src/hemspolicy/coordinate.py` so that the two can be run side by
side on the same instances. Nothing under `src/` is modified.

```
dw/coordinator.py   the DW coordinator (master LP + the existing DPs as pricing oracles)
dw/compare.py       ADMM vs DW vs exact references, one objective, one table per scenario
dw/webapi.py        JSON backend for the DW app (same payload shape as hemspolicy.webapi)
dw/gui/             the DW app: server.py (stdlib HTTP, port 8766) + index.html
tests/test_dw.py    bound ordering, parity with ADMM, exact grid limit (skips without scipy)
```

```bash
.venv/bin/pip install scipy            # optional: only for HiGHS cross-checks, bench/ and the MILP reference
.venv/bin/python dw/compare.py         # scenarios A-F, ~4 min
.venv/bin/python dw/compare.py --only A --milp   # adds the HiGHS MILP reference (~70 s more)
.venv/bin/python -m pytest tests/test_dw.py
.venv/bin/python dw/gui/server.py     # the DW app on http://127.0.0.1:8766 (numpy only)
.venv/bin/python gui/server.py        # the ADMM app on http://127.0.0.1:8765, for comparison
```

The app is the DW counterpart of `gui/` (the ADMM app on port 8765), and both
can run at once. It shows:

- **An iteration strip**, in place of ADMM's rounds. Click an iteration to plot
  that iteration's master solution: the convex mix of each device's plans,
  with fractional devices flagged. **Σ** is the final relaxed master and **★**
  the recovered, implementable plan.
- **A convergence chart**: master value, running lower bound, the recovered
  plan's objective and the final certified bound, with the pool size on the
  right axis.
- **A price panel**: the tariff, the master's meter price π, and battery 1's λ
  from the master's SoE duals, with its no-trade band. Hover it to read π, λ,
  and whether the meter is importing, exporting, on the kink, or held at a limit.
- **The DW settings as controls**: battery in master, smoothing, heuristic
  columns, polish, recovery method and iteration cap, so the §5 D ablation can
  be reproduced by hand.

- **Tank element**: on/off (as the ADMM app and the bench references model
  it), or 3–13 duty levels per slot (`WaterHeaterConfig.n_duty_levels`). The
  tank's temperature grid is refined to resolve one duty step
  (`states_for_duty_levels`); without that, a 13-level plan cost 0.18 more
  than its DP claimed. A modulating tank keeps its blend at recovery: the
  mixed duty is re-simulated through the tank physics and the tank's weights
  stay continuous in the recovery MILP. Only HVAC (mode blends) and on/off
  devices are still rounded to one plan. Measured on 2 batteries + tank + HVAC
  with a 7 kW import limit: on/off −1.243, quarters −1.376, twelfths −1.402
  (dynamic tariff); 1.377 → 1.218 → 1.194 (day/night). The gap to the bound
  falls from 0.02–0.03 to about 0.001.

Deep links work the same way as in `gui/`, plus `view=relaxed|<iteration>`, e.g.
`/?n_batteries=2&max_import_kw=3&tariff=day_night&solar_peak=9&view=relaxed`.
One feature is deliberately left out: the ADMM app's hover policy replay and
forced-action pricing (they need a DP snapshot at the final π; see §7b).

**Short version.** Handle the meter in a small linear programme. Keep the DPs
for the devices, and give each one a single price vector to respond to. The
recovered plans capture **98.4–99.8% of the savings that can be proven
available**. ADMM captures **88.3–97.5%** on the same instances, and DW runs in
comparable or less time (0.3–1.0 s against 0.04–3.7 s). Every DW solve also
reports a lower bound. That gives a certificate the ADMM loop lacks, and it can
be applied to ADMM's plans too. The cost: DW needs an LP/MILP solver
(HiGHS via scipy), and the certificate is exact only up to the device DPs'
own discretisation (§6.3).

---

## 0. Where the last thread left off

> **Since replaced.** The ADMM loop described in this section and compared in
> §6 (Jacobi rounds against the others' last plans, a z-step on the kink, dual
> ascent on limit multipliers) has been removed. The package's ADMM is now
> proximal message passing (`hemspolicy.exchange`); `docs/theory.tex` describes
> both coordinators as they are and compares them.

The previous sessions finished with PR #8: compiled DP kernels and a wasm
wheel, 3.3× faster in the browser. Before that the coordinator got two things.
The GUI can now show every round. Grid breaches are priced into the objective
($\kappa_g = 10$) so rounds are ranked on one number. The coordinator now:

1. warm-starts each device from a free ($\rho = 0$) solve;
2. runs Jacobi rounds: each device's DP sees the others' last plans as load
   (`dp_load`), and the battery also gets a proximal term $\tfrac\rho2(a-\tilde p+\nu)^2$;
3. applies a closed-form z-step on the meter kink and a scaled dual update, with adaptive $\rho$;
4. runs projected dual ascent on grid-limit multipliers $\mu$ (step $\zeta$, ≥40 rounds when a limit is set);
5. keeps the best round by objective, applies the thermostat fallback, then
   re-solves the battery at $\rho = 0$ so $\lambda$ is not polluted by $\rho$.

The known weak points are in `docs/NOTES.md` §6 and `theory.tex` §"What is guaranteed":

- **No optimality certificate.** The cost of coordination has been measured
  against references, never bounded.
- **Binary devices get no proximal term**, so they have no convergence story.
- **Grid limits rely on a step-size heuristic.** It needs 13–40 rounds and fails
  near the physical floor.
- **λ is conditional** on the other devices' plans. It is not a system-wide dual.

Dantzig–Wolfe addresses each of these directly (measured in §5, argued in §6.2).

---

## 1. What the 1960 paper says

Dantzig & Wolfe, *Decomposition Principle for Linear Programs*, Operations
Research 8(1):101–111 (1960). Notation below is the paper's.

**Structure.** The paper splits an LP into *parts* $L_t: A_t X_t = b_t$,
coupled by a few *linking* rows $\sum_t A_t' X_t + P_0 x_0 = b$. Its
"angular system" (eq. 37) is exactly this shape.

**Reformulation.** Each part's feasible set is a polytope, so any feasible
$X_t$ is a convex combination of its extreme points $X_{ti}$. The coordinating
programme (eq. 41) picks weights $\lambda_{ti} \ge 0$ with
$\sum_i \lambda_{ti} = 1$ per part, so that the linking rows hold at least cost.
Its rows are the $m$ linking rows plus one *convexity* row per part. **It is
small however large the parts are.**

**Pricing.** The coordinator's simplex multipliers are $(\pi, s_1, \dots)$:
prices on the linking rows and a scalar per part. The paper calls these scalars
"a special subsidy for each part that just balances the cost". The part
receives the objective $\pi A_t' X_t$ and solves its own LP (eq. 44). If
$\min z_t < s_t$ for some part, its minimiser is a better "proposal" and enters
the basis. Otherwise the current solution is optimal (Theorem 1).

**Termination.** The process is finite, because there are finitely many extreme
points and the objective improves monotonically when non-degenerate. The optimal
$X_t$ is recovered as $\sum_i \lambda_{ti} X_{ti}$ (eq. 33).

**Generalisation.** Columns may be drawn "freely from given convex sets" (the
*generalised programming problem*, Theorems 2–3). The pricing oracle doesn't
have to be an LP; it only has to return the best point of its set at the given
prices.

**Multi-stage systems.** The last section nests the decomposition over time
stages and warns about "a great deal of jockeying up and down the various
stages".

**The economic reading.** This is the part closest to this project:

> "The coordinator works out a system of 'prices' … A bonus … is then offered
> the management of each part if he can offer, based on these prices, a new
> feasible program for his part with lower cost … The essential idea is that
> **old offers are never forgotten** by the central agency … the former are
> mixed with the new offers to form new prices."

The credit goes to Ford & Fulkerson's 1958 multicommodity-flow proposal.

The mapping to a home is direct. The *parts* are the devices. The *linking
rows* are the meter, one per time slot. The *prices* are the marginal cost of
energy at the meter. The *proposals* are device plans.

---

## 2. What the literature has added since

Sources marked † were read for this note. The rest are standard references,
cited for their well-known results.

**DW is the dual of Kelley's cutting-plane method** (Kelley 1960). The
coordinator's LP dual is Kelley's method applied to the Lagrangian dual of the
linking rows. The best bound DW can reach is the Lagrangian bound, which equals
optimising over the convex hull of each part's feasible set (Geoffrion 1974).
The consequence for us: DW needs neither convexity nor linearity in the
*parts*. A DP that returns the best plan at given prices is a legitimate
pricing oracle, and the bound it yields is a valid lower bound for the
non-convex problem. [Lübbecke & Desrosiers, *Selected Topics in Column Generation*, OR 53(6) 2005](https://www.researchgate.net/publication/200035352_Selected_Topics_in_Column_Generation).

**Plain column generation is unstable.** The duals oscillate between
iterations and progress tails off near the optimum. The standard remedies are:

- smoothing the prices toward the best-bound point (Wentges 1997);
- box or penalty stabilisation (du Merle et al. 1999);
- a general explicit stabilising term in the dual (Ben Amor, Desrosiers & Frangioni†).

[Ben Amor, Desrosiers & Frangioni](https://pages.di.unipi.it/frangio/papers/StabCG.pdf)
show that a stabilising term in the dual "amounts at considering a (generalized)
**Augmented Lagrangian** of the primal Master Problem". **So ADMM and
stabilised DW are two ends of one family.** ADMM keeps no memory of past
proposals and uses a quadratic stabiliser. DW keeps every proposal and uses a
piecewise-linear or box stabiliser. §6.4 turns this observation into a design
choice.

**Integer recovery.** When parts are integer, the master's mixture can be
fractional. Two established routes:

- Branch-and-price (Barnhart et al. 1998) is exact.
- *Price-and-branch* is a heuristic: solve the master once more with integer
  weights over the columns generated.

For fleets of devices, [Vujanic et al., Automatica 67 (2016)](https://arxiv.org/abs/1411.1973)
tighten the linking constraints so that a dual-derived primal is feasible, with
a proven performance bound.

**The Shapley–Folkman effect.** The loss from convexifying shrinks relative to
the whole as parts are added. It is bounded by the number of *active linking
constraints*, not by the number of parts
([Udell & Boyd 2016](https://arxiv.org/abs/1410.4158)). For one home with 96
meter rows and 2–4 devices that bound is loose. For an aggregator with
thousands of homes it is tight (§7d).

**Power systems had this first.** Lagrangian-relaxation unit commitment prices
a demand constraint and solves one DP per generating unit (Muckstadt & Koenig
1977; Bertsekas et al. 1983). That is our structure exactly, with generators
in place of appliances. Column generation with DP subproblems followed
(e.g. Shiina & Birge 2004, stochastic UC). The electricity-market
*convex hull price* (Gribik, Hogan & Pope 2007, already cited in `theory.tex`)
is the optimal dual of exactly this master. **DW computes convex-hull prices
directly**, so the "expensive" construction `theory.tex` §milpduals describes
comes out of the coordinator as a by-product.

**Energy and DER applications:**

- [Sokoler, Standardi, Edlund, Poulsen, Madsen, Jørgensen, J. Process Control 2014](http://people.compute.dtu.dk/jbjo/publications/DTU_MPC_2014_14/AcceptedManuscript.pdf)†:
  economic MPC of dynamically decoupled power units, warm-started DW against a
  structure-exploiting ADMM. Averages were **7–13 DW iterations against 84–149
  for ADMM**, with a worst case of 485 for ADMM. At 128 units, DW was under 1%
  suboptimal after 0.3 s, while ADMM "only after 10 seconds". They also report
  that warm-starting DW barely cuts iterations but sharply improves
  early-terminated solutions: at most 5% suboptimal, against up to 30% cold.
- [Najafi & Fripp 2023, arXiv:2302.00166](https://arxiv.org/pdf/2302.00166)†:
  DW for price-responsive EVs and water heaters across households, with the
  coordinator as master. They note that DW "gives an optimality gap at each
  iteration", and leave integer devices (dishwashers, washers) to future work.
- Hierarchical EV-charging aggregators with DW (IEEE, 2017); DW for community
  HEMS where a monolithic MILP "breaks down" beyond about 3,000 homes (DOE BTO
  peer review, 2023).

**What none of these do.** None puts a *Bellman recursion with a published
value function* in the pricing slot. `theory.tex` makes the same observation
about ADMM: its literature puts convex programmes in the device slot. Here the
DP brings the value-function and λ properties of this package, and DW brings
the certificate.

---

## 3. The HEMS problem as a Dantzig–Wolfe master

Notation follows `theory.tex`. There are $n$ slots of length $\Delta t$ and the
fixed net demand is $d_t$. Device $k$ has plan $p^k \in \mathcal P^k$ and
private cost $f_k$: comfort for thermal devices, horizon-edge energy for
storage and thermal alike.

### 3.1 Master (restricted to the columns found so far)

Let $\mathcal J_k$ be the pool of plans proposed by device $k$. Each plan
$j$ has a power profile $p^{kj} \in \mathbb R^n$ and private cost $f_{kj}$.

$$
\begin{aligned}
\min\;& \sum_t \Delta t\Big(c^{imp}_t z^{+}_t + (1+\kappa_g)c^{imp}_t o^{+}_t
      - c^{exp}_t z^{-}_t - (c^{exp}_t - \kappa_g c^{imp}_t) o^{-}_t\Big)
      + \sum_k\sum_{j\in\mathcal J_k} f_{kj}\,\lambda_{kj}
      + \sum_{b} \big(-\pi^T_b\, s^b_n\big) \\
\text{s.t.}\;& z^{+}_t + o^{+}_t - z^{-}_t - o^{-}_t - \chi_t
      - \sum_{k,j} p^{kj}_t \lambda_{kj} - \sum_b (c^b_t - e^b_t) = d_t
      && [\pi_t] \quad \text{meter, one row per slot}\\
& \sum_{j} \lambda_{kj} = 1 && [\sigma_k] \quad \text{convexity, one per DW device}\\
& s^b_{t+1} = s^b_t + \Delta t(\eta_c c^b_t - e^b_t/\eta_d) && [\lambda^b_t] \quad \text{battery } b \text{ in the master}\\
& 0 \le z^{+} \le \bar z^{imp},\; 0 \le z^{-} \le \bar z^{exp},\; o^{\pm} \ge 0,\;
  0 \le \chi_t \le g_t,\; \lambda \ge 0,\; \text{battery box constraints.}
\end{aligned}
$$

- **The kink is exact.** $z^+$ and $z^-$ carry different prices. When
  $c^{exp} \le c^{imp}$ the LP never uses both at once, so no binary is needed.
- **Grid limits are exact and soft.** $o^\pm$ is the breach, priced at the same
  $\kappa_g$ as `grid_penalty`. The master is therefore always feasible. No
  Phase I, artificial columns or Farkas pricing are needed: any single column
  per device is a feasible start.
- **Curtailment is a variable** ($\chi_t \le g_t$, free). It is decided inside
  the optimisation, rather than applied after the fact with a `max(sell, 0)`
  correction to the device prices. The master's price takes care of
  Proposition `prop:curtail` without special-casing.
- **Batteries go in the master as LP variables** (the "hybrid master"). A
  battery *is* an LP. Turning it into columns makes the master rediscover its
  polytope one extreme point at a time, which is where plain DW tails off:
  §5 D, "battery as columns, bare" captures only 28–70%. Only non-linear or binary devices
  (tank, HVAC, and later EVs with minimum current, appliances with start menus)
  stay as DW blocks. This is the paper's generalised programme: some columns
  are fixed (the battery's), others are drawn from sets.

### 3.2 Pricing: the existing DP, given a linear price

With duals $(\pi, \sigma)$, device $k$'s pricing problem is

$$
\min_{p^k \in \mathcal P^k}\; f_k(p^k) + \sum_t \pi_t\, p^k_t \;-\; \sigma_k .
$$

This is `solve_water_heater` / `solve_hvac` / `solve_battery` called with
`buy = sell = π/Δt` and **no `dp_load`**. With equal import and export prices
and no load to cross, the kink disappears and the DP's bill term is exactly
$\pi \cdot p$. **No new solver is needed; the DPs are only called differently.**
Two things must stay pinned to the tariff rather than drift with $\pi$:

- the comfort price, which the thermal DPs derive from `np.mean(buy)`;
- the battery terminal price, which defaults to `min(buy)`.

The prototype pins these without touching `src/` (`_PriceVector`,
`replace(terminal_price=…)`). The real fix is a `ref_price` argument on the
solvers.

A negative reduced cost means a new column. The DP returns the optimal plan and
a value function, so λ and the counterfactual machinery keep working at the
final prices.

### 3.3 Two kinds of oracle

The master accepts **any feasible plan** as a column, whoever proposed it. The
prototype runs two oracles per device per iteration:

| oracle | call | role |
|---|---|---|
| **exact pricing** | DP at $\pi$, linear, no `dp_load` | improves the dual; yields the bound |
| **load-aware (heuristic)** | today's ADMM-style DP: tariff $\max(c^{imp},\pi)$ / $\min(c^{exp},\pi)$, `dp_load` = $d$ + others' current mix | proposes self-consumption-shaped plans that linear prices reach only by mixing many extreme points |

The second row is the ADMM device step with the grid multiplier $\mu$ replaced
by the master's premium $\pi - c^{imp}$. It serves as a column generator.

### 3.4 The bound

At **any** price vector $\pi$, whether it came from the master or not:

$$
L(\pi) = \pi^\top d
+ \sum_k \min_{p^k}\big(f_k + \pi^\top p^k\big)
+ \sum_b \min_{\text{battery }b}\big(-\pi^T_b s_n + \pi^\top(c-e)\big)
+ \sum_{\text{meter vars } v}\sum_t \min\big(0, (c_{v,t} - \pi_t a_v)\,u_{v,t}\big)
\;\le\; J^\star .
$$

The last term is closed form: each meter variable is a box, so its minimum is
at a bound. $L$ is $-\infty$ outside the "dual box". With no limits that box is
$\pi_t/\Delta t \in [\min(c^{exp}_t, 0), c^{imp}_t]$, which is the meter kink's
subgradient range, so the duals are **naturally boxed**.
The gap $J(\text{plan}) - \max L$ is a certificate on **any** plan, including
the one ADMM returns.

---

## 4. The algorithm, step by step

As implemented in `dw/coordinator.py::DWCoordinator.run`.

```
INPUT  site config, forecasts; max_iter=40, gap_tol=1e-3, smoothing α=0.5
0. PIN     comfort reference price = mean(tariff); battery terminal price = min(tariff)
1. SEED    per DW device: its thermostat baseline (tank, HVAC) or idle (battery-as-columns)
           + exact-pricing columns at π = c_imp and at π = max(c_exp, 0)   (the dual box corners)
           ⇒ master feasible from the first iteration (meter + breach variables absorb any mismatch)
2. LOOP  it = 1 … max_iter
   2a. MASTER    solve the restricted master LP (HiGHS) → value RMP, weights λ, duals π (meter), σ (convexity)
                 [with batteries in the master: their plans, and SoE duals = the LP costate]
   2b. PRICE     for π_it ∈ {π,  α·π_center + (1−α)·π}   (Wentges smoothing; π_center = best-bound price)
                   for each DW device k:  col = DP_k(buy = sell = π_it/Δt)  → add to pool if new
                   L(π_it) via §3.4, per device min over {col} ∪ pool   (guards DP pricing error, §6.3)
                   if L(π_it) > best: best, π_center ← L(π_it), π_it
   2c. HEURISTIC for each DW device k: col = load-aware DP against d + Σ_{j≠k} mix_j (+ battery LP plans)
   2d. STOP      if RMP − best ≤ gap_tol·max(1,|RMP|)   or no new column was added
3. RELAX   re-solve the master: value = the convexified optimum; count devices with fractional weights
4. BOUND   re-evaluate L at every priced point against the FINAL pool   (makes LB ≤ relaxed hold exactly)
5. RECOVER solve the master once more with λ binary (one plan per DW device; batteries stay continuous)
           — a tiny MILP (~2–3 × 25–60 binaries), i.e. price-and-branch
6. POLISH  Gauss–Seidel load-aware best response from the recovered plan, accepted only if the
           true objective falls (monotone; measured: never fires after step 5)
7. OUTPUT  plan (one column per device + battery LP plans), UB = objective(plan), LB, relaxed value,
           meter price π/Δt, battery SoE duals, and each device's DP at the final π (V and POL for the
           execution tier)
```

The fixed per-iteration cost is one master LP (~100–300 rows) plus 2–3 DP
solves per DW device. It came to 4–14 iterations on every hybrid run in §5.

---

## 5. Measured (synthetic forecasts, 24 h, 15-min slots)

All methods are scored on **one** number, `extended_objective`: `total_objective`
plus the thermal horizon-edge terms that every DP and both exact references
already optimise. The *capture* column is (baseline − plan) / (baseline − DW
lower bound). 100% would be provably optimal, **up to the DP's own
discretisation** (§6.3).

### A. Battery (50×21) + tank: the fixture with exact references

| tariff | ADMM | DW columns | **DW hybrid** | joint DP | MILP (HiGHS) |
|---|---|---|---|---|---|
| flat | 3.6025 · 93.0% · 42 ms | 3.3627 · 98.7% · 1.0 s | **3.3286 · 99.5% · 0.29 s** | 3.4907 · 95.6% | 3.3190 · 38 s |
| day/night | 2.5301 · 88.3% | 1.8573 · 98.3% · 2.5 s | **1.8013 · 99.1% · 0.47 s** | 2.0474 · 95.4% | 1.7461 · 20 s |
| dynamic | −0.0710 · 93.2% | −0.1040 · 93.8% · 2.2 s | **−0.3744 · 99.2% · 0.43 s** | −0.2550 · 96.8% | −0.4387 · 14 s |

- DW hybrid **beats the "exact" joint DP** on all three tariffs. The joint DP is
  exact on its *product grid*, but bilinear interpolation over that coarse grid
  costs it 3–5 points. This is the discretisation `NOTES.md` §3 already
  attributes to it.
- Measured against the MILP's savings (baseline − MILP), DW hybrid captures
  **99.8 / 99.2 / 98.7%**, against ADMM's 93.3 / 88.4 / 92.7%, at about 1/50
  of the MILP's time.
- Caveat on the MILP comparison: part of the hybrid's lead over ADMM (a point or
  so) comes from the battery being continuous in the master rather than on a DP
  grid. "DW columns" keeps the battery on the DP grid, and still captures
  93.8–98.7% against ADMM's 88.3–93.2%.

### B. Shipped grids (battery 200×81 + tank + HVAC): no exact reference affordable

| tariff | ADMM | **DW hybrid** | DW lower bound |
|---|---|---|---|
| flat | 3.7812 · 94.0% · 0.88 s (15 rounds) | **3.5706 · 98.9% · 1.0 s** (14 it) | 3.5206 |
| day/night | 2.3951 · 92.6% · 0.87 s | **1.9929 · 98.4% · 0.60 s** (9 it) | 1.8785 |
| dynamic | 0.0616 · 93.5% · 0.18 s | **−0.1890 · 98.4% · 0.48 s** (7 it) | −0.2721 |

This is the first **bound on ADMM's gap on the shipped configuration**: it is
provably within 6.0–7.4% of available savings of the optimum (up to §6.3).
`NOTES.md` §6 currently says "Coordination has no optimality certificate. The
costs are measured, not bounded." That can now change.

### C. Two batteries + tank + HVAC, 9 kW PV, day/night, binding import limit

| import limit | ADMM | **DW hybrid** |
|---|---|---|
| 6 kW | −0.7656 · 94.8% · 0.84 s, 8 rounds | **−1.0937 · 99.4% · 0.31 s**, 4 it |
| 4 kW | −0.7656 · 96.5% · 0.84 s | **−1.0937 · 99.6% · 0.31 s**, peak **exactly 4.00** |
| 3 kW | −0.6678 · 97.5% · **3.7 s, 40 rounds, breach 0.223 kW** | **−1.0937 · 99.8% · 0.31 s**, peak **exactly 3.00** |

The limit is a row of the master, so it is either met exactly or its breach is
bought at $\kappa_g$ as a deliberate decision. There is no step size, no 40-round
floor, and no "holds to within ~20% of the physical floor". At 3 kW, ADMM
exhausts its budget and ships a breach, while DW meets the limit at no extra
cost in 4 iterations.

### D. Ablation (shipped grids, 3 devices): which ingredients matter

| variant | flat | day/night | dynamic |
|---|---|---|---|
| ADMM | 94.1% | 93.1% | 94.1% |
| **hybrid, full** | **98.9%** | **98.9%** | **99.0%** |
| − heuristic (load-aware) columns | 98.1% | 99.5% | 99.2% |
| − smoothing | 98.8% | 99.0% | 99.0% |
| − MILP recovery (max-weight column instead) | 96.0% | 93.6% | 93.9% |
| − polish | 98.9% | 98.9% | 99.0% |
| hybrid, bare (none of the four) | 91.5% | 85.5% | 92.7% |
| battery as columns, full | 99.4% (3.2 s) | 99.3% (8.2 s) | 97.0% (6.8 s) |
| battery as columns, bare | 70.2% | 36.9% | 27.7% |

**What carries the result:**

1. **The battery in the master.** Without it, plain DW is textbook
   tailing-off: 40 iterations, gaps of 1.5–4.8.
2. **Integer recovery over the pool.** Rounding to the heaviest column gives up
   most of the gain; the MILP over about 50 columns costs milliseconds.

Smoothing, heuristic columns and polish are within ±1 point of each other.
Stabilisation matters little here because the dual box $[c^{exp}, c^{imp}]$
already stabilises π (§3.4). This is where HEMS differs from the vehicle-routing
problems the stabilisation literature was written for.

---

## 6. ADMM vs Dantzig–Wolfe: pros and cons

### 6.1 Side by side

| | ADMM (today) | Dantzig–Wolfe (hybrid) |
|---|---|---|
| **what the coordinator solves** | closed-form z-step on the kink, a heuristic reading of it (§splitting) | an LP with the kink, limits and curtailment *exact* |
| **optimality certificate** | none | Lagrangian lower bound every iteration, and on any plan |
| **monotonicity** | none, hence best-round selection | master value never increases; best bound tracked |
| **memory** | last round only | every proposal kept ("old offers never forgotten") |
| **binary devices** | excluded from the prox term; price-coordinated only | uniform: pricing needs only an oracle |
| **grid limits** | dual ascent, step ζ, ≥40 rounds, fails near the floor | a row of the LP: exact, or a deliberate priced breach |
| **curtailment** | post-hoc, plus `max(sell,0)` device pricing | a variable of the master |
| **device interface** | DP with `dp_load` kink + quadratic prox (battery) | DP with a *linear* price (strictly simpler) |
| **third-party devices** | must accept a prox term and a load profile | anything that answers "at these prices, your plan and cost?" |
| **tuning** | ρ, adapt ratio/factor, ρ_min, ζ, max_rounds, κ_g, converge_tol | gap_tol, smoothing α (insensitive), κ_g |
| **integrality** | never fractional (each round is a real plan) | fractional mix; needs recovery (tiny MILP) or duty-cycle interpretation |
| **dependency** | numpy only | LP + small MILP (HiGHS via scipy) |
| **per-iteration cost** | constant | grows with the pool (LP re-solve); negligible at home scale |
| **parallelism** | device solves independent per round | same, plus master |
| **λ / prices** | per battery, conditional on others' plans; needs a ρ = 0 re-solve | meter price π is **system-wide** (convex-hull price); battery λ = master SoE dual, exact LP costate |
| **theory at home scale** | convex, 2-block convergence only; ours is neither | finite termination for LP parts; valid bound for non-convex parts |
| **measured capture** | 88.3–97.5% | 98.4–99.8% |

### 6.2 Where DW wins

1. **Certificates.** "Our plan is within X of optimal" becomes true and
   checkable on every solve, and it can be applied to other planners' plans
   too. It also enables **early termination with a known quality**: Sokoler et
   al.'s main operational point, and valuable on a 15-minute re-solve cadence or
   in the browser.
2. **Exact coupling constraints.** Grid limits, export caps and curtailment
   stop being heuristics. The ≥40-round special case, the ζ tuning and the
   "20% above the floor" caveat all disappear.
3. **A price that means what it says.** π is the meter's marginal price from a
   solved dual (the convex-hull price of Gribik–Hogan–Pope), not a min over
   routes. `meter_price` / `sensor.hems_worth_running` can become π directly,
   and its interpretation is sharper: under a binding import limit it shows the
   limit's premium explicitly. This addresses the "meter price is a min over
   routes, not a solved dual" caveat in `NOTES.md` §2.
4. **λ stops being conditional.** With batteries in the master, all batteries
   see the same π, and each battery's SoE dual is the exact LP costate. That is
   the ground truth `bench/duals.py` validates against today. The "two batteries
   of differing efficiency differ by 0.015/kWh" caveat becomes a consistent
   system-wide statement. (Not yet measured in the prototype: `batt_lambda` is
   computed but not compared.)
5. **Binary and black-box devices are first-class.** A dishwasher with five
   admissible start times is a five-column oracle. The "dryer modelled nowhere"
   of the README can *bid* instead of only reading a price.
6. **Fewer knobs, and they're insensitive** (ablation D).

### 6.3 Where DW loses, or needs care

1. **It needs an LP/MILP solver.** The library is numpy-only, and that matters
   for the Pyodide build. Options, from cheapest to most work:
   - Pyodide ships scipy, but the bundle grows by several MB.
   - Write a small dense simplex for the master. It has ~100–300 rows and
     ≤ ~100 columns, which is well within a few hundred lines.
   - Solve the *dual* directly: maximise the concave piecewise-linear $L(\pi)$
     over the box with a proximal bundle method, which is numpy-only. That is
     the Frangioni view of the same algorithm.

   The recovery MILP is the harder part to replace. With 2–4 DW devices and
   ~50 columns, enumerating pairs ranked by λ with a greedy repair would likely
   do, but that is untested.
2. **The certificate is only as exact as the DP oracle.** On a grid, the DP's
   rollout plan is not exactly the minimiser of $f + \pi p$. Measured: a pool
   column sometimes undercuts the DP by **0.006–0.073**, mostly the tank, whose
   value function is pessimistic by ~0.05 because of the thermal-saturation
   truncation (`NOTES.md` §4). The prototype guards the ordering LB ≤ relaxed ≤ UB
   exactly (step 4). What it cannot guard against is a plan no oracle proposed
   that beats the DP. Evidence that this is small: on the dynamic tariff the
   continuous-model MILP (−0.4387) sits 0.024 below the DW bound (−0.4143),
   about 0.5% of available savings. **Read the certificate as "within the DP's
   discretisation", the same ±~1% qualifier λ already carries.** Refining the
   thermal grids at the final iteration only would tighten it cheaply.
3. **Fractional mixtures.** The relaxation mixes 1–2 thermal plans at the
   optimum. Integer recovery is required (step 5), and **it is the single most
   important step** (ablation D). The gap between the relaxed and recovered
   values is the price of integrality: 0.02–0.08 here.
4. **The master grows.** Each iteration adds columns. At home scale that is
   25–60 columns and irrelevant. At aggregator scale, pool management (ageing,
   purging on reduced cost) becomes necessary.
5. **Plain DW tails off badly** when a part's optimum lies in the interior of
   its polytope, which is exactly the battery's situation during
   self-consumption. The hybrid master is essential, not optional
   ("columns, bare": 28–70%).
6. **Linear-price responses are degenerate.** Under flat π a battery DP is
   indifferent across many schedules, so a price alone does not tell a device
   what to do; the *mixture* does. This matters for the execution tier (§7b).
   It is the same "prices do not decentralise a non-convex optimum" point that
   `theory.tex` §milpduals makes about MILPs.
7. **Non-convex tariffs.** If $c^{exp}_t > c^{imp}_t$ in some slot, the meter
   term is non-convex and the LP would import and export at once. The
   prototype refuses such inputs. `NOTES.md` §2 names this regime; handling it
   needs one binary per affected slot in the master, or pricing it as its own
   DW block.
8. **Wall-clock time in pure Python** is similar to ADMM, not dramatically
   better (0.3–1.0 s against 0.18–0.88 s on shipped grids). The win is quality
   and certification, not speed. It becomes a speed win when ADMM would need its
   40-round grid-limit budget (3.7 s against 0.31 s in scenario C).

### 6.4 The bridge between them

Stabilised DW with a quadratic term *is* a proximal bundle method on the dual,
and that is an augmented Lagrangian of the master (Ben Amor–Desrosiers–Frangioni).
ADMM *is* an augmented-Lagrangian method with an alternating minimisation and
no memory. So the choice is less "DW or ADMM" than two independent dials:

- **memory**: last iterate only, or every proposal kept;
- **what solves the coupling**: a heuristic closed form, or an exact LP.

The prototype turns both to maximum. The ablation says memory plus an exact LP
is where the gain comes from, and the stabiliser shape barely matters.

---

## 7. Further directions

**(a) ADMM as a column generator, DW as referee: a zero-risk migration.**
Every ADMM round proposes a plan per device, and any feasible plan is a legal
column. Feed all rounds' plans into the DW master, solve it once, and recover.
The result can be no worse than ADMM's best round. You also get the bound (with
one exact-pricing call per device), and the ADMM code does not change. This is
the cheapest possible first integration: `coordinate()` stays, and a post-pass
adds a certificate and a better recombination.

**(b) Two price layers for the execution tier.** After DW converges:

- publish **π/Δt as the meter price** (system-wide, solved);
- publish **each battery's SoE dual as λ**;
- run each device's DP **at the final π** to get the value function that the
  sensor-cadence policy uses.

Because linear prices are degenerate (§6.3.6), the execution tier should track
the recovered *plan* and use V only to handle deviations and counterfactuals.
One way is a DP at π plus a small proximal pull toward the recovered plan: the
ADMM prox term reused as a tracking regulariser rather than a coordination
device.

**(c) Devices as bidders.** This is the paper's decentralised firm made
literal. The HA integration or evcc publishes π. Any device (an EV loadpoint
with `c_min`, a heat pump, a dishwasher with a start-time menu, a pool pump)
answers with a plan and its private cost. The coordinator mixes and recovers.
Semi-continuous chargers, minimum run-times and start windows are the
constraints that make MILPs branch. Here they live inside the proposing
device's oracle and never reach the master. The evcc `c_min` semi-continuous
floor (`NOTES.md` §7) is a natural first customer.

**(d) The same master one level up: communities and aggregators.** Put homes
in the pricing slot and a feeder or transformer limit in the master rows. Each
home's pricing oracle is *this whole HEMS*: a nested DW, which is Dantzig &
Wolfe's multi-stage idea turned sideways. At that scale Shapley–Folkman works
in our favour: the convexification gap is bounded by the number of binding
feeder rows, not homes, so fractional mixing affects only a vanishing share of
homes. Najafi & Fripp and the community-HEMS work above are in this regime.
Privacy improves too: homes reveal proposals, not models.

**(e) Rolling-horizon warm start: keep the pool across re-solves.** At each
15-minute re-plan, shift every stored column by one slot, drop the elapsed slot,
and extend the tail with a heuristic. The pool then starts nearly complete.
Sokoler et al. found warm start barely cuts iterations but sharply improves
early-terminated solutions (≤5% against ≤30% suboptimal). For HEMS that means
a guaranteed-quality plan even if a re-solve is cut short.

**(f) Scenarios.** Make the master two-stage: first-slot decisions shared, later
slots per forecast scenario. The *devices* need not change: they price one
scenario at a time. This is the natural route to forecast robustness, and it
fits the SDDP / water-value lineage `theory.tex` already cites. Benders (nested
DW's dual) is its other face.

**(g) Richer device physics without touching the master.** The tank
stratification that `theory.tex` lists as omitted (N-layer state) inflates only
the *tank's* pricing oracle. For such a device a small MILP, or a
coarse-to-fine DP, could be the oracle instead. DW does not care how a column
was found.

**(h) Duty-cycle reading of the relaxation.** A fractional mix of tank plans is
a duty-cycle schedule, and `theory.tex` itself notes a duty fraction in [0,1] is
"at least as defensible" as binary over 15 minutes. Where a device can modulate,
recovery is unnecessary and the relaxed value (0.02–0.08 better here) becomes
achievable.

**(i) Closing the gap exactly.** With 2–4 DW devices, branch-and-price
(branching on "device k runs in slot t") is cheap. That would give a
DP-exact optimum for the product problem without the product state space: an
alternative to the joint DP as a reference, one that scales past two devices.

---

## The app in one file (no server)

`dw/wasm/multi_device_optimizer_standalone.html` is the whole DW app in one
file (about 145 KB): the UI, the Python (flattened, renamed, stripped,
zlib+base64) and the compiled DP-kernels wheel. It runs in a browser tab via
Pyodide, in a web worker so the page never freezes, and needs no install. It
must be served over HTTP, not opened as `file://`.

```bash
dw/wasm/build.sh            # rebuild: bundle -> page -> standalone
dw/wasm/build.sh --serve    # ...and serve it on http://127.0.0.1:8767
```

It is the same pipeline as the ADMM app's `wasm/build.sh`:

| step | from | to |
|---|---|---|
| `dw/wasm/make_bundle.py` | `dw/` + `src/hemspolicy/` | `dw_bundle.py`: one flat module; `dw.*`/`hemspolicy.*` imports removed (aliases become assignments); only `build_site` taken from `hemspolicy.webapi`, whose other names would collide; any collision fails the build |
| `dw/wasm/make_page.py` | `dw/gui/index.html` | `dw_page.html`: the ADMM page's Pyodide-worker bootstrap, retargeted; the solver choice dropped (no scipy in the browser) |
| `wasm_batt_optimizer/make_standalone.py` | page + bundle + `wasm/build/dist/*.whl` | the standalone, with names kept per `wasm/keep_names.py` |

`tests/test_dw_standalone.py` regenerates the bundle and compares it, then
executes both the bundle **and the obfuscated payload inside the shipped
page**, and checks they return the same numbers as the package. Obfuscation
breaks at call time, not build time. Measured in Chrome: a cold load and
first solve (2 batteries + tank + HVAC, 24 h) takes about 8 s, including the
Pyodide download.

## 7½. Variants implemented

`DWCoordinator(site, fc, battery_in_master=..., tank_in_master=...).run(pool=..., seed_admm=...)`.
Every option is also a control in the app. `dw/compare.py --only E` compares them.

| variant | option | what it does |
|---|---|---|
| **tank in the master** | `tank_in_master=True`; app: Tank element = *continuous (LP)* | The tank sends its model, not proposals. With the element free to run any fraction of a slot, the tank is an LP exactly: outflow `C·r·(T − T∞)` with r capped at 1/Δt, the cut-out as `T ≤ t_max`, comfort shortfall as a linear inequality. Same physics as the DP, with continuous duty and no grid. It is the limit of the "send price sensitivities" idea: the most complete sensitivity is the model itself. |
| **used + newest pool** | `pool="active"`; app: Proposal pool | After each master solve, drop proposals with zero weight except last iteration's. Memory stays near the size of the LP basis. |
| **seeded from ADMM** | `seed_admm=True`; app: Seed from ADMM | Runs `coordinate()` first and puts every round's device plans in the pool, keeping ADMM's plan as the incumbent. DW can never return worse than ADMM, and its bound certifies ADMM's plan. This is the zero-risk migration of §7(a). |
| **incumbent** | always on with `anytime=True` (the app) | Recovers a runnable plan at every iteration and returns the best one seen. Recovery from the final pool is heuristic and was once worse than a plan from iteration 2 (1.168 against 0.637). |

Measured (2 batteries + tank + HVAC, 7 kW import limit, 24 h; flat / day-night / dynamic):

| | objective | iterations | proposals | time |
|---|---|---|---|---|
| ADMM, on/off tank | 2.334 / 1.687 / −0.408 | 3–12 rounds | – | 0.7–2.1 s |
| DW, on/off tank as DP | 2.156 / 1.377 / −1.243 | 5–6 | 13–21 | 0.6–0.8 s |
| … seeded from ADMM | 2.128 / 1.365 / −1.243 | 5–6 | 14–29 | 1.3–2.9 s (includes the ADMM run) |
| … used + newest pool | 2.156 / 1.377 / −1.243 | 5–6 | 9–10 | 0.6–0.7 s |
| DW, twelfths tank as DP | 1.876 / 1.194 / −1.402 | 7–12 | 22–37 | 1.8–3.1 s |
| **DW, tank in master** | **1.848 / 1.175 / −1.411** | **1–2** | **2–4** (HVAC only) | **0.15–0.24 s** |

The ADMM seed certifies ADMM's own plan as within 0.24 / 0.35 / 0.86 of the optimum for the on/off tank. The on/off rows cannot reach the continuous tank's bound by construction: the difference between them is the value of letting the element modulate.

### Device response: price sensitivity from the DP (`response=`; app: Device response)

This is the idea of a device sending not only its plan but how it would
respond to a change. The DP already holds the answer (`dw/sensitivity.py`):

1. Solve the pricing DP at the master's price, as before.
2. For every hour t, force one step more (and, separately, one step less):
   - tank: one duty level;
   - HVAC: each other mode;
   - a battery proposed as columns: a quarter of its power.
3. Replay the stored policy from there. No re-solve is needed, and all 2n
   variants run in one batched pass.

Each variant is a plan the device can run: "more at t, so less at 18:00". Its
cost at the price is the device's marginal cost of that deviation, with every
knock-on included.

| mode | what the device sends | memory |
|---|---|---|
| `proposals` | its best plan (the classic method) | grows |
| `both` | the plan plus its cheapest `sens_keep` (24) variants per direction | grows faster |
| `sensitivity` | the same, but each iteration the pool is folded into one **aggregate** per device (Kiwiel-style) plus the heaviest runnable plan | bounded (~85) |

The app's **flexibility chart** shows, per device and hour, the cheapest "more"
and "less" deviation at the current prices. Near zero means the device is
flexible there. For the tank, heating *less* is expensive around 06:00–08:00
and 17:00–19:00 (the hot-water draws), and heating *more* is cheap almost
everywhere.

Measured (`compare.py --only F`: 2 batteries in the master + tank + HVAC,
7 kW import limit):

| | proposals | + sensitivity | sensitivity only |
|---|---|---|---|
| day/night, on/off tank | 1.3767 · 17 proposals · 0.8 s | 1.3414 · 322 · 1.8 s | **1.3162** · 85 · 2.0 s |
| day/night, twelfths tank | 1.1943 · 37 · 3.0 s | **1.1550** · 589 · 4.0 s | 1.1640 · 83 · 11.4 s (cap) |
| dynamic, on/off tank | −1.2427 · 21 · 0.8 s | **−1.2799** · 329 · 1.8 s | −1.2777 · 84 · 1.1 s |
| dynamic, twelfths tank | −1.4024 · 25 · 1.8 s | **−1.4130** · 390 · 2.5 s | −1.4115 · 84 · 11.1 s (cap) |

What this showed:

- **Sensitivity finds better plans in every case**, by 0.01–0.06, at 1.3–2×
  the time.
- **Much of the gain is the DP's own discretisation.** The tank and HVAC
  rollouts read the policy at the nearest grid state. "Force one change, then
  re-plan" variants regularly beat the DP's own plan at its own price: the
  chart flags these as negative costs. So sensitivity is also a local search
  that recovers what the grid left behind. It also means the bound from those
  DPs alone was not tight: sensitivity-only once returned a plan 0.03 *below*
  it.
- **Keeping the bound valid needs bookkeeping.** Every proposal ever made is
  now scored at every price point priced so far (a running minimum, no
  archive), so the bound accounts for proposals the aggregated pool forgets.
- **An aggregate of an on/off device is not runnable.** It is excluded from
  recovery. A modulating tank's aggregate is re-simulated through the physics
  before use.
- **Sensitivity only** keeps memory bounded but needs more iterations when
  the tank modulates: the aggregate slows the master's convergence. It is the
  most frugal mode for on/off devices.
- **It does not rescue batteries proposed as columns.** They still tail off.
  Batteries belong in the master.

Not implemented: a sensitivity passed as a local *linear model* to a
coordinator QP (ALADIN proper), rather than as runnable variant plans; and
using the same variants to repair the DP rollouts themselves.

## 7¾. The recommended configuration, and the built-in solver

After all the variants, the recommendation is: **everything linear in one LP
at the meter; only devices that aren't linear bid plans.** In practice that
means:

- batteries in the master (always);
- the tank in the master whenever its element can modulate;
- HVAC modes, on/off heaters and start-time appliances bidding plans;
- the used + newest pool, recovery by best combination (MILP), and keeping
  the best plan seen.

These are now the defaults of `DWCoordinator.run` and of the app.

**Built-in solver (`dw/lpsolver.py`, numpy only).** The master is solved by a
primal-dual interior-point method (Mehrotra predictor-corrector), which
returns the equality duals (the meter prices) with HiGHS' sign convention.
Details:

- **Fixed variables are removed first.** Curtailment at night has an upper
  bound of 0, and so do plans excluded in a branch.
- **The best iterate is kept.** Once the complementarity is tiny, the normal
  equations become singular, so later steps can only lose accuracy.
- **Block elimination.** The batteries' and the tank's rows touch each other
  only through the meter and convexity rows, so the normal-equations matrix is
  block-arrow shaped. Each block is eliminated separately and the small border
  system (the Schur complement) is factored: about 8× faster at 6 batteries
  over 48 h.
- **Recovery** is a small branch-and-bound over "one plan per on/off device",
  with a rounding heuristic, best-first order and node/time limits. The
  per-iteration "plan if stopped here" uses a 6-node budget; the incumbent
  keeps the best anyway.

Checked against HiGHS on 300 random bounded LPs: worst relative objective
difference 4e-7, 14–18 iterations each (`tests/test_lpsolver.py`). On the DW
runs it is 1–4× slower than HiGHS and gives identical plans wherever both
solvers take the same path.

Master LPs are degenerate, so their prices are not unique. The interior-point
method returns "central" prices where HiGHS returns "corner" ones, so the two
loops can collect different plans and end at different, equally valid plans.
Both were checked to be physically feasible. `solver="highs"` (app:
Advanced → LP solver) remains available for cross-checking. The DW package no
longer needs scipy.

## 8. What I would do next, in order

1. **Put the migration of 7(a) into `coordinate()`** behind a flag: ADMM rounds
   become columns, followed by a DW master pass, recovery and a bound. It adds a
   certificate with no change to the default path. Report `lower_bound` and
   `gap` in `CoordinationResult` and the GUI rounds strip.
2. **Add `ref_price` / `terminal_price` arguments to the solvers**, replacing
   the prototype's `_PriceVector` workaround.
3. ~~**Add `method="dw"` as a first-class coordinator**, returning a
   `CoordinationResult`.~~ Done: `hemspolicy.plan(site, fc, method="dw")`
   (via `dw/integrate.py`) is now the default for Home Assistant and evcc.
   Batteries with EV extras (charge floor, SoC goal, charger minimum, SoC
   gates) bid plans instead of sitting in the master; see `is_plain_battery`.
4. **Validate λ from the master SoE duals** against `bench/duals.py`
   (the LP-dual ground truth) and against today's λ.
5. **Test rolling-horizon warm start (7e)** over a simulated week, measuring
   iterations and early-termination quality.
6. **Build a numpy-only master** (small dense simplex, or bundle method on the
   dual) and a numpy-only recovery. Measure the Pyodide cost before choosing
   between them and shipping scipy.
7. **Update `NOTES.md` §6** once (1) lands: "no optimality certificate"
   becomes "certified within X, up to DP discretisation".

---

## References

- G. B. Dantzig, P. Wolfe. *Decomposition Principle for Linear Programs.* Operations Research 8(1):101–111, 1960.
- J. E. Kelley. *The cutting-plane method for solving convex programs.* J. SIAM 8(4), 1960.
- A. M. Geoffrion. *Lagrangean relaxation for integer programming.* Math. Programming Study 2, 1974.
- M. E. Lübbecke, J. Desrosiers. *Selected Topics in Column Generation.* Operations Research 53(6):1007–1023, 2005.
- H. Ben Amor, J. Desrosiers, A. Frangioni. *On the choice of explicit stabilizing terms in column generation.* Discrete Applied Math 157, 2009 ([preprint read](https://pages.di.unipi.it/frangio/papers/StabCG.pdf)).
- O. du Merle, D. Villeneuve, J. Desrosiers, P. Hansen. *Stabilized column generation.* Discrete Math 194, 1999.
- P. Wentges. *Weighted Dantzig–Wolfe decomposition for linear mixed-integer programming.* ITOR 4(2), 1997.
- C. Barnhart et al. *Branch-and-price: column generation for solving huge integer programs.* Operations Research 46(3), 1998.
- J. A. Muckstadt, S. A. Koenig. *An application of Lagrangian relaxation to scheduling in power-generation systems.* Operations Research 25(3), 1977.
- D. P. Bertsekas, G. S. Lauer, N. R. Sandell, T. A. Posbergh. *Optimal short-term scheduling of large-scale power systems.* IEEE TAC 28(1), 1983.
- T. Shiina, J. R. Birge. *Stochastic unit commitment problem.* ITOR 11(1), 2004.
- P. R. Gribik, W. W. Hogan, S. L. Pope. *Market-clearing electricity prices and energy uplift.* 2007.
- M. Udell, S. Boyd. *Bounding duality gap for separable problems with linear constraints.* Comput. Optim. Appl. 64, 2016. [arXiv:1410.4158](https://arxiv.org/abs/1410.4158).
- R. Vujanic, P. Mohajerin Esfahani, P. Goulart, S. Mariéthoz, M. Morari. *A decomposition method for large scale MILPs, with performance guarantees and a power system application.* Automatica 67, 2016. [arXiv:1411.1973](https://arxiv.org/abs/1411.1973).
- L. E. Sokoler, L. Standardi, K. Edlund, N. K. Poulsen, H. Madsen, J. B. Jørgensen. *A Dantzig–Wolfe decomposition algorithm for linear economic model predictive control of dynamically decoupled subsystems.* J. Process Control 24(8), 2014. [manuscript read](http://people.compute.dtu.dk/jbjo/publications/DTU_MPC_2014_14/AcceptedManuscript.pdf).
- F. Najafi, M. Fripp. *Market-Based Coordination of Price-Responsive Demand Using Dantzig–Wolfe Decomposition Method.* [arXiv:2302.00166](https://arxiv.org/pdf/2302.00166), 2023 (read).
- *Hierarchical Electric Vehicle Charging Aggregator Strategy Using Dantzig–Wolfe Decomposition.* [IEEE, 2017](https://ieeexplore.ieee.org/document/8057800/).
- S. Boyd et al. *Distributed optimization and statistical learning via ADMM.* 2011 (as cited in `theory.tex`).
