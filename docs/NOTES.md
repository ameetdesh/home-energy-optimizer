# Findings

What was measured, and what it means for using this package. Method notes appear
only where the result would be misread without them.

Reproduce with `bench/run_benchmark.py`, `bench/run_lambda_study.py`, and the
test suite.

---

## 1. λ is a real shadow price

`marginal_value` is validated against two independent ground truths: an LP dual
computed by HiGHS (the exact discrete costate — the LP is given explicit
state-of-energy variables so the multiplier on the transition equality *is*
dV/ds), and the costate of an exact joint 2-D dynamic program.

| compared against | mean error | worst | correlation |
|---|---|---|---|
| LP dual | 0.007–0.031 /kWh | 0.031 | 0.75–0.91 |
| exact joint DP costate | 0.006–0.039 /kWh | 0.039 | 0.83–0.95 |

The LP reference validates itself first: its dual lies inside the analytic
no-arbitrage band `[sell/η_c, buy·η_d]` 100% of the time, and on a flat tariff
hits both edges exactly (0.088889 = 0.08/0.9; 0.270000 = 0.30×0.9).

Residual error is discretisation and shrinks with the grid (day/night: 0.0190 →
0.0091 → 0.0087 across 50×21, 200×81, 400×161). Saturated states are not the
driver.

**Practical consequence:** λ ships with a **±0.04/kWh uncertainty band**, and
`sensor.hems_lambda` carries `confident_above` / `confident_below`. Decisions
inside that band are not reliable and should not be automated on λ alone.

**Scope:** λ is per battery, from a value function solved with the other devices'
power as exogenous load — a marginal value *conditional on their plans*, not a
full system-wide dual. Measured gap between two batteries of differing
efficiency sharing a meter: mean 0.015 /kWh, max 0.044.

---

## 2. Three prices, and which to use

Easy to conflate; not interchangeable.

| | formula | answers | consumer |
|---|---|---|---|
| **λ** `marginal_value` | ∂V/∂s | what a kWh *inside a battery* is worth | internal; one per battery |
| **reservation prices** `reservation_prices` | `import_below = λ·η_c`<br>`export_above = λ/η_d` | grid prices at which trading pays | dashboards, import/export decisions |
| **meter price** `meter_price` | `min(buy, λ/η_d, sell if surplus)` | what one more kWh of *consumption* costs | a flexible load deciding to run |

### Reservation prices

A bid/ask spread around λ. Buying 1 kWh stores only `η_c` of it; selling 1 kWh
drains `1/η_d`. The gap is exactly the round-trip loss — a genuine no-trade
band, zero for a lossless battery, wider for a lossier one.

**Compare against the prevailing meter price**: `buy` while importing, `sell`
while exporting. A discharge that displaces household load realises the *import*
price; comparing only against `sell` scores 2/16 and looks like a broken rule.

Agreement with the optimiser's own per-slot actions:

| tariff | agreement |
|---|---|
| high export price (grid arbitrage pays) | 93.8% |
| day/night, low export | 89.6% |
| dynamic with PV | 94.8% |

Residual is slots at a bound, or where the tariff itself contains an arbitrage
(`sell > buy`) so both directions pay at once.

On a normal tariff (import 0.15–0.40, export 0.08 flat) the export price never
clears the ask, so the battery never sells to the grid — it only shifts load.
The band reporting "hold" for grid trading is correct there.

### Meter price

Not the tariff. With storage present, extra demand is served by whatever is
cheapest at the margin. On a day/night tariff at 21:00 the meter reads 0.40/kWh
while an extra kWh costs **0.167** — the battery serves it and refills overnight.
A load gating on `buy[t]` idles through hours it should run.

Accuracy against an LP finite difference: mean 0.0007 (day/night), 0.0046
(dynamic), 0.0053 (flat); worst case 0.043. It is a min over routes, not a
solved dual, so it is an approximation.

The export route is keyed on whether **surplus generation exists**
(`dp_load < 0`), not on the battery's own net position — that flips on the
discrete action grid.

---

## 3. What the approach costs against an exact solve

### Discretisation

Battery DP against a continuous LP over the identical model:

| grid | savings capture |
|---|---|
| 50 × 21 | 94.3–96.6% |
| 200 × 81 (default) | 98.4–99.7% |
| 400 × 161 | >99% |

Tunable, not fundamental. 200×81 costs 37 ms — free against a 15-minute
re-solve cadence — so it is the default.

### Multi-device coordination

Per-device DPs coordinated by price, against an exact joint 2-D DP over the same
model (battery + hot-water tank, 96 slots):

| tariff | decomposed | exact joint | capture |
|---|---|---|---|
| flat | 3.5545 | 3.4907 | 98.4% |
| day/night | 2.5301 | 2.0474 | 92.5% |
| dynamic | −0.0710 | −0.2550 | 96.2% |

**The joint solve is only 5.6× slower** (240 ms vs 43 ms), so at one or two
devices it is worth solving jointly and skipping the approximation. Decomposition
is only forced at three or more devices, where the product state space becomes
impractical.

### Against a mixed-integer solver

A HiGHS MILP over the same model — the class of solver used by comparable tools.
Only expressible because comfort is priced linearly (§4).

| tariff | decomposed | joint DP | MILP | decomposed capture |
|---|---|---|---|---|
| flat | 3.5545 | 3.4907 | 3.3190 | 94.4% |
| day/night | 2.5301 | 2.0474 | 1.7245 | 88.1% |
| dynamic | −0.0710 | −0.2550 | −0.4387 | 92.7% |

The MILP is a tighter reference than the joint DP, being continuous where the DP
is discretised — so the joint DP's own 97–99% is the discretisation cost
reappearing from another direction.

**Caveat:** the MILP closes to a 1e-4 relative gap on all three tariffs at 96
slots, in 14–38 s. That formulation is untuned, so its runtime is *not* evidence
about MILP speed in general and should not be quoted as such. (It got
substantially easier when the tank's draw term became linear — the old draw
factor needed a big-M clip on every draw slot, and that was most of the
branching.)

---

## 4. Model choices that matter

### Comfort is priced in currency

Discomfort costs `discomfort_multiplier` × the cost of restoring the setpoint:
`heat_capacity × energy_price`, divided by COP for a heat pump. The multiplier is
the only taste parameter, and it reads plainly.

An arbitrary °C² weight instead produces a penalty two orders of magnitude larger
than the bill it is added to, distorting every comparison built on the objective.
With currency pricing the optimiser is cheaper *and* more comfortable than a
thermostat; with an arbitrary weight it raised the flat-tariff bill (savings
−0.104 vs +0.562). `comfort_mode="quadratic"` still reaches the old behaviour.

### Terminal value is linear

`V[N] = terminal_price · S`, defaulting to the cheapest import price on the
horizon. A quadratic pull toward a target state of charge instead hoards energy
at the horizon edge for reasons unrelated to prices — measured cost 29% of
savings — and forces dV/ds to zero at the target, corrupting λ exactly where most
batteries sit.

Linear terminal value does encourage end-of-horizon dumping, so run a 48 h
horizon and act only on the first 24 h.

### Storage bounds carry the efficiencies

The admissible action set is state dependent and the efficiencies belong in it:

```
a  in  [ -s*eta_d/dt ,  (cap - s)/(eta_c*dt) ]   intersect  [-P_dis, P_chg]
```

With those bounds `s + psi(a)*dt` lands inside `[0, cap]` by construction and no
projection is needed. Using the un-adjusted `(0-s)/dt` and `(cap-s)/dt` instead
lets a nearly empty store discharge past zero, where clamping silently absorbs
the deficit and the plan reports delivering energy that never existed — 0.111
kWh in one slot at `s = 1` kWh, `eta = 0.9`. The charge bound is symmetrically
too tight, so the store approaches capacity without reaching it.

The same correction applies to the joint-DP reference in `bench/`, which shared
the error; the two partly cancelled, so the reported gaps moved little
(discretisation 98.4–99.1%, decomposition 91.7–97.0%). Pinned by
`test_soe_bounds_need_no_projection`.

Thermal saturation still uses a projection. The action is binary, so the
admissible set cannot be narrowed continuously; a slot that would carry the tank
past its ceiling is billed in full but credited only the truncated rise. The
bias is *pessimistic* and bounded by one slot of rated power, and it does bind —
the tank reaches its ceiling in every scenario tested.

### Efficiency is split

`eta_charge` / `eta_discharge` are first-class (`eta` sets both). Required for
interoperability, and it is what makes the reservation band asymmetric.

---

## 5. Grid limits and curtailment

A grid limit binds the **sum** of every device plus the inflexible load, so no
per-device DP can enforce it. It is priced instead: a per-timestep multiplier
raises the effective import price until the plan fits, updated by dual ascent
inside the coordination loop. Devices need no change — they see a different
price.

Two identical 15 kWh batteries, 10 kW evening load:

| import limit | rounds | net peak | excess |
|---|---|---|---|
| 12 kW | 5 | 11.17 | 0.000 |
| 10 kW | 9 | 10.00 | 0.000 |
| 8 kW | 13 | 8.00 | 0.000 |
| 6 kW | 20 | 6.00 | 0.000 |
| 5 kW | 40 | 7.17 | **2.167** |

The physical floor for that site is **4.87 kW** (50 kWh of peak-window load
against 25.7 kWh deliverable), so it holds to within ~20% of a bound no method
could beat. Below that the result reports the breach rather than shipping an
infeasible plan — `grid_import_excess` must be believed.

**Tuning:** a binding limit needs 13–20 rounds where pure cost coordination needs
2, so `max_rounds` is raised to ≥40 automatically. The step size under-steps at
0.5 and converges at 1.0–2.0; the default is 1.0.

### PV curtailment

Without it an export cap is unenforceable: once storage fills, surplus must go to
the grid and no price can prevent it. Curtailment reduces export 1:1 and is
applied after the device solves, which is exact rather than approximate — it only
discards energy already destined for export, so it cannot change what a device
should have done.

Two triggers only: an export cap storage cannot absorb, and a negative export
price. Measured: a 5 kW cap against saturating PV goes from 0.70 kW over to
exactly 5.00 kW with 8.2 kWh curtailed; a negative-price window goes from paying
0.236 to export, to zero.

---

## 6. Known limitations

- **Never run on real hardware.** Every number here comes from synthetic
  forecasts or a test rig.
- **λ is per battery and conditional** on the other devices' plans (§1).
  `PolicySnapshot` carries battery 0's value function only.
- **Coordination has no optimality certificate.** The costs are measured (§3),
  not bounded.
- **Three or more batteries** run, but coordination quality at that size is
  unmeasured.
- **The MILP reference is untuned** (§3).
- **Home Assistant states are pushed over REST**, so they do not survive a
  restart; a durable deployment needs MQTT discovery or a custom component.
- **Grid export limits depend on curtailment** being enabled (§5).
- **Thermal saturation is truncated, not modelled** (§4): pessimistic, bounded
  by one slot of rated power, and it binds in practice.

---

## 7. Interoperability notes

`evcc`'s optimizer HTTP contract is implemented in `src/hemspolicy/evcc.py`,
transcribed from the generated client in `github.com/evcc-io/optimizer` and
verified against that client and a live instance. Points worth knowing:

- Units are **W and Wh**, prices per Wh, and the arrays are energies per slot
  rather than powers. The first slot is usually short — the remainder of the
  current quarter-hour — so each slot's own duration must be used.
- A **loadpoint arrives as a battery** that only charges (`d_max = 0`), carrying
  its requirement in `p_demand` and its charger's minimum power in `c_min`. The
  latter is *semi-continuous*: off, or at least `c_min`, never between.
- A self-hosted endpoint receives the **full forecast length** (672 slots
  observed) rather than the 2-day cap applied to the hosted service.
- The optimizer needs a **30-day home energy profile**; a fresh instance fails
  with `optimizer: meter profile incomplete`, which does not explain itself.
- `sponsor.IsAuthorized()` is a general project-wide feature gate (67 call sites,
  61 of them charger drivers), not specific to the optimizer.
