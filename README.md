# hems-optim

Home energy optimisation that produces a **price signal**, not just a schedule.

The Python package in this repository is `hems-policy` (import `hemspolicy`); it
feeds [Home Assistant](https://www.home-assistant.io) and [evcc](https://evcc.io).
Try the planner in your browser:
[Smart Home Energy Optimizer](https://ameetdesh.github.io/multi_device_optimizer_standalone.html).

It plans a day of battery, hot-water and HVAC operation against a tariff and a
PV forecast — and because it solves by dynamic programming rather than linear
programming, it also answers questions a schedule cannot:

- **What is a stored kWh worth right now?** (λ)
- **At what grid price should I import, or export?** (a bid/ask band)
- **What would one more kWh of consumption actually cost me?** (often far below
  the tariff, because storage serves it)
- **What if I'm not in the state the plan predicted?** (there is an optimal
  action for every state, not only the planned one)

Those come from the value function, so they are lookups rather than re-solves —
microseconds, no solver in the loop.

---

## Install

```bash
git clone https://github.com/ameetdesh/hems-optim && cd hems-optim
python3 -m venv .venv
.venv/bin/pip install -e .          # numpy only
```

`scipy` is needed only for the benchmarks in `bench/`:

```bash
.venv/bin/pip install -e ".[test]" scipy
.venv/bin/python -m pytest          # 210 tests (182 without scipy)
```

## Theory notes

`docs/theory.tex` sets out the problem, the device interface and both
coordinators, with the evidence behind every number. Build the PDF with

```bash
tools/build-theory          # -> docs/theory.pdf (not committed)
tools/build-theory --figs   # regenerate docs/figs/ first (needs the bench extras)
```

It uses [tectonic](https://tectonic-typesetting.github.io) if installed (it
fetches missing LaTeX packages itself), else `latexmk`, else three `pdflatex`
passes.

Rebuilding the single-file browser pages (`wasm/build.sh`, `dw/wasm/build.sh`)
needs `make_standalone.py`, which is not in this repository yet; point
`MAKE_STANDALONE` at it. The committed pages and the tests do not need it.

---

## Three ways to use it

### 1. As a library

```python
from hemspolicy import (
    SiteConfig, BatteryConfig, Horizon, PolicySnapshot,
    coordinate, demo_forecasts, action, marginal_value, reservation_prices,
)

site = SiteConfig(horizon=Horizon(dt=0.25, hours=48.0),
                  battery=BatteryConfig(capacity_kwh=10.0),
                  water_heater=None, hvac=None)
fc   = demo_forecasts(site.horizon, tariff="dynamic")

# Slow tier: solve on the cadence forecasts actually change (15 min is fine)
result = coordinate(site, fc)
snap   = PolicySnapshot.from_result(site, fc, result)
snap.save("policy.npz")

# Fast tier: a process that never runs the solver
snap = PolicySnapshot.load("policy.npz")
t    = snap.step_for(hours_from_start=6.5)

action(snap, t, soe=4.2)                    # kW, + = charge
marginal_value(snap, t, soe=4.2)            # λ, currency/kWh
reservation_prices(snap, t, soe=4.2)        # {'import_below':…, 'export_above':…}
```

Safety limits are clamped **outside** the solver, never left to a penalty term:

```python
from hemspolicy import clamp, HardLimits
safe, bound_by = clamp(action(snap, t, soe), other_load_kw=3.1,
                       limits=HardLimits(max_import_kw=17.25))
```

### 2. Local GUIs: ADMM or Dantzig–Wolfe

There are two testbed UIs, one per coordinator. They use different ports, so
both can run at once and be compared on the same settings.

| | ADMM coordinator (the default in the package) | Dantzig–Wolfe coordinator (`dw/`) |
|---|---|---|
| start | `.venv/bin/python gui/server.py` | `.venv/bin/python dw/gui/server.py` |
| opens | <http://127.0.0.1:8765> | <http://127.0.0.1:8766> |
| needs | numpy | numpy (the DW master uses the built-in solver in `dw/lpsolver.py`; scipy only for the optional HiGHS cross-check) |
| shows | coordination rounds, λ, policy replay on hover | column-generation iterations, lower bound, meter price π, λ from the master |

```bash
# terminal 1 - ADMM
.venv/bin/python gui/server.py            # http://127.0.0.1:8765

# terminal 2 - Dantzig-Wolfe
.venv/bin/python dw/gui/server.py         # http://127.0.0.1:8766
```

Both servers open a browser tab on start. Pass `--no-browser` to the DW server
(`dw/gui/server.py 8766 --no-browser`) to skip that, and a number to either
to change the port. Stop them with Ctrl-C. Both UIs accept deep links such as
`?n_batteries=2&max_import_kw=7&tariff=day_night`. The DW UI also accepts
`view=relaxed` or `view=<iteration>`. What the DW UI shows, and how to read
it, is in `dw/README.md`. In the DW UI (and its standalone page) the import
and export prices, the household load and the hot-water draw can be edited:
drag the hourly dots (on the bold lines) on the Prices and Power charts, and it
re-solves on release. The draw is the heat taken from the tank, in kW. Export is kept at
or below import; **reset** (or a new tariff or horizon) restores the preset.
The same page can plan with **ADMM** instead (Method), with a slider for its
tether ρ and an "Adapt ρ" switch, so the two can be compared on one site.
While a solve runs, the runnable plan, lower bound and gap update after every
iteration (ADMM has no bound). With **Batteries in LP** off, DW keeps every
plan and smooths prices harder ("auto" pool and smoothing), which closes the
gap several times faster (`bench/dw_accel.py`).

The rest of this section describes the ADMM UI.

Sliders for capacity, power, efficiency, solar, horizon, number of batteries
(1–3) and grid import/export limits; toggles for each device and for PV
curtailment. Charts for prices with the no-trade band shaded, battery state of
energy, power flows and thermal state.

Hovering the battery chart replays the optimal policy from whatever state the
cursor is over, with **no re-solve** — the value function used directly.
Clicking prices a forced action against the optimum.

### 3. Home Assistant

Publishes the plan and the price signals as HA sensors. Walkthrough below.

### Which coordinator plans

```python
from hemspolicy import plan
res = plan(site, fc)                  # Dantzig-Wolfe (default)
res = plan(site, fc, method="admm")   # the ADMM loop
res.gap, res.lower_bound              # DW only: the plan is within `gap` of the best
res.meter_price                       # DW only: cost of one more kWh at the meter, per slot
```

Both return the same `CoordinationResult`, so `PolicySnapshot`, Home Assistant
and evcc work unchanged. DW puts plain batteries (and a modulating tank) into a
small LP and lets everything else (on/off tank, HVAC, EVs with a charger
minimum or a SoC goal) bid plans from its own DP. If DW cannot run (an export
price above the import price in some slot), `plan()` falls back to ADMM and
says why in `res.note`. `docs/theory.tex` §6–9 explains both and compares
them; `bench/run_planners.py` reproduces the comparison.

---

## Home Assistant, from scratch

### Step 1 — run Home Assistant

```bash
docker run -d --name ha -p 8123:8123 -v "$PWD/ha-config:/config" \
    -e TZ=UTC homeassistant/home-assistant:stable
```

Give it a minute, then open <http://localhost:8123>.

### Step 2 — create an account

Complete the onboarding wizard: create a user, set a location, skip device
discovery. Any username and password will do for a local trial.

### Step 3 — get a long-lived access token

In Home Assistant: click your **user avatar** (bottom left) → **Security** tab →
scroll to **Long-lived access tokens** → **Create token**. Copy it; it is shown
only once.

### Step 4 — create the dashboard

```bash
HA_TOKEN=<your token> .venv/bin/python tools/ha-lambda-demo/setup_dashboard.py
# -> dashboard ready at http://127.0.0.1:8123/hems-policy
```

Doing this explicitly matters: entities published through the REST API are
*orphan states* with no entity-registry entry, and Home Assistant's
auto-generated Overview dashboard may not list them. The script creates a
dashboard with three views — Trade, Power flows, Thermal — over the websocket
API.

### Step 5 — stream data into it

```bash
HA_TOKEN=<your token> .venv/bin/python tools/ha-lambda-demo/run.py --speed 2
```

The slow tier plans with Dantzig–Wolfe; add `--method admm` for the ADMM loop.
`--speed 2` runs a simulated day in about 12 real minutes, slow enough for the
history graphs to draw curves. `--speed 400` is a day in 4 seconds, useful as a
smoke test. Then open <http://localhost:8123/hems-policy>.

### What lands in Home Assistant

| entity | meaning |
|---|---|
| `sensor.hems_import_below` | **import while the grid price is below this** |
| `sensor.hems_export_above` | **export while the grid price is above this** |
| `sensor.hems_lambda` | λ, the value of a stored kWh, with a `forecast` array |
| `sensor.hems_worth_running` | run a flexible load while its value/kWh exceeds this |
| `sensor.hems_battery_action` | battery setpoint now (kW, + = charge) |
| `sensor.hems_battery_soe` | battery state of energy (kWh, plus `soc_percent`) |
| `sensor.hems_import_price` | the prevailing tariff |
| `sensor.hems_meter_price` | DW: the plan's cost of one more kWh at the meter, whole house (with `forecast`) |
| `sensor.hems_plan_gap` | DW: the plan is at most this far from the best possible; `unknown` under ADMM |
| `sensor.hems_pv`, `_load`, `_net_grid` | power flows (kW) |
| `sensor.hems_water_heater_temp`, `_power` | tank temperature and heater draw |
| `sensor.hems_hvac_temp`, `_power` | room temperature and HVAC draw |
| `sensor.hems_outdoor_temp` | outdoor temperature |

### Using it in an automation

A load worth about 0.25/kWh to run:

```yaml
automation:
  - alias: Run the dryer when energy is cheap enough
    trigger:
      - platform: state
        entity_id: sensor.hems_worth_running
    condition:
      - condition: numeric_state
        entity_id: sensor.hems_worth_running
        below: 0.25          # a kWh currently costs less than it is worth to us
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.tumble_dryer
```

The dryer is **modelled nowhere in the optimiser** — it just reads a price. That
is the point: adding devices does not grow the optimisation.

For a battery you control directly, gate on the band instead:

```yaml
      - condition: template
        value_template: >
          {{ states('sensor.hems_import_price')|float
             < states('sensor.hems_import_below')|float }}
```

### Charting the forecast

`sensor.hems_lambda` carries a `forecast` attribute, so ApexCharts works:

```yaml
type: custom:apexcharts-card
header: {show: true, title: Marginal value of stored energy}
series:
  - entity: sensor.hems_lambda
    data_generator: |
      return entity.attributes.forecast.map(p =>
        [Date.now() + p.hours_ahead * 3600000, p.lambda]);
```

### Caveats for the Home Assistant path

- The demo uses synthetic forecasts. Point it at real data with
  `hemspolicy.feeds` (keyless Open-Meteo PV, CSV, or a list from any tariff
  integration).
- States are pushed over the REST API, so they disappear when Home Assistant
  restarts. A durable deployment wants MQTT discovery or a custom component.

---

## The three prices, and which to use

Easy to conflate, and not interchangeable.

| | formula | answers |
|---|---|---|
| **λ** `marginal_value` | ∂V/∂s | what a kWh *inside a battery* is worth. One per battery |
| **reservation prices** `reservation_prices` | `λ·η_c` / `λ/η_d` | the grid prices at which importing or exporting starts to pay |
| **meter price** `meter_price` | `min(buy, λ/η_d, sell if surplus)` | what one more kWh *of consumption* costs |

Use the **reservation band** for trading decisions and the **meter price** for
whether to run a load. The band's width is exactly the round-trip loss, so it is
a genuine no-trade region: inside it, holding beats both directions.

Compare the band against the price actually faced — the import tariff while
importing, the export price while exporting. A discharge that displaces
household load realises the *import* price, not the export price.

---

## Also supported

**evcc.** `src/hemspolicy/evcc.py` implements the optimizer HTTP contract used
by [evcc](https://evcc.io) (`POST /optimize/charge-schedule`), so a local server
can serve its `OPTIMIZER_URI` endpoint. Handles multiple batteries, loadpoints
modelled as charge-only batteries, per-slot charge demands and state-of-charge
goals, and a charger's minimum power as a semi-continuous floor. `docs/PLAN.md`
lists what is and is not mapped. Requests are planned with Dantzig–Wolfe: an
EV bids charging plans from its DP, evcc's grid limits are enforced in the
plan, and the response carries an extra `_hems_policy_certificate`. Start the
server with `HEMS_METHOD=admm` for the ADMM loop.

**Real forecasts.** `hemspolicy.feeds` provides keyless Open-Meteo PV and
temperature, CSV and list ingestion, and measured-value blending — anchoring the
first forecast slot to what was just measured and decaying back over four slots.

---

## Layout

```
src/hemspolicy/
  types.py         config + result dataclasses
  dp_battery.py    battery DP; returns the value function and policy
  dp_thermal.py    hot-water and HVAC DPs, and their thermostat baselines
  coordinate.py    multi-device coordination (ADMM), grid limits, curtailment
  planner.py       plan(): Dantzig-Wolfe (default) or ADMM, one result type
  policy.py        value function -> actions, prices, counterfactuals
  feeds.py         real forecast inputs
  ha.py            Home Assistant publishing
  evcc.py          evcc optimizer wire contract
  profiles.py      synthetic forecasts for tests and demos
bench/             exact references: continuous LP, joint DP, MILP, duals
gui/               local web UI (ADMM coordinator)
dw/                Dantzig-Wolfe coordinator, its comparison runner and its UI (dw/gui/);
                   dw/integrate.py hands its plan to Home Assistant and evcc
tools/             HA dashboard setup, evcc compatibility checks
docs/theory.tex    theory notes (tools/build-theory makes the PDF)
docs/NOTES.md      measured findings
docs/PLAN.md       state and next steps
```

---

## Validation

All measured; method in `docs/NOTES.md`.

**λ is a real shadow price**, checked against two independent ground truths — an
LP dual computed by HiGHS, and an exact joint dynamic program:

| compared against | worst mean error | correlation |
|---|---|---|
| LP dual (the exact costate) | 0.031 /kWh | 0.75–0.91 |
| exact joint 2-D DP costate | 0.039 /kWh | 0.83–0.95 |

It is published with a **±0.04/kWh uncertainty band**, because that is what the
measurement supports. Decisions inside that band are not reliable.

**Reservation prices predict the optimiser's own actions** in 89.6–94.8% of
slots across three tariffs.

**The discretisation cost is small and tunable**: 94.3% of available savings on
a 50×21 state grid, 98.4–99.7% on 200×81 (37 ms).

**Multi-device coordination costs 2–8%** against an exact joint solve, and
**9–17%** against a mixed-integer solver over the same model. The exact joint
solve is only 3.8× slower, so at one or two devices it is worth doing directly.

**Grid import limits are met** down to about 20% above the physically achievable
floor; below that the result reports the breach rather than shipping an
infeasible plan.

---

## Licence

MIT; see [LICENSE](LICENSE).
