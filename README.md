# home-energy-optimizer

**Makes a home's energy devices work together.**

A home's distributed energy resources (DERs) — a battery, an EV charger, a
hot-water tank, a heat pump, solar panels — often come from different makers,
and each optimises for itself. Planned one at a time they work against each
other: two batteries charge into the same solar surplus and import to do it;
the water heater runs from the grid while the battery could have covered it.

home-energy-optimizer lets them cooperate. Each device keeps its own model and
its own optimiser — a linear or quadratic programme, a dynamic programme, a
manufacturer's black box — and answers the same few questions through one
interface. A coordinator turns the answers into one plan for the whole house,
around solar output, tariffs and grid limits:

- **Dantzig–Wolfe**, the default, blends the devices' plans in a small linear
  programme and bounds how far the result can be from the best possible plan.
- **ADMM** steers the devices with one shared price, and needs no LP solver.

The house reaches a better outcome than its devices would alone, and the saving
is split among the solar and each device by what it contributed, so every
device is fairly rewarded for cooperating. No single solver is the point: each
device uses whichever answers its questions best. A dynamic programme, for
instance, suits a device with one state variable and also gives a policy for
every state; a linear programme solves a battery exactly.

```mermaid
flowchart LR
  M["Dantzig–Wolfe coordinator<br/>an LP over the meter, the grid limits and the batteries<br/>picks the cheapest blend of offers"]
  D["Each other device (tank, heat pump, EV)<br/>its best plan at those prices,<br/>by its own solver"]
  M -- "prices, one per slot" --> D
  D -- "offers: a plan and its private cost" --> M
  M -.-> R(["one plan per device,<br/>with a bound on how far it is from the best"])
```

```mermaid
flowchart LR
  T["Every device, in parallel<br/>its cheapest plan near a target,<br/>by its own solver"]
  H["ADMM coordinator (the house)<br/>averages the imbalance,<br/>raises the price where it persists"]
  T -- "plans" --> H
  H -- "imbalance and price" --> T
  H -.-> R(["the best runnable plan, polished"])
```

It feeds [Home Assistant](https://www.home-assistant.io) and
[evcc](https://evcc.io). Try it in your browser:
[Smart Home Energy Optimizer](https://ameetdesh.github.io/multi_device_optimizer_standalone.html).
The theory, with the evidence behind every number, is in
[docs/theory.pdf](https://github.com/ameetdesh/home-energy-optimizer/blob/main/docs/theory.pdf). The Python package is
`home-energy-optimizer` (import `home_energy_optimizer`).

---

## Install

```bash
git clone https://github.com/ameetdesh/home-energy-optimizer && cd home-energy-optimizer
python3 -m venv .venv
.venv/bin/pip install -e .          # numpy only
```

`scipy` is needed only for the benchmarks in `bench/`:

```bash
.venv/bin/pip install -e ".[test]" scipy
.venv/bin/python -m pytest          # the full suite; a few reference checks skip without scipy or osqp
```

## Theory notes

[docs/theory.pdf](https://github.com/ameetdesh/home-energy-optimizer/blob/main/docs/theory.pdf) sets out the problem, the device interface
and both coordinators, with the evidence behind every number. It is built from
`docs/theory.tex`:

```bash
tools/build-theory          # -> docs/theory.pdf
tools/build-theory --figs   # regenerate docs/figs/ first (needs the bench extras)
```

It uses [tectonic](https://tectonic-typesetting.github.io) if installed (it
fetches missing LaTeX packages itself), else `latexmk`, else three `pdflatex`
passes.

The single-file browser pages rebuild with `admm/wasm/build.sh` and
`dw/wasm/build.sh`, which embed the Python with `tools/make_standalone.py` and,
if one has been built (`wasm/build_wheel.sh`, needs Docker), the compiled-kernel
wheel. The embedded Python is obfuscated; that is kept as a proof of concept,
since the source is here.

---

## Three ways to use it

### 1. As a library

Plan a site, then act between plans from the battery's own solver - here its
dynamic programme, whose value function answers "what now, from this state?"
without a re-solve. `plan()` (below) picks the coordinator.

```python
from home_energy_optimizer import (
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
from home_energy_optimizer import clamp, HardLimits
safe, bound_by = clamp(action(snap, t, soe), other_load_kw=3.1,
                       limits=HardLimits(max_import_kw=17.25))
```

### 2. Local GUIs: ADMM or Dantzig–Wolfe

There are two testbed UIs, one per coordinator. They use different ports, so
both can run at once and be compared on the same settings.

| | ADMM coordinator, and the policy/evcc endpoints | Dantzig–Wolfe coordinator, with ADMM as an option (`dw/`) |
|---|---|---|
| start | `.venv/bin/python admm/gui/server.py` | `.venv/bin/python dw/gui/server.py` |
| opens | <http://127.0.0.1:8765> | <http://127.0.0.1:8766> |
| needs | numpy | numpy (the DW master uses the built-in solver in `src/home_energy_optimizer/dw/lpsolver.py`; scipy only for the optional HiGHS cross-check) |
| shows | ADMM iterations, λ, policy replay on hover | column-generation iterations, lower bound, meter price π, λ from the master |

```bash
# terminal 1 - ADMM
.venv/bin/python admm/gui/server.py       # http://127.0.0.1:8765

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
The room's comfort band is edited the same way on the Room chart: one
low and one high dot per hour, flat 22–26 °C until dragged, with its own
**reset** under Advanced (a new horizon restores it too).
The same page can plan with **ADMM** instead (Method), with a slider for its
tether ρ, an "Adapt ρ" switch, each battery's step (its DP or its exact LP) and
a cold or warm start, so the two can be compared on one site.
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

**One battery, slow and fast.** `battery/` is a third, smaller page: a single
battery's DP, with its policy drawn as a flow field over hour and state of
charge. The slow tier (the full solve) and the fast tier (one decision from any
state, read from the stored value function) are timed side by side. Hover to
replay the policy from the pointer.

```bash
.venv/bin/python battery/gui/server.py    # http://127.0.0.1:8768
battery/wasm/build.sh                     # or as one self-contained page
```

`battery/README.md` has the build steps and what each part of the page shows.

### 3. Home Assistant

Publishes the plan and the price signals as HA sensors. Walkthrough below. To
plan a real house's devices instead of publishing a simulated one, the
EMHASS adapter — **EMHASS: a whole house, two solvers at once**,
below — coordinates EMHASS's own solver and this package's, device by device.

### Which coordinator plans

```python
from home_energy_optimizer import plan
res = plan(site, fc)                  # Dantzig-Wolfe (default)
res = plan(site, fc, method="admm")   # ADMM (proximal message passing)
res.gap, res.lower_bound              # DW only: the plan is within `gap` of the best
res.meter_price                       # DW only: cost of one more kWh at the meter, per slot
```

Both return the same `CoordinationResult`, so `PolicySnapshot`, Home Assistant
and evcc work unchanged. DW puts plain batteries (and a modulating tank) into a
small LP and lets everything else (on/off tank, HVAC, EVs with a charger
minimum or a SoC goal) bid plans from its own DP. If DW cannot run (an export
price above the import price in some slot), `plan()` falls back to ADMM and
says why in `res.note`. `docs/theory.tex` explains both - the device interface
they share (section 2), each coordinator as a problem and an algorithm (sections
3-4) - and compares them (section 8); `bench/run_planners.py` reproduces the
comparison.

---

## Home Assistant, from scratch

The two scripts below talk to Home Assistant over its websocket API, which
needs one dependency beyond the core install:

```bash
.venv/bin/pip install -e ".[ha]"    # adds websockets
```

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
# -> dashboard ready at http://127.0.0.1:8123/home-energy-optimizer
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

The slow tier plans with Dantzig–Wolfe; add `--method admm` for ADMM (each re-solve starts from the last one's state).
`--speed 2` runs a simulated day in about 12 real minutes, slow enough for the
history graphs to draw curves. `--speed 400` is a day in 4 seconds, useful as a
smoke test. Then open <http://localhost:8123/home-energy-optimizer>.

### What lands in Home Assistant

| entity | meaning |
|---|---|
| `sensor.hems_import_below` | import while the grid price is below this (from the battery's DP) |
| `sensor.hems_export_above` | export while the grid price is above this (from the battery's DP) |
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
  `home_energy_optimizer.feeds` (keyless Open-Meteo PV, CSV, or a list from any tariff
  integration).
- States are pushed over the REST API, so they disappear when Home Assistant
  restarts. A durable deployment wants MQTT discovery or a custom component.

---

## EMHASS: a whole house, two solvers at once

The Home Assistant walkthrough above streams a *simulated* day into HA. This is
the other direction: [EMHASS](https://emhass.readthedocs.io) is a Home Assistant
add-on that already pulls live PV, load, prices and battery SoC out of HA, so the
adapter in `src/home_energy_optimizer/integrations/emhass.py` plans those real
inputs. It replaces EMHASS's single whole-house MILP with **one subproblem per
device**, coordinated by Dantzig–Wolfe at the meter.

The point is that each device keeps its own model and its own solver. A
deferrable load stays with EMHASS's MILP, which is good at it. A hot water tank
or a heat pump goes to this package's dynamic programme, which prices comfort and
returns a value function rather than one trajectory. The coordinator only makes
the plans agree on the meter, and reports what each device is worth.

> **Status.** The EMHASS side is a draft PR —
> [davidusb-geek/emhass#1158](https://github.com/davidusb-geek/emhass/pull/1158)
> — which adds the `optimization_backend` dispatch and the `participants` option.
> Released EMHASS has no `optimization_backend` key, so on stock EMHASS this
> config does nothing. The adapter in this package is complete and tested
> (`tests/test_emhass_adapter.py`); it imports EMHASS only when it plans, so it
> costs nothing if you never use it.

**To run it** next to your own Home Assistant, `tools/emhass-coordination/`
builds EMHASS from that branch with this package, starts it in Docker, plans,
and publishes to Home Assistant: a battery planned here, two deferrable loads
planned by EMHASS, and each one's share of the saving as a sensor.

```bash
HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py up
HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py run --every 30
```

Its README walks through each step, and where EMHASS's configuration is changed.

### A four-DER house

Solar, a battery, two deferrable loads, a hot water tank and a heat pump — split
across both solvers, and wired as they are in the house: the PV and the battery
on a 4 kW hybrid inverter; the two loads on a garage panel (7.4 kW, no
backfeed); on that panel, a 3.5 kW breaker with the tank and the heat pump; and
the two loads on a 3 kW budget. This is
[`tools/emhass-coordination/config_four_der.json`](tools/emhass-coordination/config_four_der.json);
the coordination's own keys are:

```json
{
  "optimization_backend": "dantzig_wolfe",
  "costfun": "profit",

  "set_use_battery": true,
  "number_of_deferrable_loads": 2,
  "nominal_power_of_deferrable_loads": [3000, 750],

  "participants": [
    {"devices": ["battery"], "solver": "emhass"},

    {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},

    {"devices": ["water_heater"], "solver": "home_energy_optimizer",
     "config": {"power_kw": 3.0, "liters": 180.0, "t_comfort": 55.0,
                "n_duty_levels": 2, "comfort_weight": 10.0}},

    {"devices": ["hvac"], "solver": "home_energy_optimizer",
     "config": {"power_kw": 1.5, "cop": 3.5, "c_room_kwh_per_k": 2.5,
                "r_wall_k_per_kw": 5.0,
                "t_comfort_low": 21.0, "t_comfort_high": 25.0}}
  ],

  "electrical_topology": {
    "nodes": [
      {"id": "inverter", "type": "hybrid_inverter", "max_import": 4000, "max_export": 4000,
       "efficiency_from_parent": 0.97, "efficiency_to_parent": 0.97},
      {"id": "garage", "type": "panel", "max_import": 7400, "max_export": 0},
      {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 3500}
    ],
    "devices": {"pv": "inverter", "battery": "inverter",
                "deferrable0": "garage", "deferrable1": "garage",
                "water_heater": "heat", "hvac": "heat"}
  },
  "set_nodischarge_to_grid": false,

  "deferrable_load_groups": [
    {"names": ["deferrable0", "deferrable1"], "max_power": 3000}
  ]
}
```

Run it next to Home Assistant with `coordinate.py up --config config_four_der.json`
then `run --pv-peak 8000` (`tools/emhass-coordination/`). On its demo day the
inverter never passes more than 4 kW and the battery stores the PV it clips, the
garage never feeds back, the tank and the heat pump never run together (their
4.5 kW would trip the 3.5 kW breaker), and the loads keep within 3 kW.

**How the tree enters.** `electrical_topology` is the whole tree, in W:

- `nodes`: connection points behind the main meter (`grid`) — a hybrid inverter,
  a panel, a breaker — each with a `parent` (another node, or `grid` by
  default), so they nest; `max_import` / `max_export` on its connection to the
  parent (`max_export: 0` is no backfeed); `efficiency_from_parent` /
  `efficiency_to_parent` for a converter.
- `devices`: each device (`battery`, `water_heater`, `hvac`, `deferrableN`) and
  the PV (`pv`) on a node; the rest are on `grid`.
- `constraints`: limits on a set of devices wherever they are (a phase, a
  shared cable), `max_import` / `max_export`.

The adapter builds the package's tree (`types.SubMeter`, `SetLimit`) from it:

| what you write | becomes | its price |
|---|---|---|
| a node | `SubMeter(id, members, parent=...)`: a bus of its own in the master | `fed_local_price_<id>` |
| a node of type `hybrid_inverter` on `grid` holding the PV and the battery | the same, and EMHASS fills its own `inverter_*` keys from it, so its own solver plans the same inverter | `fed_local_price_<id>`, and `P_hybrid_inverter` |
| a constraint | `SetLimit(name, members, ...)`: a row of its own | `fed_limit_price_<name>` (a premium) |
| `deferrable_load_groups` (EMHASS's key), all its loads in one participant | kept in that participant's own EMHASS model, exactly as EMHASS holds it | — |
| `deferrable_load_groups`, its loads across participants | a `SetLimit` over those participants (`max_power`) | `fed_limit_price_deferrable0+deferrable1` |

An EMHASS participant group sits on one node, and a constraint holds all of its
devices or none (the coordinator sees only its total). Without
`electrical_topology`, EMHASS's own inverter keys still describe one hybrid
inverter. EMHASS has one PV forecast and one meter, so through EMHASS the PV is
one array on one node; the library itself takes several (below).

Keep whatever other EMHASS options those deferrable loads already use; they are
passed through to EMHASS's own model untouched. The battery is read from
EMHASS's `plant_conf` as it stands — capacity, the SoC window, both
efficiencies, both power limits and `battery_target_state_of_charge`.

| device | `solver` | planned by | in the coordination |
|---|---|---|---|
| solar | — | EMHASS's PV forecast | not a device; a player in the saving split |
| `battery` | `emhass` | EMHASS's linear battery model, held **exactly** in the master LP | continuous, no integers |
| `deferrable0`, `deferrable1` | `emhass` | EMHASS's own MILP, one solve per price query | bids plans |
| `water_heater` | `home_energy_optimizer` | this package's tank DP | bids on/off plans |
| `hvac` | `home_energy_optimizer` | this package's room DP | bids on/off plans |

The battery is held in the master LP only while EMHASS's battery is exactly its
linear constraints. `set_battery_dynamic`, a non-zero `weight_battery_charge` or
`weight_battery_discharge`, `battery_stress_cost`, `battery_soc_deficit_cost`,
`battery_soc_surplus_cost` or `battery_charge_power_derating` each take it out of
the master and make it a black-box participant instead — answered by EMHASS's own
model on every price query. That still plans; it is just slower. Move the battery
to `"solver": "home_energy_optimizer"` to have it planned by this package's DP
from the same `plant_conf` numbers, which gives a value function — λ at whatever
state the battery is in — rather than a plan with λ only along it.

Three details that are easy to get wrong:

- **One `config` per group.** A group's `config` is applied to every device in
  it, and the two thermal configs share field names (`power_kw`, `t_min`,
  `n_duty_levels`, …). Grouping `water_heater` and `hvac` together would give the
  heat pump the tank's `power_kw`. Give each its own participant, as above.
- **`water_heater` and `hvac` are not EMHASS loads.** They do not come from
  `number_of_deferrable_loads` and have no EMHASS entry; they exist only through
  a `home_energy_optimizer` participant, configured entirely by its `config`
  block. Devices this package knows: `battery`, `water_heater`, `hvac`.
- **Thermal devices need thermal forecasts.** The adapter reads
  `outdoor_temperature_forecast` (°C) and `hot_water_demand_kw` (kW) from
  EMHASS's input DataFrame, falling back to a flat 20 °C and no draw. Without
  them the tank and the heat pump plan against constants.

Any EMHASS device you leave out of `participants` becomes its own EMHASS group,
so a partial list is fine.

### What comes back

EMHASS's usual `opt_res` columns — `P_PV`, `P_Load`, `P_grid`, `P_deferrable0`,
`P_batt`, `SOC_opt`, `unit_load_cost`, `cost_fun_profit` — in EMHASS's own units,
so existing automations and charts keep working. Plus:

| column | meaning |
|---|---|
| `P_water_heater`, `temp_water_heater` | the tank's power (W) and temperature (°C) |
| `P_hvac`, `temp_hvac` | the heat pump's power (W) and room temperature (°C) |
| `fed_meter_price` | the cost of one more kWh at the meter, per slot — the master's dual |
| `fed_lower_bound`, `fed_gap` | how far this plan can be, at most, from the best possible one |
| `fed_stop_reason`, `fed_iterations` | why the coordinator stopped (`converged`, `stalled`, `no new proposals`, `iteration cap`) and after how many rounds |
| `fed_share_<player>` | each player's share of the saving over the horizon, in currency |
| `fed_local_price_<id>`, `fed_node_power_<id>` | each node's own price per slot, and its power to its parent (W, + = up the tree) |
| `fed_limit_price_<name>` | each constraint's premium per slot while it binds |
| `fed_local_price_<name>` | the price of one more kWh behind a limit the coordinator holds (above), per slot |

`optim_status` is `Optimal` only when the plan is proven within 0.1% of its
lower bound; otherwise `Optimal_Inaccurate` — a runnable plan, which EMHASS
publishes as usual, without that proof. When the adapter declines a
configuration it logs why and sets `fed_fallback_reason` on EMHASS's
`Optimization` object, and EMHASS's own MILP plans.

The shares are the part no single MILP can give you: `fed_share_solar`,
`fed_share_battery`, `fed_share_water_heater`, `fed_share_hvac` and
`fed_share_deferrable0+deferrable1` (a group is one player, named by its
devices). Each device is first reimbursed what coordination cost it privately,
then the surplus is split — an Owen value between solar and the devices,
Aumann–Shapley among the devices — so the shares and reimbursements add up to the
saving exactly. `src/home_energy_optimizer/dw/attribution.py` has the derivation.

### When it declines

A home automation loop must always get a plan, so the adapter returns `None` and
lets EMHASS run its default solver — logging the option responsible — rather
than raising. `unsupported()` lists the cases: a `costfun` other than `profit` or
`cost`, `set_total_pv_sell`, `set_nocharge_from_grid` with a battery,
`set_battery_first_priority`, more than one battery,
`heat_topology`, shared thermal tanks, `deferrable_load_groups` with
`mutual_exclusion` across participants, a node or constraint that splits an
EMHASS participant group, `group_limits` (read by 0.2.6 from a draft of
EMHASS's backend; now a node or a constraint of `electrical_topology`),
`cost_forecast_per_deferrable_load`, `set_deferrable_startup_penalty`,
`deferrable_load_max_cost`, capacity charges, and the `soc_target` family of
runtime arguments. An export price above the import price in some slot, or a
coordinator failure, falls back the same way.

A **hybrid inverter** is planned — a `hybrid_inverter` node of
`electrical_topology`, or EMHASS's own `inverter_is_hybrid` keys without one —
and the plan carries `P_hybrid_inverter` with EMHASS's meaning (+ DC to AC). It
still falls back with `set_nodischarge_to_grid` (EMHASS ties the battery to the
meter's direction then; likewise the battery and the PV on one node),
`inverter_stress_cost`, an inverter rated only by `pv_inverter_model` name, or
the battery inside an EMHASS participant group.

---

## The electrical tree: inverters, panels, breakers, several meters

Some devices reach the meter through their own connection: the PV and batteries
on a hybrid inverter's DC bus, a heat pump and an EV charger behind one breaker,
a sub-panel behind another, a second hybrid inverter on the first one's backup
port, a heat pump on a meter of its own with its own tariff. The site carries
that tree:

```python
from dataclasses import replace
from home_energy_optimizer import (BatteryConfig, HvacConfig, Horizon, SiteConfig, SubMeter,
                                   WaterHeaterConfig, demo_forecasts, group_limit, hybrid_inverter)
from home_energy_optimizer.dw.integrate import dw_plan
from home_energy_optimizer.types import GridConnection, SetLimit

h = Horizon(dt=0.5, hours=24)
site = SiteConfig(horizon=h, battery=BatteryConfig(), batteries=(BatteryConfig(capacity_kwh=6.0),),
    water_heater=WaterHeaterConfig(), hvac=HvacConfig(),
    submeters=(
        hybrid_inverter(("battery",), max_output_kw=5.0, eta_dc_ac=0.97, eta_ac_dc=0.97),
        group_limit("garage", (), max_kw=7.4, min_kw=0.0),               # a panel, no backfeed
        group_limit("heat", ("water_heater",), max_kw=3.5, parent="garage"),
        SubMeter("inv2", ("battery1", "carport"), max_export_kw=3.0, max_import_kw=3.0,
                 eta_export=0.96, eta_import=0.96, parent="garage"),      # a second inverter
    ),
    connections=(GridConnection("hp_meter", members=("hvac",)),),        # its own meter and tariff
    set_limits=(SetLimit("L1", ("battery", "water_heater"), max_import_kw=5.0),),  # a phase
)
fc = demo_forecasts(h, solar_peak_kw=8.0)                                # its solar is "pv"
fc = replace(fc, pv_arrays={"carport": 0.4 * fc.solar},
             tariffs={"hp_meter": (0.8 * fc.buy, 0.0 * fc.sell)})
res = dw_plan(site, fc)                    # or coordinate(site, fc): ADMM
```

- **Nodes** (`SubMeter`) nest to any depth through `parent`; each connection
  has ratings each way and a conversion efficiency each way (1: a panel).
- **Grid connections** (`GridConnection`) are further meters, each with its own
  tariff (`Forecasts.tariffs`) and limits; the main one is `"grid"`.
- **Set limits** (`SetLimit`) bound what a set of devices draw together, wherever
  they are, and may overlap each other and the tree.
- **PV arrays**: `Forecasts.solar` is `"pv"`; further arrays are
  `Forecasts.pv_arrays`, each placed by naming it in a node's members.

Every planner holds it. Dantzig–Wolfe gives each node and connection a balance
row and each set limit a row of its own, so a device is priced at **its bus's
price** (`DWResult.local_prices`) plus its set limits' premiums
(`DWResult.limit_prices`): its parent's price through the connection's
efficiency while the connection has headroom, apart from it at a rating — 0
while PV is being clipped, so a battery there stores PV that would otherwise be
lost. ADMM makes each bus and each set limit a net of its own, so its iterations
steer by them; the device DPs and the fast tier score each action up their chain
of nodes to their meter, at that meter's tariff; and the saving split prices
each device's kWh at its own local price. On battery-only sites the plan matches
an independent LP of the same tree exactly (`tests/test_topology.py`).
`docs/theory.tex`, "Sub-meters and local prices"; `bench/hybrid_inverter.py`
sweeps an inverter's rating. The DW page (Advanced → Site) has a **Hybrid
inverter** switch and an **Inverter limit** slider, and draws the bus price μ.

---

## Prices from a battery's dynamic programme

A convenience of one device solver, not the heart of the method. When the
battery is solved by a dynamic programme, its value function gives three prices
at whatever state the battery is in, which the Home Assistant path publishes
between plans (a battery solved as an LP gives its plan, and λ only along it).
They are easy to conflate, and not interchangeable.

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

**evcc.** `src/home_energy_optimizer/evcc.py` implements the optimizer HTTP contract used
by [evcc](https://evcc.io) (`POST /optimize/charge-schedule`), so a local server
can serve its `OPTIMIZER_URI` endpoint. Handles multiple batteries, loadpoints
modelled as charge-only batteries, per-slot charge demands and state-of-charge
goals, and a charger's minimum power as a semi-continuous floor. `docs/PLAN.md`
lists what is and is not mapped. Requests are planned with Dantzig–Wolfe: an
EV bids charging plans from its DP, evcc's grid limits are enforced in the
plan, and the response carries an extra `_hems_policy_certificate`. Start the
server with `HEMS_METHOD=admm` for ADMM.

**Real forecasts.** `home_energy_optimizer.feeds` provides keyless Open-Meteo PV and
temperature, CSV and list ingestion, and measured-value blending — anchoring the
first forecast slot to what was just measured and decaying back over four slots.

---

## Roadmap

### Remote participants: black-box solvers over the network

Today every participant runs in the coordinator's process, as Python
(`interface.Participant`: EMHASS's model, this package's DPs). Next, a solver
in another process, container or device — a vendor's EV-charging optimiser, a
heat pump's controller, EMHASS in its own container — should join without
sharing its model. The contract is the one the coordinator already speaks:

- **JSON Schemas** (stable, packaged): `schemas/device-query.v1.json` (a
  price response, a best response, or ADMM's proximal step: prices per slot,
  the horizon) and `schemas/device-answer.v1.json` (the plan in kW per slot, its
  private cost, its state trajectory, a status). `interface.query_to_dict` /
  `answer_from_dict` convert.
- **OpenAPI 3.1** (draft): `schemas/participant-api.v1.json` puts them on HTTP —
  `GET /v1/describe` (its key, its devices, its most power, whether it
  modulates or is on/off, which query kinds it answers, its slot lengths),
  `POST /v1/baseline` (a plan with no price), `POST /v1/query` (one what-if
  question), optional `POST /v1/blend` (a weighted mix of its plans, for a
  modulating participant) and `POST /v1/commit` (the plan chosen: the only call
  with an effect). `interface.schema("participant-api")` loads it.

What a remote solver must promise:

- **Queries are what-ifs.** Answering never changes what the device does; only
  a commit does. EMHASS's `dry_run` is this for EMHASS.
- **Always answer.** If its solve fails it answers with a plan it can run and
  status `fallback`; the coordinator uses the plan but proves no bound from it.
  A timeout is treated the same way, with its last good plan.
- **Answer price responses quickly** — a plan takes a few dozen of them — and
  the same question the same way.
- **Report its private cost honestly** (comfort, wear, energy left in store, a
  missed goal), in currency; the coordinator's bound and the saving split rely
  on it.

And what the coordinator promises it: prices only. A third-party participant
is never sent `best_response` (it carries the rest of the house's load); the
transport is local (mutual TLS on the LAN, or a paired token), and the
coordinator keeps no more than its answers.

### Which standards make sense to connect, and how

None of these is implemented yet; this is where the coordinator would meet
each, and why. The coordinator's question (a price response) and its answer
(a plan with its private cost) stay the same throughout; a standard is a
transport for it, or a source of the tree and its limits.

| standard | why it makes sense | what it would carry |
|---|---|---|
| [Matter](https://csa-iot.org/all-solutions/matter/) 1.3+ energy management | local only, mutually authenticated, and already in Home Assistant; devices ship it (EVSE, heat pumps, batteries, solar, water heaters in 1.4) | **the device edge**: the committed plan to the device; a node's local price handed to a device as its tariff, its forecast read back as its plan (a price response without a private cost, so no certificate); Power Topology (which endpoints a measurement covers) to seed the tree |
| [S2](https://s2standard.org) (EN 50491-12-2) | built for exactly this split - an energy manager and per-device resource managers, locally - and device-agnostic | **a participant per S2 resource manager**: its flexibility model (fill rate, operation modes, power envelopes) answers the coordinator's price responses on the coordinator's side |
| [EEBus](https://www.eebus.org) (SHIP/SPINE) | heat pumps and wallboxes in Germany speak it, and §14a obliges the limits | **§14a limits** (LPC/LPP) as connection limits; an incentive table out and a charging plan back is a price response |
| [OpenADR 3](https://www.openadr.org/openadr-3-0) | how utilities and aggregators send prices and capacity limits to a site | **into the root**: tariffs into `Forecasts`, `IMPORT_CAPACITY_LIMIT` / `EXPORT_CAPACITY_LIMIT` into the meter's limits; later, the house answering as one participant of a neighbourhood |
| [IEEE 2030.5](https://standards.ieee.org/ieee/2030.5/11216/) / CSIP-AUS | mandated for DER in California and Australia; carries dynamic operating envelopes | **into the root**: the site's `opModImpLimW` / `opModExpLimW` as the main connection's limits, per slot |

The site's tree can also be read rather than typed: evcc's `circuits` (parent,
`maxPower`) map one-to-one onto nodes, and an attested connection - the [IES
ElectricityCredential](https://india-energy-stack.gitbook.io/docs/schemas/electricitycredential/v1.2)
(sanctioned import and export, meters, registered DERs and their parents), or a
utility's CIM model - gives the root's limits and a starting tree.

Not planned: building-automation and charger protocols such as KNX and OCPP.
They execute a plan on one device, which Home Assistant's own integrations
already do; they do not carry the coordinator's question.

### Further

- **The tree through EMHASS**: several PV arrays and several meters (the
  library has them; EMHASS carries one PV forecast and one meter).
- **Phases**: a phase per device and per-phase limits at each node.
- **Mutual exclusion across participants**: an on/off constraint in the
  recovery step.
- **Network-side feeds**: a meter fed from either of two transformers (an open
  ring: the parent switches), and meshed feeds as constraints with sensitivity
  factors (PTDF) — the path from one house to a neighbourhood coordinator in
  which each house is a participant.
- **ADMM in EMHASS**: a proximal step in EMHASS's model (`docs/coordinated_backend.md`).
- **ADMM on deep trees**: its runnable-plan recovery is weak where a
  no-backfeed panel holds both storage and an on/off load (the relaxed plan is
  close to Dantzig-Wolfe's; the recovered one is not). Dantzig-Wolfe, the
  default, recovers jointly and is unaffected.

---

## Layout

```
src/home_energy_optimizer/     the package (import home_energy_optimizer)
  types.py         config + result dataclasses
  dp_battery.py    battery DP; returns the value function and policy
  dp_thermal.py    hot-water and HVAC DPs, and their thermostat baselines
  coordinate.py    plan scoring shared by both coordinators; coordinate() runs ADMM
  meter.py         a device's view of the meter: grid limits, and its bus behind a sub-meter
  submeter.py      the electrical tree: the meters from every device's power, through every node
  schemas/         the device query and answer (JSON Schema), the remote participant API (OpenAPI, draft)
  planner.py       plan(): Dantzig-Wolfe (default) or ADMM, one result type
  policy.py        value function -> actions, prices, counterfactuals
  feeds.py         real forecast inputs
  ha.py            Home Assistant publishing
  integrations/    emhass.py: EMHASS's devices as coordinated participants
  battery/         webapi.py: the single-battery page's backend
  evcc.py          evcc optimizer wire contract
  profiles.py      synthetic forecasts for tests and demos
  dw/              Dantzig-Wolfe: coordinator, LP solver, integrate (to HA and evcc),
                   attribution (who saves what), webapi (the DW app's backend)
  admm/            ADMM: coordinator, battery_qp (exact LP battery step),
                   webapi (the ADMM app's backend and the policy API)
dw/                the DW app: gui/ (server + page), wasm/ (single-file page), design notes
battery/           the single-battery page: one DP, its slow and fast tiers, the policy as a flow field
admm/              the ADMM app: gui/ (server + page; also the evcc endpoint), wasm/
wasm/              shared browser-build tooling: the compiled-kernel wheel, keep_names.py
bench/             exact references (continuous LP, joint DP, MILP, duals) and studies
tools/             theory build, page builder, HA dashboard setup, evcc checks
docs/theory.tex    theory notes (tools/build-theory makes the PDF)
docs/NOTES.md      measured findings
docs/PLAN.md       state and next steps
```

---

## Evidence

All measured; each number and the script behind it is in
[docs/theory.pdf](https://github.com/ameetdesh/home-energy-optimizer/blob/main/docs/theory.pdf), Appendix C.

**Cooperation pays.** For a battery and an on/off water heater over a day,
against an exact MILP of the same model (our own formulation, solved with
HiGHS), Dantzig–Wolfe captures 98.7–99.4% of the available savings in 0.6–0.7 s
and ADMM 97.3–99.1% in 4–8 s. On the Home Assistant demo site (battery, water
heater, HVAC) Dantzig–Wolfe captures 98–99% and ADMM 97–98%, and every
Dantzig–Wolfe plan carries its certified gap.

**Grid limits hold.** Dantzig–Wolfe meets a limit exactly or prices and reports
the breach; ADMM left at most 5 W over on 72 test sites.

**The split is fair.** Against the Shapley value, which needs a plan for every
coalition, the savings split came within 0.46 (within 0.13 for the batteries)
on six test cases, from one extra plan.

**The device solvers are accurate.** A battery's dynamic programme captures
98.4–99.2% of what its exact LP does; its λ lies within 0.007–0.032/kWh of the
LP's dual, and is published with a ±0.04/kWh band.

---

## Licence

MIT; see [LICENSE](https://github.com/ameetdesh/home-energy-optimizer/blob/main/LICENSE).
