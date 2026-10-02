# λ in Home Assistant

Phase 5 step 1 of [`docs/PLAN.md`](../../docs/PLAN.md): put the marginal value
of a stored kWh in front of a user, in the one place where an automation can
already act on it.

This is the first time the package's *differentiating* capability reaches
anyone. The evcc integration replaces a planner — useful, but competing on the
thing we are only 91–96% as good at. λ is the thing neither EMHASS nor evcc has.

## Run it

```bash
pip install -e ".[ha]"   # websockets, for the HA websocket API

docker run -d --name ha -p 8123:8123 -v /tmp/ha-config:/config \
    homeassistant/home-assistant:stable
# complete onboarding, then Profile -> Security -> Long-lived access tokens

HA_TOKEN=<token> python tools/ha-lambda-demo/run.py --speed 400
```

### Seeing it in the browser

```bash
HA_TOKEN=<token> python tools/ha-lambda-demo/setup_dashboard.py
# -> dashboard ready at http://127.0.0.1:8123/home-energy-optimizer
```

Then open <http://localhost:8123/home-energy-optimizer>, or use the sidebar entry.

**Do not rely on the default Overview dashboard.** Entities pushed through the
REST `/api/states` endpoint are *orphan states* — they have no entry in HA's
entity registry, so the auto-generated Overview may not list them at all. The
states are there (`/api/states` shows them); nothing renders them. Hence the
explicit dashboard.

Three views, mirroring the local GUI:

| view | shows |
|---|---|
| **Energy value** | λ against the import price — the signal |
| **Power flows** | PV, household load, net grid, battery setpoint and state of energy |
| **Thermal** | tank, room and outdoor temperatures; heater and HVAC power |

Pick a `--speed` you can watch: `--speed 2` runs a simulated day in ~12 real
minutes, which is slow enough for the history graph to draw a curve. `--speed 400`
is a day in 4 seconds — good for a smoke test, too fast to watch.

```
 sim time   price   lambda   action     soe  tier
      0:00   0.150   0.1009    -0.40    5.00  lookup
      5:45   0.150   0.1042    -0.80    2.17  lookup
      6:00   0.150   0.1042    -1.00    1.94  SOLVE
     21:00   0.400   0.1500    -1.20    5.06  lookup
```

The two tiers the package argues for are visible in the last column: `SOLVE`
re-runs the optimiser on the cadence forecasts actually change on, `lookup`
publishes from the stored value function — a table lookup, no solver.

## What lands in Home Assistant

| entity | meaning |
|---|---|
| `sensor.hems_lambda` | **marginal value of a stored kWh**, with a `forecast` attribute over the horizon |
| `sensor.hems_worth_running` | the threshold a flexible load compares itself against |
| `sensor.hems_battery_action` | optimal battery setpoint now (kW, + = charge) |
| `sensor.hems_battery_soe` | state of energy (kWh, plus `soc_percent`) |
| `sensor.hems_import_price` | prevailing tariff, for context |
| `sensor.hems_pv` | PV production (kW) |
| `sensor.hems_load` | inflexible household load (kW) |
| `sensor.hems_net_grid` | net exchange, + import / − export (kW) |
| `sensor.hems_water_heater_temp` / `_power` | tank temperature and heater draw |
| `sensor.hems_hvac_temp` / `_power` | room temperature and HVAC draw |
| `sensor.hems_outdoor_temp` | outdoor temperature |

Thermal and PV values come from the plan rather than a simulation: those devices
have no feedback loop here, so their planned trajectory *is* what would happen
absent a disturbance. Only the battery is advanced by the policy's own action.

Measured curve from a live run (day/night tariff, 0.15 off-peak → 0.40 peak,
0.08 export):

| hour | import | λ | |
|---|---|---|---|
| 00:00 | 0.150 | 0.1350 | overnight |
| 12:00 | 0.400 | **0.0889** | PV spilling — a stored kWh is only worth the export price, `sell/η` = 0.08/0.9 |
| 18:00 | 0.400 | **0.3564** | evening peak — worth the import it displaces, `buy·η` = 0.40×0.9 |

λ sits at its theoretical bounds at both extremes. That is the signal working.

## The point: pricing a device the optimizer has never heard of

λ is published **with an uncertainty band**, because validation put it within
~0.04/kWh of two independent ground truths (`docs/NOTES.md` §3). A decision
inside that band is not reliable, so the sensor carries `confident_above` and
`confident_below` rather than pretending to a precision it does not have.

A tumble dryer worth ~0.25/kWh to run now:

```yaml
automation:
  - alias: Run the dryer when stored energy is cheap enough
    trigger:
      - platform: state
        entity_id: sensor.hems_lambda
    condition:
      # Compare against the CONFIDENT edge, not the bare number - inside the
      # band the signal cannot tell you which way to go.
      - condition: numeric_state
        entity_id: sensor.hems_lambda
        attribute: confident_above
        below: 0.25
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.tumble_dryer
```

On the curve above that runs the dryer at midday (λ 0.089, well below 0.25) and
holds it in the evening (λ 0.356, well above). **The dryer is not modelled
anywhere in the optimiser** — it just consumes a price. That is the whole
argument: the optimizer's complexity stops growing with the number of devices.

## Chart it

`forecast` follows EMHASS's attribute shape, so ApexCharts configs carry over:

```yaml
type: custom:apexcharts-card
header:
  show: true
  title: Marginal value of stored energy
series:
  - entity: sensor.hems_lambda
    data_generator: |
      return entity.attributes.forecast.map(p =>
        [Date.now() + p.hours_ahead * 3600000, p.lambda]);
  - entity: sensor.hems_lambda
    name: Import price
    data_generator: |
      return entity.attributes.forecast.map(p =>
        [Date.now() + p.hours_ahead * 3600000, p.import_price]);
```

## Caveats

- The demo uses `demo_forecasts` synthetics. Point it at real feeds with
  `home_energy_optimizer.feeds` (keyless Open-Meteo PV, CSV, or a list from any tariff
  integration).
- It writes states over the REST API, so entities vanish on an HA restart.
  A real deployment would use MQTT discovery or a custom component.
- A single battery, no thermal devices — enough to make λ legible.
