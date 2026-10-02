# EMHASS and home-energy-optimizer, coordinated, next to Home Assistant

One plan for the house, made by two different optimisers:

- the **battery** is planned by home-energy-optimizer (its battery model);
- the **two deferrable loads** are planned by EMHASS's own model (its MILP);
- a Dantzig–Wolfe coordinator makes the plans agree at the meter, and splits
  the saving between solar, the battery and the loads.

EMHASS runs the coordination, in its own container. Home Assistant sees the
same sensors EMHASS always publishes (`sensor.p_batt_forecast`,
`sensor.p_deferrable0`, ...), plus the coordination's own:
`sensor.coordination_share_<player>`, `sensor.coordination_meter_price`,
`sensor.coordination_gap`.

> The coordinated backend is EMHASS PR
> [davidusb-geek/emhass#1158](https://github.com/davidusb-geek/emhass/pull/1158),
> not yet in a released EMHASS, so EMHASS is built from that branch. It needs
> home-energy-optimizer **0.2.2** or later.

## What you need

- Home Assistant running, and a **long-lived access token**: your profile
  (bottom left) → *Security* → *Long-lived access tokens* → *Create token*.
- Docker and git.
- This repository, with its `.venv` (any Python 3.10+ works; the script uses
  only the standard library).

This works with **Home Assistant Container** (the Docker image), which has no
add-on store, and with any other install that can reach EMHASS over HTTP.

## Quick start

```bash
export HA_TOKEN=<your token>            # HA_URL defaults to http://127.0.0.1:8123

python tools/emhass-coordination/coordinate.py up      # build EMHASS + start it on :5050
python tools/emhass-coordination/coordinate.py run     # plan once, publish to HA
python tools/emhass-coordination/coordinate.py run --every 30   # keep re-planning (MPC)
python tools/emhass-coordination/coordinate.py down    # stop EMHASS
```

The first `up` builds EMHASS's image, which takes several minutes, and its
first start takes a minute or two more. Later runs are quick.

`run` prints the coordinator's own line, then the plan and the shares:

```
coordinated: dantzig_wolfe: 1 iterations, gap -0.0000, 11 participant solves, 0.22 s

plan for 48 steps from Fri 14:30 PDT  (0.3 s)
  time   PV W  load W  batt W   SoC  def0 W  def1 W  grid W   buy  meter
 14:30   4719     506   -1113   55%       0       0   -3100  0.20  0.050
 ...
share of the saving over the horizon (currency):
  solar                           4.584
  battery                         1.285
  deferrable0+deferrable1         0.420
```

If it says **"EMHASS planned WITHOUT the coordinator"** instead, EMHASS fell
back to its single MILP and the next line names why (see *When it falls back*).

## Where the coordination is switched on

In **EMHASS's configuration file**, `config.json`. Inside the container it is
`/share/config.json`; here it is [`config.json`](config.json), next to the
script, which `up` copies into `.run/config.json` and mounts. Two keys do it:

```json
"optimization_backend": "dantzig_wolfe",
"participants": [
  {"devices": ["battery"], "solver": "home_energy_optimizer"},
  {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"}
]
```

- `optimization_backend`: `cvxpy` (EMHASS's default, one MILP) or
  `dantzig_wolfe` (coordinated).
- `participants`: which solver plans which devices. EMHASS's device names are
  `battery` and `deferrable0`, `deferrable1`, … (as many as
  `number_of_deferrable_loads`). A device you leave out is planned by EMHASS on
  its own.

The rest of `config.json` is ordinary EMHASS configuration: the battery
(`battery_nominal_energy_capacity` in Wh, the power limits in W, the SoC
window, the efficiencies), the deferrable loads (`nominal_power_of_deferrable_loads`,
`operating_hours_of_each_deferrable_load`), `optimization_time_step` (minutes)
and `costfun`.

To change it, edit `tools/emhass-coordination/config.json` and run `up` again.
EMHASS's web UI (<http://127.0.0.1:5050/configuration>) shows both new
settings too; saving there takes effect at once but only lasts until the next
`up`, which copies the file again.

## What `up` does, step by step

So you can do it by hand, or adapt it:

1. **Get EMHASS from the PR branch.**
   `git clone --branch federated-all https://github.com/ameetdesh/emhass.git`
   (or pass an existing checkout with `--emhass-src`).
2. **Build its image.**
   `docker build --build-arg TARGETARCH=aarch64 -t emhass-coordinated-base <checkout>`
   (`amd64` on an Intel/AMD machine).
3. **Add home-energy-optimizer.** EMHASS's Dockerfile installs no optional
   extras, so a one-line image on top does:
   `FROM emhass-coordinated-base` / `RUN uv pip install "home-energy-optimizer>=0.2.2"`.
   (Outside Docker, `pip install "emhass[federated]"` from the branch does both.)
4. **Write EMHASS's secrets** (`.run/secrets_emhass.yaml`, readable only by
   you): Home Assistant's URL and your token, and the time zone and location
   read from Home Assistant itself. Inside the container, `127.0.0.1` is
   `host.docker.internal`.
5. **Start it**: port 5050 on the host, `.run/config.json` as
   `/share/config.json`, the secrets as `/app/secrets_emhass.yaml`, and
   `.run/data` as `/data` (where EMHASS keeps its last plan).

## What `run` does

1. Builds 24 h of forecasts at EMHASS's 30-minute step, aligned to Home
   Assistant's clock: PV, house load, import and export prices. By default a
   plausible demo day; `--forecasts my_day.json` passes your own
   (`pv_power_forecast` and `load_power_forecast` in W,
   `load_cost_forecast` and `prod_price_forecast` per kWh, one value per step).
2. `POST /action/naive-mpc-optim` with those forecasts, the battery's state of
   charge now (`--soc`, 0-1) and each load's required hours (`--hours`).
   Passing the forecasts means EMHASS needs no sensor history.
3. `POST /action/publish-data`: EMHASS publishes its sensors to Home Assistant.
4. Reads the plan back (`GET /api/v1/plan`), prints it, and publishes the
   coordination's sensors.
5. Checks EMHASS's log for the coordinator's line, so a silent fallback is
   visible.

## Running it from Home Assistant instead

`run --every 30` is the simplest automation. To have Home Assistant trigger
the plan itself, the usual EMHASS pattern is a `rest_command` plus an
automation, in Home Assistant's `configuration.yaml`:

```yaml
rest_command:
  emhass_coordinated_mpc:
    url: http://host.docker.internal:5050/action/naive-mpc-optim
    method: POST
    content_type: application/json
    payload: >-
      {"prediction_horizon": 48,
       "soc_init": {{ states('sensor.battery_state_of_charge') | float(50) / 100 }},
       "soc_final": 0.5}
  emhass_publish:
    url: http://host.docker.internal:5050/action/publish-data
    method: POST
    content_type: application/json
    payload: "{}"

automation:
  - alias: "Coordinated plan every 30 minutes"
    trigger: {platform: time_pattern, minutes: "/30"}
    action:
      - service: rest_command.emhass_coordinated_mpc
      - service: rest_command.emhass_publish
```

**This needs real sensors.** Without forecasts in the request, EMHASS reads the
house load's history from Home Assistant (`sensor_power_load_no_var_loads` in
`config.json`, the load without the deferrable loads, in W) and forecasts PV
from your location. On a Home Assistant without those sensors the request
fails, so use `run` until you have them. Point `sensor_power_load_no_var_loads`,
`sensor_power_photovoltaics` and `sensor_battery_state_of_charge` in `config.json` at
yours, and give them a couple of days of history.

## When it falls back

EMHASS plans with its single MILP, and logs one line naming the option, when
the configuration uses something the coordinator cannot split per device yet:
`costfun: self-consumption`, `set_total_pv_sell`, `set_nocharge_from_grid`,
`set_battery_first_priority`, a hybrid inverter, more than one battery,
thermal tanks shared between loads, deferrable load groups, startup penalties,
capacity charges, or the runtime `soc_target`. A participant that names the
battery while `set_use_battery` is off falls back too. A plan is always
published either way.

## Troubleshooting

- **Port 5000.** EMHASS listens on 5000 inside the container, but on macOS the
  host's 5000 belongs to AirPlay Receiver, hence 5050. Use `up --port N` and
  `EMHASS_URL=http://127.0.0.1:N` for `run` to change it.
- **Linux.** `--add-host host.docker.internal:host-gateway` (which `up`
  passes) lets the container reach Home Assistant on the host.
- **Logs.** `docker logs -f emhass-coordinated`.
