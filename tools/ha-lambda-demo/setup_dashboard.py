"""Create the hems-policy dashboard in a running Home Assistant.

    HA_TOKEN=<token> python tools/ha-lambda-demo/setup_dashboard.py

Entities pushed through the REST `/api/states` endpoint are *orphan states* -
they have no entry in HA's entity registry, so the auto-generated Overview
dashboard may not show them at all. Creating a dashboard explicitly avoids
depending on that behaviour.

Three views, mirroring what the local GUI plots:
  Energy value  - lambda against the import price
  Power flows   - PV, load, net grid, battery
  Thermal       - tank, room and outdoor temperatures, and device power
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets") from None

P = "sensor.hems_"

CONFIG = {
    "views": [
        {
            "title": "Trade", "path": "trade", "icon": "mdi:swap-vertical",
            "cards": [
                {"type": "markdown", "content":
                 "## When to import and export\n\n"
                 "**Import** while the grid price is below `import_below`.\n"
                 "**Export** while it is above `export_above`.\n"
                 "In between, hold - the gap is the round-trip loss, a genuine "
                 "no-trade band.\n\n"
                 "`import_below = lambda * eta_c` and `export_above = lambda / eta_d`. "
                 "Compare against the price you actually face: the import tariff while "
                 "importing, the export price while exporting.\n\n"
                 "**Meter price** is the planner's price for one more kWh drawn at "
                 "the meter, for the whole house. **Plan gap** is how far the plan "
                 "can be, at most, from the best possible one."},
                {"type": "history-graph", "title": "Import price vs the no-trade band",
                 "hours_to_show": 1,
                 "entities": [P + "import_price", P + "import_below",
                              P + "export_above", P + "lambda"]},
                {"type": "entities", "title": "Now", "entities": [
                    {"entity": P + "import_below", "name": "Import while below"},
                    {"entity": P + "export_above", "name": "Export while above"},
                    {"entity": P + "lambda", "name": "lambda  value of a stored kWh"},
                    {"entity": P + "import_price", "name": "Grid import price"},
                    {"entity": P + "worth_running", "name": "Run a flexible load below"},
                    {"entity": P + "meter_price", "name": "Meter price (plan)"},
                    {"entity": P + "plan_gap", "name": "Plan gap (at most)"}]},
                {"type": "history-graph", "title": "Meter price vs the tariff",
                 "hours_to_show": 1,
                 "entities": [P + "import_price", P + "meter_price", P + "worth_running"]},
                {"type": "history-graph", "title": "What the battery does about it",
                 "hours_to_show": 1,
                 "entities": [P + "battery_action", P + "battery_soe"]},
            ],
        },
        {
            "title": "Power flows", "path": "power", "icon": "mdi:transmission-tower",
            "cards": [
                {"type": "history-graph", "title": "PV, load and net grid",
                 "hours_to_show": 1, "entities": [P + "pv", P + "load", P + "net_grid"]},
                {"type": "history-graph",
                 "title": "Battery: setpoint and state of energy", "hours_to_show": 1,
                 "entities": [P + "battery_action", P + "battery_soe"]},
                {"type": "gauge", "entity": P + "battery_soe", "name": "Battery",
                 "min": 0, "max": 10, "unit": "kWh",
                 "severity": {"green": 6, "yellow": 3, "red": 0}},
                {"type": "entities", "title": "Now", "entities": [
                    {"entity": P + "pv", "name": "PV production"},
                    {"entity": P + "load", "name": "Household load"},
                    {"entity": P + "net_grid", "name": "Net grid (+ import / − export)"},
                    {"entity": P + "battery_action", "name": "Battery setpoint"},
                    {"entity": P + "battery_soe", "name": "Battery state of energy"}]},
            ],
        },
        {
            "title": "Thermal", "path": "thermal", "icon": "mdi:thermometer",
            "cards": [
                {"type": "history-graph",
                 "title": "Temperatures: tank, room, outdoor", "hours_to_show": 1,
                 "entities": [P + "water_heater_temp", P + "hvac_temp",
                              P + "outdoor_temp"]},
                {"type": "history-graph", "title": "Thermal device power",
                 "hours_to_show": 1,
                 "entities": [P + "water_heater_power", P + "hvac_power"]},
                {"type": "entities", "title": "Now", "entities": [
                    {"entity": P + "water_heater_temp",
                     "name": "Hot water tank (setpoint 55 °C)"},
                    {"entity": P + "water_heater_power", "name": "Heater"},
                    {"entity": P + "hvac_temp", "name": "Room (comfort 22–26 °C)"},
                    {"entity": P + "hvac_power", "name": "HVAC"},
                    {"entity": P + "outdoor_temp", "name": "Outdoor"}]},
            ],
        },
    ]
}


async def main() -> None:
    token = os.environ.get("HA_TOKEN", "")
    url = os.environ.get("HA_URL", "http://127.0.0.1:8123").rstrip("/")
    if not token:
        raise SystemExit("set HA_TOKEN (Profile -> Long-lived access tokens)")
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        if json.loads(await ws.recv())["type"] != "auth_ok":
            raise SystemExit("authentication failed")

        i = 0

        async def call(msg: dict) -> dict:
            nonlocal i
            i += 1
            msg["id"] = i
            await ws.send(json.dumps(msg))
            while True:
                r = json.loads(await ws.recv())
                if r.get("id") == i:
                    return r

        r = await call({"type": "lovelace/dashboards/create", "url_path": "hems-policy",
                        "title": "hems-policy", "require_admin": False,
                        "show_in_sidebar": True, "mode": "storage", "icon": "mdi:flash"})
        if not r.get("success"):
            print("dashboard exists already:", r.get("error", {}).get("message", ""))

        r = await call({"type": "lovelace/config/save", "url_path": "hems-policy",
                        "config": CONFIG})
        if not r.get("success"):
            raise SystemExit(f"save failed: {r.get('error')}")
        print(f"dashboard ready at {url}/hems-policy")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
