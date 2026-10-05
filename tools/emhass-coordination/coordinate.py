#!/usr/bin/env python3
"""Run EMHASS and home-energy-optimizer as one coordinated planner, next to
Home Assistant.

    HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py up     # build + start EMHASS on :5050
    HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py run    # plan once, publish to HA
    HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py run --every 30
    python tools/emhass-coordination/coordinate.py down                    # stop EMHASS

    # a hybrid inverter (PV and battery on its DC bus, 4 kW AC), 8 kW of PV:
    HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py up --config config_hybrid.json
    HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py run --pv-peak 8000

    # four DERs on two solvers, on an electrical topology (an inverter, a garage
    # panel with no backfeed, a 3.5 kW breaker under it):
    HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py up --config config_four_der.json
    HA_TOKEN=<token> python tools/emhass-coordination/coordinate.py run --pv-peak 8000

`up` builds EMHASS from the branch of davidusb-geek/emhass#1158 (the
coordinated backend is not in a released EMHASS yet), adds
home-energy-optimizer, writes EMHASS's secrets from Home Assistant's own
settings, and starts the container with `config.json` (next to this script)
mounted as EMHASS's configuration. That file is where the coordination is
switched on: `optimization_backend` and `participants`.

`run` asks EMHASS for a 24 h plan (naive MPC), with the forecasts passed in
the request, so it needs no sensor history: a demo day aligned to Home
Assistant's clock, or your own with `--forecasts`. EMHASS publishes its usual
sensors; this adds the coordination's own - each participant's share of the
saving and the price at the meter - and checks the coordinated backend
actually ran rather than falling back to EMHASS's single solver.

Stdlib only. Needs Docker, git, and Home Assistant reachable at HA_URL
(default http://127.0.0.1:8123).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BUILD, RUN = HERE / ".build", HERE / ".run"
EMHASS_REPO = "https://github.com/ameetdesh/emhass.git"
EMHASS_BRANCH = "federated-all"            # davidusb-geek/emhass#1158
PACKAGE = "home-energy-optimizer>=0.2.7"
IMAGE, CONTAINER = "emhass-coordinated", "emhass-coordinated"
# EMHASS listens on 5000 inside the container. On the host, 5000 is taken by
# macOS (AirPlay Receiver), so it is published on 5050 by default.
PORT = 5050
STEP_MIN, HORIZON = 30, 48                 # EMHASS's optimization_time_step; 24 h


# ---------------------------------------------------------------- plumbing
def sh(*cmd: str, check: bool = True, capture: bool = False) -> str:
    out = subprocess.run(cmd, text=True, stdout=subprocess.PIPE if capture else None,
                         stderr=subprocess.STDOUT if capture else None)
    if check and out.returncode:
        sys.exit(f"failed: {' '.join(cmd[:3])} ...\n{(out.stdout or '').strip()}")
    return out.stdout or ""


def http(method: str, url: str, body: Any = None, token: str | None = None,
         timeout: float = 600) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        raw = r.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def ha_settings() -> tuple[str, str]:
    token = os.environ.get("HA_TOKEN", "")
    if not token:
        sys.exit("set HA_TOKEN (Home Assistant: your profile -> Security -> Long-lived access tokens)")
    return os.environ.get("HA_URL", "http://127.0.0.1:8123").rstrip("/"), token


def emhass_url() -> str:
    return os.environ.get("EMHASS_URL", f"http://127.0.0.1:{PORT}").rstrip("/")


# ---------------------------------------------------------------- up / down
def up(args: argparse.Namespace) -> None:
    ha, token = ha_settings()
    try:
        conf = http("GET", f"{ha}/api/config", token=token, timeout=10)
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"cannot reach Home Assistant at {ha}: {exc}")

    # 1. EMHASS, from the PR branch
    src = Path(args.emhass_src) if args.emhass_src else BUILD / "emhass"
    if not args.emhass_src:
        if (src / ".git").exists():
            sh("git", "-C", str(src), "fetch", "--depth", "1", "origin", EMHASS_BRANCH)
            sh("git", "-C", str(src), "checkout", "-q", "FETCH_HEAD")
        else:
            BUILD.mkdir(exist_ok=True)
            sh("git", "clone", "-q", "--depth", "1", "--branch", EMHASS_BRANCH, EMHASS_REPO, str(src))
    arch = "aarch64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    print(f"building EMHASS from {src} ({arch}); the first build takes several minutes")
    sh("docker", "build", "-q", "--build-arg", f"TARGETARCH={arch}", "-t", f"{IMAGE}-base", str(src))

    # 2. ...plus home-energy-optimizer, which EMHASS's Dockerfile does not
    #    install (it installs no optional extras)
    #    - from PyPI, or from a local checkout (`--package-src`), built to a wheel
    BUILD.mkdir(exist_ok=True)
    for old in BUILD.glob("*.whl"):
        old.unlink()
    if args.package_src:
        sh(sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(BUILD), args.package_src)
        wheel = next(BUILD.glob("home_energy_optimizer-*.whl")).name
        (BUILD / "Dockerfile").write_text(f"FROM {IMAGE}-base\nCOPY {wheel} /tmp/\nRUN uv pip install /tmp/{wheel}\n")
    else:
        (BUILD / "Dockerfile").write_text(f'FROM {IMAGE}-base\nRUN uv pip install "{PACKAGE}"\n')
    sh("docker", "build", "-q", "-t", IMAGE, str(BUILD))

    # 3. EMHASS's secrets, from Home Assistant's own settings. Inside the
    #    container, the host's 127.0.0.1 is host.docker.internal.
    RUN.mkdir(exist_ok=True)
    (RUN / "data").mkdir(exist_ok=True)
    hass_url = ha.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    secrets = RUN / "secrets_emhass.yaml"
    secrets.touch(mode=0o600)
    secrets.write_text(
        f"server_ip: 0.0.0.0\nhass_url: {hass_url}/\nlong_lived_token: {token}\n"
        f"time_zone: {conf['time_zone']}\nLatitude: {conf['latitude']}\n"
        f"Longitude: {conf['longitude']}\nAltitude: {conf.get('elevation', 0)}\n")
    secrets.chmod(0o600)

    # 4. start it, with a copy of config.json as EMHASS's configuration. A copy,
    #    because saving in EMHASS's web UI rewrites the mounted file; the one
    #    next to this script stays the source of truth, re-copied on every `up`.
    config = Path(args.config) if Path(args.config).is_absolute() else HERE / args.config
    (RUN / "config.json").write_text(config.read_text())
    print(f"EMHASS's configuration: {config.name}")
    sh("docker", "rm", "-f", CONTAINER, check=False, capture=True)
    sh("docker", "run", "-d", "--name", CONTAINER, "--restart", "unless-stopped",
       "-p", f"{args.port}:5000", "--add-host", "host.docker.internal:host-gateway",
       "-v", f"{RUN / 'config.json'}:/share/config.json",
       "-v", f"{secrets}:/app/secrets_emhass.yaml:ro",
       "-v", f"{RUN / 'data'}:/data", IMAGE, capture=True)   # fails with Docker's message, e.g. a busy port
    # The first start syncs EMHASS's environment (uv run --frozen), a minute or two.
    print("waiting for EMHASS to start (the first start takes a minute or two)")
    for _ in range(300):
        try:
            http("GET", f"http://127.0.0.1:{args.port}/healthz", timeout=2)
            break
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    else:
        sys.exit(f"EMHASS did not start; see: docker logs {CONTAINER}")
    have = sh("docker", "exec", CONTAINER, "/app/.venv/bin/python", "-c",
              "import home_energy_optimizer as h; print(h.__version__)", check=False, capture=True).strip()
    print(f"EMHASS is up on http://127.0.0.1:{args.port} (home-energy-optimizer {have}), "
          f"talking to Home Assistant at {hass_url} ({conf['time_zone']})")


def down(_args: argparse.Namespace) -> None:
    sh("docker", "rm", "-f", CONTAINER, check=False, capture=True)
    print(f"stopped {CONTAINER}")


# ---------------------------------------------------------------- run
def demo_forecasts(start: datetime, pv_peak: float = 5000.0) -> dict[str, list[float]]:
    """A plausible day from `start`, one value per step, by local hour: PV up
    to `pv_peak` W at noon, a morning and an evening load, a day/night tariff
    with an evening peak, a flat export price, and a cool day outside (5-15 C,
    for a heat pump planned by home-energy-optimizer)."""
    hrs = [(start + timedelta(minutes=STEP_MIN * i)) for i in range(HORIZON)]
    hod = [t.hour + t.minute / 60 for t in hrs]
    return {
        "pv_power_forecast": [round(max(0.0, pv_peak * math.sin(math.pi * (h - 6) / 14))) if 6 <= h <= 20 else 0.0
                              for h in hod],
        "load_power_forecast": [round(500 + 1000 * math.exp(-((h - 19) / 2) ** 2)
                                      + 300 * math.exp(-((h - 7.5) / 1) ** 2)) for h in hod],
        "load_cost_forecast": [0.30 if 17 <= h < 21 else 0.12 if (h < 6 or h >= 23) else 0.20 for h in hod],
        "prod_price_forecast": [0.05] * HORIZON,
        "outdoor_temperature_forecast": [round(10 + 5 * math.sin(math.pi * (h - 9) / 12), 1) for h in hod],
    }


def run_once(args: argparse.Namespace, ha: str, token: str) -> None:
    em = emhass_url()
    tz = ZoneInfo(http("GET", f"{ha}/api/config", token=token, timeout=10)["time_zone"])
    now = datetime.now(tz)
    start = now.replace(minute=(now.minute // STEP_MIN) * STEP_MIN, second=0, microsecond=0)
    if args.forecasts:
        fc = json.loads(Path(args.forecasts).read_text())
    else:
        fc = demo_forecasts(start, args.pv_peak)
    body = {**fc, "prediction_horizon": len(fc["load_cost_forecast"]),
            "soc_init": args.soc, "soc_final": args.soc,
            "operating_hours_of_each_deferrable_load": args.hours}

    since = datetime.now().astimezone().isoformat()
    t0 = time.perf_counter()
    http("POST", f"{em}/action/naive-mpc-optim", body)
    secs = time.perf_counter() - t0
    http("POST", f"{em}/action/publish-data", {})

    # Did the coordinated backend run, or did EMHASS fall back to its MILP?
    log = sh("docker", "logs", "--since", since, CONTAINER, check=False, capture=True)
    ran = [l for l in log.splitlines() if "optimization_backend=dantzig_wolfe:" in l]
    fell = [l for l in log.splitlines() if "using the default solver" in l]
    if fell or not ran:
        print("!! EMHASS planned WITHOUT the coordinator:")
        print("   " + (fell[-1].split("] ", 1)[-1] if fell else "no coordinator line in its log"))
    else:
        print("coordinated:", ran[-1].split("optimization_backend=", 1)[-1])

    plan = http("GET", f"{em}/api/v1/plan")["plan"]
    first = plan[0]
    shares = {k[len("fed_share_"):]: float(v) for k, v in first.items() if k.startswith("fed_share_")}
    local = lambda ts: datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)  # noqa: E731
    hybrid = "P_hybrid_inverter" in first
    thermal = [(c, label) for c, label in (("P_water_heater", "tank W"), ("P_hvac", "hp W")) if c in first]
    print(f"\nplan for {len(plan)} steps from {local(first['timestamp']):%a %H:%M %Z}  ({secs:.1f} s); "
          f"{first.get('optim_status', '?')}"
          + (f", the coordinator {first['fed_stop_reason']} after {first['fed_iterations']} rounds"
             if "fed_stop_reason" in first else ""))
    print(f"{'time':>6} {'PV W':>6} {'load W':>7} {'batt W':>7} {'SoC':>5} {'def0 W':>7} {'def1 W':>7} "
          + "".join(f"{label:>6} " for _, label in thermal)
          + (f"{'inv W':>6} {'clip W':>6} " if hybrid else "") + f"{'grid W':>7} {'buy':>5} {'meter':>6}")
    for r in plan[:: max(1, len(plan) // 12)]:
        print(f"{local(r['timestamp']):%H:%M}".rjust(6) + f" {r['P_PV']:>6.0f} {r['P_Load']:>7.0f} {r['P_batt']:>7.0f} "
              f"{100 * r['SOC_opt']:>4.0f}% {r.get('P_deferrable0', 0):>7.0f} {r.get('P_deferrable1', 0):>7.0f} "
              + "".join(f"{r[c]:>6.0f} " for c, _ in thermal)
              + (f"{r['P_hybrid_inverter']:>6.0f} {r.get('P_PV_curtailment', 0):>6.0f} " if hybrid else "")
              + f"{r['P_grid']:>7.0f} {r['unit_load_cost']:>5.2f} {r.get('fed_meter_price', float('nan')):>6.3f}")
    if hybrid:
        dt = STEP_MIN / 60
        clipped = sum(r.get("P_PV_curtailment", 0) for r in plan) * dt / 1000
        print(f"\nhybrid inverter: at most {max(r['P_hybrid_inverter'] for r in plan):.0f} W to the house "
              f"(inv W: + DC to AC), {clipped:.1f} kWh of PV not used")
    local_cols = {k[len("fed_local_price_"):]: k for k in first if k.startswith("fed_local_price_")}
    limit_cols = {k[len("fed_limit_price_"):]: k for k in first if k.startswith("fed_limit_price_")}
    node_cols = {k[len("fed_node_power_"):]: k for k in first if k.startswith("fed_node_power_")}
    if node_cols:
        print("\nnodes of the topology: power to the parent (W, + = up the tree), over the horizon")
        for name, col in node_cols.items():
            vals = [float(r[col]) for r in plan]
            print(f"  {name:<28} {min(vals):>8.0f} .. {max(vals):.0f}")
    if local_cols or limit_cols:
        print("\nlocal prices (currency/kWh) on each node the coordinator holds; the meter's for comparison:")
        meter = [float(r["fed_meter_price"]) for r in plan]
        print(f"  {'meter':<28} {min(meter):>6.3f} .. {max(meter):.3f}")
        for name, col in local_cols.items():
            vals = [float(r[col]) for r in plan]
            print(f"  {name:<28} {min(vals):>6.3f} .. {max(vals):.3f}")
        for name, col in limit_cols.items():
            vals = [float(r[col]) for r in plan]
            print(f"  {name + ' (premium)':<28} {min(vals):>6.3f} .. {max(vals):.3f}")
    if shares:
        print("\nshare of the saving over the horizon (currency):")
        for player, v in shares.items():
            print(f"  {player:<28} {v:>8.3f}")

    # The coordination's own sensors, beside EMHASS's
    unit = "$"
    for player, v in shares.items():
        slug = player.replace("+", "_").replace(" ", "_")
        http("POST", f"{ha}/api/states/sensor.coordination_share_{slug}", token=token, body={
            "state": round(v, 3), "attributes": {
                "friendly_name": f"Saving share: {player}", "unit_of_measurement": unit,
                "icon": "mdi:handshake"}})
    if "fed_meter_price" in first:
        http("POST", f"{ha}/api/states/sensor.coordination_meter_price", token=token, body={
            "state": round(float(first["fed_meter_price"]), 4), "attributes": {
                "friendly_name": "Coordinated meter price", "unit_of_measurement": f"{unit}/kWh",
                "forecasts": [{"date": r["timestamp"], "price": round(float(r["fed_meter_price"]), 4)}
                              for r in plan]}})
        http("POST", f"{ha}/api/states/sensor.coordination_gap", token=token, body={
            "state": round(float(first["fed_gap"]), 4), "attributes": {
                "friendly_name": "Coordination gap (cost above the lower bound)",
                "unit_of_measurement": unit}})
    for kind, cols, title in (("local_price", local_cols, "Local price"), ("limit_price", limit_cols, "Limit premium")):
        for name, col in cols.items():
            slug = name.replace("+", "_")
            http("POST", f"{ha}/api/states/sensor.coordination_{kind}_{slug}", token=token, body={
                "state": round(float(first[col]), 4), "attributes": {
                    "friendly_name": f"{title}: {name}", "unit_of_measurement": f"{unit}/kWh",
                    "forecasts": [{"date": r["timestamp"], "price": round(float(r[col]), 4)} for r in plan]}})
    print(f"\npublished to {ha}: EMHASS's sensor.p_batt_forecast, sensor.p_deferrable0/1, "
          f"sensor.soc_batt_forecast, ...; and sensor.coordination_share_*, "
          f"sensor.coordination_meter_price, sensor.coordination_gap"
          + "".join(f", sensor.coordination_local_price_{n.replace('+', '_')}" for n in local_cols)
          + "".join(f", sensor.coordination_limit_price_{n.replace('+', '_')}" for n in limit_cols))


def run(args: argparse.Namespace) -> None:
    ha, token = ha_settings()
    while True:
        run_once(args, ha, token)
        if not args.every:
            return
        print(f"\nnext plan in {args.every} min (Ctrl-C to stop)\n")
        time.sleep(args.every * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="build EMHASS (PR branch) + home-energy-optimizer, and start it")
    u.add_argument("--emhass-src", help="an existing EMHASS checkout of the PR branch (default: clone it)")
    u.add_argument("--port", type=int, default=PORT, help=f"host port for EMHASS (default {PORT}; "
                   "set EMHASS_URL to match for `run` if you change it)")
    u.add_argument("--config", default="config.json",
                   help="EMHASS configuration to start with, next to this script or a path "
                        "(config_hybrid.json: a hybrid inverter)")
    u.add_argument("--package-src", help="install home-energy-optimizer from this local checkout "
                   f"instead of PyPI ({PACKAGE})")
    u.set_defaults(fn=up)
    r = sub.add_parser("run", help="plan, publish to Home Assistant, report the coordination")
    r.add_argument("--forecasts", help="JSON with pv_power_forecast, load_power_forecast (W), "
                   "load_cost_forecast, prod_price_forecast (per kWh), one value per 30 min")
    r.add_argument("--soc", type=float, default=0.5, help="battery state of charge now, 0-1")
    r.add_argument("--hours", type=float, nargs="+", default=[3, 4],
                   help="hours each deferrable load must run")
    r.add_argument("--every", type=float, default=0, help="re-plan every N minutes (MPC)")
    r.add_argument("--pv-peak", type=float, default=5000.0, help="the demo day's PV peak, W (default 5000)")
    r.set_defaults(fn=run)
    d = sub.add_parser("down", help="stop and remove the EMHASS container")
    d.set_defaults(fn=down)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
