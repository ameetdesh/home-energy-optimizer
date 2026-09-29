"""Local GUI for exercising the headless core, and a prototype of the Phase 3 sidecar.

Run:  .venv/bin/python admm/gui/server.py    then open http://127.0.0.1:8765

Deliberately stdlib-only (`http.server`), so it adds no dependency to a package
whose whole point is being embeddable. It runs the *same* code the tests run -
no Pyodide, no WASM, no second implementation to drift.

The endpoints are the ones docs/PLAN.md Phase 3 specifies, so this is a working
draft of the sidecar rather than a throwaway demo:

    POST /api/solve      full config      -> plan + lambda curve + costs
    GET  /api/round      ?r=              -> lambda for one coordination round
    GET  /api/policy     ?t=&soe=         -> optimal action at a state
    GET  /api/lambda     ?t=&soe=         -> marginal value of a stored kWh
    POST /api/evaluate   {t, soe, action} -> opportunity cost of an override
    GET  /api/rollout    ?t=&soe=         -> counterfactual trajectory
    POST /api/setpoint   {t, soe, ...}    -> the SAFE action an actuator should use

It also speaks evcc's optimizer contract, so a stock evcc binary can use this
solver with no code change at all:

    POST /optimize/charge-schedule   evcc OptimizationInput -> OptimizationResult
    GET  /optimize/health

    OPTIMIZER_URI=http://127.0.0.1:8765 evcc

evcc requests are planned with Dantzig-Wolfe; start the server with
HEMS_METHOD=admm to plan them with the ADMM loop instead.

`/api/evaluate` and `/api/rollout` are the interesting ones: neither EMHASS nor
evcc can answer them.

`/api/setpoint` is the one an actuator should actually call. The DP enforces
SoC gates as big-M penalties, which is fine for preferences and wrong for
safety, so fuse limits and grid-operator dimming are clamped OUTSIDE the solver
(`policy.clamp`) and the response says which limit bound the answer. It also
ramp-limits: evcc's 3-minute disable / 1-minute enable hysteresis exists
because chargers cannot be cycled quickly, so "fast" has to mean smoother
setpoints, not faster switching.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from home_energy_optimizer.admm import webapi  # noqa: E402
from home_energy_optimizer.evcc import (  # noqa: E402
    ContractError,
    health as evcc_health,
    optimize_charge_schedule,
)

HERE = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    """HTTP in front of `admm.webapi`.

    Every /api/<name> path maps straight onto webapi.ROUTES, so this class
    knows nothing about the solver - which is what lets the standalone page in
    admm/wasm/ call the same functions in the browser with no server at all.
    """

    def log_message(self, *_args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _api(self, name: str, payload: dict) -> None:
        self._json(webapi.call(name, payload))

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/optimize/health":
                self._json(evcc_health())
            elif u.path.startswith("/api/"):
                self._api(u.path[len("/api/"):], q)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # surfaced in the UI rather than swallowed
            self._json({"error": str(exc)}, 400)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if u.path == "/optimize/charge-schedule":
                # Capture real evcc requests when asked, so the contract can be
                # checked against what evcc actually sends rather than a fixture.
                dump = os.environ.get("HEMS_DUMP_REQUESTS")
                if dump:
                    with open(dump, "a") as fh:
                        fh.write(json.dumps(payload) + "\n")
                # evcc's contract. Errors use evcc's Error shape, not ours.
                try:
                    self._json(optimize_charge_schedule(payload))
                except ContractError as exc:
                    self._json({"message": str(exc)}, 400)
            elif u.path.startswith("/api/"):
                self._api(u.path[len("/api/"):], payload)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    url = f"http://127.0.0.1:{port}"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"home-energy-optimizer GUI on {url}   (Ctrl-C to stop)")
    try:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
