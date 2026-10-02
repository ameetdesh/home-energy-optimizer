"""Local server for the single-battery page (battery/gui/index.html).

Run:  python3 battery/gui/server.py        then open http://127.0.0.1:8768

The battery counterpart of admm/gui/server.py (8765) and dw/gui/server.py
(8766): stdlib HTTP over `home_energy_optimizer.battery.webapi`, which holds
every call the page makes. No solver logic lives here.

    POST /api/solve     settings + hourly curves -> plan, flow field, timing (the slow tier)
    POST /api/rollout   {hour, soe}              -> the policy from there (the fast tier)
"""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_energy_optimizer.battery import webapi  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path.startswith("/api/"):
                self._json(webapi.call(self.path[len("/api/"):], payload))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)


def main() -> None:
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8768)
    url = f"http://127.0.0.1:{port}"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"home-energy-optimizer battery page on {url}   (Ctrl-C to stop)")
    try:
        if "--no-browser" not in sys.argv:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
