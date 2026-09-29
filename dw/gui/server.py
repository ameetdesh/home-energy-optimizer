"""Local GUI for the Dantzig-Wolfe coordinator (src/home_energy_optimizer/dw/coordinator.py).

Run:  python3 dw/gui/server.py        then open http://127.0.0.1:8766

The DW counterpart of admm/gui/server.py, kept separate so the two can run side by
side (ADMM on 8765, DW on 8766). Stdlib HTTP only; the solve itself needs numpy
and scipy (HiGHS), which the DW master uses.

    POST /api/solve   config -> recovered plan, relaxed master, every iteration,
                                bounds, meter price, battery costates
"""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_energy_optimizer.dw import webapi  # noqa: E402


class Handler(BaseHTTPRequestHandler):
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
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8766)
    url = f"http://127.0.0.1:{port}"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"home-energy-optimizer DW GUI on {url}   (Ctrl-C to stop)")
    try:
        if "--no-browser" not in sys.argv:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
