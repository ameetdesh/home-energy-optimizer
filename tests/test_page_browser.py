"""The testbed page in a real browser: switching coordinators and options
redraws the charts, and nothing throws.

The Python tests check what the page's backend returns; they cannot see a
script error that stops the page from drawing it. This drives the page
(dw/gui/index.html, served by dw/gui/server.py) with headless Chrome -
tests/browser/page_flow.cjs - and is skipped where node or puppeteer is not
installed. PUPPETEER_MODULE may name puppeteer's path.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "tests" / "browser" / "page_flow.cjs"
GLOBAL = Path("/opt/homebrew/lib/node_modules/@mermaid-js/mermaid-cli/node_modules/puppeteer")


def _puppeteer() -> str | None:
    if os.environ.get("PUPPETEER_MODULE"):
        return os.environ["PUPPETEER_MODULE"]
    node = shutil.which("node")
    if node and subprocess.run([node, "-e", "require('puppeteer')"], capture_output=True).returncode == 0:
        return "puppeteer"
    return str(GLOBAL) if GLOBAL.exists() else None


@pytest.fixture(scope="module")
def flow():
    node, module = shutil.which("node"), _puppeteer()
    if not node or not module:
        pytest.skip("needs node and puppeteer")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, str(ROOT / "dw" / "gui" / "server.py"), str(port), "--no-browser"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        out = subprocess.run([node, str(FLOW), f"http://127.0.0.1:{port}/"], capture_output=True, text=True,
                             timeout=900, env={**os.environ, "PUPPETEER_MODULE": module})
        return json.loads(out.stdout.strip().splitlines()[-1])
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_nothing_throws(flow):
    assert "fatal" not in flow, flow.get("fatal")
    assert flow["errors"] == []


def test_switching_coordinator_redraws_the_charts(flow):
    s = {x["tag"]: x for x in flow["steps"]}
    assert s["dw"]["method"] == "dw" and s["admm"]["method"] == "admm"
    for chart in ("conv", "power", "soc"):
        assert s["admm"][chart] != s["dw"][chart], chart
        assert s["dw again"][chart] != s["admm, warm start"][chart], chart


def test_dragging_the_comfort_band_replans_against_it(flow):
    s = {x["tag"]: x for x in flow["steps"]}
    dw, dragged, reset = s["dw"], s["band dragged"], s["band reset"]
    assert dw["band"].endswith("drag the dots") and dw["low7"] == 22.0
    assert dragged["band"] == "custom" and dragged["low7"] > 22.0    # the backend planned against the dragged floor
    assert dragged["room"] != dw["room"]
    assert reset["band"] == dw["band"] and reset["low7"] == 22.0
    assert reset["room"] == dw["room"]


def test_clicking_an_iteration_redraws_the_plan(flow):
    s = {x["tag"]: x for x in flow["steps"]}
    assert s["admm, an iteration clicked"]["power"] != s["admm"]["power"]


def test_the_admm_options_reach_the_status(flow):
    s = {x["tag"]: x for x in flow["steps"]}
    assert "LP batteries" in s["admm, LP batteries"]["status"]
    assert "warm start" in s["admm, warm start"]["status"]


def test_pause_draws_the_best_plan_so_far_and_says_so(flow):
    s = {x["tag"]: x for x in flow["steps"]}
    paused, before = s["paused"], s["admm before pause"]
    assert paused["button"] == "Resume" and "Paused" in paused["status"]
    assert paused["pausedTag"].startswith("paused at iteration")
    assert paused["iterations"] < 60
    assert paused["conv"] != before["conv"]           # the paused solve's own iterations, drawn


def test_resume_carries_on_to_the_end(flow):
    s = {x["tag"]: x for x in flow["steps"]}
    running, done, paused = s["resumed, running"], s["resumed, done"], s["paused"]
    assert running["pausedTag"] == ""                  # the marker goes as soon as the solve resumes
    assert "resumed from iteration" in running["status"]
    assert done["pausedTag"] == "" and "Paused" not in done["status"]
    assert done["iterations"] > paused["iterations"]
    assert done["conv"] != paused["conv"]
