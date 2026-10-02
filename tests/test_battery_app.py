"""The single-battery page (battery/): its backend, its bundle, its artifact.

The page exists to show the package's two tiers on one battery - a slow DP
solve, then fast reads of the policy it leaves behind - so the tests pin the
claims it makes: the plan is the package's own solver's, and replaying the
policy from the starting state reproduces that plan exactly.
"""

from __future__ import annotations

import ast
import base64
import json
import pathlib
import re
import subprocess
import sys
import zlib

import numpy as np
import pytest

from home_energy_optimizer import policy
from home_energy_optimizer.battery import webapi
from home_energy_optimizer.dp_battery import solve_battery
from home_energy_optimizer.types import BatteryConfig, Horizon

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAKER = ROOT / "battery" / "wasm" / "make_bundle.py"
BUNDLE = ROOT / "battery" / "wasm" / "battery_bundle.py"
STANDALONE = ROOT / "battery" / "wasm" / "battery_standalone.html"

SETTINGS = {"grid": "coarse", "capacity_kwh": 20.0, "p_charge_kw": 4.0, "p_discharge_kw": 3.0,
            "reserve_pct": 20.0}
# timings differ run to run; everything else must match exactly
TIMING = {"slow_ms", "decision_us", "replay_ms"}


def _untimed(out: dict) -> dict:
    return {k: v for k, v in out.items() if k not in TIMING}


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------

def test_the_defaults_are_the_original_pages():
    """An evening peak in the import price, flat export, a flat 3 kW load."""
    c = webapi.default_curves(48)
    assert len(c["buy_h"]) == 49
    assert c["buy_h"][19] == 0.25 and c["buy_h"][12] == 0.15 and c["buy_h"][43] == 0.25
    assert set(c["sell_h"]) == {0.05} and set(c["load_h"]) == {3.0}


def test_the_plan_is_the_package_solvers():
    """No second DP: the page's plan is dp_battery.solve_battery's, number for number."""
    out = webapi.call("solve", dict(SETTINGS))
    h = Horizon(dt=0.25, hours=48.0)
    c = webapi.default_curves(48)
    slots = np.arange(h.steps) * h.dt
    buy, sell, load = (np.interp(slots, np.arange(49), c[k]) for k in ("buy_h", "sell_h", "load_h"))
    cfg = BatteryConfig(capacity_kwh=20.0, p_charge_max_kw=4.0, p_discharge_max_kw=3.0, eta=0.9,
                        n_states=50, n_actions=21, soc_initial_frac=0.6, soe_min_frac=0.2,
                        terminal_mode="linear", terminal_price=float(np.mean(buy)))
    ref = solve_battery(cfg, h, buy, sell, dp_load=load)
    assert out["plan"]["power"] == [round(float(v), 4) for v in ref.power]
    assert out["plan"]["soe"] == [round(float(v), 4) for v in ref.trajectory]


def test_the_fast_tier_from_the_start_is_the_slow_tiers_plan():
    """Reading the policy from the starting state reproduces the plan exactly:
    the plan IS the policy replayed, which is the page's whole argument."""
    out = webapi.call("solve", dict(SETTINGS))
    ro = webapi.call("rollout", {"hour": 0.0, "soe": out["soe_start_kwh"]})
    assert ro["start_step"] == 0
    assert ro["power"] == out["plan"]["power"]
    assert ro["soe"] == out["plan"]["soe"]
    assert ro["import"] == out["plan"]["import"]


def test_the_fast_tier_answers_from_a_state_the_plan_never_visits():
    out = webapi.call("solve", dict(SETTINGS))
    ro = webapi.call("rollout", {"hour": 10.3, "soe": 4.5})       # 22.5 %, off the plan
    assert ro["start_step"] == 41 and ro["replay_steps"] == out["steps"] - 41
    soe, power = np.array(ro["soe"]), np.array(ro["power"])
    assert soe[0] == pytest.approx(4.5)
    assert np.all(soe >= out["soe_floor_kwh"] - 1e-9) and np.all(soe <= out["capacity_kwh"] + 1e-9)
    assert np.all(power <= 4.0 + 1e-9) and np.all(power >= -3.0 - 1e-9)
    # its numbers are the policy layer's
    snap = webapi._state["snapshot"]
    assert ro["action_kw"] == round(policy.action(snap, 41, 4.5), 4)
    assert ro["lambda"] == round(policy.marginal_value(snap, 41, 4.5), 5)
    assert ro["action_kw"] == ro["power"][0]


def test_a_minimum_state_of_charge_is_met():
    """A gate ("at least 90% by 18:00") reaches the plan through the DP."""
    out = webapi.call("solve", {**SETTINGS, "gates": [{"hour": 18.0, "soc_frac": 0.9}]})
    assert out["plan"]["soe"][int(18.0 / 0.25)] >= 0.9 * 20.0 - 1e-6


def test_the_battery_never_costs_money():
    """Idling is always a plan, so the best one saves at least nothing."""
    for grid in ("coarse", "fine"):
        assert webapi.call("solve", {**SETTINGS, "grid": grid})["savings_per_day"] >= -1e-9


def test_the_flow_field_is_the_policy_table():
    out = webapi.call("solve", dict(SETTINGS))
    f = out["field"]
    assert len(f["policy"]) == out["steps"] and len(f["policy"][0]) == len(f["states"]) == 50
    snap = webapi._state["snapshot"]
    assert f["policy"][30] == [round(float(v), 3) for v in snap.policy[30]]


def test_bad_requests_say_what_is_wrong():
    webapi._state.clear()
    with pytest.raises(ValueError, match="solve first"):
        webapi.call("rollout", {"hour": 1.0, "soe": 5.0})
    with pytest.raises(ValueError, match="grid must be one of"):
        webapi.call("solve", {"grid": "huge"})
    with pytest.raises(ValueError, match="no such route"):
        webapi.call("nope", {})


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("battery") / "bundle.py"
    subprocess.run([sys.executable, str(MAKER), "-o", str(out)], check=True, capture_output=True)
    return out.read_text()


def test_the_committed_bundle_is_current(fresh):
    """Regenerate and compare. If this fails, run battery/wasm/build.sh."""
    assert BUNDLE.read_text() == fresh, "battery/wasm/battery_bundle.py is stale; run battery/wasm/build.sh"


def test_the_bundle_carries_no_package_imports(fresh):
    for i, line in enumerate(fresh.splitlines(), 1):
        assert not line.lstrip().startswith(("from .", "from home_energy_optimizer")), f"line {i}: {line!r}"


def _drive(api) -> tuple[dict, dict]:
    solved = json.loads(api("solve", json.dumps(SETTINGS)))
    assert "error" not in solved, solved.get("error")
    rolled = json.loads(api("rollout", json.dumps({"hour": 10.3, "soe": 4.5})))
    assert "error" not in rolled, rolled.get("error")
    assert "error" in json.loads(api("nope", "{}"))          # errors come back as JSON
    return solved, rolled


def _package() -> tuple[dict, dict]:
    return webapi.call("solve", dict(SETTINGS)), webapi.call("rollout", {"hour": 10.3, "soe": 4.5})


def test_the_bundle_matches_the_package(fresh):
    ns: dict = {}
    exec(compile(fresh, "battery_bundle.py", "exec"), ns)
    got, ref = _drive(ns["_api"]), _package()
    assert [_untimed(x) for x in got] == [_untimed(x) for x in ref]


# --------------------------------------------------------------------------
# the shipped artifact
# --------------------------------------------------------------------------

def _payload() -> str:
    m = re.search(r'const _pysrc = "([^"]+)"', STANDALONE.read_text())
    assert m, "no _pysrc payload in the standalone page"
    return zlib.decompress(base64.b64decode(m.group(1))).decode()


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_shipped_page_answers_like_the_package():
    """Obfuscation breaks at call time, not build time, so run the artifact."""
    ns: dict = {}
    exec(compile(_payload(), "standalone", "exec"), ns)
    got, ref = _drive(ns["_api"]), _package()
    assert [_untimed(x) for x in got] == [_untimed(x) for x in ref]


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_shipped_payload_is_obfuscated():
    src = _payload()
    defs = [l for l in src.splitlines() if l.startswith("def ")]
    renamed = [d for d in defs if len(d.split("(")[0].removeprefix("def ").strip().lstrip("_")) == 5]
    assert len(renamed) >= 15, f"only {len(renamed)} of {len(defs)} top-level defs look renamed"
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            assert not node.value.value.strip(), f"prose survived at line {node.lineno}"


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_solver_runs_off_the_main_thread_and_coalesces():
    html = STANDALONE.read_text()
    assert "new Worker(" in html, "Pyodide is not in a worker; the tab will freeze"
    assert "RESOLVE_WANTED" in html, "solve requests are not coalesced"
