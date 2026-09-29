"""The single-file DW app (dw/wasm/) must not drift from dw/ and must work.

Mirrors tests/test_wasm_bundle.py for the ADMM app: the bundle is regenerated
and compared, then both the bundle and the obfuscated payload shipped inside
the standalone page are executed - obfuscation breaks at call time, not build
time, so the artifact itself has to run.
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

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
MAKER = ROOT / "dw" / "wasm" / "make_bundle.py"
BUNDLE = ROOT / "dw" / "wasm" / "dw_bundle.py"
STANDALONE = ROOT / "dw" / "wasm" / "multi_device_optimizer_standalone.html"

PARAMS = {"hours": 24, "grid": 60, "tariff": "day_night", "solar_peak": 9,
          "max_import_kw": 4.0, "n_batteries": 2, "tank_levels": 2}


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("dwwasm") / "bundle.py"
    subprocess.run([sys.executable, str(MAKER), "-o", str(out)], check=True, capture_output=True)
    return out.read_text()


def run(src: str, params: dict) -> dict:
    ns: dict = {}
    exec(compile(src, "dw_bundle", "exec"), ns)
    return json.loads(ns["_api"]("solve", json.dumps(params)))


def test_the_committed_bundle_is_current(fresh):
    assert BUNDLE.read_text() == fresh, "dw/wasm/dw_bundle.py is stale; run dw/wasm/build.sh"


def test_the_bundle_carries_no_package_imports(fresh):
    for i, line in enumerate(fresh.splitlines(), 1):
        s = line.lstrip()
        assert not s.startswith(("from .", "from dw", "from home_energy_optimizer", "import dw", "import home_energy_optimizer")), \
            f"package import at line {i}: {line!r}"


def test_the_bundle_matches_the_package(fresh):
    from home_energy_optimizer.dw import webapi
    flat = run(fresh, PARAMS)
    pkg = webapi.call("solve", dict(PARAMS))
    assert "error" not in flat, flat.get("error")
    for k in ("upper", "lower", "relaxed", "iterations", "stop_reason"):
        assert flat["summary"][k] == pkg["summary"][k], k


def payload() -> str:
    html = STANDALONE.read_text()
    m = re.search(r'const _pysrc = "([^"]+)"', html)
    assert m, "no _pysrc payload in the standalone page"
    return zlib.decompress(base64.b64decode(m.group(1))).decode()


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_shipped_payload_is_obfuscated():
    src = payload()
    defs = [l for l in src.splitlines() if l.startswith("def ")]
    renamed = [d for d in defs if len(d.split("(")[0].removeprefix("def ").strip().lstrip("_")) == 5]
    assert len(renamed) > 40, f"only {len(renamed)} of {len(defs)} top-level defs renamed"
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            assert not node.value.value.strip(), f"prose survived at line {node.lineno}"


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
@pytest.mark.parametrize("extra", [{}, {"tank_levels": "lp"}, {"response": "both"},
                                   {"method": "admm", "max_iter": 15},        # textbook ADMM
                                   {"method": "admm", "max_iter": 15, "battery_step": "lp"}])
def test_the_shipped_payload_solves_like_the_package(extra):
    from home_energy_optimizer.dw import webapi
    params = {**PARAMS, **extra}
    got = run(payload(), params)
    assert "error" not in got, got.get("error")
    ref = webapi.call("solve", dict(params))
    for k in ("upper", "lower", "iterations"):
        assert got["summary"][k] == ref["summary"][k], k


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_page_is_browser_ready():
    html = STANDALONE.read_text()
    assert "new Worker(" in html, "Pyodide is not in a worker; the tab would freeze"
    assert "window.__HEMS_API__" in html, "the page would still call a server"
    assert "HiGHS" not in html, "scipy is not loaded in the browser; HiGHS must not be offered"
    if "const _whl" in html:
        assert "micropip.install" in html, "wheel embedded but never installed"
