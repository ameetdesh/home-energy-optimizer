"""The flattened browser bundle must not drift from the package.

`wasm/hemspolicy_bundle.py` is generated and committed, which is a standing
invitation for the two to diverge. These tests close that: the bundle is
regenerated here and compared, and then actually exercised.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAKER = ROOT / "wasm" / "make_bundle.py"
BUNDLE = ROOT / "wasm" / "hemspolicy_bundle.py"

pytestmark = pytest.mark.skipif(not MAKER.exists(), reason="wasm/ not present")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("wasm") / "bundle.py"
    subprocess.run([sys.executable, str(MAKER), "-o", str(out)], check=True,
                   capture_output=True)
    return out.read_text()


def test_the_committed_bundle_is_current(fresh):
    """Regenerate and compare. If this fails, run wasm/build.sh."""
    assert BUNDLE.exists(), "wasm/hemspolicy_bundle.py is missing; run wasm/build.sh"
    assert BUNDLE.read_text() == fresh, (
        "wasm/hemspolicy_bundle.py is stale relative to src/hemspolicy; "
        "run wasm/build.sh and commit the result"
    )


def test_the_bundle_carries_no_package_imports(fresh):
    """A flat module has no package to resolve `from .x import y` against.

    One of these hid inside a function body and only failed at call time, so
    this checks every line rather than the top of the file.
    """
    for i, line in enumerate(fresh.splitlines(), 1):
        assert not line.lstrip().startswith("from ."), f"relative import at line {i}: {line!r}"
        assert not line.lstrip().startswith("from hemspolicy"), f"package import at line {i}"


def test_the_bundle_solves_and_answers_every_route(tmp_path, fresh):
    """Exec the bundle in a bare namespace - the way Pyodide does - and drive
    it through the same `_api` entry point the page uses."""
    ns: dict = {}
    exec(compile(fresh, "hemspolicy_bundle.py", "exec"), ns)
    api = ns["_api"]

    plan = json.loads(api("solve", json.dumps(
        {"hours": 24, "grid": 60, "tariff": "dynamic", "solar_peak": 5}
    )))
    assert "error" not in plan
    assert len(plan["rounds"]) == plan["meta"]["rounds_run"]
    assert plan["rounds"][plan["selected_round"]]["selected"]

    for route, payload in (
        ("round", {"r": 0}),
        ("policy", {"t": 4, "soe": 5.0}),
        ("lambda", {"t": 4, "soe": 5.0}),
        ("rollout", {"t": 4, "soe": 5.0}),
        ("evaluate", {"t": 4, "soe": 5.0, "action": 2.0}),
        ("setpoint", {"t": 4, "soe": 5.0, "max_import_kw": 3.0}),
    ):
        got = json.loads(api(route, json.dumps(payload)))
        assert "error" not in got, f"{route}: {got}"

    # errors come back as JSON, not as an exception through the JS boundary
    assert "error" in json.loads(api("nope", "{}"))


def test_the_bundle_matches_the_package_numerically(fresh):
    """Same inputs, same objective. Flattening must not change an answer."""
    from hemspolicy import webapi

    params = {"hours": 24, "grid": 60, "tariff": "day_night",
              "solar_peak": 9, "max_import_kw": 2.0}
    ns: dict = {}
    exec(compile(fresh, "hemspolicy_bundle.py", "exec"), ns)

    flat = json.loads(ns["_api"]("solve", json.dumps(params)))
    pkg = webapi.call("solve", dict(params))
    assert flat["costs"] == pkg["costs"]
    assert flat["selected_round"] == pkg["selected_round"]
    assert [r["total_objective"] for r in
            (x["costs"] for x in flat["rounds"])] == \
           [r["total_objective"] for r in (x["costs"] for x in pkg["rounds"])]


# --------------------------------------------------------------------------
# the shipped artifact, not just its inputs
# --------------------------------------------------------------------------

STANDALONE = ROOT / "wasm" / "hems_policy_standalone.html"


def _payload() -> str:
    """The Python actually embedded in the standalone page."""
    import base64
    import re
    import zlib

    html = STANDALONE.read_text()
    m = re.search(r'const _pysrc = "([^"]+)"', html)
    assert m, "no _pysrc payload in the standalone page"
    return zlib.decompress(base64.b64decode(m.group(1))).decode()


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_shipped_payload_is_obfuscated():
    """Renaming and stripping both actually happened.

    Worth asserting because --no-rename silently produces a working page, so
    losing the obfuscation would otherwise go unnoticed.
    """
    import ast

    src = _payload()
    defs = [l for l in src.splitlines() if l.startswith("def ")]
    renamed = [d for d in defs
               if len(d.split("(")[0].removeprefix("def ").strip().lstrip("_")) == 5]
    assert len(renamed) > 40, f"only {len(renamed)} of {len(defs)} top-level defs look renamed"

    # No prose survives. The obfuscator empties a docstring in first position;
    # a module docstring that has been flattened into the middle of the file is
    # not in first position any more, so make_bundle drops those itself. This
    # caught every one of them shipping verbatim.
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            assert not node.value.value.strip(), (
                f"prose survived into the payload at line {node.lineno}: "
                f"{node.value.value[:60]!r}"
            )


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_shipped_payload_still_answers_every_route():
    """Obfuscation breaks at CALL time, not build time.

    Renaming a @property or a name that collides with a dataclass field gives
    a page that loads fine and dies on the first solve, so the artifact itself
    has to be executed - checking its inputs is not enough.
    """
    from hemspolicy import webapi

    ns: dict = {}
    exec(compile(_payload(), "standalone", "exec"), ns)
    api = ns["_api"]

    params = {"hours": 24, "grid": 60, "tariff": "day_night",
              "solar_peak": 9, "max_import_kw": 2.0}
    got = json.loads(api("solve", json.dumps(params)))
    assert "error" not in got, got.get("error")

    ref = webapi.call("solve", dict(params))
    assert got["costs"] == ref["costs"]
    assert got["selected_round"] == ref["selected_round"]

    for route, payload in (
        ("round", {"r": 0}),
        ("policy", {"t": 4, "soe": 5.0}),
        ("lambda", {"t": 4, "soe": 5.0}),
        ("rollout", {"t": 4, "soe": 5.0}),
        ("evaluate", {"t": 4, "soe": 5.0, "action": 2.0}),
        ("setpoint", {"t": 4, "soe": 5.0, "max_import_kw": 3.0}),
    ):
        out = json.loads(api(route, json.dumps(payload)))
        assert "error" not in out, f"{route}: {out['error']}"


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_solver_runs_off_the_main_thread():
    """The freeze this fixed: a multi-second solve on the main thread makes
    the tab unresponsive, and a slider drag was enough to trigger it."""
    html = STANDALONE.read_text()
    assert "new Worker(" in html, "Pyodide is not in a worker; the tab will freeze"
    assert "RESOLVE_WANTED" in html, "solve requests are not coalesced"


@pytest.mark.skipif(not STANDALONE.exists(), reason="standalone not built")
def test_the_standalone_embeds_the_compiled_wheel():
    """The wheel is what makes the browser build usable (3.3x), and build.sh
    silently falls back to pure Python when build/dist is empty - so a
    forgotten wheel build would ship a slow page that looks fine."""
    html = STANDALONE.read_text()
    if "const _whl" not in html:
        pytest.skip("built without a wheel; run wasm/build_wheel.sh then wasm/build.sh")
    assert "micropip.install" in html, "wheel embedded but never installed"
    assert "emfs:///tmp/wheels/" in html, "wheel installed from the wrong filesystem"


def test_the_kernels_use_no_platform_sized_buffers():
    """`long` is 64-bit on the host and 32-bit on wasm32.

    A `long[:, ::1]` buffer passed every native test and failed only in the
    browser, so this reads the .pyx directly rather than trusting a run.
    """
    pyx = ROOT / "wasm" / "build" / "dp_kernels.pyx"
    if not pyx.exists():
        pytest.skip("kernel source not present")
    for i, line in enumerate(pyx.read_text().splitlines(), 1):
        code = line.split("#")[0]
        if "[" in code and "]" in code and "def " not in code and ":" in code:
            continue
        assert " long[" not in code and " int[" not in code, (
            f"platform-sized buffer type at line {i}: {line.strip()!r}"
        )
