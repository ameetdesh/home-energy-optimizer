# The testbed, in a browser tab

`hems_policy_standalone.html` is the whole thing in one file: the UI, the
solver, and a zlib+base64 copy of the Python. Open it and it runs — no server,
no install, no Python on the machine. Pyodide and numpy come from a CDN on
first load and are cached after.

    wasm/build.sh --serve     # rebuild and open on :8765

It must be **served over HTTP**, not opened as `file://` — browsers block ES
module imports from `file://`, and Pyodide is loaded as a module.

## What is generated, and from what

| File | From | Committed |
|---|---|---|
| `hemspolicy_bundle.py` | `src/hemspolicy/*.py` via `make_bundle.py` | yes, and checked |
| `hems_policy.html` | `gui/index.html` via `make_page.py` | yes |
| `hems_policy_standalone.html` | both, via `make_standalone.py` | yes — this is the artifact |

`tests/test_wasm_bundle.py` regenerates the bundle and fails if it differs from
the committed copy, so the two cannot drift apart quietly. It also unpacks the
shipped `hems_policy_standalone.html`, execs the payload, and checks it still
answers every route with numbers identical to the package — because the two
ways obfuscation can break this both build cleanly and fail only at call time.

## Obfuscation

The payload is renamed as well as stripped: 69 of the 73 module-level
functions ship as `_wdpgx`-style names, with comments and docstrings gone.

Renaming needs a keep-list, which `keep_names.py` derives rather than
hand-maintains. `make_standalone.py` renames every `FunctionDef` and rewrites
the name wherever it appears, **including after a dot**, and that breaks two
things:

- A method or `@property` is reached as `horizon.times()`. Renaming the `def`
  and not the attribute gives `'Horizon' object has no attribute 'times'`.
- A module-level function can share a name with a dataclass field.
  `dp_battery.terminal_price` and `BatteryConfig.terminal_price` collide, and
  renaming the function rewrote `batt.terminal_price` along with it.

Both are one rule: **a name ever reached through an attribute must not be
renamed.** `keep_names.py` collects every `ast.Attribute` target plus every
class member, and `build.sh` passes them as `--keep`. What is left — the solver
internals — is renamed.

## How it stays one implementation

The point of the exercise was not to have a browser version of the solver. It
was to have the *same* solver reachable two ways.

- `hemspolicy.webapi` holds every call the UI makes — `solve`, `round`,
  `policy`, `rollout`, `evaluate`, `setpoint` — as ordinary functions over
  JSON-safe dicts, with no knowledge of HTTP.
- `gui/server.py` is a thin HTTP shell over `webapi.ROUTES`. It is ~140 lines
  and contains no solver logic.
- `gui/index.html` routes every backend call through one `api(route, payload)`
  function. It uses `fetch` normally, and `window.__HEMS_API__` when that
  exists.
- `make_page.py` supplies `window.__HEMS_API__` by booting Pyodide and calling
  the same `webapi` functions in the browser.

So the served GUI and the standalone page run identical Python, and the UI does
not know which transport it is on.

`make_bundle.py` flattens the package because the embedding tool takes a single
module. It concatenates in dependency order, strips intra-package imports, and
**fails on a name collision** rather than letting one module silently shadow
another. `feeds`, `ha` and `evcc` are excluded: they talk to the network or to
another process, which a browser tab cannot do and the testbed does not need.

## Cost

| | |
|---|---|
| standalone file | 127 KB (21 KB of base64 Python + 67 KB of base64 wheel) |
| first load | a few seconds for Pyodide + numpy from the CDN |
| solve, 2 batteries + tank + HVAC, 24 h, 40 rounds | **1.35 s** with the kernels, 4.5 s without |

That last number is the honest cost of the port: the same solve is ~0.9 s
natively, so the browser is now within ~1.5x of it. (It was 15-20 s before the
worker — not because a worker computes faster, but because on the main thread
the solve was competing with the browser's own rendering and unresponsiveness
handling.)

## Compiled kernels

`build/dp_kernels.pyx` is a Cython translation of the three backward
recursions, cross-compiled to a Pyodide wasm32 wheel and embedded alongside the
Python. The page installs it with micropip, and `hemspolicy._kernels`
dispatches to it when it imports.

    wasm/build_wheel.sh       # Docker; slow the first time (emsdk + xbuildenv)
    wasm/build.sh             # picks up build/dist/*.whl automatically

Measured on the same page, best of four warm runs, 2 batteries + tank + HVAC
over 24 h at a 2 kW import limit (40 rounds):

| | solve | objective |
|---|---|---|
| pure Python | 4513 ms | -2.193 |
| compiled kernels | **1351 ms** | -2.193 |

**3.3x, and the same answer.** Worth knowing: the same kernels give **no
speedup at all natively** (922 ms vs 897 ms) — numpy's vectorised ops over the
(states x actions) matrix are already about as fast as a scalar C loop. The win
is specific to Pyodide, where both the interpreter and numpy run in wasm and
the kernel removes the interpreter from the inner loop entirely. Native
profiling would have said this was not worth doing.

Numerical identity is the contract, and `build/verify_kernels.py` checks it
bitwise across 18 combinations of tariff, device and draw. A DP stores an
argmax, so a value differing in the last bit flips a stored action, and that
action is what the execution tier replays for the next fifteen minutes.

That check runs **natively**, which is not sufficient on its own: `long` is
64-bit there and 32-bit on wasm32, so a `long[:, ::1]` policy buffer passed
every native test and failed in the browser with `Buffer dtype mismatch,
expected 'long' but got 'long long'`. Every buffer is float64 now, and the
callers convert the policy back to int64. The cross-compiled wheel has to be
exercised in a browser, not just verified on the host.

Without a wheel in `build/dist/` everything still works — `build.sh` says so
and falls back to the pure-Python path. Pyodide's numpy is the bottleneck, and the DP inner loops are exactly
the shape that suffers.

**Pyodide runs in a Web Worker**, which is not an optimisation but a
correctness fix. A solve is seconds of straight-line numpy, and on the main
thread that is seconds of frozen tab — moving a slider produced "page not
responding". In a worker the UI keeps painting and the controls keep working
while the solve runs. The page also coalesces requests: a slider drag fires a
change per stop, and rather than queue solves whose answers are stale before
they arrive, `solve()` notes that another is wanted and runs exactly one more
when the current one lands. `wasm_batt_optimizer` solves this by compiling the
kernels to a wasm wheel via Cython — `make_standalone.py --build-wheel` — and
the same treatment would apply here. It has not been done; the pure-Python path
is fast enough to explore with, and slow enough that you would not run a house
on it.
