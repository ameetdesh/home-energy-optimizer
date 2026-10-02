# battery/ — one battery, solved slow and read fast

A single battery against a price curve, drawn so you can see the two tiers the
whole package is built on:

- **Slow tier — solve the DP.** The backward recursion over every 15-minute slot,
  every state of charge and every action: `dp_battery.solve_battery`. It leaves
  behind a value function `V[t, s]` and a policy over the *whole* state space, not
  one trajectory.
- **Fast tier — read the policy.** From any time and state of charge, what the
  battery does next is one look-ahead against the stored `V`:
  `policy.action`. Replaying that slot by slot to the end of the horizon is
  `policy.rollout`. No re-solve.

Hover the state-of-charge chart and the path redraws from wherever the pointer
is. That path, and the timing beside it, is the fast tier. The animated flow
behind it is the policy table itself: particles drift through hour and state of
charge the way the battery is pushed, blue where it charges, red where it
discharges. It runs by default; untick "animate" to stop it.

What the timings look like, 48 h, measured natively. In the browser (Pyodide,
compiled kernels embedded) the full solve measured about 3.5x slower - 248 ms
on the default grid - and a decision 2-5x, 100-200 µs:

| DP grid (states × actions) | slow tier: full solve | fast tier: one decision | fast tier: replay 152 slots |
|---|---|---|---|
| 50 × 21 (coarse, the original page's class) | 15 ms | 47 µs | 3.9 ms |
| 100 × 41 | 26 ms | 39 µs | 4.0 ms |
| 200 × 81 (the package default) | 70 ms | 36 µs | 4.1 ms |

The solve grows with both grid sizes; a single decision barely moves. That
difference is the whole case for the two tiers: plan on the cadence forecasts
change, and act at sensor cadence from the stored value function.

## Run it

**Served** (Python on your machine, page in your browser):

```bash
.venv/bin/python battery/gui/server.py        # opens http://127.0.0.1:8768
```

**Standalone** (one HTML file, Python in the browser via Pyodide, no server):

```bash
battery/wasm/build.sh            # -> battery/wasm/battery_standalone.html
battery/wasm/build.sh --serve    # ...and serve it on http://127.0.0.1:8769/battery_standalone.html
```

It must be served over HTTP, not opened as `file://`: browsers block the ES
module import Pyodide is loaded through. Any static host works, which is how it
is published.

### Build it step by step

`build.sh` is three steps, each runnable on its own:

1. **Flatten the Python.** `battery/wasm/make_bundle.py` concatenates the
   modules the page needs — `_kernels`, `interp`, `types`, `meter`,
   `dp_battery`, `policy`, `battery/webapi` — into one module,
   `battery_bundle.py`, removing the package imports. A name defined in two
   modules fails the build. (The flattening is `dw/wasm/make_bundle.py`'s,
   loaded rather than copied.)
2. **Make the page boot Pyodide.** `battery/wasm/make_page.py` replaces the
   `HEMS-BOOT` block of `battery/gui/index.html` with the Pyodide worker
   bootstrap from `admm/wasm/make_page.py`, giving `battery_page.html`. The page
   itself is unchanged: every call goes through one `api(route, payload)`, which
   uses HTTP when served and the worker when not.
3. **Inline everything.** `tools/make_standalone.py` embeds the bundle
   (renamed, stripped, zlib + base64) and, if one has been built, the
   compiled-kernels wheel, giving `battery_standalone.html`. The names that
   must survive renaming come from `wasm/keep_names.py`.

The compiled kernels are optional. Without a wheel in `wasm/build/dist/`, the
page runs the same numpy code, more slowly. To build one (Docker, slow the first time):

```bash
wasm/build_wheel.sh && battery/wasm/build.sh
```

To publish, copy `battery/wasm/battery_standalone.html` to the site. It
replaces the original `pyodide_optimizer_standalone.html` on
ameetdesh.github.io.

## What is generated, and what keeps it honest

| File | From | Checked by |
|---|---|---|
| `battery/wasm/battery_bundle.py` | `src/home_energy_optimizer/` via `make_bundle.py` | `tests/test_battery_app.py` regenerates it and fails if the committed copy differs |
| `battery/wasm/battery_page.html` | `battery/gui/index.html` via `make_page.py` | — (an intermediate) |
| `battery/wasm/battery_standalone.html` | both, via `make_standalone.py` | the tests unpack it, run its Python, and require the same numbers as the package |

After changing anything under `src/` that the page uses, run `battery/wasm/build.sh`
and commit the regenerated files, or the bundle test fails.

The tests also pin what the page claims:
- the plan is `solve_battery`'s, number for number;
- replaying the policy from the starting state reproduces that plan exactly;
- the fast tier's action and λ are `policy.action`'s and `policy.marginal_value`'s.

## The page

| Control | Meaning |
|---|---|
| capacity, max charge / discharge power | the battery; "Linked" moves the two powers together |
| reserve | the state of charge the plan never goes below (`BatteryConfig.soe_min_frac`); the battery starts halfway between it and full |
| DP grid | states × actions for the slow tier: the cost/accuracy trade-off, live |
| price and load chart | drag the hourly points; a negative load is PV surplus. "Import prices" takes 24 h of numbers or a CSV with a Price column (per-MWh prices are converted), repeated over the horizon |
| shift+click on the state-of-charge chart | add a minimum ("at least 80% by 18:00", a `SocGate`); drag to move it, double-click to remove it |
| saved per day | the plan's bill against no battery, with any change in stored energy valued at the average import price, which is also how the DP values energy left at the end |
| value of a stored kWh | λ = dV/ds where the pointer is (`policy.marginal_value`). Near a minimum that can't be met from there, it is the penalty's price, and the page says so |

The backend is `src/home_energy_optimizer/battery/webapi.py`: two routes, `solve`
(the slow tier) and `rollout` (the fast tier), plain functions over JSON. The
server (`battery/gui/server.py`) and the browser worker both call them.

## Where it came from

This is the Pyodide proof of concept the package grew out of (`ORIGIN.md`),
originally `ideas/wasm_batt_optimizer/utils.py` with its own copy of the DP.
It now runs on the package's solver, so the page and the library can't drift
apart. The model is the package's, which differs from the original in three
small ways:

| | original page | now |
|---|---|---|
| energy left at the end | valued at the last slot's import price | at the average import price (and the savings figure uses the same) |
| grid | 50 states × 10 actions | 200 × 81 by default; `docs/NOTES.md` measures the coarse grid giving up 3-6% of the savings |
| fast tier | its own one-step look-ahead | `policy.action` / `policy.rollout`, the functions Home Assistant and evcc use |
