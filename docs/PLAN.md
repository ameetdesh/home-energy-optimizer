# Plan

> Written when the package was `hems-policy` (import `hemspolicy`); it is now
> `home-energy-optimizer` (import `home_energy_optimizer`), laid out as in the README.

Execution plan from `optimsurvey_notes/poc-integration-analysis.md` §7, updated
with what Phase 1 actually found.

---

## Where to go next

Ordered by what unblocks adoption per unit of effort. The state as of now:
phases 0–4b are done, Phase 5 step 1 (λ in HA) is done at demo grade, step 3 is
a draft PR sitting in a fork.

### Immediate — days

1. **Publish the repo.** It has no git remote; it exists only on one laptop.
   The evcc PR says "I have a working local backend" and nobody can see or try
   it. Everything downstream is blocked on this and nothing else is.
2. **Approach evcc with an issue rather than a PR.** `NOTES.md` §7 records what
   the sponsorship check actually covers; the useful, uncontroversial reports
   are the undiscoverable failure modes (silent no-op without a token, and the
   `meter profile incomplete` message that does not explain itself).

### Short — weeks

3. **Run it on a real house.** The highest-value missing thing by a wide margin.
   Everything measured so far is synthetic forecasts or a test rig. Two weeks of
   real tariff, real PV, real battery, with λ recorded against what actually
   happened, answers the one question no benchmark can: *is this signal useful
   to a person?* Needs nobody's permission.
4. **Make the HA integration survive a restart.** It currently pushes orphan
   states over REST that vanish on reboot. MQTT discovery is the small version;
   a custom component distributed through HACS is the real one, and HACS is the
   natural distribution channel.

### Medium

5. EMHASS analysis module (Phase 5 step 2).
6. evcc `PolicyController` (Phase 5 step 4).

### Loose ends, none blocking

- The MILP formulation is untuned — do not quote its 120 s runtime until it is.
- `strategy` and `c_priority` are accepted and ignored.
- Grid limits are clamped post-hoc rather than optimised.

### The strategic call, revised

**Home Assistant, not evcc, now looks like the primary target** — which inverts
the original recommendation in `poc-integration-analysis.md` §6.

evcc integration replaces a *planner*, where this package is 91–96% of a MILP
and the win is "local instead of cloud". Real, but modest, and gated by a
maintainer whose funding model it touches. λ in HA delivers the actual
differentiator, to a larger audience, with no gatekeeper — and the property that
makes it interesting (a device pricing itself with zero optimizer involvement)
only shows up there.

evcc stays worth doing. It should no longer lead.

---

**No fork of evcc is involved.** `OPTIMIZER_URI` is already an environment
variable in evcc (`core/site_optimizer.go:341`), so a local sidecar can be
substituted for the cloud optimizer with **zero upstream changes**. A fork only
becomes relevant at Phase 5 step 4.

---

## Phase 0 — Headless core ✅ done

Prerequisite for everything. The POC mixed DP solving with canvas drawing and
imported `js.document` at module scope.

- [x] Config dataclasses replace module globals (`DT`, `STEPS`, `ETA`, device params)
- [x] All DOM/Pyodide coupling removed; runs under plain CPython
- [x] **Thermal DPs now return `V` and `POL`** — the POC's compiled path returned
      only `(temp, on)`, discarding two-thirds of the value functions at the
      WASM boundary
- [x] `policy.py`: `action`, `marginal_value`, `price_signal`, `evaluate`,
      `rollout`, `clamp`, plus `.npz` persistence
- [x] 73 tests, golden files, determinism check
- [x] Three inherited defects found, fixed, and pinned (`docs/NOTES.md` §4)

**Exit criterion met:** `pytest` green with no browser.

---

## Phase 1 — Validate against an exact optimum ✅ done, and it moved the goalposts

Two error sources measured separately, against references that reimplement
*this package's own model* (comparing to EMHASS's MILP would conflate algorithm
differences with model differences).

| gap | measured | verdict |
|---|---|---|
| Discretisation (DP grid vs continuous LP) | 94–97% capture at 50×21; >99% at 200×81 | **fine, and tunable** |
| Decomposition (ADMM vs exact joint DP) | 88.6–89.5% with a price spread; **negative on flat** | **worse than hoped** |

### Consequences

1. **Change the default grid to 200×81.** 36 ms is free against a 15-minute
   re-solve cadence and it buys ~5 points of capture.
2. **At 1–2 devices, solve jointly.** The joint DP is only 3.7× slower (150 ms
   vs 40 ms) and it is exact. The decomposition is not earning its 11%.
3. **The core proposal is unaffected.** `policy.py` works identically on a joint
   DP — λ, counterfactuals and fast lookup do not depend on how the value
   function was produced. Only the *coordination heuristic* is in question.
4. **Do not lead with savings claims.** Lead with λ and the fast tier, exactly
   as the analysis doc argued.

### Still open

- [x] **Does λ survive decomposition?** ✅ **Yes.** Settled against two
      independent ground truths — the HiGHS LP dual and the exact joint DP
      costate. Decomposition error (≤0.033/kWh) is the same order as
      discretisation error (≤0.031/kWh), correlation 0.88–0.96. `NOTES.md` §3.
      Ships with the caveat that decisions within ±0.03/kWh of the margin are
      not reliable.
- [ ] Sweep more scenarios: negative prices, no PV, high PV, EV-sized battery.
- [ ] Test whether more ADMM rounds / better ρ adaptation closes the gap.
- [x] **HiGHS MILP reference** (`bench/milp_full.py`) — the "what a solver would
      give you today" bar that EMHASS sets in HA. Decomposed captures **91–96%**
      of the MILP optimum; the joint DP 97–99%. `NOTES.md` §3.
- [ ] Tune the MILP formulation before quoting its runtime as evidence: it hit a
      120 s limit with a gap at 96 slots, but it is untuned (big-M draw clip, no
      tightening, weak LP relaxation) and EMHASS solves comparable problems in
      seconds.

---

## Phase 2 — Real inputs ✅ done

- [x] `feeds.py`: **Open-Meteo PV and temperature** (keyless, verified live
      against Berlin — 4.02 kW peak, 67.3 kWh/48 h), plus `from_series` /
      `from_csv` so any Solcast / tariff / ML feed drops in. Short feeds are
      padded explicitly, since silent misalignment yields a plan that reports
      `Optimal` with every timestep offset.
- [x] **evcc's measured-value blending** (`core/optimizer.md`): `blend_measured`
      with `replace` (base load) and `scale` (solar) modes, decaying over four
      slots. The `scale` path no-ops at night rather than dividing by ~0.
- [x] **Comfort priced in currency.** Restoring 1 K costs
      `heat_capacity × energy_price` (÷ COP for the heat pump), so discomfort is
      `discomfort_multiplier` × that. Removed the flat-tariff bill regression:
      savings **−0.104 → +0.562**, comfort penalty 608 arbitrary units → 0.25
      currency. `comfort_mode="quadratic"` still reaches the old behaviour, and
      a test pins that it is still wrong.
- [x] Reconciled the efficiency convention: `eta_charge` / `eta_discharge` are
      now first-class (`eta` remains the shorthand). This matches EMHASS's
      `battery_charge_efficiency` / `battery_discharge_efficiency` and is
      *required* by evcc's contract, which sends `eta_c` / `eta_d`.
- [ ] Richer PV model — the Open-Meteo path is GHI × efficiency, ignoring tilt,
      azimuth, shading and temperature derating. EMHASS's pvlib ModelChain is
      the thorough version; the seam is `feeds.open_meteo_pv`.

---

## Phase 3 — The sidecar service (mostly done via the GUI)

- [x] All five endpoints, in `admm/gui/server.py` (stdlib-only). The GUI *is* the
      sidecar draft: `/api/solve`, `/api/policy`, `/api/lambda`, `/api/rollout`,
      `/api/evaluate`, `/api/setpoint`.
- [x] Persist `V`/`POL` between solves (`.npz`)
- [ ] Split the service out of `admm/gui/` into its own module once a second consumer
      exists; add per-site keying instead of one global snapshot.
- [x] Hard-constraint clamp + ramp limiting, exposed as `POST /api/setpoint` —
      the endpoint real hardware should call. Reports every binding limit.
- [ ] Cython → WASM for the hot path (`skills/cython-wasm-standalone-html.md`);
      golden files make the port verifiable

---

## Phase 4 — First consumer: evcc, via the env-var seam ✅ done, end to end

**Zero upstream changes**, exactly as planned — `OPTIMIZER_URI` is an ordinary
environment variable (`core/site_optimizer.go:341`).

- [x] Wire contract obtained from **source**, not captured traffic: the
      generated client in `github.com/evcc-io/optimizer/client`, pinned by
      commit in evcc's `go.mod`. `POST /optimize/charge-schedule` and
      `GET /optimize/health`.
- [x] Implemented in `src/hemspolicy/evcc.py`, served by `admm/gui/server.py`.
      27 tests in `tests/test_evcc_contract.py` cover the unit conversions
      (W/Wh and per-Wh prices vs kW/kWh and per-kWh), the signed-to-split power
      mapping, the s_min offset, and evcc's short first slot.
- [x] **Verified against evcc's own Go client** (`tools/evcc-client-check/`):
      it deserialises our response cleanly, `status="Optimal"`, all arrays the
      right length. This is the real client code evcc imports, not a mock.
- [x] **Ran a live evcc against this server** (`tools/evcc-live-test/`). evcc
      called us, parsed the response, and published derived state from *our*
      `state_of_charge` array:
      `battery.forecast.highest = 95% at 04:15`, `lowest = 20% at 13:45`,
      `evopt-batteries: full 04:31:30, empty 14:01:30`. Charge overnight on the
      0.15 rate, discharge into the 0.30 peak — correct.
- [ ] Run it for a *week* alongside the hosted optimizer and compare
      suggestions. The local side needs nothing; the comparison side needs
      access to the hosted service.

### What actually blocked the end-to-end test

Not the sponsorship check — that is one line in an MIT-licensed codebase, and
`NOTES.md` §7 records what it covers project-wide.

**The blocker was the 30-day home energy profile.** `site.homeProfile` reads
`Collector.EnergyProfile` from the metrics database, and a fresh instance fails
with `optimizer: meter profile incomplete` before it ever builds a request.
`tools/evcc-live-test/seed_metrics.py` seeds it.

### What the live run taught us

- **672 slots, not 96.** evcc caps the horizon at 2 days *only for its own cloud
  endpoint* (`if optimizerURI() == OPTIMIZER_URI`). A self-hosted optimizer gets
  the full forecast length. Solve cost 808 ms at the 200×81 grid — fine against
  a 15-minute cadence, but 7× the size the cloud service ever sees.
- `dt[0] = 810 s`, confirming the short first slot the contract tests assume.
- `grid: {}` — limits may be entirely absent.
- `s_min=2000, s_max=9500, s_capacity=10000` — the usable-window offset is real,
  not theoretical.
- `strategy` arrives populated and is currently ignored.

The captured request is checked in at `tests/fixtures/evcc_request_real.json`.

### Phase 4b — make the mapping usable ✅ done

Upstreaming something that refuses evcc's main use case was never viable, so
this came before Phase 5.

- [x] **Multi-battery.** evcc sends one `BatteryConfig` per stationary battery
      *and one per loadpoint* (`core/site_optimizer.go:532,564`), so every user
      with a charger sends ≥2 and the previous refusal made the integration a
      no-op for them. `SiteConfig.batteries` + dynamic device keys in the
      coordinator; results keep request order, because evcc matches them back
      positionally.
- [x] **`p_demand`** — minimum charge *energy* per slot. A loadpoint carries its
      entire charging requirement here, so ignoring it means never charging an
      EV. Enforced as a per-slot floor on the action grid.
- [x] **`s_goal`** — per-slot target stored energy, how evcc says "60% by
      07:00". Soft, priced at `soc_goal_penalty`, so an unreachable goal
      degrades instead of returning infeasible.
- [x] **`c_min`** — found only by capturing a live loadpoint request. It is a
      *semi-continuous* floor (`Voltage × minCurrent × minPhases`): a charger is
      off, or at ≥ c_min, never between. Planning 2 kW into a 4.14 kW wallbox is
      a schedule the hardware cannot execute. A DP enforces this by deleting
      actions from the grid; an LP needs a binary per timestep, which is
      precisely EMHASS's `treat_deferrable_load_as_semi_cont`.

### Still-known gaps

- **Non-uniform `dt`**: `Horizon` is uniform; we solve on the modal step and
  report it. evcc's first slot is normally partial.
- **`strategy`** (`charge_before_export`, peak attenuation) — accepted, ignored.
- **`c_priority`** — accepted, ignored, and noted in the response.
- **Grid limits**: now priced inside the coordination as a dual variable
  (`GridLimits`, `NOTES.md` §1), not clamped post-hoc. Import limits are met
  down to ~20% above the physical floor. **Export limits remain advisory**
  because the model has no PV curtailment — once the battery fills, surplus
  must go to the grid.
- The live capture with a loadpoint returned only the loadpoint, not the
  stationary battery — evcc skips a battery whose `Capacity`/`Soc` measurement
  is not populated in that tick (`site_optimizer.go:560`). A config artifact of
  the minimal test rig, not a contract issue; the multi-battery path is covered
  by unit tests and a synthetic home-battery + EV scenario.

---

## Phase 5 — Upstream, in increasing order of ask

1. [x] **Publish λ as a Home Assistant sensor.** ✅ Done —
       `src/hemspolicy/ha.py` + `tools/ha-lambda-demo/`. Verified against a
       live HA container: five sensors, λ carrying a `forecast` array in
       EMHASS's attribute shape and an explicit uncertainty band.
       Measured curve: λ = **0.0889** under PV surplus (`sell/η`) and
       **0.3564** at the evening peak (`buy·η`) — its theoretical bounds at
       both ends. The README shows an automation running an *unmodelled*
       tumble dryer off the signal, which is the composability argument made
       concrete.
2. [ ] **EMHASS analysis module** in the `pv_bias_calibration.py` mould —
       reports, does not touch the LP. Low friction given that precedent.
3. [~] **Local optimizer backend for evcc.** Draft PR on a personal fork
       (`ameetdesh/evcc#1`): 16 lines adding `selfHostedOptimizer()` so the
       sponsorship check is skipped when `OPTIMIZER_URI` points elsewhere.
       Not sent upstream. `NOTES.md` §7 explains why the original justification
       was too narrow; the smaller and more useful asks are the diagnostics.
4. [ ] **Fast execution tier in evcc** — a `PolicyController` consulted between
       the 30 s cycles. Largest architectural ask; only after 1–3 have landed.

---

## Open questions carried forward

1. ~~Does λ survive decomposition?~~ ✅ settled — `NOTES.md` §3
2. Would evcc accept a self-hosted optimizer backend upstream?
3. ~~Is `optimizer.OptimizationInput` a stable contract?~~ Transcribed and implemented; still pinned by commit hash upstream, so re-run `tools/evcc-client-check` after any bump.
4. How much does discretisation chatter cost in real execution? 21 power levels
   on a 10 kW battery is ~500 W granularity — fine for planning, possibly
   visible in control.
5. ~~What is the right currency price for comfort?~~ ✅ derived from restoration cost — `NOTES.md` §4
