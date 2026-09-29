# End-to-end test against a live evcc

Proves a **running evcc instance** uses this solver, rather than only that the
wire format parses. Reproduces the Phase 4 result in `docs/PLAN.md`.

## The gate is not the obstacle

`sponsor.IsAuthorized()` is `len(Subject) > 0` on a package variable
(`util/sponsor/auth.go:53`), so it is a one-line local change in an MIT-licensed
codebase. It is also not a paywall being circumvented: the sponsorship gates
evcc's **cloud** optimizer at `https://optimizer.evcc.io`, and pointing
`OPTIMIZER_URI` at localhost means that service is never called. Nothing
sponsor-funded is consumed.

(If this work is ever proposed upstream, the sponsorship model is a conversation
to have with the maintainers. That is a social question, not a technical one,
and it has no bearing on local development.)

**The real obstacle is different**: the optimizer needs a 30-day home energy
profile (`site.homeProfile` -> `Collector.EnergyProfile`). A fresh instance has
none and fails with `optimizer: meter profile incomplete`. That is seedable.

## Steps

1. **Build evcc with a dev escape hatch** (branch `local-optimizer-test` in the
   sibling `evcc` clone):

   ```go
   // util/sponsor/auth.go, top of ConfigureSponsorship
   if os.Getenv("EVCC_LOCAL_SPONSOR") != "" {
       Subject = "local-dev"
       return nil
   }
   ```

   `main.go` has `//go:embed dist`, so a placeholder `dist/index.html` is needed
   to build without a Node UI build:

   ```bash
   mkdir -p dist && echo '<!doctype html>' > dist/index.html
   go build -o /tmp/evcc-local ./
   ```

2. **Config** (`evcc-test.yaml` here): const-plugin grid/PV/battery meters, a
   `fixed` grid tariff with an off-peak zone, and a `fixed` feed-in tariff. Day
   names must be `mon-sun`, not `Mo-Su`.

3. **Enable the optimizer.** Two DB-backed settings, and the REST routes are not
   registered in config-file mode, so write them directly:

   ```bash
   sqlite3 evcc.db "insert or replace into settings (key,value)
                    values ('experimental','true'),('optimizer','true');"
   ```

4. **Seed 30+ days of metrics** — see `seed_metrics.py`. Without this the
   optimizer aborts before it ever builds a request.

5. **Run both:**

   ```bash
   HEMS_DUMP_REQUESTS=/tmp/evcc-requests.jsonl python gui/server.py &
   EVCC_LOCAL_SPONSOR=1 OPTIMIZER_URI=http://127.0.0.1:8765 \
     /tmp/evcc-local --config evcc-test.yaml --disable-auth
   ```

## What it showed

evcc called us, parsed the response, and published derived state from **our**
`state_of_charge` array (`/api/state`):

```
battery.forecast.highest = 95% at 04:15   (= s_max 9500 of 10000)
battery.forecast.lowest  = 20% at 13:45   (= s_min 2000)
evopt-batteries: full 04:31:30, empty 14:01:30
```

Charge overnight on the 0.15 rate, discharge into the 0.30 peak. Correct.

## Findings worth keeping

- **672 slots, not 96.** evcc caps the horizon at 2 days *only for its own cloud
  endpoint* (`if optimizerURI() == OPTIMIZER_URI`). A self-hosted optimizer gets
  the full forecast length — here a 7-day `fixed` tariff. Solve cost 808 ms at
  the 200x81 grid, comfortably inside a 15-minute cadence, but it is 7x the size
  the cloud service ever sees.
- **`dt[0] = 810 s`**, confirming the short first slot the contract tests assume.
- **`grid: {}`** — limits may be entirely absent.
- **`s_min=2000, s_max=9500, s_capacity=10000`** — the usable-window offset is
  used in practice, not a theoretical case.
- **`strategy`** arrives populated (`charge_before_export`,
  `discharge_before_import`) and is currently ignored.

The captured request is checked in at `tests/fixtures/evcc_request_real.json`
and is exercised by `tests/test_evcc_contract.py`.
