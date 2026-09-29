# evcc client compatibility check

Drives **evcc's own generated Go client** (`github.com/evcc-io/optimizer/client`,
the package evcc itself imports) against a running `hems-policy` server.

This is the difference between "we implemented a JSON API that looks right" and
"a stock evcc binary can actually use this". The Python tests in
`tests/test_evcc_contract.py` check our side of the wire; this checks that
evcc's own deserialiser accepts what we emit.

```bash
.venv/bin/python gui/server.py &          # from the repo root
cd tools/evcc-client-check
go run . http://127.0.0.1:8765
```

Expected:

```
health: HTTP 200 status="ok" message="hems-policy 0.1.0"
schedule: HTTP 200 status="Optimal" objective=0.4113
  batteries=1 charging=96 discharging=96 soc=96
  flow=96 gridImport=96 gridExport=96
  total charge=10526 Wh discharge=10478 Wh soc[0]=4441 soc[last]=3684
  limitViolations: import=false exportHit=false
OK: evcc's generated client parsed the response
```

The request deliberately includes evcc's **short first slot** (420 s rather than
900 s), because that is the case most likely to be mishandled: the same Wh in a
shorter slot is a *higher* average power, and dividing by a fixed step reads it
as a dip in demand.

The optimizer module is pinned to the same commit evcc's `go.mod` uses. If evcc
bumps it, re-run this first — the contract lives in a separate repo pinned by
commit hash, which usually means it is not yet stable.
