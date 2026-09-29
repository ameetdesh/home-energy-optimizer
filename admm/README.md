# admm/ — the ADMM app

The ADMM coordinator is `home_energy_optimizer.admm`: ADMM in exchange form,
proximal message passing (Kraning, Chu, Lavaei and Boyd, 2013), with each
device's own solver as its proximal step. `docs/theory.tex`, "The second
coordinator: ADMM", gives the problem, the algorithm as implemented and where it
departs from the paper. This folder holds its app.

```
src/home_energy_optimizer/admm/coordinator.py   the ADMM loop (ExchangeRun: pause and resume, warm starts);
                      home_energy_optimizer.coordinate() and plan(method="admm") call it
src/home_energy_optimizer/admm/battery_qp.py    a battery's exact LP step, by a small interior point method
src/home_energy_optimizer/admm/webapi.py        JSON backend for the ADMM app, and the policy API
admm/gui/             the ADMM app: server.py (stdlib HTTP, port 8765) + index.html;
                      the server also answers evcc's optimizer contract
admm/wasm/            the app as one self-contained browser page (build.sh)
```

Dantzig–Wolfe is laid out the same way: `home_energy_optimizer.dw` and `dw/`.
