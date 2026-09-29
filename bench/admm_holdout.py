"""Held-out check for the ADMM grid-limit step size and polish (bench/admm_variants.py --limits).

A different site from the one the step size was chosen on: 13.5 + 5 kWh batteries, 3 kW PV, limits 3.5 / 4.5 kW.
    .venv/bin/python bench/admm_holdout.py
"""

import sys, time, pickle; sys.path.insert(0, "bench")
import numpy as np
from dataclasses import replace
import admm_variants as A
from dw.integrate import dw_plan
from hemspolicy import demo_forecasts, SiteConfig, Horizon, BatteryConfig, WaterHeaterConfig, HvacConfig, GridLimits
def site2(lim):
    return SiteConfig(horizon=Horizon(dt=0.25, hours=24.0),
        battery=BatteryConfig(capacity_kwh=13.5, p_charge_max_kw=5.0, p_discharge_max_kw=5.0, soc_initial_frac=0.3, n_states=100, n_actions=51),
        batteries=(BatteryConfig(capacity_kwh=5.0, p_charge_max_kw=2.5, p_discharge_max_kw=2.5, soc_initial_frac=0.8, n_states=100, n_actions=51),),
        water_heater=WaterHeaterConfig(), hvac=HvacConfig(), grid=GridLimits(max_import_kw=lim))
def bound(cfg, fc):
    try: return dw_plan(cfg, fc).lower_bound, "numpy"
    except RuntimeError:
        from dw.coordinator import DWCoordinator
        from dw.integrate import to_coordination_result
        co = DWCoordinator(cfg, fc, solver="highs"); return to_coordination_result(co, co.run()).lower_bound, "HiGHS (numpy stalled)"
res = {}
for tariff in ["flat", "day_night", "dynamic"]:
    for lim in [3.5, 4.5]:
        cfg = site2(lim); fc = demo_forecasts(cfg.horizon, tariff=tariff, solar_peak_kw=3.0)
        lb, how = bound(cfg, fc)
        line = f"{tariff:9s} <= {lim} kW [bound: {how}]"
        for step in [1.0, 2.0, 4.0]:
            fl = replace(A.AS_IMPLEMENTED, price_step=step)
            tr = A.run(cfg, fc, fl, rounds=40); st = A.stop_round(tr, cfg.coordination, True)
            tr = A.run(cfg, fc, fl, rounds=st); _, P, T = A.run.best
            at = min(t["obj"] for t in tr) - lb; br = min(tr, key=lambda t: t["obj"])["breach_kwh"]
            pol = A.polish(cfg, fc, P, T, A.run.last["mu_imp"], A.run.last["mu_exp"])[0] - lb
            res.setdefault(step, []).append((at, pol, st))
            line += f"\n     step {step:g}: {at:.3f} -> polish {pol:.3f}  (stop {st:2d}, breach {br:.2f} kWh)"
        print(line, flush=True)
for step, r in res.items():
    print(f"step {step:g}: mean above bound {np.mean([x[0] for x in r]):.3f} -> with polish {np.mean([x[1] for x in r]):.3f}; mean stop round {np.mean([x[2] for x in r]):.1f}")
