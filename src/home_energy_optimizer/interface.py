"""The device interface: one query in, a plan and its private cost back.

A participant is one device, or a group of devices solved together by one
solver. Every participant answers the same query (docs/theory.pdf, "The device
interface"), so a coordinator never sees a model: the package's own solvers,
EMHASS's model of a device (integrations/emhass.py) and a remote agent look
alike to it.

This module is the contract in Python. `schemas/device-query.v1.json` and
`schemas/device-answer.v1.json` carry the same fields as JSON, for participants
in another process; `query_to_dict` / `answer_from_dict` convert.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

SCHEMA_VERSION = 1
KINDS = ("price_response", "best_response", "proximal")
STATUSES = ("ok", "infeasible", "fallback")


@dataclass(frozen=True)
class Query:
    """What a coordinator asks a participant.

    price_response  its cheapest plan when its energy costs `price_draw` drawn
                    and `price_supply` supplied (equal: Dantzig-Wolfe's query).
    best_response   the same at the real tariff, metered under `residual_kw`,
                    the rest of the house (a polish step, or an extra offer).
    proximal        no price, a pull of weight `rho` toward `target_kw` (ADMM).

    Prices are currency/kWh per slot; powers kW, + = drawn from the meter.
    """

    kind: str
    price_draw: np.ndarray | None = None
    price_supply: np.ndarray | None = None
    residual_kw: np.ndarray | None = None
    target_kw: np.ndarray | None = None
    rho: float = 0.0


@dataclass
class Answer:
    """A plan (kW per slot, + = drawn), its private cost (everything but the
    bill, in currency) and its state trajectory (one point more than the plan).
    `detail` is the solver's own result, kept in-process and never serialised."""

    plan_kw: np.ndarray
    private_cost: float
    trajectory: np.ndarray
    status: str = "ok"
    detail: object = None


@runtime_checkable
class Participant(Protocol):
    """What a coordinator needs from a participant.

    Two optional attributes refine how the coordinator treats it:
    `onoff` (bool) - True if its plans are on/off schedules, so the coordinator
    fixes it to one plan before re-pricing the others (DWCoordinator.run's
    dive); taken as `not modulating` when absent. `blend(weights, details)` -
    its own result for a weighted mix of its plans, given each plan's
    `Answer.detail`; called only when `modulating` is True.
    """

    key: str              # unique name in the coordinator (e.g. "battery", "deferrable0+deferrable1")
    max_power_kw: float   # the most it can draw or supply in one slot (bounds the meter)
    modulating: bool      # can a weighted blend of its plans run as it is?

    def respond(self, query: Query) -> Answer:
        """Its answer to `query`: the plan it would run and that plan's private
        cost. If its solve fails it still answers with a plan it can run (for
        example its last good one), with status "fallback" or "infeasible"; the
        coordinator then uses that plan but proves no bound from it."""
        ...

    def baseline(self) -> Answer:
        """A feasible plan to start from, with no price given (for example its
        thermostat or its plan at the import tariff)."""
        ...


# --------------------------------------------------------------------- JSON
def _series(x: npt.ArrayLike | None) -> list[float] | None:
    """An array (or None) as a flat list of floats for JSON (or None)."""
    return None if x is None else [float(v) for v in np.asarray(x, dtype=float).ravel()]


def query_to_dict(q: Query, step_minutes: float, start: str | None = None) -> dict:
    """The query as a device-query.v1 JSON object (a dict ready for json.dumps).

    `q`: the query. `step_minutes`: the slot length. `start`: the first slot's
    start, ISO 8601, if known. The slot count is read from `price_draw`, or from
    `target_kw` for a proximal query; one of them must be set.
    """
    out: dict = {"version": SCHEMA_VERSION, "kind": q.kind,
                 "horizon": {"step_minutes": float(step_minutes),
                             "slots": len(next(a for a in (q.price_draw, q.target_kw) if a is not None))}}
    if start is not None:
        out["horizon"]["start"] = start
    for name in ("price_draw", "price_supply", "residual_kw", "target_kw"):
        v = _series(getattr(q, name))
        if v is not None:
            out[name] = v
    if q.kind == "proximal":
        out["rho"] = float(q.rho)
    return out


def query_from_dict(d: dict) -> Query:
    """A device-query.v1 JSON object (as parsed by json.loads) back as a Query.
    The horizon is not kept: a Query's arrays carry the slot count."""
    arr = lambda k: None if d.get(k) is None else np.asarray(d[k], dtype=float)   # noqa: E731
    return Query(kind=d["kind"], price_draw=arr("price_draw"), price_supply=arr("price_supply"),
                 residual_kw=arr("residual_kw"), target_kw=arr("target_kw"), rho=float(d.get("rho", 0.0)))


def answer_to_dict(a: Answer, solver: str | None = None) -> dict:
    """The answer as a device-answer.v1 JSON object (a dict ready for json.dumps).

    `a`: the answer; its `detail` stays behind (it is in-process only).
    `solver`: an optional label for what produced it (for logs).
    """
    out = {"version": SCHEMA_VERSION, "plan_kw": _series(a.plan_kw),
           "private_cost": float(a.private_cost), "trajectory": _series(a.trajectory),
           "status": a.status}
    if solver is not None:
        out["solver"] = solver
    return out


def answer_from_dict(d: dict) -> Answer:
    """A device-answer.v1 JSON object (as parsed by json.loads) back as an
    Answer, with no `detail`. A missing status reads as "ok"."""
    return Answer(plan_kw=np.asarray(d["plan_kw"], dtype=float), private_cost=float(d["private_cost"]),
                  trajectory=np.asarray(d["trajectory"], dtype=float), status=d.get("status", "ok"))


def schema(name: str) -> dict:
    """The packaged JSON Schema as a dict.

    `name`: "device-query" or "device-answer". Raises FileNotFoundError for
    any other name.
    """
    text = resources.files("home_energy_optimizer").joinpath(f"schemas/{name}.v{SCHEMA_VERSION}.json").read_text()
    return json.loads(text)
