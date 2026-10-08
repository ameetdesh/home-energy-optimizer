"""Participants over the network: a solver in another process, container or
device, answering the participant API (schemas/participant-api.v1.json).

`RemoteParticipant` is the coordinator's side: an interface.Participant whose
answers come over HTTP. `ParticipantService` is the solver's side - the API's
handlers, for any HTTP host (EMHASS serves them from its own web server);
`serve` runs them on the standard library's threaded HTTP server, for tests
and for this package's own models.

Queries have no effect; only a commit does. A session is the horizon a
coordinator plans - its start, slot length and slot count - so queries carry
no session of their own: the solver builds its model for a horizon once, and
answers every query on it from that model.

Privacy: a remote is asked what it would do at given prices. A best response
also carries the rest of the house's load (its `residual_kw`), so it is sent
only to a remote that lists "best_response" among the kinds it answers;
otherwise the same prices are asked as a price response.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from home_energy_optimizer.interface import (
    KINDS,
    SCHEMA_VERSION,
    Answer,
    Participant,
    Query,
    answer_from_dict,
    answer_to_dict,
    query_from_dict,
    query_to_dict,
)

# method, URL, headers, body -> HTTP status, body
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]


@dataclass(frozen=True)
class PlanWindow:
    """The horizon a coordinator plans: `slots` slots of `step_minutes`
    each, the first starting at `start` (ISO 8601) when known."""

    step_minutes: float
    slots: int
    start: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The API's Horizon object."""
        out: dict[str, Any] = {"step_minutes": float(self.step_minutes), "slots": int(self.slots)}
        if self.start is not None:
            out["start"] = self.start
        return out

    @classmethod
    def from_dict(cls, d: Any) -> PlanWindow:
        """A Horizon object back; ValueError if it is not one."""
        if not isinstance(d, dict) or set(d) - {"step_minutes", "slots", "start"}:
            raise ValueError("horizon must be an object with step_minutes, slots and start")
        step, slots, start = d.get("step_minutes"), d.get("slots"), d.get("start")
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not 0 < step <= 1440:
            raise ValueError("horizon.step_minutes must be in (0, 1440]")
        if isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 10_000:
            raise ValueError("horizon.slots must be an integer in [1, 10000]")
        if start is not None and not (isinstance(start, str) and len(start) <= 64):
            raise ValueError("horizon.start must be an ISO 8601 date-time")
        return cls(float(step), slots, start)


class RemoteError(RuntimeError):
    """A remote refused a call, or could not be reached. `status` is the HTTP
    status, or 0 when nothing came back."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class RemoteUnavailable(RemoteError):
    """A remote that has not answered once: there is no plan to stand in."""


class WindowRefused(Exception):
    """Raised by a ParticipantService's `build` for a horizon it cannot plan
    (another slot length, too long): the caller gets 409 and why."""


def urllib_transport(timeout: float) -> Transport:
    """The standard library's HTTP client as a Transport (`timeout` seconds
    per call)."""
    def send(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - http(s) only, checked by the caller
                return int(r.status), r.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()
    return send


class RemoteParticipant:
    """A participant answered over HTTP (interface.Participant).

    `key`: its name in the coordinator. `url`: the API's base, below which
    /v1/describe and the rest live (for EMHASS: http://host:5000/participant).
    `window`: the horizon being planned. `token`: the bearer token it expects,
    if any. `transport`: how calls are sent (urllib by default, `timeout`
    seconds each).

    Describes itself on construction (RemoteError if it cannot, or speaks
    another version). An answer's `detail` is {"query_id": ...}, which a
    commit names. If a call fails later, the last answer that solved stands
    in, with status "fallback", so the coordinator proves no bound from it.
    """

    modulating = False          # a blend of its plans is not asked for (/v1/blend)

    def __init__(self, key: str, url: str, window: PlanWindow, token: str | None = None,
                 timeout: float = 30.0, transport: Transport | None = None) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{key}: the url must be http(s)")
        self.key, self.url, self.window = key, url.rstrip("/"), window
        self._token = token
        self._send = transport or urllib_transport(timeout)
        d = self._call("GET", "/v1/describe")
        if d.get("version") != SCHEMA_VERSION:
            raise RemoteError(f"{key}: the remote speaks version {d.get('version')!r} of the API, not {SCHEMA_VERSION}")
        kinds = d.get("kinds") or []
        if "price_response" not in kinds:
            raise RemoteError(f"{key}: the remote does not answer price responses")
        self.kinds = tuple(k for k in kinds if k in KINDS)
        # whether it may be sent the rest of the house's load (a best response)
        self.takes_residual = "best_response" in self.kinds
        self.max_power_kw = float(d["max_power_kw"])
        self.onoff = bool(d.get("onoff", not d.get("modulating", False)))
        self.devices = tuple(str(x) for x in d.get("devices") or ())
        self.solver = str(d.get("solver", ""))
        self.solves, self.solve_s = 0, 0.0
        self._good: Answer | None = None
        self._last: tuple[tuple[Any, ...], Answer] | None = None
        self._baseline: Answer | None = None

    def _call(self, method: str, path: str, body: Any = None, query_id: str | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if query_id:
            headers["Query-Id"] = query_id
        data = None if body is None else json.dumps(body).encode()
        try:
            status, raw = self._send(method, self.url + path, headers, data)
        except OSError as exc:                     # refused, unreachable, timed out
            raise RemoteError(f"{self.key}: {exc}") from exc
        try:
            out = json.loads(raw or b"{}")
        except ValueError:
            out = {}
        if status != 200 or not isinstance(out, dict):
            why = out.get("detail") or out.get("title") if isinstance(out, dict) else ""
            raise RemoteError(f"{self.key}: HTTP {status}{': ' + str(why) if why else ''}", status)
        return out

    def _answer(self, path: str, body: dict[str, Any]) -> Answer:
        """POST `body` to `path` and read the answer; a failure answers with
        the last good plan (status "fallback"), or raises RemoteUnavailable."""
        qid = uuid.uuid4().hex
        t0 = time.perf_counter()
        try:
            a = answer_from_dict(self._call("POST", path, body, qid))
            if a.plan_kw.shape != (self.window.slots,) or a.trajectory.shape != (self.window.slots + 1,):
                raise RemoteError(f"{self.key}: an answer of the wrong length")
            if a.status not in ("ok", "infeasible", "fallback") or not np.all(np.isfinite(a.plan_kw)):
                raise RemoteError(f"{self.key}: an answer that is not a plan")
        except (RemoteError, KeyError, ValueError, TypeError) as exc:
            if self._good is None:
                raise RemoteUnavailable(str(exc), getattr(exc, "status", 0)) from exc
            g = self._good
            return Answer(g.plan_kw, g.private_cost, g.trajectory, status="fallback", detail=g.detail)
        finally:
            self.solves += 1
            self.solve_s += time.perf_counter() - t0
        a.detail = {"query_id": qid}
        if a.status == "ok":
            self._good = a
        return a

    def respond(self, query: Query) -> Answer:
        """Its answer to `query` (interface.Participant). A best response goes
        out as a price response unless the remote takes best responses; a
        query equal to the last one is not sent again."""
        if query.kind == "best_response" and "best_response" not in self.kinds:
            query = Query("price_response", query.price_draw, query.price_supply)
        if query.kind not in self.kinds:
            raise NotImplementedError(f"{self.key} does not answer {query.kind} queries")
        key = (query.kind,) + tuple(None if a is None else np.asarray(a, dtype=float).tobytes()
                                    for a in (query.price_draw, query.price_supply, query.residual_kw,
                                              query.target_kw)) + (query.rho,)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        a = self._answer("/v1/query", query_to_dict(query, self.window.step_minutes, self.window.start))
        self._last = (key, a)
        return a

    def baseline(self) -> Answer:
        """Its plan with no price given (interface.Participant); asked once."""
        if self._baseline is None or self._baseline.status != "ok":
            self._baseline = self._answer("/v1/baseline", self.window.to_dict())
        return self._baseline

    def commit(self, detail: Any, plan_kw: np.ndarray) -> tuple[bool, str]:
        """Ask it to run the plan the coordinator chose: `detail`, the chosen
        answer's (it names the query), and that plan. Returns (accepted,
        reason); an unreachable remote is (False, why)."""
        body: dict[str, Any] = {"version": SCHEMA_VERSION, "horizon": self.window.to_dict(),
                                "plan_kw": [float(x) for x in np.asarray(plan_kw, dtype=float)]}
        if isinstance(detail, dict) and isinstance(detail.get("query_id"), str):
            body["query_id"] = detail["query_id"]
        try:
            r = self._call("POST", "/v1/commit", body)
        except RemoteError as exc:
            return False, str(exc)
        return bool(r.get("accepted")), str(r.get("reason", ""))


# ---------------------------------------------------------------- the solver's side
@dataclass
class _Session:
    participant: Participant
    created: float
    answers: OrderedDict[str, Answer] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _problem(status: int, title: str, detail: str = "") -> tuple[int, dict[str, Any]]:
    """An RFC 9457 problem body."""
    return status, {"type": "about:blank", "title": title, "status": status, "detail": detail}


class ParticipantService:
    """The participant API's handlers, for any HTTP host.

    `describe`: returns the Description (key, devices, max_power_kw,
    modulating, onoff, ...); version and kinds are filled in if absent.
    `build`: the participant for a horizon, built when a horizon is first
    asked about. `commit`: runs the chosen answer - given the session's
    participant and its own Answer, with the `detail` it kept - and returns
    (accepted, reason); None: commits are refused. `token`: the bearer token callers must present; None: open (for
    tests and hosts that authenticate themselves). Sessions live `ttl_s`, at
    most `max_sessions` at once.
    """

    def __init__(self, describe: Callable[[], dict[str, Any]], build: Callable[[PlanWindow], Participant],
                 commit: Callable[[Participant, Answer], tuple[bool, str]] | None = None,
                 token: str | None = None, ttl_s: float = 3600.0, max_sessions: int = 4) -> None:
        self._describe, self._build, self._commit = describe, build, commit
        self._token = token
        self.ttl_s, self.max_sessions = ttl_s, max_sessions
        self._sessions: OrderedDict[PlanWindow, _Session] = OrderedDict()
        self._lock = threading.Lock()

    def _session(self, window: PlanWindow, create: bool = True) -> _Session | None:
        now = time.monotonic()
        with self._lock:
            for w in [w for w, s in self._sessions.items() if now - s.created > self.ttl_s]:
                del self._sessions[w]
            s = self._sessions.get(window)
            if s is not None or not create:
                return s
        participant = self._build(window)              # outside the lock: it may take a while
        with self._lock:
            s = self._sessions.setdefault(window, _Session(participant, now))
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return s

    @staticmethod
    def _keep(s: _Session, qid: str, a: Answer) -> None:
        s.answers[qid] = a
        while len(s.answers) > 1024:
            s.answers.popitem(last=False)

    def handle(self, method: str, path: str, headers: Mapping[str, str], body: bytes | None
               ) -> tuple[int, dict[str, Any]]:
        """One request: `path` is below the API's base ("/v1/query").
        Returns (HTTP status, JSON body)."""
        h = {k.lower(): v for k, v in headers.items()}
        if self._token is not None and h.get("authorization", "") != f"Bearer {self._token}":
            return _problem(401, "Unauthorized", "a bearer token is required")
        try:
            if (method, path) == ("GET", "/v1/describe"):
                d = dict(self._describe())
                d.setdefault("version", SCHEMA_VERSION)
                d.setdefault("kinds", ["price_response"])
                return 200, d
            if method != "POST" or path not in ("/v1/baseline", "/v1/query", "/v1/commit"):
                return _problem(404, "Not Found", f"{method} {path}")
            try:
                req = json.loads(body or b"null")
            except ValueError:
                return _problem(400, "Bad Request", "the body is not JSON")
            qid = h.get("query-id") or uuid.uuid4().hex
            if len(qid) > 64:
                return _problem(400, "Bad Request", "Query-Id is at most 64 characters")
            if path == "/v1/baseline":
                window = PlanWindow.from_dict(req)
                s = self._session(window)
                assert s is not None
                with s.lock:
                    a = s.participant.baseline()
                    self._keep(s, qid, a)
                return 200, answer_to_dict(a)
            if path == "/v1/query":
                if not isinstance(req, dict) or req.get("version") != SCHEMA_VERSION:
                    return _problem(400, "Bad Request", f"a device-query version {SCHEMA_VERSION} is expected")
                window = PlanWindow.from_dict(req.get("horizon"))
                q = query_from_dict(req)
                if q.kind not in KINDS:
                    return _problem(400, "Bad Request", f"unknown kind {q.kind!r}")
                for name in ("price_draw", "price_supply", "residual_kw", "target_kw"):
                    v = getattr(q, name)
                    if v is not None and (v.shape != (window.slots,) or not np.all(np.isfinite(v))):
                        return _problem(400, "Bad Request", f"{name} must be {window.slots} finite numbers")
                s = self._session(window)
                assert s is not None
                with s.lock:
                    a = s.participant.respond(q)
                    self._keep(s, qid, a)
                return 200, answer_to_dict(a)
            # /v1/commit
            if not isinstance(req, dict) or req.get("version") != SCHEMA_VERSION:
                return _problem(400, "Bad Request", f"a commit version {SCHEMA_VERSION} is expected")
            window = PlanWindow.from_dict(req.get("horizon"))
            s = self._session(window, create=False)
            chosen = None if s is None else s.answers.get(str(req.get("query_id")))
            if s is None or chosen is None:
                return _problem(409, "Conflict", "no answer of this horizon has that query_id")
            plan = np.asarray(req.get("plan_kw"), dtype=float)
            if plan.shape != chosen.plan_kw.shape or not np.allclose(plan, chosen.plan_kw, atol=1e-6):
                return _problem(409, "Conflict", "plan_kw is not the plan that query was answered with")
            if self._commit is None:
                return 200, {"accepted": False, "reason": "this participant does not run plans"}
            ok, reason = self._commit(s.participant, chosen)
            return 200, {"accepted": bool(ok), "reason": str(reason)}
        except WindowRefused as exc:
            return _problem(409, "Conflict", str(exc))
        except ValueError as exc:
            return _problem(400, "Bad Request", str(exc))
        except NotImplementedError as exc:
            return _problem(422, "Unprocessable Content", str(exc))
        except Exception as exc:  # noqa: BLE001 - the host's model failed: say so, never a bare 500
            return _problem(503, "Service Unavailable", f"{type(exc).__name__}: {exc}")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    service: ParticipantService
    url: str


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def _reply(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 4_000_000:
            status, out = _problem(413, "Content Too Large")
        else:
            body = self.rfile.read(n) if n else None
            base = self.path.split("?", 1)[0]
            status, out = self.server.service.handle(self.command, base, dict(self.headers), body)
        raw = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/problem+json" if status >= 400 else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = _reply

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the stdlib's name
        pass


def serve(service: ParticipantService, host: str = "127.0.0.1", port: int = 0) -> _Server:
    """Serve `service` on `host`:`port` (0: any free port) from a daemon
    thread. Returns the server: its `url` is the API's base; `shutdown()`
    stops it."""
    server = _Server((host, port), _Handler)
    server.service = service
    server.url = f"http://{host}:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
