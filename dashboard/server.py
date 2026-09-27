"""FastAPI server: serves the dashboard, pushes every bus event over /ws, exposes the demo controls (plan 02 §6).

Bus callbacks fire on audio/worker threads; they never touch sockets. They hand the event to the server loop with
`loop.call_soon_threadsafe`, and each client drains its own bounded queue, so a slow browser can't stall a driver.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parent.parent
# arena.html's match data when no CallGuard match has run yet: a saved Keyguard match, else the vendored sample
ARMS_RACE_FALLBACKS = (ROOT / "runs" / "keyguard" / "arms_race_data.js", ROOT / "keyguard" / "web" / "arms_race_data.js")
log = logging.getLogger(__name__)
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _hostname(value: str | None) -> str | None:
    """'http://localhost:8765' / 'localhost:8765' / '[::1]:8765' -> 'localhost' / '::1'."""
    if not value:
        return None
    v = value.split("://", 1)[-1].split("/", 1)[0]
    if v.startswith("["):
        return v[1:v.find("]")]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def is_local(headers: Any, allowed: frozenset[str] = LOCAL_HOSTS) -> bool:
    """The page and the request both belong to this machine. The dashboard shows what an eavesdropper reads from
    your keyboard, so another site in the same browser must not read /ws (WebSockets skip CORS) and a DNS-rebinding
    name must not reach the API. Requests with no Origin (curl, tests) only need a local Host."""
    origin = headers.get("origin")
    return _hostname(headers.get("host")) in allowed and (origin is None or _hostname(origin) in allowed)
CLIENT_QUEUE = 512  # events buffered per browser; beyond this the oldest are dropped


class ShieldCmd(BaseModel):
    mode: Literal["off", "dsp", "adversarial"]


class SecretCmd(BaseModel):
    action: Literal["allow", "arm", "disarm", "auto"]


class MeetCmd(BaseModel):
    action: Literal["join", "leave"]
    url: str | None = None


class ScenarioCmd(BaseModel):
    action: Literal["start", "stop"]
    name: str = "ai_caller"


def _jsonable(o: Any) -> Any:
    """numpy scalars/arrays and dataclasses show up in payloads; turn them into plain JSON."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        return {k: getattr(o, k) for k in o.__dataclass_fields__}
    return str(o)


def _as_msg(ev: Any) -> dict:
    if isinstance(ev, dict):
        return {"topic": ev.get("topic"), "t": ev.get("t", time.time()), "data": ev.get("data", {})}
    return {"topic": ev.topic, "t": ev.t, "data": ev.data}


def create_app(bus: Any, state_provider: Callable[[], dict] | None = None, controls: Any = None,
               allowed_hosts: frozenset[str] = LOCAL_HOSTS) -> FastAPI:
    """bus: anything with subscribe(glob, fn) -> unsubscribe. state_provider: returns the snapshot
    {topic: {"t", "data"}}; default = the latest event per topic seen here. controls: set_shield(mode),
    scenario(action, name), secret(action), meet(action, url)."""
    latest: dict[str, dict] = {}
    clients: set[asyncio.Queue] = set()

    def snapshot() -> dict:
        return state_provider() if state_provider else dict(latest)

    def fan_out(msg: dict) -> None:  # runs on the server loop
        for q in list(clients):
            if q.full():
                q.get_nowait()
            q.put_nowait(msg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()

        def on_event(ev: Any) -> None:  # runs on the publisher's thread: never raise back into it
            try:
                msg = _as_msg(ev)
                latest[msg["topic"]] = {"t": msg["t"], "data": msg["data"]}
                loop.call_soon_threadsafe(fan_out, msg)
            except Exception:
                pass

        unsubscribe = bus.subscribe("*", on_event)
        try:
            yield
        finally:
            if callable(unsubscribe):
                unsubscribe()

    app = FastAPI(title="CallGuard", lifespan=lifespan)

    @app.middleware("http")
    async def local_only(request, call_next):
        if not is_local(request.headers, allowed_hosts):
            return PlainTextResponse("CallGuard only answers pages served from this machine", status_code=403)
        return await call_next(request)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state() -> Response:
        return Response(json.dumps(snapshot(), default=_jsonable), media_type="application/json")

    @app.get("/api/health")  # the desktop shell waits on this before opening the window
    def health() -> dict:
        return {"ok": True, "mode": getattr(controls, "mode", None)}

    def _control(method: str, *args):
        fn = getattr(controls, method, None)
        if fn is None:
            raise HTTPException(503, f"control '{method}' not available")
        try:
            out = fn(*args)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "result": out}

    @app.post("/api/control/shield")
    def control_shield(cmd: ShieldCmd):
        return _control("set_shield", cmd.mode)

    @app.post("/api/control/secret")
    def control_secret(cmd: SecretCmd):
        return _control("secret", cmd.action)

    @app.post("/api/control/meet")
    def control_meet(cmd: MeetCmd):
        return _control("meet", cmd.action, cmd.url)

    @app.post("/api/control/scenario")
    def control_scenario(cmd: ScenarioCmd):
        return _control("scenario", cmd.action, cmd.name)

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        if not is_local(sock.headers, allowed_hosts):
            await sock.close(code=1008)
            return
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(CLIENT_QUEUE)
        clients.add(q)  # before the snapshot, so nothing published in between is lost

        async def pump() -> None:
            await sock.send_text(json.dumps({"topic": "snapshot", "t": time.time(), "data": snapshot()},
                                            default=_jsonable))
            while True:
                await sock.send_text(json.dumps(await q.get(), default=_jsonable))

        async def drain() -> None:  # only here to notice the browser leaving
            while True:
                await sock.receive_text()

        tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            clients.discard(q)
            for t in tasks:
                t.cancel()
                if t.done() and not t.cancelled():
                    t.exception()  # disconnects end up here; retrieve so asyncio doesn't log them

    @app.get("/keyguard/static/arms_race_data.js")  # before the mount, so it shadows Keyguard's static file
    def arms_race_data() -> Response:
        match = latest.get("keyguard.arms_race")  # the bus subscription in lifespan keeps this current
        if match:
            body = "window.ARMS_RACE = " + json.dumps(match["data"], default=_jsonable) + ";\n"
        else:
            src = next((p for p in ARMS_RACE_FALLBACKS if p.exists()), None)
            body = src.read_text() if src else "window.ARMS_RACE = null;\n"
        return Response(body, media_type="application/javascript", headers={"Cache-Control": "no-store"})

    _mount_keyguard(app)
    return app


def _mount_keyguard(app: FastAPI) -> None:
    """The vendored Keyguard console (population arena, runs, pipelines, arms-race replay) at /keyguard/. Its pages
    use relative api/ and static/ paths, so link it with the trailing slash. If it can't load, /keyguard says why."""
    try:
        from keyguard.server import app as keyguard_app
    except Exception as e:  # missing deps or a broken vendored copy must not take the dashboard down
        log.warning("Keyguard console not mounted: %s", e)
        reason = f"Keyguard console unavailable: {type(e).__name__}: {e}"

        @app.get("/keyguard", include_in_schema=False)
        @app.get("/keyguard/{rest:path}", include_in_schema=False)
        def keyguard_unavailable(rest: str = "") -> PlainTextResponse:
            return PlainTextResponse(reason, status_code=503)
        return
    app.mount("/keyguard", keyguard_app, name="keyguard")
