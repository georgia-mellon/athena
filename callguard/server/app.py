"""FastAPI server: serves the dashboard, pushes every bus event over /ws, exposes the demo controls (plan 02 §6).

Bus callbacks fire on audio/worker threads; they never touch sockets. They hand the event to the server loop with
`loop.call_soon_threadsafe`, and each client drains its own bounded queue, so a slow browser can't stall a driver.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"
CLIENT_QUEUE = 512  # events buffered per browser; beyond this the oldest are dropped


class ShieldCmd(BaseModel):
    mode: Literal["off", "dsp", "adversarial"]


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


def create_app(bus: Any, state_provider: Callable[[], dict] | None = None, controls: Any = None) -> FastAPI:
    """bus: anything with subscribe(glob, fn) -> unsubscribe. state_provider: returns the snapshot
    {topic: {"t", "data"}}; default = the latest event per topic seen here. controls: set_shield(mode),
    scenario(action, name)."""
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
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state() -> Response:
        return Response(json.dumps(snapshot(), default=_jsonable), media_type="application/json")

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

    @app.post("/api/control/scenario")
    def control_scenario(cmd: ScenarioCmd):
        return _control("scenario", cmd.action, cmd.name)

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
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

    return app
