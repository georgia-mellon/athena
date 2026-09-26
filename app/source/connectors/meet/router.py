"""The Meet bridge's server side: /meet/mic and /meet/far WebSockets, the local test room, and its static files.

Wire format both ways: raw little-endian float32 mono 16 kHz, one BLOCK (320 samples, 20 ms) per binary message.
/meet/mic answers every block with exactly one processed block of the same size, in order (the bridge measures the
round trip from that). A block of the wrong size is echoed unchanged: the bridge must never lose its mic. Text
messages on /meet/mic are the bridge's stats ({"rtt_ms": ...}).

One page owns each stream at a time (first come): a second tab's socket is closed with 1013 "try again later", so
its bridge passes the mic through raw and retries with backoff until the first one leaves. Two tabs interleaving
blocks would scramble the shield's and the spotter's timeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app.source.config import REPO
from app.source.types import BLOCK

HERE = Path(__file__).parent
STATIC = {"bridge.js": "text/javascript", "testroom.html": "text/html"}   # everything /meet/static/ serves
DEMO_AUDIO = REPO / "demo" / "audio"
MEET_ORIGIN = "https://meet.google.com"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _host(value: str) -> str:
    """'http://localhost:8765' -> 'localhost', 'http://[::1]:80' -> '::1'."""
    v = value.split("://", 1)[-1].split("/", 1)[0]
    if v.startswith("["):
        return v[1:v.find("]")]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def origin_ok(origin: str | None) -> bool:
    """Meet itself or a page served from this machine. No Origin = not a browser (a local process): allowed, it
    could connect anyway. Anything else is another website trying to listen to (or inject into) your mic."""
    if origin is None:
        return True
    return origin == MEET_ORIGIN or (origin.startswith(("http://", "https://")) and _host(origin) in LOCAL_HOSTS)


def make_router(pipeline) -> APIRouter:
    """`pipeline`: meet_mic(block) -> block, meet_far(block), optional meet_link(kind, delta, rtt_ms)."""
    r = APIRouter()
    owner: dict[str, WebSocket | None] = {"mic": None, "far": None}
    stats = {"mic_in": 0, "mic_out": 0, "far_in": 0, "rejected": 0}
    r.meet_stats = stats                                 # for tests and the e2e check

    def link(kind: str, delta: int = 0, rtt_ms: float | None = None) -> None:
        fn = getattr(pipeline, "meet_link", None)
        if fn is not None:
            fn(kind, delta, rtt_ms)

    async def claim(sock: WebSocket, kind: str) -> bool:
        if not origin_ok(sock.headers.get("origin")):
            stats["rejected"] += 1
            await sock.close(code=1008)
            return False
        if owner[kind] is not None:                      # another tab has it
            await sock.accept()
            await sock.close(code=1013)
            return False
        owner[kind] = sock
        await sock.accept()
        link(kind, +1)
        return True

    def release(sock: WebSocket, kind: str) -> None:
        if owner[kind] is sock:
            owner[kind] = None
            link(kind, -1)

    @r.websocket("/meet/mic")
    async def mic(sock: WebSocket) -> None:
        if not await claim(sock, "mic"):
            return
        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None:                         # text: the bridge's stats
                    try:
                        link("mic", 0, float(json.loads(msg.get("text") or "{}")["rtt_ms"]))
                    except (ValueError, KeyError, TypeError):
                        pass
                    continue
                stats["mic_in"] += 1
                if len(data) == BLOCK * 4:
                    # ponytail: the shield runs on the server loop (it's the audio-thread budget, < 1 ms for DSP);
                    # move it to a thread per socket if a heavier shield ever stalls the dashboard sockets.
                    y = pipeline.meet_mic(np.frombuffer(data, "<f4"))
                    data = np.asarray(y, "<f4").tobytes()
                await sock.send_bytes(data)
                stats["mic_out"] += 1
        except WebSocketDisconnect:
            pass
        finally:
            release(sock, "mic")

    @r.websocket("/meet/far")
    async def far(sock: WebSocket) -> None:
        if not await claim(sock, "far"):
            return
        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data and len(data) % 4 == 0:
                    stats["far_in"] += 1
                    pipeline.meet_far(np.frombuffer(data, "<f4"))
        except WebSocketDisconnect:
            pass
        finally:
            release(sock, "far")

    @r.get("/meet/testroom")
    def testroom() -> FileResponse:
        return FileResponse(HERE / "testroom.html", media_type="text/html")

    @r.get("/meet/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in STATIC:
            raise HTTPException(404)
        return FileResponse(HERE / name, media_type=STATIC[name], headers={"cache-control": "no-store"})

    @r.get("/meet/audio")
    def audio_list() -> JSONResponse:
        """The demo's WAVs (demo/audio, gitignored) the test room can play as the remote participant."""
        files = sorted(p.relative_to(DEMO_AUDIO).as_posix() for p in DEMO_AUDIO.rglob("*.wav")) \
            if DEMO_AUDIO.is_dir() else []
        return JSONResponse(files)

    @r.get("/meet/audio/{path:path}")
    def audio_file(path: str) -> FileResponse:
        p = (DEMO_AUDIO / path).resolve()
        if p.suffix.lower() != ".wav" or not p.is_file() or not p.is_relative_to(DEMO_AUDIO.resolve()):
            raise HTTPException(404)
        return FileResponse(p, media_type="audio/wav")

    return r
