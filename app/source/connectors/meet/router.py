"""The Meet bridge's server side: /meet/mic and /meet/far WebSockets, the local test room, and its static files.

Wire format both ways: raw little-endian float32 mono 16 kHz, one BLOCK (320 samples, 20 ms) per binary message.
/meet/mic answers every block with exactly one processed block of the same size, in order (the bridge measures the
round trip from that). A block of the wrong size is echoed unchanged: the bridge must never lose its mic. Text
messages on /meet/mic are the bridge's stats ({"rtt_ms": ...}).

One page owns each stream at a time: a second tab's socket is closed with 1013 "try again later", so its bridge
passes the mic through raw and retries with backoff until the first one leaves. Two tabs interleaving blocks would
scramble the shield's and the spotter's timeline. Exception: https://meet.google.com takes a stream over from a local
page (a real meeting beats a stray test-room tab); the local one gets the 1013. The owner's origin goes to the
pipeline (meet_link), which shows it in meet.state and flushes its delay line for every new mic owner.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app.source.config import REPO
from app.source.types import BLOCK

HERE = Path(__file__).parent
EXTENSION = HERE / "extension"                  # the Chrome extension that injects bridge.js into Meet (load unpacked)
STATIC = {"bridge.js": (EXTENSION / "bridge.js", "text/javascript"),       # everything /meet/static/ serves
          "testroom.html": (HERE / "testroom.html", "text/html")}
DEMO_AUDIO = REPO / "demo" / "audio"
MEET_ORIGIN = "https://meet.google.com"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def origin_ok(origin: str | None, port: int | None) -> bool:
    """Meet itself, or a page this server serves (a local host on `port`; None = any port). No Origin = not a
    browser (a local process): allowed, it could connect anyway. Anything else is another website, or another local
    server, trying to listen to (or inject into) your mic."""
    if origin is None or origin == MEET_ORIGIN:
        return True
    try:
        p = urlsplit(origin)
        host, oport = p.hostname, p.port or {"http": 80, "https": 443}.get(p.scheme)
    except ValueError:                                   # a malformed port
        return False
    return p.scheme in ("http", "https") and host in LOCAL_HOSTS and port in (None, oport)


def make_router(pipeline, port: int | None = None) -> APIRouter:
    """`pipeline`: meet_mic(block) -> block, meet_far(block), optional meet_link(kind, delta, rtt_ms, owner).
    `port`: the server's own port, the only one local pages may connect from (default: pipeline.meet_port, else
    pipeline.cfg.server.port, looked up per connection)."""
    r = APIRouter()
    owner: dict[str, WebSocket | None] = {"mic": None, "far": None}
    origins: dict[str, str | None] = {"mic": None, "far": None}
    stats = {"mic_in": 0, "mic_out": 0, "far_in": 0, "rejected": 0}
    r.meet_stats = stats                                 # for tests and the e2e check

    def own_port() -> int | None:
        cfg = getattr(pipeline, "cfg", None)
        return port or getattr(pipeline, "meet_port", None) or getattr(getattr(cfg, "server", None), "port", None)

    def link(kind: str, delta: int = 0, rtt_ms: float | None = None) -> None:
        fn = getattr(pipeline, "meet_link", None)
        if fn is not None:
            fn(kind, delta, rtt_ms, origins[kind])

    async def claim(sock: WebSocket, kind: str) -> bool:
        origin = sock.headers.get("origin")
        if not origin_ok(origin, own_port()):
            stats["rejected"] += 1
            await sock.close(code=1008)
            return False
        old = owner[kind]
        if old is not None and not (origin == MEET_ORIGIN and origins[kind] != MEET_ORIGIN):
            await sock.accept()                          # another tab has it
            await sock.close(code=1013)
            return False
        owner[kind], origins[kind] = sock, origin or "local process"
        if old is not None:                              # Meet takes over from a local page
            try:
                await old.close(code=1013)
            except Exception:  # noqa: BLE001 - it was already going away
                pass
        await sock.accept()
        link(kind, +1)                                   # mic: the pipeline flushes its delay line (this thread)
        return True

    def release(sock: WebSocket, kind: str) -> None:
        if owner[kind] is sock:
            owner[kind] = origins[kind] = None
            link(kind, -1)

    @r.websocket("/meet/mic")
    async def mic(sock: WebSocket) -> None:
        if not await claim(sock, "mic"):
            return
        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect" or owner["mic"] is not sock:   # or taken over
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
        except (WebSocketDisconnect, RuntimeError):     # RuntimeError: closed under us by a takeover
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
                if msg["type"] == "websocket.disconnect" or owner["far"] is not sock:
                    break
                data = msg.get("bytes")
                if data and len(data) % 4 == 0:
                    stats["far_in"] += 1
                    pipeline.meet_far(np.frombuffer(data, "<f4"))
        except (WebSocketDisconnect, RuntimeError):
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
        path, media = STATIC[name]
        return FileResponse(path, media_type=media, headers={"cache-control": "no-store"})

    @r.get("/meet/audio")
    def audio_list() -> JSONResponse:
        """The test voices (demo/audio/testclips, gitignored; demo/build_testclips.py) the test room plays as the
        remote participant."""
        clips = DEMO_AUDIO / "testclips"
        files = sorted(p.relative_to(DEMO_AUDIO).as_posix() for p in clips.glob("*.wav")) if clips.is_dir() else []
        return JSONResponse(files)

    @r.get("/meet/audio/{path:path}")
    def audio_file(path: str) -> FileResponse:
        p = (DEMO_AUDIO / path).resolve()
        media = {".wav": "audio/wav", ".mp3": "audio/mpeg"}.get(p.suffix.lower())
        if media is None or not p.is_file() or not p.is_relative_to(DEMO_AUDIO.resolve()):
            raise HTTPException(404)
        return FileResponse(p, media_type=media)

    return r
