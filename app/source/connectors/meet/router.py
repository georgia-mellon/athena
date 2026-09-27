"""The Meet bridge's server side: /meet/mic and /meet/far WebSockets, the local test room, and its static files.

Wire format both ways: raw little-endian float32 mono 16 kHz, one BLOCK (320 samples, 20 ms) per binary message.
/meet/mic answers each block with one processed block of the same size, in order. A wrong-size block is echoed
unchanged so the bridge never loses its mic. Text on /meet/mic is the bridge's stats ({"rtt_ms": ...}).

One page owns each stream at a time; a second tab's socket is closed with 1013 and retries with backoff, so two
tabs can't interleave blocks. Exception: meet.google.com takes over from a local page (a real meeting beats a
stray test-room tab). The owner's origin goes to the pipeline via meet_link.
"""
from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app.source.config import REPO
from app.source.types import BLOCK

HERE = Path(__file__).parent
EXTENSION = HERE / "extension"  # the Chrome extension that injects bridge.js into Meet (load unpacked)
STATIC = {"bridge.js": (EXTENSION / "bridge.js", "text/javascript"),  # everything /meet/static/ serves
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
    except ValueError:  # a malformed port
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
    r.meet_stats = stats  # for tests and the e2e check
    pages: dict[int, tuple[WebSocket, str | None, asyncio.AbstractEventLoop]] = {}   # /meet/status sockets
    ids = itertools.count(1)

    def command(msg: dict) -> bool:
        """Send a command (leave, open) to the newest Meet page, else the newest page. Any thread."""
        if not pages:
            return False
        meet = [k for k, (_, o, _) in pages.items() if o == MEET_ORIGIN]
        sock, _, loop = pages[max(meet or pages)]
        asyncio.run_coroutine_threadsafe(sock.send_text(json.dumps(msg)), loop)
        return True
    pipeline.meet_command = command  # the dashboard's Join / Leave reach the page through this

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
            await sock.accept()  # another tab has it
            await sock.close(code=1013)
            return False
        owner[kind], origins[kind] = sock, origin or "local process"
        if old is not None:  # Meet takes over from a local page
            try:
                await old.close(code=1013)
            except Exception:  # noqa: BLE001 - it was already going away
                pass
        await sock.accept()
        link(kind, +1)  # mic: the pipeline flushes its delay line (this thread)
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
                if msg["type"] == "websocket.disconnect" or owner["mic"] is not sock:  # or taken over
                    break
                data = msg.get("bytes")
                if data is None:  # text: the bridge's stats
                    try:
                        link("mic", 0, float(json.loads(msg.get("text") or "{}")["rtt_ms"]))
                    except (ValueError, KeyError, TypeError):
                        pass
                    continue
                stats["mic_in"] += 1
                if len(data) == BLOCK * 4:
                    # shield runs on the server loop (< 1 ms for DSP); move to a thread per socket if a
                    # heavier shield ever stalls the dashboard sockets.
                    y = pipeline.meet_mic(np.frombuffer(data, "<f4"))
                    data = np.asarray(y, "<f4").tobytes()
                await sock.send_bytes(data)
                stats["mic_out"] += 1
        except (WebSocketDisconnect, RuntimeError):  # RuntimeError: closed under us by a takeover
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

    @r.websocket("/meet/status")
    async def status(sock: WebSocket) -> None:
        """Every page with the bridge: {"site", "in_call", ...} each second (the dashboard's meeting indicator)."""
        origin = sock.headers.get("origin")
        if not origin_ok(origin, own_port()):
            stats["rejected"] += 1
            await sock.close(code=1008)
            return
        await sock.accept()
        pid = next(ids)
        pages[pid] = (sock, origin, asyncio.get_running_loop())
        page = getattr(pipeline, "meet_page", None)
        try:
            while True:
                text = await sock.receive_text()
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                if page is not None and isinstance(msg, dict):
                    page(pid, origin, msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            pages.pop(pid, None)
            if page is not None:
                page(pid, origin, None)

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
