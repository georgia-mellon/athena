"""Open Google Meet in CallGuard's own Chrome (or Edge) window with the audio bridge injected.

The browser runs a dedicated profile (runs/meet-profile, gitignored: sign in to Google there once, or join as a
guest) with --remote-debugging-port=0; the port it picked is read from <profile>/DevToolsActivePort. Chrome 136+
only allows remote debugging with a non-default --user-data-dir, which we always pass. Over the DevTools protocol
(the installed `websockets` package, on a private asyncio loop) every page target gets, before any of its scripts
run: Page.setBypassCSP (Meet's CSP would block ws://127.0.0.1 and the blob: worklet) and
Page.addScriptToEvaluateOnNewDocument(bridge.js). Browser-level Target.setAutoAttach with waitForDebuggerOnStart
covers tabs opened later; bridge.js itself only activates on meet.google.com and local pages.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from app.source.config import REPO

HERE = Path(__file__).parent
PROFILE = REPO / "runs" / "meet-profile"
MEET_HOME = "https://meet.google.com/"
_CODE = re.compile(r"^[a-z]{3}-[a-z]{4}-[a-z]{3}$")
# Local Network Access (Chrome 142+) would prompt before meet.google.com may reach ws://127.0.0.1; this window
# exists for exactly that connection. Unknown feature names are ignored by older builds.
FLAGS = ["--no-first-run", "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required",
         "--disable-features=LocalNetworkAccessChecks"]


def meet_url(url: str | None) -> str:
    """None -> Meet's home; 'abc-defg-hij' -> that meeting; https://meet.google.com/... or a local http(s) page (the
    test room) as is. Anything else -> ValueError: this window only ever runs Meet."""
    if not url or not url.strip():
        return MEET_HOME
    url = url.strip()
    if _CODE.match(url.lower()):
        return MEET_HOME + url.lower()
    if url.startswith("meet.google.com"):
        url = "https://" + url
    p = urlsplit(url)
    if p.scheme == "https" and p.hostname == "meet.google.com":
        return url
    if p.scheme in ("http", "https") and p.hostname in ("127.0.0.1", "localhost", "::1"):
        return url
    raise ValueError(f"not a Google Meet link: {url!r} (expected https://meet.google.com/abc-defg-hij)")


def find_browser() -> Path | None:
    """CALLGUARD_BROWSER, then Chrome, then Edge (Windows install paths, macOS, PATH)."""
    env = os.environ.get("CALLGUARD_BROWSER")
    if env and Path(env).is_file():
        return Path(env)
    roots = [os.environ.get(k) for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    for rel, names in ((r"Google\Chrome\Application\chrome.exe", ("google-chrome", "google-chrome-stable", "chrome",
                                                                  "chromium", "chromium-browser")),
                       (r"Microsoft\Edge\Application\msedge.exe", ("msedge", "microsoft-edge"))):
        for root in filter(None, roots):
            if (p := Path(root) / rel).is_file():
                return p
        mac = {"chrome.exe": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
               "msedge.exe": "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"}[Path(rel).name]
        if sys.platform == "darwin" and Path(mac).is_file():
            return Path(mac)
        for n in names:
            if w := shutil.which(n):
                return Path(w)
    return None


def _merge_features(args: list[str]) -> list[str]:
    """Chrome only honours the last --disable-features=...: fold them all into one."""
    off = [f for a in args if a.startswith("--disable-features=") for f in a.split("=", 1)[1].split(",") if f]
    rest = [a for a in args if not a.startswith("--disable-features=")]
    return rest + ([f"--disable-features={','.join(dict.fromkeys(off))}"] if off else [])


def bridge_source(port: int) -> str:
    return (f"window.__callguardPort={int(port)};window.__callguardBridgeSource='cdp';\n"
            + (HERE / "bridge.js").read_text(encoding="utf-8"))


class MeetSession:
    """A running browser window + its DevTools connection. close() ends both."""

    def __init__(self, proc: subprocess.Popen, browser: str, ws_url: str, source: str):
        self.proc, self.browser, self.url = proc, browser, None
        self._source, self._main = source, None
        self._pending: dict[int, asyncio.Future] = {}
        self._n = 0
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="callguard-cdp", daemon=True).start()
        self._call(self._connect(ws_url))

    # --- plumbing ---------------------------------------------------------------------------------------------
    def _call(self, coro, timeout: float = 20.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self, ws_url: str) -> None:
        import websockets
        self._ws = await websockets.connect(ws_url, max_size=None, ping_interval=None)
        self._reader = asyncio.create_task(self._read())
        await self._send("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=True, flatten=True)
        for t in (await self._send("Target.getTargets"))["targetInfos"]:   # pages that existed before auto-attach
            if t["type"] == "page" and not t.get("attached"):
                await self._send("Target.attachToTarget", targetId=t["targetId"], flatten=True)
        for _ in range(100):
            if self._main:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("no browser tab to attach to")

    async def _send(self, method: str, session: str | None = None, **params):
        self._n += 1
        fut = self._loop.create_future()
        self._pending[self._n] = fut
        msg = {"id": self._n, "method": method, "params": params}
        if session:
            msg["sessionId"] = session
        await self._ws.send(json.dumps(msg))
        return await asyncio.wait_for(fut, 15)

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                m = json.loads(raw)
                if "id" in m:
                    fut = self._pending.pop(m["id"], None)
                    if fut is not None and not fut.done():
                        if "error" in m:
                            fut.set_exception(RuntimeError(f"CDP: {m['error'].get('message')}"))
                        else:
                            fut.set_result(m.get("result", {}))
                elif m.get("method") == "Target.attachedToTarget":
                    asyncio.create_task(self._attached(m["params"]))
                elif m.get("method") == "Target.detachedFromTarget" and m["params"].get("sessionId") == self._main:
                    self._main = None
        except Exception:  # noqa: BLE001 - the browser went away; `alive` reports it
            pass

    async def _attached(self, p: dict) -> None:
        sid, info = p["sessionId"], p["targetInfo"]
        try:
            if info["type"] == "page":
                await self._send("Page.enable", sid)
                await self._send("Page.setBypassCSP", sid, enabled=True)
                await self._send("Page.addScriptToEvaluateOnNewDocument", sid, source=self._source)
                self._main = self._main or sid
        except Exception:  # noqa: BLE001 - best effort per tab: a tab without the bridge just isn't protected
            pass
        finally:
            if p.get("waitingForDebugger"):                  # paused until we let it go, whatever the type
                try:
                    await self._send("Runtime.runIfWaitingForDebugger", sid)
                except Exception:  # noqa: BLE001
                    pass

    # --- API --------------------------------------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def navigate(self, url: str) -> None:
        """Load `url` in the main tab (or a new tab if it was closed)."""
        async def go():
            if self._main:
                try:
                    await self._send("Page.navigate", self._main, url=url)
                    return
                except RuntimeError:
                    self._main = None
            await self._send("Target.createTarget", url=url)
        self._call(go())
        self.url = url

    def focus(self) -> None:
        if self._main:
            self._call(self._send("Page.bringToFront", self._main))

    def evaluate(self, expression: str, timeout: float = 15.0):
        """Run JS in the main tab and return its (JSON) value (tests, the real-Meet check)."""
        async def ev():
            r = await self._send("Runtime.evaluate", self._main, expression=expression, returnByValue=True,
                                 awaitPromise=True, userGesture=True)
            if "exceptionDetails" in r:
                raise RuntimeError(r["exceptionDetails"].get("text", "evaluate failed"))
            return r["result"].get("value")
        return self._call(ev(), timeout)

    def close(self) -> None:
        if self.alive:
            try:
                self._call(self._send("Browser.close"), timeout=3)
            except Exception:  # noqa: BLE001 - kill it instead
                pass
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._loop.call_soon_threadsafe(self._loop.stop)


def launch(url: str | None, port: int, profile_dir: Path | None = None,
           extra_args: list[str] | None = None) -> MeetSession:
    """Start the browser, inject the bridge (pointing at ws://127.0.0.1:{port}/meet/...), open `url` (Meet's home
    if None). Raises FileNotFoundError (no browser) or RuntimeError (it didn't come up)."""
    url = meet_url(url)
    exe = find_browser()
    if exe is None:
        raise FileNotFoundError("Chrome or Edge not found (set CALLGUARD_BROWSER to the browser's .exe)")
    profile = Path(profile_dir or PROFILE)
    profile.mkdir(parents=True, exist_ok=True)
    active = profile / "DevToolsActivePort"
    active.unlink(missing_ok=True)
    proc = subprocess.Popen([str(exe), f"--user-data-dir={profile}", "--remote-debugging-port=0",
                             *_merge_features(FLAGS + list(extra_args or [])), "about:blank"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    while not active.exists() or len(active.read_text().split()) < 2:
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"{exe.name} didn't open a DevTools port (is a CallGuard Meet window already open "
                               f"on {profile}? close it first)")
        time.sleep(0.1)
    dev_port, path = active.read_text().split()[:2]
    try:
        s = MeetSession(proc, exe.stem, f"ws://127.0.0.1:{dev_port}{path}", bridge_source(port))
        s.navigate(url)
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"couldn't drive {exe.name} over DevTools: {e}") from e
    return s
