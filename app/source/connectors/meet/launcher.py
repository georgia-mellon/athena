"""Open Google Meet in Athena's own Chrome (or Edge) window with the audio bridge injected.

Runs a dedicated profile outside the repo (see PROFILE), so a zip of it never carries a signed-in Google session;
sign in there once or join as a guest. The browser starts with --remote-debugging-port=0 and we read the port it
picked from <profile>/DevToolsActivePort (Chrome 136+ only allows remote debugging with a non-default profile).

DevTools runs over a TCP port on 127.0.0.1, not a pipe. Any local process running as you could drive this window,
but it could read the profile's cookies anyway.

Over DevTools every page gets bridge.js injected before its scripts run, and its document requests intercepted.
Each main-frame navigation is classified by `where`: Meet and local pages get Page.setBypassCSP (Meet's CSP blocks
ws://127.0.0.1 and the blob: worklet); other Google pages (sign-in) load without it; anything else opens in the
system browser. Switching bypass state cancels and restarts the navigation (GET only).

Local Network Access (Chrome 142+) would prompt before Meet may reach ws://127.0.0.1, so we grant it to
https://meet.google.com only; a browser that can't do that is relaunched with LNA checks off (see MeetSession.lna).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

log = logging.getLogger(__name__)
HERE = Path(__file__).parent


def _data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


PROFILE = _data_dir() / "Athena" / "meet-profile"
MEET_ORIGIN = "https://meet.google.com"
MEET_HOME = MEET_ORIGIN + "/"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
_CODE = re.compile(r"^[a-z]{3}-[a-z]{4}-[a-z]{3}$")
FLAGS = ["--no-first-run", "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required",
         "--remote-debugging-address=127.0.0.1"]
LNA_FLAG = "--disable-features=LocalNetworkAccessChecks"  # fallback only (old builds ignore unknown names)
LNA_PERMISSIONS = ("local-network-access", "local-network", "loopback-network")  # the 142+ name, then the later split
DOC_REQUESTS = [{"resourceType": "Document", "requestStage": "Request"}]


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
    if (p.scheme == "https" and p.hostname == "meet.google.com") or \
            (p.scheme in ("http", "https") and p.hostname in LOCAL_HOSTS):
        return url
    raise ValueError(f"not a Google Meet link: {url!r} (expected https://meet.google.com/abc-defg-hij)")


def where(url: str) -> str | None:
    """Where a main-frame navigation may go. 'bridge': Meet or a local page (CSP bypassed, the bridge runs);
    'window': Google sign-in, about:blank, error pages (no bypass); None: not this window (the system browser)."""
    p = urlsplit(url)
    if p.scheme not in ("http", "https"):
        return "window"
    host = p.hostname or ""
    if host == "meet.google.com" or host in LOCAL_HOSTS:  # http://meet is upgraded to https (HSTS)
        return "bridge"
    if host == "google.com" or host.endswith(".google.com") or host == "accounts.youtube.com":
        return "window"  # sign-in (accounts.youtube.com sets its cookie)
    return None


def find_browser() -> Path | None:
    """ATHENA_BROWSER, then Chrome, then Edge (Windows install paths, macOS, PATH)."""
    env = os.environ.get("ATHENA_BROWSER")
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
               "msedge.exe": "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"}[PureWindowsPath(rel).name]
        if sys.platform == "darwin" and Path(mac).is_file():
            return Path(mac)
        for n in names:
            if w := shutil.which(n):
                return Path(w)
    return None


def open_tab(url: str) -> str:
    """Open `url` as a normal tab in the user's own Chrome / Edge (their profile, no automation). The Athena
    extension (extension/, loaded unpacked once) puts the bridge in the page. Returns what opened it."""
    b = find_browser()
    if b is None:
        import webbrowser
        webbrowser.open(url)
        return "the default browser"
    subprocess.Popen([str(b), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return b.stem


def show_extension() -> str:
    """Open the extension folder in the file manager (for chrome://extensions > Load unpacked)."""
    path = HERE / "extension"
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - a local folder
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    return str(path)


def _merge_features(args: list[str]) -> list[str]:
    """Chrome only honours the last --disable-features=...: fold them all into one."""
    off = [f for a in args if a.startswith("--disable-features=") for f in a.split("=", 1)[1].split(",") if f]
    rest = [a for a in args if not a.startswith("--disable-features=")]
    return rest + ([f"--disable-features={','.join(dict.fromkeys(off))}"] if off else [])


def bridge_source(port: int) -> str:
    return (f"window.__athenaPort={int(port)};window.__athenaBridgeSource='cdp';\n"
            + (HERE / "extension" / "bridge.js").read_text(encoding="utf-8"))


class MeetSession:
    """A running browser window + its DevTools connection. close() ends both."""

    def __init__(self, proc: subprocess.Popen, browser: str, ws_url: str, source: str):
        self.proc, self.browser, self.url = proc, browser, None
        self._source, self._main = source, None
        self._targets: dict[str, str] = {}  # session id -> target id, page targets
        self._bypass: dict[str, bool] = {}  # session id -> Page.setBypassCSP state
        self.lna: str | None = None  # 'grant' | 'flag': how Meet may reach 127.0.0.1
        self._pending: dict[int, asyncio.Future] = {}
        self._n = 0
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="athena-cdp", daemon=True).start()
        self._call(self._connect(ws_url))

    # --- plumbing ---------------------------------------------------------------------------------------------
    def _call(self, coro, timeout: float = 20.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self, ws_url: str) -> None:
        import websockets
        self._ws = await websockets.connect(ws_url, max_size=None, ping_interval=None)
        self._reader = asyncio.create_task(self._read())
        await self._send("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=True, flatten=True)
        for t in (await self._send("Target.getTargets"))["targetInfos"]:  # pages from before auto-attach
            if t["type"] == "page" and not t.get("attached"):
                await self._send("Target.attachToTarget", targetId=t["targetId"], flatten=True)
        for _ in range(100):
            if self._main:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("no browser tab to attach to")

    async def _send(self, method: str, session: str | None = None, **params):
        return await asyncio.wait_for(await self._post(method, session, **params), 15)

    async def _post(self, method: str, session: str | None = None, **params) -> asyncio.Future:
        """Send a command; return the future of its reply without waiting for it."""
        self._n += 1
        fut = self._loop.create_future()
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # an unawaited error is fine
        self._pending[self._n] = fut
        msg = {"id": self._n, "method": method, "params": params}
        if session:
            msg["sessionId"] = session
        await self._ws.send(json.dumps(msg))
        return fut

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
                elif m.get("method") == "Fetch.requestPaused":
                    asyncio.create_task(self._paused(m.get("sessionId"), m["params"]))
                elif m.get("method") == "Target.detachedFromTarget" and m["params"].get("sessionId") == self._main:
                    self._main = None
        except Exception:  # noqa: BLE001 - the browser went away; `alive` reports it
            pass

    async def _attached(self, p: dict) -> None:
        sid, info = p["sessionId"], p["targetInfo"]
        try:
            if info["type"] == "page":
                self._targets[sid] = info["targetId"]
                await self._send("Page.enable", sid)
                await self._send("Fetch.enable", sid, patterns=DOC_REQUESTS)
                await self._send("Page.addScriptToEvaluateOnNewDocument", sid, source=self._source)
                self._main = self._main or sid
        except Exception:  # noqa: BLE001 - best effort per tab: a tab without the bridge just isn't protected
            pass
        finally:
            if p.get("waitingForDebugger"):  # paused until we let it go, whatever the type
                try:
                    await self._send("Runtime.runIfWaitingForDebugger", sid)
                except Exception:  # noqa: BLE001
                    pass

    async def _paused(self, sid: str, p: dict) -> None:
        """A document request, held until we answer. Main frame: set the CSP bypass for where it is going (before
        its response can commit), or hand it to the system browser. Subframes, and anything that fails: continue."""
        rid, url = p["requestId"], p["request"]["url"]
        try:
            tid = self._targets.get(sid)
            if tid is not None and p.get("frameId") == tid:  # main frame id equals the target id
                kind = where(url)
                if kind is None:
                    await self._send("Fetch.failRequest", sid, requestId=rid, errorReason="BlockedByClient")
                    log.info("opening a non-Meet link (%s) in the system browser", urlsplit(url).hostname)
                    await asyncio.get_running_loop().run_in_executor(None, webbrowser.open, url)
                    if sid != self._main:  # a tab that only existed for this link
                        await self._send("Target.closeTarget", targetId=tid)
                    return
                bypass = kind == "bridge"
                if bypass != self._bypass.get(sid, False) and p["request"].get("method", "GET") == "GET":
                    # a held navigation keeps the state it started with: cancel it, switch bypass, navigate again.
                    await self._send("Fetch.failRequest", sid, requestId=rid, errorReason="Aborted")
                    await self._send("Page.setBypassCSP", sid, enabled=bypass)
                    self._bypass[sid] = bypass
                    await self._post("Page.navigate", sid, url=url)
                    return
        except Exception:  # noqa: BLE001 - never leave a navigation hanging
            pass
        try:
            await self._send("Fetch.continueRequest", sid, requestId=rid)
        except Exception:  # noqa: BLE001 - the tab went away, or it was already answered
            pass

    async def _grant_lna(self) -> bool:
        """Let https://meet.google.com (only) reach the local network. False if this browser knows none of the
        permission names."""
        ok = False
        for name in LNA_PERMISSIONS:
            try:
                await self._send("Browser.setPermission", permission={"name": name}, setting="granted",
                                 origin=MEET_ORIGIN)
                ok = True
            except RuntimeError:  # unknown to this build
                pass
        return ok

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
    """Start the browser, inject the bridge (pointing at ws://127.0.0.1:{port}/meet/...), let Meet reach it (Local
    Network Access), open `url` (Meet's home if None). Raises FileNotFoundError (no browser) or RuntimeError (it
    didn't come up)."""
    url = meet_url(url)
    exe = find_browser()
    if exe is None:
        raise FileNotFoundError("Chrome or Edge not found (set ATHENA_BROWSER to the browser's .exe)")
    profile = Path(profile_dir or PROFILE)
    profile.mkdir(parents=True, exist_ok=True)
    extra = list(extra_args or [])
    s = _start(exe, profile, port, extra)
    try:
        if s._call(s._grant_lna()):
            s.lna = "grant"
        else:
            s.close()
            s = _start(exe, profile, port, extra + [LNA_FLAG])
            s.lna = "flag"
        log.info("Local Network Access: %s", f"granted to {MEET_ORIGIN} only" if s.lna == "grant"
                 else "this browser can't grant it per origin; checks are off for the whole Meet window")
        s.navigate(url)
    except Exception as e:
        s.proc.kill()
        raise RuntimeError(f"couldn't drive {exe.name} over DevTools: {e}") from e
    return s


def _start(exe: Path, profile: Path, port: int, extra_args: list[str]) -> MeetSession:
    active = profile / "DevToolsActivePort"
    active.unlink(missing_ok=True)
    proc = subprocess.Popen([str(exe), f"--user-data-dir={profile}", "--remote-debugging-port=0",
                             *_merge_features(FLAGS + extra_args), "about:blank"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    while not active.exists() or len(active.read_text().split()) < 2:
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"{exe.name} didn't open a DevTools port (is an Athena Meet window already open "
                               f"on {profile}? close it first)")
        time.sleep(0.1)
    dev_port, path = active.read_text().split()[:2]
    try:
        return MeetSession(proc, exe.stem, f"ws://127.0.0.1:{dev_port}{path}", bridge_source(port))
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"couldn't drive {exe.name} over DevTools: {e}") from e
