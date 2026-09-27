"""Desktop shell, headless: engine + server come up in-process, the dashboard is served, shutdown is clean."""
import json
import socket
import threading
import time
import urllib.request

import pytest

from app.source import desktop


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_headless_serves_dashboard_and_stops(monkeypatch):
    for slot in ("VOICE", "ATTACKER", "SHIELD", "SECRET"):
        monkeypatch.setenv(f"CALLGUARD_DRIVERS_{slot}", "mock")
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("rc", desktop.main(["--no-window", "--port", str(port)])),
                         daemon=True)
    t.start()
    health = None
    end = time.monotonic() + 60
    while time.monotonic() < end and health is None:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=1) as r:
                health = json.load(r)
        except OSError:
            time.sleep(0.2)
    try:
        assert health and health["ok"] is True
        with urllib.request.urlopen(url, timeout=5) as r:
            page = r.read().decode()
        assert "CallGuard" in page and 'id="meet-bar"' in page
        want, state, end = {"shield.state", "secret.state", "meet.state"}, {}, time.monotonic() + 5
        while not want <= set(state) and time.monotonic() < end:           # startup state reaches the dashboard
            with urllib.request.urlopen(url + "api/state", timeout=5) as r:  # (announced just after health is up)
                state = json.load(r)
            time.sleep(0.05)
        assert want <= set(state)
    finally:
        desktop.STOP.set()
        t.join(timeout=20)
    assert not t.is_alive() and out["rc"] == 0


def test_meet_url_opens_a_normal_tab(monkeypatch):
    """`callguard app --meet-url ...` opens the meeting as a normal tab in the user's browser (the extension connects
    it), never an automated window."""
    from app.source.connectors.meet import launcher
    for slot in ("VOICE", "ATTACKER", "SHIELD", "SECRET"):
        monkeypatch.setenv(f"CALLGUARD_DRIVERS_{slot}", "mock")
    seen = []
    monkeypatch.setattr(launcher, "open_tab", lambda url: seen.append(url) or "chrome")
    monkeypatch.setattr(launcher, "launch", lambda *a, **k: pytest.fail("automated browser launched"))
    port = _free_port()
    t = threading.Thread(target=lambda: desktop.main(["--no-window", "--port", str(port), "--meet-url",
                                                     "abc-defg-hij"]), daemon=True)
    t.start()
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end and not seen:
            time.sleep(0.1)
        assert seen == ["https://meet.google.com/abc-defg-hij"]
    finally:
        desktop.STOP.set()
        t.join(timeout=20)
