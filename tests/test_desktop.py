"""Desktop shell, headless: engine + server come up in-process, the dashboard is served, shutdown is clean."""
import json
import socket
import threading
import time
import urllib.request

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
        with urllib.request.urlopen(url + "api/state", timeout=5) as r:     # startup state reached the dashboard
            state = json.load(r)
        assert {"shield.state", "secret.state", "meet.state"} <= set(state)
    finally:
        desktop.STOP.set()
        t.join(timeout=20)
    assert not t.is_alive() and out["rc"] == 0


def test_meet_bridge_dials_the_app_port(monkeypatch):
    """`callguard app --port N --meet-url ...`: the injected bridge must point at N, or Meet's mic runs unprotected."""
    from app.source.connectors.meet import launcher
    for slot in ("VOICE", "ATTACKER", "SHIELD", "SECRET"):
        monkeypatch.setenv(f"CALLGUARD_DRIVERS_{slot}", "mock")
    seen = []

    class FakeSession:
        alive, url = True, None

        def close(self):
            self.alive = False
    monkeypatch.setattr(launcher, "launch", lambda url, port, *a, **k: seen.append(port) or FakeSession())
    port = _free_port()
    t = threading.Thread(target=lambda: desktop.main(["--no-window", "--port", str(port), "--meet-url",
                                                     "abc-defg-hij"]), daemon=True)
    t.start()
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end and not seen:
            time.sleep(0.1)
        assert seen == [port]
    finally:
        desktop.STOP.set()
        t.join(timeout=20)
