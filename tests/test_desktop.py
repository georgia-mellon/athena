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
    finally:
        desktop.STOP.set()
        t.join(timeout=20)
    assert not t.is_alive() and out["rc"] == 0
