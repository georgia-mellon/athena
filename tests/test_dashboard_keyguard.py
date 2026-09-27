"""The vendored Keyguard console mounted at /keyguard, the live arms-race data it replays, and the Ares vs Athena
panel on the CallGuard dashboard. Offline: no LLM keys, no models needed."""
import json
import sys

import pytest
from fastapi.testclient import TestClient

from app.source.types import Event
from dashboard.server import create_app
from tests.test_server import LOCAL, FakeBus

MATCH = {"line": "my password is hunter2", "secret": "HUNTER2", "decoy": "DRAGON7", "kind": "password",
         "backend": "gemini", "route": "backboard:google/gemini-2.5-flash", "source": "callguard",
         "rounds": [{"round": 0, "mode": "deceive", "span_read": "DRAGON7", "reads_true_secret": False, "stoi": 0.99}],
         "moves": [], "agents": {"ctc": {"before": "HUNTER2", "after": "DRAGON7"}}, "protected": True}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BACKBOARD_API_KEY", raising=False)


@pytest.mark.parametrize("path", ["/keyguard/", "/keyguard/static/arena.html", "/keyguard/api/arena",
                                  "/keyguard/api/runs", "/keyguard/api/pipelines"])
def test_keyguard_console_routes(path):
    with TestClient(create_app(FakeBus()), base_url=LOCAL) as c:
        assert c.get(path).status_code == 200


def test_arms_race_data_serves_latest_bus_match():
    bus = FakeBus()
    with TestClient(create_app(bus), base_url=LOCAL) as c:
        before = c.get("/keyguard/static/arms_race_data.js")  # no match yet: a saved/vendored sample
        assert before.status_code == 200 and before.text.startswith("window.ARMS_RACE")
        bus.publish(Event("keyguard.arms_race", MATCH))
        r = c.get("/keyguard/static/arms_race_data.js")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/javascript")
    body = r.text.strip()
    assert body.startswith("window.ARMS_RACE = ") and body.endswith(";")
    assert json.loads(body[len("window.ARMS_RACE = "):-1]) == MATCH


def test_keyguard_unavailable_is_503(monkeypatch):
    monkeypatch.setitem(sys.modules, "keyguard.server", None)  # import fails -> dashboard still up
    with TestClient(create_app(FakeBus()), base_url=LOCAL) as c:
        assert c.get("/").status_code == 200
        r = c.get("/keyguard/")
        assert r.status_code == 503 and "Keyguard console unavailable" in r.text


def test_index_has_ares_vs_athena_panel():
    with TestClient(create_app(FakeBus()), base_url=LOCAL) as c:
        html = c.get("/").text
    for needle in ('id="kg"', "Ares ⚔️ vs Athena 🦉", 'id="kg-moves"', 'id="kg-agents"', 'id="kg-rounds"',
                   'href="/keyguard/static/arena.html"', 'href="/keyguard/"'):
        assert needle in html


def test_ws_forwards_keyguard_move():
    bus = FakeBus()
    move = {"t_audio": 1.5, "agent": "⚔️ ARES", "title": "Opening read", "reasoning": "", "action": "read",
            "result": "HUNTER2", "tag": "exposed"}
    with TestClient(create_app(bus), base_url=LOCAL) as c:
        with c.websocket_connect("ws://127.0.0.1:8765/ws") as ws:
            assert ws.receive_json()["topic"] == "snapshot"
            bus.publish(Event("keyguard.move", move, t=2.0))
            assert ws.receive_json() == {"topic": "keyguard.move", "t": 2.0, "data": move}
