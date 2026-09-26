"""Server contract: page served, snapshot, bus events reach the WebSocket, controls reach the controls object."""
import fnmatch

import numpy as np
from fastapi.testclient import TestClient

from callguard.server.app import create_app
from callguard.types import Event


class FakeBus:
    def __init__(self):
        self.subs = []

    def subscribe(self, glob, fn):
        entry = (glob, fn)
        self.subs.append(entry)
        return lambda: self.subs.remove(entry)

    def publish(self, ev):
        for glob, fn in list(self.subs):
            if fnmatch.fnmatch(ev.topic, glob):
                fn(ev)


class FakeControls:
    def __init__(self):
        self.calls = []

    def set_shield(self, mode):
        self.calls.append(("shield", mode))

    def scenario(self, action, name):
        self.calls.append(("scenario", action, name))
        return "started"

    def secret(self, action):
        self.calls.append(("secret", action))
        return action


LOCAL = "http://127.0.0.1:8765"


def make():
    bus, ctl = FakeBus(), FakeControls()
    return bus, ctl, create_app(bus, None, ctl)


def test_index_and_static():
    _, _, app = make()
    with TestClient(app, base_url=LOCAL) as c:
        r = c.get("/")
        assert r.status_code == 200 and "CallGuard" in r.text and "/static/app.js" in r.text
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/static/style.css").status_code == 200


def test_state_tracks_latest_event_and_unsubscribes():
    bus, _, app = make()
    with TestClient(app, base_url=LOCAL) as c:
        assert c.get("/api/state").json() == {}
        bus.publish(Event("threat.update", {"score": np.float32(42.0), "level": "WATCH", "reasons": []}, t=1.0))
        s = c.get("/api/state").json()
        assert s["threat.update"]["data"]["score"] == 42.0 and s["threat.update"]["t"] == 1.0
    assert bus.subs == []  # lifespan shutdown unsubscribed


def test_state_provider_overrides_cache():
    app = create_app(FakeBus(), lambda: {"x": {"t": 0, "data": {"a": 1}}}, None)
    with TestClient(app, base_url=LOCAL) as c:
        assert c.get("/api/state").json() == {"x": {"t": 0, "data": {"a": 1}}}


def test_websocket_gets_snapshot_then_events():
    bus, _, app = make()
    with TestClient(app, base_url=LOCAL) as c:
        bus.publish(Event("shield.state", {"mode": "dsp"}, t=2.0))
        with c.websocket_connect("ws://127.0.0.1:8765/ws") as ws:
            snap = ws.receive_json()
            assert snap["topic"] == "snapshot" and snap["data"]["shield.state"]["data"] == {"mode": "dsp"}
            bus.publish(Event("voice.verdict", {"p_synthetic": 0.9, "margin": 2.0, "latency_ms": 80.0,
                                                "speech_fraction": 0.7}, t=3.0))
            msg = ws.receive_json()
            assert msg == {"topic": "voice.verdict", "t": 3.0,
                           "data": {"p_synthetic": 0.9, "margin": 2.0, "latency_ms": 80.0, "speech_fraction": 0.7}}


def test_controls():
    _, ctl, app = make()
    with TestClient(app, base_url=LOCAL) as c:
        assert c.post("/api/control/shield", json={"mode": "adversarial"}).json()["ok"]
        assert c.post("/api/control/shield", json={"mode": "loud"}).status_code == 422
        r = c.post("/api/control/scenario", json={"action": "start", "name": "ai_caller"})
        assert r.json() == {"ok": True, "result": "started"}
        c.post("/api/control/scenario", json={"action": "stop", "name": "ai_caller"})
        assert c.post("/api/control/secret", json={"action": "allow"}).json()["result"] == "allow"
        assert c.post("/api/control/secret", json={"action": "read"}).status_code == 422
    assert ctl.calls == [("shield", "adversarial"), ("scenario", "start", "ai_caller"), ("scenario", "stop", "ai_caller"),
                         ("secret", "allow")]


def test_controls_missing_is_503():
    with TestClient(create_app(FakeBus(), None, None), base_url=LOCAL) as c:
        assert c.post("/api/control/shield", json={"mode": "off"}).status_code == 503


def test_other_sites_are_refused():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    _, _, app = make()
    with TestClient(app, base_url=LOCAL) as c:
        assert c.get("/api/state", headers={"origin": "http://127.0.0.1:8765"}).status_code == 200
        assert c.get("/api/state", headers={"origin": "https://evil.example"}).status_code == 403
        assert c.post("/api/control/shield", json={"mode": "off"}, headers={"origin": "null"}).status_code == 403
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("ws://127.0.0.1:8765/ws", headers={"origin": "https://evil.example"}) as ws:
                ws.receive_text()
    with TestClient(app, base_url="http://rebind.example:8765") as c:     # DNS rebinding: foreign Host
        assert c.get("/").status_code == 403
