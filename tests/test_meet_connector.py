"""Google Meet connector: the /meet WebSockets over TestClient (mock drivers, no devices), the secret shield in meet
mode, and (if Chrome/Edge is installed) the real bridge end to end in a headless browser against the test room."""
import json
from types import SimpleNamespace
import os
import socket
import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.hearsay.mock import MockVoice
from app.keystroke_guard.mock import MockAttacker, MockShield
from app.secret_shield.mock import MockSpotter
from app.source import config
from app.source.audio.keys import ScriptedKeyClock
from app.source.bus import EventBus
from app.source.connectors.meet import launcher
from app.source.connectors.meet.router import make_router, origin_ok
from app.source.pipeline import ArrivalAnchor, Pipeline
from app.source.types import BLOCK, SR

MEET = {"origin": "https://meet.google.com"}
DELAY = SR // 2 // BLOCK                            # the secret delay line, in blocks (500 ms)


def _pipe(spans=(), **secret):
    cfg = config.load(env={})
    cfg.secret.style = "mute"
    for k, v in secret.items():
        setattr(cfg.secret, k, v)
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=MockVoice(), attacker=MockAttacker(), shield=MockShield(),
                    spotter=MockSpotter(spans, lag_s=0.05), spotter_in=MockSpotter(mode="inbound"))
    states = []
    bus.subscribe("meet.state", lambda e: states.append(e.data))
    pipe.start_meet(keyclock=ScriptedKeyClock([]))
    pipe.meet_port = 8765          # the local-origin tests assume this port; pin it so a dev's athena.toml can't change it
    return pipe, bus, states


def _client(pipe):
    app = FastAPI()
    router = make_router(pipe)
    app.include_router(router)
    return TestClient(app), router


def _blocks(n, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, (n, BLOCK)).astype("<f4")


def test_mic_round_trip_same_size_in_order():
    pipe, bus, states = _pipe()
    client, router = _client(pipe)
    x = _blocks(60)
    with client.websocket_connect("/meet/mic", headers=MEET) as ws:
        out = []
        for b in x:
            ws.send_bytes(b.tobytes())
            out.append(np.frombuffer(ws.receive_bytes(), "<f4"))
        ws.send_bytes(b"\x00\x00\x80\x3f" * 7)              # wrong size: echoed as is, never dropped
        assert ws.receive_bytes() == b"\x00\x00\x80\x3f" * 7
        ws.send_text('{"rtt_ms": 42.5}')
        time.sleep(0.4)                                      # the watcher publishes within 0.25 s
    assert all(len(o) == BLOCK for o in out)
    # the constant secret delay line: block k out == block k - 25 in (the mock shield passes through)
    np.testing.assert_array_equal(np.concatenate(out[DELAY:]), x[:-DELAY].reshape(-1))
    assert not np.any(np.concatenate(out[:DELAY]))
    assert router.meet_stats["mic_in"] == 61 and router.meet_stats["mic_out"] == 61
    assert pipe.mic.position == 60 * BLOCK
    pipe.stop()
    bus.flush()
    assert any(s["connected"] and s["mic"] and s["latency_ms"] >= 500 + 42 and s["owner"] == MEET["origin"]
               for s in states)
    assert set(states[-1]) == {"page", "in_call", "connected", "owner", "url", "mic", "far", "browser", "latency_ms"}
    assert not states[-1]["connected"] and states[-1]["owner"] is None
    bus.close()


def test_far_frames_reach_the_far_ring():
    pipe, bus, states = _pipe()
    client, router = _client(pipe)
    with client.websocket_connect("/meet/far", headers={"origin": "http://127.0.0.1:8765"}) as ws:
        for b in _blocks(25):
            ws.send_bytes(b.tobytes())
        ws.send_bytes(b"\x00\x00\x00\x00" * BLOCK)
        deadline = time.monotonic() + 2
        while pipe.far.total < 26 * BLOCK and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.4)
    assert pipe.far.total == 26 * BLOCK and router.meet_stats["far_in"] == 26
    pipe.stop()
    bus.flush()
    assert any(s["far"] for s in states)
    bus.close()


def test_origin_checks():
    assert origin_ok("https://meet.google.com", 8765) and origin_ok("http://localhost:8765", 8765)
    assert origin_ok("http://[::1]:1", 1) and origin_ok("http://localhost", 80) and origin_ok("https://127.0.0.1", 443)
    assert origin_ok(None, 8765)                             # not a browser: a local process
    for bad in ("https://evil.example", "https://meet.google.com.evil.example", "http://meet.google.com",
                "https://127.0.0.1.evil.example", "null", "http://localhost:5173", "http://127.0.0.1",
                "http://127.0.0.1:99999", "file://"):
        assert not origin_ok(bad, 8765), bad
    pipe, bus, _ = _pipe()
    client, router = _client(pipe)
    # another website, and another local server (a dev server, a local page someone else runs)
    for origin in ("https://evil.example", "http://127.0.0.1:5173"):
        for path in ("/meet/mic", "/meet/far"):
            with pytest.raises(WebSocketDisconnect) as e:
                with client.websocket_connect(path, headers={"origin": origin}) as ws:
                    ws.receive_bytes()
            assert e.value.code == 1008
    assert router.meet_stats["rejected"] == 4 and pipe.mic.position == 0
    pipe.meet_port = 9000                                   # the port follows the server Athena runs on
    with client.websocket_connect("/meet/far", headers={"origin": "http://localhost:9000"}) as ws:
        ws.send_bytes(_blocks(1).tobytes())
    app = FastAPI()
    app.include_router(make_router(pipe, port=9100))        # or an explicit one
    with TestClient(app).websocket_connect("/meet/far", headers={"origin": "http://127.0.0.1:9100"}) as ws:
        ws.send_bytes(_blocks(1).tobytes())
    pipe.stop()
    bus.close()


def test_second_tab_waits_its_turn():
    pipe, bus, _ = _pipe()
    client, _ = _client(pipe)
    with client.websocket_connect("/meet/mic", headers=MEET) as first:
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect("/meet/mic", headers=MEET) as second:
                second.receive_bytes()
        assert e.value.code == 1013
        first.send_bytes(_blocks(1).tobytes())
        assert len(first.receive_bytes()) == BLOCK * 4
    with client.websocket_connect("/meet/mic", headers=MEET) as again:      # free once the first one left
        again.send_bytes(_blocks(1).tobytes())
        assert len(again.receive_bytes()) == BLOCK * 4
    pipe.stop()
    bus.close()


def test_meet_takes_over_from_a_local_page():
    """A real meeting beats a stray test-room tab; the reverse (and Meet vs Meet) waits its turn."""
    pipe, bus, states = _pipe()
    client, _ = _client(pipe)
    local = {"origin": "http://127.0.0.1:8765"}
    with client.websocket_connect("/meet/mic", headers=local) as room:
        room.send_bytes(_blocks(1).tobytes())
        room.receive_bytes()
        assert pipe.meet_owners["mic"] == local["origin"]
        with client.websocket_connect("/meet/mic", headers=MEET) as meet:
            with pytest.raises(WebSocketDisconnect) as e:
                room.receive_bytes()
            assert e.value.code == 1013
            assert pipe.meet_owners["mic"] == MEET["origin"]
            for other in (local, MEET):
                with pytest.raises(WebSocketDisconnect) as e:
                    with client.websocket_connect("/meet/mic", headers=other) as late:
                        late.receive_bytes()
                assert e.value.code == 1013
            meet.send_bytes(_blocks(1).tobytes())
            assert len(meet.receive_bytes()) == BLOCK * 4
            time.sleep(0.3)
    assert pipe.meet_owners == {"mic": None, "far": None}   # the displaced page's close didn't clear Meet's claim
    pipe.stop()
    bus.flush()
    owners = [s["owner"] for s in states]
    assert local["origin"] in owners and MEET["origin"] in owners and owners[-1] is None
    bus.close()


def test_new_mic_socket_flushes_the_delay_line():
    """Audio still in the secret delay line when a socket drops must not come out ~0.5 s into the next one."""
    pipe, bus, _ = _pipe()
    client, _ = _client(pipe)
    with client.websocket_connect("/meet/mic", headers=MEET) as ws:
        for b in np.full((DELAY + 10, BLOCK), 0.25, "<f4"):
            ws.send_bytes(b.tobytes())
            ws.receive_bytes()
    out = []
    with client.websocket_connect("/meet/mic", headers=MEET) as ws:
        for b in np.zeros((DELAY + 10, BLOCK), "<f4"):
            ws.send_bytes(b.tobytes())
            out.append(np.frombuffer(ws.receive_bytes(), "<f4"))
    assert not np.any(np.concatenate(out))                  # nothing from before the reconnect
    assert pipe.mic.position == 2 * (DELAY + 10) * BLOCK   # the stream's clock runs on
    pipe.stop()
    bus.close()


def test_owner_survives_start_meet_and_never_goes_negative():
    """A socket opened before start_meet (the app starts meet mode while a page is already connected) and closed
    after it: meet.state follows the router's owner, no counter to drift."""
    cfg = config.load(env={})
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=MockVoice(), attacker=MockAttacker(), shield=MockShield(),
                    spotter=MockSpotter(), spotter_in=MockSpotter(mode="inbound"))
    states = []
    bus.subscribe("meet.state", lambda e: states.append(e.data))
    client, _ = _client(pipe)
    with client.websocket_connect("/meet/far", headers=MEET):
        time.sleep(0.1)
        pipe.start_meet(keyclock=ScriptedKeyClock([]))
        pipe.start_meet(keyclock=ScriptedKeyClock([]))      # idempotent
        time.sleep(0.4)
    time.sleep(0.4)
    for _ in range(2):
        with client.websocket_connect("/meet/mic", headers=MEET):
            pass
    time.sleep(0.4)
    pipe.stop()
    bus.flush()
    assert states[0]["connected"] and states[0]["owner"] == MEET["origin"]
    assert not states[-1]["connected"] and states[-1]["owner"] is None
    bus.close()


def test_arrival_anchor_rejects_jitter():
    """Blocks arrive late by a constant 30 ms plus bursty jitter (0-150 ms, some bursts): the anchor recovers the
    constant part within a block's worth, and follows a slow clock drift."""
    rng = np.random.default_rng(1)
    anchor, t0 = ArrivalAnchor(window_s=3.0), 100.0
    errs = []
    for k in range(1, 1000):                                # 20 s of blocks
        s = k * BLOCK
        drift = 0.001 * s / SR                              # 1 ms per second
        jitter = rng.exponential(0.02) + (0.15 if k % 50 < 5 else 0.0)
        est = anchor(s, t0 + s / SR + 0.03 + drift + jitter)
        if k > 150:
            errs.append(est - (t0 + s / SR + 0.03 + drift))
    errs = np.array(errs)
    assert np.abs(errs).max() < 0.005, (errs.min(), errs.max())   # the raw arrival time is off by up to ~250 ms
    anchor.reset()
    assert anchor(BLOCK, 5.0) == pytest.approx(5.0)


def test_armed_secret_is_redacted_in_the_meet_stream():
    """Manual Arm (the dashboard button) in meet mode; the spotter marks 1.0-1.5 s: those samples come back muted
    after the delay line, everything else comes back untouched."""
    pipe, bus, _ = _pipe(spans=[(1.0, 1.5, "digits", 2)])
    pipe.secret("arm")
    client, _ = _client(pipe)
    x = np.full((150, BLOCK), 0.25, "<f4")                  # 3 s of a constant "voice"
    out = []
    with client.websocket_connect("/meet/mic", headers=MEET) as ws:
        for b in x:
            ws.send_bytes(b.tobytes())
            out.append(np.frombuffer(ws.receive_bytes(), "<f4"))
            time.sleep(0.004)                                # ~5x realtime: the spotter worker keeps up
    y = np.concatenate(out)
    lag = pipe._shield_lag()
    rms = lambda a, b: float(np.sqrt(np.mean(y[int(a * SR) + lag: int(b * SR) + lag] ** 2)))  # noqa: E731
    assert pipe.armed and pipe.armed_by == "manual"
    assert rms(1.02, 1.48) < 1e-3                            # the code: muted
    assert rms(0.2, 0.95) == pytest.approx(0.25) and rms(1.55, 2.4) == pytest.approx(0.25)
    assert pipe.redactor.leaked_samples == 0
    pipe.stop()
    bus.close()


def test_status_channel_drives_the_meeting_state_and_leave():
    """/meet/status: the extension's page reports whether a call is live (the dashboard's pill), and the dashboard's
    Leave / Join reach that page as commands."""
    pipe, bus, states = _pipe()
    calls = []
    bus.subscribe("meet.call", lambda e: calls.append(e.data["event"]))
    client, _ = _client(pipe)
    with client.websocket_connect("/meet/status", headers=MEET) as ws:
        ws.send_text(json.dumps({"site": "meet", "in_call": False}))
        time.sleep(0.2)
        bus.flush()
        assert states[-1]["page"] == "meet" and states[-1]["in_call"] is False
        ws.send_text(json.dumps({"site": "meet", "in_call": True}))
        time.sleep(0.2)
        bus.flush()
        assert states[-1]["in_call"] is True
        assert pipe.meet("leave") == "leaving the call"
        assert json.loads(ws.receive_text()) == {"cmd": "leave"}
        assert pipe.meet("join", "abc-defg-hij") == "opened https://meet.google.com/abc-defg-hij in your Meet tab"
        assert json.loads(ws.receive_text()) == {"cmd": "open", "url": "https://meet.google.com/abc-defg-hij"}
    time.sleep(0.2)
    bus.flush()
    assert states[-1]["page"] is None and states[-1]["in_call"] is False                # the tab closed
    assert calls == ["meet_open", "joined", "left"]              # the event log: tab opened, joined, left (tab closed)
    with pytest.raises(WebSocketDisconnect):                                         # another site: rejected
        with client.websocket_connect("/meet/status", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()
    pipe.stop()
    bus.close()


def test_speech_gate_and_levels():
    pipe, bus, _ = _pipe()
    seen = {"level": [], "system": []}
    bus.subscribe("audio.level", lambda e: seen["level"].append(e.data))
    bus.subscribe("system.state", lambda e: seen["system"].append(e.data))
    with pytest.raises(ValueError):
        pipe.set_speech_db(-5)
    assert pipe.set_speech_db(-60) == -60.0
    for b in _blocks(50):
        pipe.meet_far(b)
    time.sleep(0.5)
    pipe.warm_up()
    bus.flush()
    assert seen["system"][-1]["speech_db"] == -60.0 and seen["system"][-1]["ready"] is True
    assert any(x["far_db"] is not None and -20 < x["far_db"] < 0 for x in seen["level"])   # uniform +-0.5: ~-10 dBFS

    # judge your own mic instead of the caller (solo tests): mic audio now reaches the voice model
    windows = []
    bus.subscribe("voice.window", lambda e: windows.append(e.data))
    with pytest.raises(ValueError):
        pipe.set_voice_source("speaker")
    assert pipe.set_voice_source("mic") == "mic"
    for b in _blocks(400, seed=3):                               # 8 s of loud audio from the page's mic
        pipe.meet_mic(b)
    time.sleep(1.0)
    bus.flush()
    assert windows and seen["system"][-1]["voice_source"] == "mic"
    pipe.set_voice_source("far")
    pipe.stop()
    bus.close()


def test_meet_controls_and_pass_through_outside_meet_mode():
    cfg = config.load(env={})
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=MockVoice(), attacker=MockAttacker(), shield=MockShield(),
                    spotter=MockSpotter(), spotter_in=MockSpotter(mode="inbound"))
    b = _blocks(1)[0]
    np.testing.assert_array_equal(pipe.meet_mic(b), b)      # idle: straight through, nothing recorded
    pipe.meet_far(b)
    assert pipe.mic.position == 0 and pipe.far.total == 0
    with pytest.raises(ValueError, match="meet mode"):
        pipe.meet("join")
    pipe.start_meet(keyclock=ScriptedKeyClock([]))
    for bad in (("dance", None), ("join", "https://evil.example/abc"), ("join", "http://meet.google.com/x")):
        with pytest.raises(ValueError):
            pipe.meet(*bad)
    with pytest.raises(ValueError, match="no Google Meet tab"):             # nothing to leave: say so
        pipe.meet("leave")
    opened = []
    launcher.open_tab, real = (lambda url: opened.append(url) or "chrome"), launcher.open_tab
    try:
        assert pipe.meet("join", "abc-defg-hij") == "opened https://meet.google.com/abc-defg-hij in chrome"
    finally:
        launcher.open_tab = real
    assert opened == ["https://meet.google.com/abc-defg-hij"]            # a normal tab, no automated window
    assert pipe.meet_session is None
    with pytest.raises(ValueError, match="replay mode"):
        pipe.scenario("start")
    pipe.stop()
    assert pipe.mode == "idle"
    bus.close()
    assert launcher.meet_url(None) == launcher.MEET_HOME
    assert launcher.meet_url("ABC-defg-hij") == "https://meet.google.com/abc-defg-hij"
    assert launcher.meet_url("meet.google.com/abc-defg-hij") == "https://meet.google.com/abc-defg-hij"
    assert launcher.meet_url("http://127.0.0.1:8765/meet/testroom") == "http://127.0.0.1:8765/meet/testroom"


def test_testroom_and_static_served():
    pipe, bus, _ = _pipe()
    client, _ = _client(pipe)
    assert "Athena test room" in client.get("/meet/testroom").text
    js = client.get("/meet/static/bridge.js")
    assert js.status_code == 200 and "__athenaBridge" in js.text
    assert client.get("/meet/static/launcher.py").status_code == 404
    assert client.get("/meet/audio/../../../pyproject.toml").status_code == 404
    assert isinstance(client.get("/meet/audio").json(), list)
    pipe.stop()
    bus.close()


# --- end to end: the real bridge in headless Chrome/Edge -------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served():
    """The real server stack (dashboard app + Meet router) on a free port, mock drivers, meet mode."""
    import uvicorn

    from dashboard.server import create_app
    pipe, bus, states = _pipe()
    app = create_app(bus, controls=pipe)
    port = _free_port()
    router = make_router(pipe, port=port)
    app.include_router(router)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    yield pipe, router, port, states, server
    server.should_exit = True
    th.join(5)
    pipe.stop()
    bus.close()


# The fake device's built-in tone is the mic: --use-file-for-fake-audio-capture delivers silence on Chrome 153 (macOS)
HEADLESS = ["--headless=new", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            "--disable-features=WebRtcHideLocalIpsWithMdns", "--mute-audio"]
needs_browser = pytest.mark.skipif(launcher.find_browser() is None or os.environ.get("ATHENA_SKIP_BROWSER") == "1",
                                   reason="no Chrome/Edge (or ATHENA_SKIP_BROWSER=1)")


def _wait(cond, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.1)
    return False


def _room_hears_both(s, timeout=3.0) -> bool:
    """Both room meters above the floor in the same 100 ms tick (the fake device beeps: half the ticks are gaps)."""
    return _wait(lambda: all(v > 0.01 for v in s.evaluate("__room.meters.map((m) => m.rms)")), timeout)


@needs_browser
def test_browser_end_to_end_testroom(served, tmp_path):
    pipe, router, port, states, server = served
    s = launcher.launch(f"http://127.0.0.1:{port}/meet/testroom?auto=1", port, profile_dir=tmp_path / "profile",
                        extra_args=HEADLESS)
    try:
        st = router.meet_stats
        assert _wait(lambda: st["mic_out"] > 100 and st["far_in"] > 50), f"no audio through the bridge: {st}"
        assert _wait(lambda: (s.evaluate("__athenaBridge.stats") or {}).get("mic") == "processed")
        a, t0 = dict(st), time.monotonic()
        time.sleep(3.0)
        b, dt = dict(st), time.monotonic() - t0
        bridge = s.evaluate("__athenaBridge")
        stats = bridge["stats"]
        mic_fps, far_fps = (b["mic_in"] - a["mic_in"]) / dt, (b["far_in"] - a["far_in"]) / dt
        print(f"\n[e2e] source={bridge['source']} mic {mic_fps:.1f} blocks/s in, "
              f"{(b['mic_out'] - a['mic_out']) / dt:.1f} out; far {far_fps:.1f} blocks/s; "
              f"bridge round trip {stats['rttMs']:.1f} ms; sent {stats['micSent']} back {stats['micBack']}")
        assert bridge["source"] == "cdp"                     # the launcher's injection won the race
        assert 40 < mic_fps < 60 and 40 < far_fps < 60      # 50 blocks/s = realtime 16 kHz
        assert b["mic_out"] - b["mic_in"] in (0, -1)         # every block answered
        assert stats["rttMs"] < 200
        # the mic audio really is the fake device's tone (after the delay line), and the room hears it
        tone = pipe.mic.raw.read_last(SR)
        peak_hz = np.argmax(np.abs(np.fft.rfft(tone))) * SR / len(tone)
        print(f"[e2e] mic at the server: rms {np.sqrt(np.mean(tone ** 2)):.3f}, peak {peak_hz:.0f} Hz")
        assert np.sqrt(np.mean(tone ** 2)) > 0.005 and 100 < peak_hz < 2000
        assert _room_hears_both(s), s.evaluate("__room.meters.map((m) => m.rms)")
        assert any(x["mic"] and x["far"] and x["connected"] for x in states)
        # FAIL OPEN: Athena goes away mid-call -> the bridge switches to the raw mic; the room still hears you
        server.should_exit = True
        assert _wait(lambda: s.evaluate("__athenaBridge.stats.mic") == "raw", 10)
        time.sleep(0.5)
        heard = _room_hears_both(s)
        print(f"[e2e] server stopped: bridge mic={s.evaluate('__athenaBridge.stats.mic')}, "
              f"room levels {s.evaluate('__room.meters.map((m) => m.rms)')}")
        assert heard
    finally:
        s.close()
    assert not s.alive


@needs_browser
@pytest.mark.skipif(os.environ.get("ATHENA_MEET_ONLINE") != "1", reason="needs network: ATHENA_MEET_ONLINE=1")
def test_browser_bridge_on_real_meet_page(served, tmp_path):
    """Real meet.google.com: the bridge is installed before Meet's scripts and its mic path reaches us from Meet's
    origin (CSP bypass, mixed content, Local Network Access). Meet's home redirects signed-out users to a marketing
    site, so this opens a meeting-code URL (the guest pre-join page, still meet.google.com); joining needs a real
    meeting, so getUserMedia is called from the page directly."""
    pipe, router, port, _, _ = served
    s = launcher.launch("abc-defg-hij", port, profile_dir=tmp_path / "profile", extra_args=HEADLESS)
    try:
        assert _wait(lambda: s.evaluate("location.hostname + ':' + document.readyState") == "meet.google.com:complete",
                     30), s.evaluate("location.href")
        assert s.evaluate("!!window.__athenaBridge && __athenaBridge.source") == "cdp"
        s.evaluate("navigator.mediaDevices.getUserMedia({audio: true}).then((m) => { window.__m = m; return 1; })")
        assert _wait(lambda: router.meet_stats["mic_out"] > 50), router.meet_stats
        print(f"\n[meet] {s.evaluate('location.href')} (local network access: {s.lna}): "
              f"{s.evaluate('__athenaBridge.stats')}")
    finally:
        s.close()


def test_caller_audio_source_system_or_tab():
    """Meet mode takes the caller from one source at a time: system audio (speaker loopback) or the Meet tab."""
    pipe, bus, _ = _pipe()                                   # tests: "tab" (conftest)
    b = _blocks(1)[0]
    pipe.meet_far(b)
    assert pipe.far.total == BLOCK
    pipe._far_in(b, "system")                                # not the selected source: ignored
    assert pipe.far.total == BLOCK
    started = []
    pipe._start_system_audio = lambda: started.append(1) or setattr(pipe, "_loopback", SimpleNamespace(last_error=None))
    assert pipe.set_far_source("system") == "system" and started == [1]
    pipe._far_in(b, "system")
    pipe.meet_far(b)                                         # the tab is ignored now
    assert pipe.far.total == 2 * BLOCK
    with pytest.raises(ValueError):
        pipe.set_far_source("zoom")
    pipe._loopback = None
    pipe.stop()
    bus.close()
