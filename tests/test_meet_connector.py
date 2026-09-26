"""Google Meet connector: the /meet WebSockets over TestClient (mock drivers, no devices), the secret shield in meet
mode, and (if Chrome/Edge is installed) the real bridge end to end in a headless browser against the test room."""
import os
import socket
import threading
import time
import wave

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
from app.source.pipeline import Pipeline
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
    assert any(s["connected"] and s["mic"] and s["latency_ms"] >= 500 + 42 for s in states)
    assert set(states[-1]) == {"connected", "url", "mic", "far", "browser", "latency_ms"}
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
    assert origin_ok("https://meet.google.com") and origin_ok("http://localhost:5173") and origin_ok("http://[::1]:1")
    assert origin_ok(None)                                   # not a browser: a local process
    for bad in ("https://evil.example", "https://meet.google.com.evil.example", "http://meet.google.com",
                "https://127.0.0.1.evil.example", "null"):
        assert not origin_ok(bad), bad
    pipe, bus, _ = _pipe()
    client, router = _client(pipe)
    for path in ("/meet/mic", "/meet/far"):
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect(path, headers={"origin": "https://evil.example"}) as ws:
                ws.receive_bytes()
        assert e.value.code == 1008
    assert router.meet_stats["rejected"] == 2 and pipe.mic.position == 0
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
    assert pipe.meet("leave") == "left"
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
    assert "CallGuard test room" in client.get("/meet/testroom").text
    js = client.get("/meet/static/bridge.js")
    assert js.status_code == 200 and "__callguardBridge" in js.text
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


def _tone_wav(path, seconds=10.0):
    t = np.arange(int(seconds * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * 330 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.tobytes())


@pytest.fixture
def served():
    """The real server stack (dashboard app + Meet router) on a free port, mock drivers, meet mode."""
    import uvicorn

    from dashboard.server import create_app
    pipe, bus, states = _pipe()
    app = create_app(bus, controls=pipe)
    router = make_router(pipe)
    app.include_router(router)
    port = _free_port()
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


HEADLESS = ["--headless=new", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            "--disable-features=WebRtcHideLocalIpsWithMdns", "--mute-audio"]
needs_browser = pytest.mark.skipif(launcher.find_browser() is None or os.environ.get("CALLGUARD_SKIP_BROWSER") == "1",
                                   reason="no Chrome/Edge (or CALLGUARD_SKIP_BROWSER=1)")


def _wait(cond, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.1)
    return False


@needs_browser
def test_browser_end_to_end_testroom(served, tmp_path):
    pipe, router, port, states, server = served
    wav = tmp_path / "tone.wav"
    _tone_wav(wav)
    s = launcher.launch(f"http://127.0.0.1:{port}/meet/testroom?auto=1", port, profile_dir=tmp_path / "profile",
                        extra_args=HEADLESS + [f"--use-file-for-fake-audio-capture={wav}"])
    try:
        st = router.meet_stats
        assert _wait(lambda: st["mic_out"] > 100 and st["far_in"] > 50), f"no audio through the bridge: {st}"
        assert _wait(lambda: (s.evaluate("__callguardBridge.stats") or {}).get("mic") == "processed")
        a, t0 = dict(st), time.monotonic()
        time.sleep(3.0)
        b, dt = dict(st), time.monotonic() - t0
        bridge = s.evaluate("__callguardBridge")
        stats = bridge["stats"]
        mic_fps, far_fps = (b["mic_in"] - a["mic_in"]) / dt, (b["far_in"] - a["far_in"]) / dt
        print(f"\n[e2e] source={bridge['source']} mic {mic_fps:.1f} blocks/s in, "
              f"{(b['mic_out'] - a['mic_out']) / dt:.1f} out; far {far_fps:.1f} blocks/s; "
              f"bridge round trip {stats['rttMs']:.1f} ms; sent {stats['micSent']} back {stats['micBack']}")
        assert bridge["source"] == "cdp"                     # the launcher's injection won the race
        assert 40 < mic_fps < 60 and 40 < far_fps < 60      # 50 blocks/s = realtime 16 kHz
        assert b["mic_out"] - b["mic_in"] in (0, -1)         # every block answered
        assert stats["rttMs"] < 200
        # the mic audio really is the fake capture's tone (after the delay line), and the room hears it
        # (Chrome's default noise suppression / AGC attenuate a steady tone, so check its pitch, not its level)
        tone = pipe.mic.raw.read_last(SR)
        peak_hz = np.argmax(np.abs(np.fft.rfft(tone))) * SR / len(tone)
        print(f"[e2e] mic at the server: rms {np.sqrt(np.mean(tone ** 2)):.3f}, peak {peak_hz:.0f} Hz")
        assert np.sqrt(np.mean(tone ** 2)) > 0.005 and abs(peak_hz - 330) < 5
        level = s.evaluate("__room.meters.map((m) => m.rms)")
        assert level and all(v > 0.01 for v in level), level
        assert any(x["mic"] and x["far"] and x["connected"] for x in states)
        # FAIL OPEN: CallGuard goes away mid-call -> the bridge switches to the raw mic; the room still hears you
        server.should_exit = True
        assert _wait(lambda: s.evaluate("__callguardBridge.stats.mic") == "raw", 10)
        time.sleep(0.5)
        level = s.evaluate("__room.meters.map((m) => m.rms)")
        print(f"[e2e] server stopped: bridge mic={s.evaluate('__callguardBridge.stats.mic')}, room levels {level}")
        assert all(v > 0.01 for v in level), level
    finally:
        s.close()
    assert not s.alive


@needs_browser
@pytest.mark.skipif(os.environ.get("CALLGUARD_MEET_ONLINE") != "1", reason="needs network: CALLGUARD_MEET_ONLINE=1")
def test_browser_bridge_on_real_meet_page(served, tmp_path):
    """Real meet.google.com: the bridge is installed before Meet's scripts and its mic path reaches us from Meet's
    origin (CSP bypass, mixed content, Local Network Access). Meet's home redirects signed-out users to a marketing
    site, so this opens a meeting-code URL (the guest pre-join page, still meet.google.com); joining needs a real
    meeting, so getUserMedia is called from the page directly."""
    pipe, router, port, _, _ = served
    wav = tmp_path / "tone.wav"
    _tone_wav(wav)
    s = launcher.launch("abc-defg-hij", port, profile_dir=tmp_path / "profile",
                        extra_args=HEADLESS + [f"--use-file-for-fake-audio-capture={wav}"])
    try:
        assert _wait(lambda: s.evaluate("location.hostname + ':' + document.readyState") == "meet.google.com:complete",
                     30), s.evaluate("location.href")
        assert s.evaluate("!!window.__callguardBridge && __callguardBridge.source") == "cdp"
        s.evaluate("navigator.mediaDevices.getUserMedia({audio: true}).then((m) => { window.__m = m; return 1; })")
        assert _wait(lambda: router.meet_stats["mic_out"] > 50), router.meet_stats
        print(f"\n[meet] {s.evaluate('location.href')}: {s.evaluate('__callguardBridge.stats')}")
    finally:
        s.close()
