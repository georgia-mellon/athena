"""WP9 spoken-secret shield (plan 06): the delay line, redaction timing, the leak counter, arming, Allow, fail-open,
and the threat input S. Mock spotter only: no model, no devices."""
import numpy as np

from app.source import config
from app.secret_shield.redactor import RAMP, Redactor
from app.source.bus import EventBus
from app.keystroke_guard.mock import MockAttacker, MockShield
from app.secret_shield.mock import MockSpotter
from app.source.pipeline import Pipeline, Scenario
from app.source.threat import ThreatEngine
from app.source.types import BLOCK, SR, SecretSpotterDriver, VoiceScore


def _run(r: Redactor, x: np.ndarray, marks=()):
    marks, out = dict(marks), []                # {feed position: (start, end)}: mark just before that block
    for s in range(0, len(x), BLOCK):
        if s in marks:
            r.mark(*marks[s])
        out.append(r.process(x[s:s + BLOCK], s))
    return np.concatenate(out)


def test_delay_line_is_constant_and_exact():
    x = np.random.default_rng(0).standard_normal(4 * SR).astype(np.float32)
    y = _run(Redactor(delay=SR // 2, style="mute"), x)
    np.testing.assert_array_equal(y[SR // 2:], x[:-SR // 2])


def test_redaction_in_time_and_leak_counter():
    x = np.ones(4 * SR, np.float32)
    a, b = SR, SR + SR // 4
    r = Redactor(delay=SR // 2, style="mute")
    y = _run(r, x, {a: (a, b)})                 # marked the moment the audio arrives: 500 ms early enough
    assert r.leaked_samples == 0
    assert np.all(y[a + SR // 2 + RAMP: b + SR // 2 - RAMP] == 0)          # muted (after the delay)
    assert np.all(y[a + SR // 2 - 2 * RAMP - BLOCK: a + SR // 2 - RAMP] == 1)  # untouched before
    r2 = Redactor(delay=SR // 2, style="mute")
    late = a + SR // 2 + 3200                   # mark arrives 200 ms after the start already left
    _run(r2, x, {late: (a, b)})
    assert r2.leaked_samples == 3200


def test_mock_spotter_contract_and_lag():
    sp = MockSpotter([(1.0, 1.5, "digits", 2)], lag_s=0.3)
    assert isinstance(sp, SecretSpotterDriver)
    got = [s for i in range(0, 2 * SR, BLOCK) for s in sp.feed(np.zeros(BLOCK, np.float32), i)]
    assert len(got) == 1 and got[0].start == SR and got[0].category == "digits" and not hasattr(got[0], "text")


class FixedVoice:
    name, sample_rate = "fixed_voice", SR

    def __init__(self, p):
        self.p = p

    def score(self, audio):
        return VoiceScore(self.p, 0.0, 0.0, 0.0)


def _scenario(seconds=20.0):
    t = np.arange(int(seconds * SR)) / SR
    far = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)      # always "speech" for the VAD
    mic = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return Scenario("secret", far, mic, keys=[])


# victim reads 6 digits at 12.0-14.5 s: spans for digits 2..6 (the spotter never spans the first one)
DIGITS = [(12.0 + 0.5 * i, 12.5 + 0.5 * i, "digits", i + 1) for i in range(1, 6)]  # word + tail, like Vosk


def _pipe(p_synth, spans=DIGITS, requests=(), **secret):
    cfg = config.load(env={})
    for k, v in {"style": "mute", **secret}.items():
        setattr(cfg.secret, k, v)
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=FixedVoice(p_synth), attacker=MockAttacker(), shield=MockShield(),
                    spotter=MockSpotter(spans), spotter_in=MockSpotter(requests, mode="inbound"))
    seen = {"blocked": [], "levels": [], "state": []}
    bus.subscribe("secret.blocked", lambda e: seen["blocked"].append(e.data))
    bus.subscribe("secret.state", lambda e: seen["state"].append(e.data))
    bus.subscribe("threat.update", lambda e: seen["levels"].append((pipe.clock(), e.data["level"], e.data["reasons"])))
    return pipe, bus, seen


def _muted(pipe, t0, t1):
    """Fraction of the output (meeting side) muted over input-time [t0, t1)."""
    lag = pipe._shield_lag()
    y = pipe.mic.shielded.read_range(int(t0 * SR) + lag, int(t1 * SR) + lag)
    frames = y[: len(y) // BLOCK * BLOCK].reshape(-1, BLOCK)
    return float(np.mean(np.sqrt(np.mean(frames ** 2, axis=1)) < 1e-3))


def test_synthetic_caller_arms_and_blocks_digits():
    pipe, bus, seen = _pipe(0.95)
    pipe.replay(_scenario(), realtime=False)
    bus.flush()
    assert pipe.armed and pipe.armed_by == "voice"
    assert _muted(pipe, 12.52, 14.98) > 0.95         # digits 2-6 redacted
    assert _muted(pipe, 12.0, 12.46) == 0.0          # the first digit passes (by design)
    assert _muted(pipe, 5.0, 11.0) == 0.0            # nothing else touched
    assert pipe.redactor.leaked_samples == 0
    assert seen["blocked"] == [dict(category="digits", length=6, armed_by="voice", allowed=False, leaked_ms=0,
                                    t_audio=12.5)]
    lv = [lv for t, lv, _ in seen["levels"] if 16.2 <= t <= 18]  # published gap_s after the last digit
    assert "WARN" in lv or "CRITICAL" in lv
    bus.close()


def test_real_caller_stays_disarmed_and_untouched():
    pipe, bus, seen = _pipe(0.05)
    pipe.replay(_scenario(), realtime=False)
    bus.flush()
    assert not pipe.armed and seen["blocked"] == [] and _muted(pipe, 12.0, 15.0) == 0.0
    bus.close()


def test_request_trigger_makes_it_critical():
    pipe, bus, seen = _pipe(0.95, requests=[(10.0, 10.8, "request", 0)])
    pipe.replay(_scenario(), realtime=False)
    bus.flush()
    after = [(lv, r) for t, lv, r in seen["levels"] if 16.2 <= t <= 18]
    assert any(lv == "CRITICAL" and "caller asked for a code" in r[0] for lv, r in after)
    bus.close()


def test_allow_bypasses_redaction():
    pipe, bus, seen = _pipe(0.95)
    sc = _scenario()
    sc.actions = []
    orig = pipe.secret_step

    def step():                                     # press Allow at 11 s (audio time)
        if abs(pipe.clock() - 11.0) < 0.011:
            pipe.secret("allow")
        return orig()
    pipe.secret_step = step
    pipe.replay(sc, realtime=False)
    bus.flush()
    assert _muted(pipe, 12.5, 15.0) == 0.0 and seen["blocked"][0]["allowed"]
    bus.close()


def test_spotter_failure_fails_open():
    class Broken(MockSpotter):
        def feed(self, block, start):
            raise RuntimeError("model crashed")

    cfg = config.load(env={})
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=FixedVoice(0.95), attacker=MockAttacker(), shield=MockShield(),
                    spotter=Broken(), spotter_in=MockSpotter(mode="inbound"))
    pipe.replay(_scenario(), realtime=False)
    bus.flush()
    assert pipe.spotter.quarantined and "spotter failed" in pipe.secret_error
    assert _muted(pipe, 1.0, 19.0) == 0.0            # audio kept flowing
    bus.close()


def test_threat_s_rules():
    class Clock:
        t = 0.0
    clk = Clock()
    bus = EventBus()
    eng = ThreatEngine(bus, config.ThreatConfig(), now=lambda: clk.t)
    from app.source.types import Event
    for i in range(8):                               # a synthetic voice, but below WARN on its own
        clk.t += 0.5
        eng.on_event(Event("voice.verdict", {"p_synthetic": 0.9}))
    eng.V = 0.6
    eng.on_event(Event("secret.blocked", {"category": "digits", "length": 6}))
    u = eng.tick()
    assert u["S"] == 1 and u["score"] >= 50 and u["reasons"][0] == "6-digit code blocked from your voice"
    eng.on_event(Event("secret.request", {}))
    assert eng.tick()["level"] == "CRITICAL"
    bus.close()
