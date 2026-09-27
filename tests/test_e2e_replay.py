"""End-to-end replay through the real Pipeline: the ai_caller story must go SAFE -> CRITICAL -> back, and the
attacker must read the raw mic better than the shielded one.

Mock run: a synthetic scenario (a 300 Hz tone = the real colleague, 600 Hz = the AI agent, clicks = keys), a voice
stand-in that tells the two apart by pitch, MockAttacker/MockShield. No models, no devices, a few seconds.
Real run: the built demo scenario with the real drivers (skipped unless the upstream repos and the demo audio exist).
"""
import os

import numpy as np
import pytest

from app.source import config
from app.source.bus import EventBus
from app.keystroke_guard.mock import MockAttacker, MockShield
from app.source.pipeline import Pipeline, Scenario, load_scenario
from app.source.types import SR, VoiceScore

CODE = "RESET4821"


def _tone(f, seconds, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


class PitchVoice:
    """600 Hz 'agent' -> synthetic, 300 Hz 'colleague' -> real."""
    name, sample_rate = "pitch_voice", SR

    def score(self, audio):
        f = np.argmax(np.abs(np.fft.rfft(audio))) * SR / len(audio)
        p = 0.95 if f > 450 else 0.05
        return VoiceScore(p, p - 0.5, 0.0, 0.0)


def mock_scenario() -> Scenario:
    # 0-12 colleague, 12-40 agent, 40-60 colleague; code typed at 22-26.5 (shield off) and 30-34.5 (shield on)
    far = np.concatenate([_tone(300, 12), _tone(600, 28), _tone(300, 20)])
    mic = (0.003 * np.random.default_rng(0).standard_normal(len(far))).astype(np.float32)
    keys = []
    for t0 in (22.0, 30.0):
        for i, ch in enumerate(CODE):
            s = int((t0 + 0.5 * i) * SR)
            mic[s:s + 200] += 0.3
            keys.append((s, ch))
    return Scenario("mock_ai_caller", far, mic, keys, actions=[(0.0, {"shield": "off"}), (28.0, {"shield": "dsp"})])


def run(pipe: Pipeline, sc: Scenario, bus: EventBus):
    seen = {"levels": [], "readouts": [], "verdicts": []}
    bus.subscribe("threat.update", lambda e: seen["levels"].append((pipe.clock(), e.data["level"], e.data["score"])))
    bus.subscribe("keys.readout", lambda e: seen["readouts"].append(e.data))
    bus.subscribe("voice.verdict", lambda e: seen["verdicts"].append((e.data["t_audio"], e.data["p_synthetic"])))
    pipe.replay(sc, realtime=False)
    bus.flush()
    return seen


def _mock_cfg():
    cfg = config.load(env={})
    cfg.drivers.secret = "mock"
    return cfg


def test_mock_replay_story():
    cfg = _mock_cfg()
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=PitchVoice(), attacker=MockAttacker(accuracy=0.95), shield=MockShield())
    seen = run(pipe, mock_scenario(), bus)

    levels = seen["levels"]
    at = lambda t: [lv for tt, lv, _ in levels if tt <= t][-1]  # noqa: E731
    assert at(11) == "SAFE"
    assert at(21) in ("WATCH", "WARN")                   # synthetic voice alone
    assert "CRITICAL" in [lv for tt, lv, _ in levels if 22 <= tt <= 28]  # agent + readable typing
    assert at(60) in ("SAFE", "WATCH")                   # agent gone, colleague back

    ro = seen["readouts"]
    assert len(ro) == 2 * len(CODE)
    before, after = ro[len(CODE) - 1], ro[-1]
    assert before["acc_raw"] > 0.8                        # shield off: both streams read the code
    last = after["shielded"][-len(CODE):]
    assert np.mean([r["exact"] for r in last]) < 0.3                 # shield on: noise
    assert np.mean([r["exact"] for r in after["raw"][-len(CODE):]]) > 0.8
    pipe.stop()
    bus.close()


def test_shield_failure_passes_audio_and_alarms():
    class Broken(MockShield):
        def process(self, block, key_events):
            raise RuntimeError("boom")

    cfg = _mock_cfg()
    bus = EventBus()
    errors, states = [], []
    bus.subscribe("driver.error", lambda e: errors.append(e.data))
    bus.subscribe("shield.state", lambda e: states.append(e.data))
    pipe = Pipeline(cfg, bus, voice=PitchVoice(), attacker=MockAttacker(), shield=Broken())
    sc = mock_scenario()
    pipe.replay(sc, realtime=False)
    bus.flush()
    assert pipe.shield.quarantined and errors and errors[-1]["quarantined"]
    assert states[-1]["failed"]
    n, lag = pipe.mic.raw.total, pipe._shield_lag()          # quarantined shield: no lookahead; secret delay only
    np.testing.assert_array_equal(pipe.mic.shielded.read_range(n - SR, n), pipe.mic.raw.read_range(n - SR - lag, n - lag))
    bus.close()


def test_controls():
    cfg = _mock_cfg()
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=PitchVoice(), attacker=MockAttacker(), shield=MockShield())
    assert pipe.set_shield("dsp") == "dsp"
    with pytest.raises(ValueError):
        pipe.set_shield("adversarial")
    with pytest.raises(ValueError):
        pipe.set_shield("bogus")
    bus.close()


def _real_available():
    try:
        from app.hearsay import driver as hearsay_real
        from app.keystroke_guard import driver as keyguard_real
        return hearsay_real.RUN.exists() and (keyguard_real.keyguard_root() / "keyguard").is_dir()
    except Exception:
        return False


@pytest.mark.skipif(not _real_available() or os.environ.get("CALLGUARD_SKIP_REAL"), reason="upstream repos missing")
def test_real_replay_ai_caller():
    try:
        sc = load_scenario("ai_caller")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    cfg = config.load(env={})
    cfg.drivers.voice = cfg.drivers.attacker = cfg.drivers.shield = "real"
    bus = EventBus()
    pipe = Pipeline(cfg, bus)
    seen = run(pipe, sc, bus)
    changes, last = [], None
    for t, lv, score in seen["levels"]:
        if lv != last:
            changes.append((round(t, 1), lv, round(score)))
            last = lv
    print("\nlevel timeline:", changes)
    for seg in sc.segments:
        ps = [p for t, p in seen["verdicts"] if seg["start"] + 4 <= t <= seg["end"]]
        print(f"  {seg['name']:<28} verdicts={len(ps)} median p={np.median(ps) if ps else float('nan'):.2f}")
    ro = seen["readouts"][-1]
    print("  raw read:", "".join(r["top1"] for r in ro["raw"]), " shielded read:",
          "".join(r["top1"] for r in ro["shielded"]))
    at = lambda t: [lv for tt, lv, _ in seen["levels"] if tt <= t][-1]  # noqa: E731
    assert at(11) == "SAFE"                                        # real colleague
    assert "CRITICAL" in [lv for tt, lv, _ in seen["levels"] if 20 <= tt < 29]  # agent + readable typing
    assert at(36) != "CRITICAL"                                    # shield on (29 s)
    assert at(sc.seconds) in ("SAFE", "WATCH")                     # agent gone
    n = len(seen["readouts"]) // 2                                 # first burst: shield off; second: on
    raw_off = sum(r["hit"]["raw"] for r in seen["readouts"][:n])
    shd_on = sum(r["hit"]["shielded"] for r in seen["readouts"][n:])
    assert raw_off > 2 * shd_on, (raw_off, shd_on)                 # top-3 hits
    pipe.stop()
    bus.close()
