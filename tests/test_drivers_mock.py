from types import SimpleNamespace

import numpy as np
import pytest

from callguard.drivers import base
from callguard.drivers.mock import CLASSES, MockAttacker, MockShield, MockVoice
from callguard.types import BLOCK, SR, KeystrokeAttackerDriver, ShieldDriver, VoiceAuthenticityDriver


def _noise(n, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 0.05).astype(np.float32)


def test_protocols():
    assert isinstance(MockVoice(), VoiceAuthenticityDriver)
    assert isinstance(MockAttacker(), KeystrokeAttackerDriver)
    assert isinstance(MockShield(), ShieldDriver)
    assert isinstance(base.guard(MockVoice()), VoiceAuthenticityDriver)
    assert isinstance(base.guard(MockAttacker()), KeystrokeAttackerDriver)
    assert isinstance(base.guard(MockShield()), ShieldDriver)


def test_voice_schedule_and_signal():
    v = MockVoice(schedule=[0.1, 0.9])
    assert [v.score(_noise(SR)).p_synthetic for _ in range(3)] == [0.1, 0.9, 0.1]
    tone = np.sin(2 * np.pi * 220 * np.arange(3 * SR) / SR).astype(np.float32)
    v = MockVoice()
    assert v.score(tone).p_synthetic > 0.9 > 0.6 > v.score(_noise(3 * SR)).p_synthetic


def _typed(n_keys=40, seed=1):
    """Mic stream with key events every 0.25 s, streamed through the shield block by block."""
    audio = _noise(n_keys * 4000 + SR, seed)
    onsets = np.arange(n_keys) * 4000 + 1000
    truths = [CLASSES[i % len(CLASSES)] for i in range(n_keys)]
    return audio, onsets, truths


def _shielded(audio, onsets):
    sh, out = MockShield(), []
    for s in range(0, len(audio), BLOCK):
        blk = audio[s: s + BLOCK]
        out.append(sh.process(blk, [int(o - s) for o in onsets if s <= o < s + BLOCK]))
    return np.concatenate(out)


def _acc(guesses):
    return np.mean([g.top[0][0] == g.truth for g in guesses])


def test_attacker_reads_raw_but_not_shielded():
    audio, onsets, truths = _typed()
    atk = MockAttacker(accuracy=0.9)
    raw = atk.read(audio, onsets, truths=truths)
    assert len(raw) == len(onsets) and _acc(raw) > 0.7
    assert atk.read(audio, onsets, truths=truths) == raw          # deterministic
    shielded = _shielded(audio, onsets)
    assert shielded.shape == audio.shape and shielded.dtype == np.float32
    assert _acc(atk.read(shielded, onsets, truths=truths)) < 0.2  # chance = 1/36
    assert all(g.truth is None for g in atk.read(audio, onsets))


def test_shield_passthrough_and_reset():
    blk = _noise(BLOCK)
    assert MockShield(perturb=False).process(blk, [0]) is blk
    sh = MockShield()
    sh.process(blk, [BLOCK - 1])
    assert not np.array_equal(sh.process(blk, []), blk)  # tail carries into the next block
    sh.reset()
    np.testing.assert_array_equal(sh.process(blk, []), blk)


class _Boom:
    name = "boom"
    sample_rate = SR

    def __init__(self):
        self.calls = 0

    def score(self, audio):
        self.calls += 1
        raise RuntimeError("model died")

    def process(self, block, key_events):
        raise RuntimeError("shield died")

    def reset(self):
        pass


def test_quarantine():
    events, drv = [], _Boom()
    q = base.QuarantinedVoice(drv, on_error=events.append, max_failures=2)
    assert q.name == "boom" and q.sample_rate == SR
    assert q.score(_noise(SR)) is None and not q.quarantined
    assert q.score(_noise(SR)) is None and q.quarantined
    q.score(_noise(SR))
    assert drv.calls == 2                                  # no longer called once quarantined
    assert [e.topic for e in events] == ["driver.error"] * 2
    assert events[-1].data["quarantined"] and "model died" in events[-1].data["error"]

    s = base.QuarantinedShield(_Boom(), on_error=lambda e: 1 / 0)  # a broken sink is swallowed too
    blk = _noise(BLOCK)
    assert s.process(blk, [3]) is blk
    assert base.QuarantinedAttacker(_Boom()).read(blk, np.array([1])) == []  # _Boom has no read -> fallback


def test_quarantine_streak_resets_on_success():
    v = MockVoice(schedule=[0.5])
    q = base.QuarantinedVoice(v, max_failures=2)
    v.score, ok = (lambda a: 1 / 0), v.score
    q.score(None)
    v.score = ok
    assert q.score(_noise(SR)).p_synthetic == 0.5 and q.failures == 0 and q.last_latency_ms >= 0


@pytest.mark.parametrize("cfg", [None, {}, {"drivers": {"voice": "mock"}}, SimpleNamespace(voice="mock")])
def test_factory_mock(cfg):
    assert isinstance(base.make_voice(cfg), MockVoice)
    assert isinstance(base.make_attacker(cfg), MockAttacker)
    assert isinstance(base.make_shield(cfg), MockShield)


def test_factory_real_is_lazy_and_passes_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(base, "_real", lambda mod, cls, **kw: seen.setdefault(cls, (mod, kw)))
    cfg = {"drivers": {"voice": "real", "attacker": "real", "shield": "real", "voice_mode": "r5", "threads": 2},
           "attacker_weights": "w.pt"}
    base.make_voice(cfg), base.make_attacker(cfg), base.make_shield(cfg)
    assert seen == {"HearsayDriver": ("hearsay_real", {"mode": "r5", "threads": 2}),
                    "KeyguardAttacker": ("keyguard_real", {"weights": "w.pt"}),
                    "KeyguardShield": ("keyguard_real", {"mode": "dsp"})}
    with pytest.raises(ValueError):
        base.make_voice({"voice": "bogus"})
