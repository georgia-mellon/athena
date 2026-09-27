"""Pillar harnesses: the placeholders fit the contracts, and a deliberately broken driver does not."""
from __future__ import annotations

import numpy as np

from app.hearsay import harness as hearsay
from app.hearsay.mock import MockVoice
from app.keystroke_guard import harness as keys
from app.keystroke_guard.mock import MockAttacker, MockShield
from app.secret_shield import harness as secret
from app.source.types import SR, KeyGuess, SecretSpan, VoiceScore


def statuses(rep) -> dict[str, str]:
    return {check: status for _, check, status, _ in rep.rows}


# --- the placeholders fit ------------------------------------------------------------------------------------------
def test_mocks_fit():
    for rep in (hearsay.check("mock", quality=False), keys.check("mock", "mock", quality=False),
                secret.check("mock", quality=False), secret.check("mock", "inbound", quality=False)):
        assert rep.ok, rep.rows
        assert "FAIL" not in statuses(rep).values()


def test_class_path_loads_and_cli_exit_code():
    assert hearsay.main(["--driver", "app.hearsay.mock:MockVoice", "--no-quality"]) == 0
    assert keys.main(["--attacker", "app.keystroke_guard.mock:MockAttacker", "--shield", "none", "--no-quality"]) == 0
    assert secret.main(["--spotter", "app.secret_shield.mock:MockSpotter", "--no-quality"]) == 0


# --- broken drivers don't -------------------------------------------------------------------------------------------
class OutOfRangeVoice(MockVoice):
    """p_synthetic outside [0, 1] and disagreeing with its own margin."""
    def score(self, audio):
        return VoiceScore(p_synthetic=1.7, margin=-3.0, threshold=0.0, latency_ms=1.0)


class NondeterministicVoice(MockVoice):
    def score(self, audio):
        p = float(np.random.rand())
        return VoiceScore(p, np.log(p / (1 - p)), 0.0, 1.0)


class UnsortedAttacker(MockAttacker):
    """Leaks the truth field, lists the worst guess first and names a key that isn't a class."""
    def read(self, audio, onsets, **kw):
        return [KeyGuess(int(o), [("A", 0.1), ("?", 0.6)], truth="A") for o in onsets]


class LeakyShield(MockShield):
    """Changes audio even with no key events (and returns a short block)."""
    def process(self, block, key_events):
        return block[:-1] * 0.5


class TextSpotter:
    """Emits a span that carries the recognized words, and repeats it every block."""
    name = "text_spotter"

    def reset(self):
        pass

    def feed(self, block, start):
        span = SecretSpan(SR // 2, SR, "digits", 2)
        span.text = "four seven two nine"   # exactly what the contract forbids
        return [span]


def test_broken_voice_fails():
    rep = hearsay.check(OutOfRangeVoice(), quality=False)
    assert not rep.ok and statuses(rep)["score(4 s) -> VoiceScore"] == "FAIL"
    rep = hearsay.check(NondeterministicVoice(), quality=False)
    assert not rep.ok and statuses(rep)["deterministic"] == "FAIL"


def test_broken_keystroke_drivers_fail():
    rep = keys.check(UnsortedAttacker(), LeakyShield(), quality=False)
    s = statuses(rep)
    assert not rep.ok
    assert s["read() -> [KeyGuess]"] == "FAIL"
    assert s["no keys -> exact pass-through"] == "FAIL"
    assert s["attacker Protocol + attributes"] == "PASS"


def test_broken_spotter_fails():
    rep = secret.check(TextSpotter(), quality=False)
    assert not rep.ok and statuses(rep)["feed() -> [SecretSpan]"] == "FAIL"


def test_not_a_driver_fails_the_protocol():
    rep = hearsay.check(object(), quality=False)
    assert not rep.ok and statuses(rep)["Protocol + attributes"] == "FAIL"
