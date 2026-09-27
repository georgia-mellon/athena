"""Keyguard drivers (WP4). Skips when KEYGUARD_ROOT is missing; the attacker test trains/caches the provisional
KeyNet on first run (~2 min CPU) and reuses runs/provisional_keynet.pt after that."""
import time

import numpy as np
import pytest

from app.keystroke_guard import driver as kr
from app.source.types import BLOCK, SR, KeystrokeAttackerDriver, ShieldDriver

pytestmark = pytest.mark.skipif(not (kr.keyguard_root() / "keyguard").is_dir(), reason="KEYGUARD_ROOT not available")


def run(shield, x, events):
    out = [shield.process(x[i:i + BLOCK], [e for e in events if i <= e < i + BLOCK]) for i in range(0, len(x), BLOCK)]
    return np.concatenate(out)[shield.lookahead:]   # undo the fixed output delay


def speechlike(n, seed=0):
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)     # syllable-rate AM
    return (0.1 * env * (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 540 * t))).astype(np.float32)


def test_shield_near_identity_without_keys_and_same_length():
    s = kr.KeyguardShield()
    assert isinstance(s, ShieldDriver)
    x = speechlike(2 * SR)
    y = run(s, x, [])
    assert len(y) + s.lookahead == len(x)
    assert np.corrcoef(x[:len(y)], y)[0, 1] > 0.99


def test_shield_attenuates_key_and_keeps_speech_far_away():
    rng = np.random.default_rng(0)
    x = speechlike(2 * SR)
    e = SR  # key at 1.0 s
    click = rng.standard_normal(800).astype(np.float32) * np.exp(-np.linspace(0, 8, 800)).astype(np.float32)
    x[e:e + 800] += 0.8 * click
    s = kr.KeyguardShield()
    t0 = time.perf_counter()
    y = run(s, x, [e])
    print(f"shield: {(time.perf_counter() - t0) * 1000:.0f} ms for {len(x) / SR:.0f} s of audio, delay {s.latency_ms:.0f} ms")
    key = slice(e, e + 800)
    assert np.sum(y[key] ** 2) < 0.5 * np.sum(x[key] ** 2)                 # transient energy removed
    far = slice(0, e - 3000)                                               # speech away from the key untouched
    assert np.corrcoef(x[far], y[far])[0, 1] > 0.95
    s.reset()
    assert np.allclose(s.process(np.ones(BLOCK, np.float32), []), 0)       # reset clears the delay line


def test_adversarial_mode_needs_trained_deltas(tmp_path):
    with pytest.raises(FileNotFoundError):         # modes themselves: tests/test_adversarial_shield.py
        kr.KeyguardShield(mode="adversarial", deltas=tmp_path / "missing.pt")
    with pytest.raises(ValueError):
        kr.KeyguardShield(mode="bogus")


def test_attacker_reads_held_out_presses_well_above_chance():
    a = kr.KeyguardAttacker()
    assert isinstance(a, KeystrokeAttackerDriver) and len(a.classes) == 36
    _, _, Xte, yte = kr.harrison_split()
    # lay held-out windows end to end and read them back through read(), exercising the onset->window cut
    from keyguard.config import KEY_WIN, PRE_S
    pre = int(PRE_S * SR)
    audio = Xte.ravel()
    onsets = np.arange(len(Xte)) * KEY_WIN + pre
    guesses = a.read(audio, onsets)
    assert len(guesses) == len(Xte) and len(guesses[0].top) == 3
    top1 = np.mean([g.top[0][0] == a.classes[y] for g, y in zip(guesses, yte)])
    top3 = np.mean([a.classes[y] in [k for k, _ in g.top] for g, y in zip(guesses, yte)])
    print(f"held-out harrison: top1={top1:.3f} top3={top3:.3f} (chance {1/36:.3f}, n={len(yte)})")
    assert top1 > 10 / 36   # >10x chance


@pytest.mark.skipif(not (kr.keyguard_root() / kr.CTC_WEIGHTS).exists(), reason="Keyguard CTC weights not available")
def test_ctc_attacker_reads_keyguard_bank_typing():
    """Keyguard's current attacker (MtlCRNN, ctc_rich_ft) on typing synthesized from the teammate's own key bank
    (its training domain; harrison is out of domain for it). Checks the onset -> frame alignment, not generalization."""
    from keyguard.ctc.data import VOCAB, random_text, synth_line
    a = kr.KeyguardCTCAttacker()
    assert isinstance(a, KeystrokeAttackerDriver) and len(a.classes) == 37
    rng = np.random.default_rng(0)
    hits = n = 0
    for _ in range(5):
        y, lab, on = synth_line(random_text(rng), rng, wpm=(40, 75), root=str(kr.keyguard_root() / "data" / "live_bank_rich.npz"))
        guesses = a.read(y, on)
        assert len(guesses) == len(on) and len(guesses[0].top) == 3
        hits += sum(g.top[0][0] == VOCAB[j] for g, j in zip(guesses, lab))
        n += len(guesses)
    print(f"keyguard bank typing: top1={hits / n:.3f} (chance {1 / 37:.3f}, n={n})")
    assert hits / n > 0.5
