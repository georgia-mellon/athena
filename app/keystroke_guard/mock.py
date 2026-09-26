"""Deterministic stand-ins for the real drivers: tests, UI work and CI run with no models or devices (spec F8).

The shield/attacker pair communicates through the audio itself, so it survives ring buffers and copies: MockShield
adds a quiet pilot tone (MARK_HZ) after every key event, and MockAttacker falls to chance on any onset where it
hears that tone. That mimics the real claim (shield on -> the eavesdropper reads noise) without a model.
"""
from __future__ import annotations

import string
from typing import Sequence

import numpy as np

from app.source.types import SR, KeyGuess

CLASSES = list(string.ascii_uppercase + string.digits)
MARK_HZ = 7_600.0     # below Nyquist, above speech; the tone the attacker listens for
MARK_AMP = 0.05
MARK_LEN = 1_600      # 100 ms of perturbation after each key event (a keystroke transient is ~ 40-100 ms)
_DETECT_LEN = 800
_DETECT_AMP = MARK_AMP / 2


def shield_marked(audio: np.ndarray, onset: int) -> bool:
    """True when MockShield's pilot tone is present just after `onset` (single-bin DFT amplitude)."""
    seg = np.asarray(audio[max(onset, 0): onset + _DETECT_LEN], np.float64)
    if len(seg) < _DETECT_LEN // 2:
        return False
    w = np.exp(-2j * np.pi * MARK_HZ / SR * np.arange(len(seg)))
    return 2 * abs(np.dot(seg, w)) / len(seg) > _DETECT_AMP


class MockAttacker:
    """Reads the true key with probability `accuracy` when truths are given, else guesses uniformly; on shielded
    onsets it always guesses uniformly. Seeded per onset, so the same input gives the same output.
    It can't hear keys, so it's the one driver the pipeline hands the truth to (`wants_truth`); real attackers never
    see it, and the pipeline attaches KeyGuess.truth after the call."""
    name = "mock_attacker"
    wants_truth = True

    def __init__(self, accuracy: float = 0.85, top_k: int = 3, seed: int = 0, classes: list[str] | None = None):
        self.accuracy, self.top_k, self.seed = accuracy, top_k, seed
        self.classes = classes or CLASSES

    def read(self, audio: np.ndarray, onsets: np.ndarray, truths: Sequence[str | None] | None = None) -> list[KeyGuess]:
        out = []
        for i, onset in enumerate(np.asarray(onsets, dtype=np.int64)):
            truth = truths[i] if truths is not None and i < len(truths) else None
            seg = np.asarray(audio[max(int(onset), 0): int(onset) + _DETECT_LEN], np.float64)
            rng = np.random.default_rng((self.seed, int(onset), int(abs(seg).sum() * 1e4) % 2**31))
            order = [self.classes[j] for j in rng.permutation(len(self.classes))[: self.top_k]]
            if truth in self.classes and not shield_marked(audio, int(onset)) and rng.random() < self.accuracy:
                order = [truth] + [k for k in order if k != truth][: self.top_k - 1]
                probs = [0.7] + [0.3 / (self.top_k - 1)] * (self.top_k - 1) if self.top_k > 1 else [1.0]
            else:
                probs = [1.0 / len(self.classes)] * self.top_k
            out.append(KeyGuess(onset=int(onset), top=list(zip(order, probs)), truth=truth))
        return out


class MockShield:
    """Adds the pilot tone for MARK_LEN samples after each key event (absolute sample indices since reset(), like
    the real shield); carries the tail and the phase across blocks. `perturb=False` is a pure pass-through."""
    name = "mock_shield"
    latency = 0  # samples of output delay (the real DSP shield has a lookahead)

    def __init__(self, perturb: bool = True):
        self.perturb = perturb
        self.reset()

    def reset(self) -> None:
        self._n, self._left = 0, 0      # samples seen (tone phase), samples of tone still owed

    def process(self, block: np.ndarray, key_events: list[int]) -> np.ndarray:
        n = len(block)
        if not self.perturb:
            self._n += n
            return block
        gate = np.zeros(n, np.float32)
        gate[: min(self._left, n)] = 1.0
        end = min(self._left, n) if self._left else 0
        for e in (int(e) - self._n for e in key_events):  # absolute -> within this block; late events keep their tail
            if e < n and e + MARK_LEN > 0:
                gate[max(e, 0): e + MARK_LEN] = 1.0
                end = max(end, e + MARK_LEN)
        self._left = max(end - n, 0)
        t = (self._n + np.arange(n)) / SR
        self._n += n
        return (block + gate * MARK_AMP * np.sin(2 * np.pi * MARK_HZ * t)).astype(np.float32)
