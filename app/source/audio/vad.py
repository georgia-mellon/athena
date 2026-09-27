"""Energy + zero-crossing VAD.

Only far-end speech is worth sending to the voice model: silence and hiss make its scores meaningless.
A frame is speech when it is loud enough and its zero-crossing rate is below what broadband noise produces.
Fixed thresholds, no adaptive noise floor; add one if venue noise trips it (tune energy_db first).
"""
from __future__ import annotations

import numpy as np

from app.source.types import BLOCK


def frame_is_speech(frame: np.ndarray, energy_db: float = -45.0, max_zcr: float = 0.35) -> bool:
    frame = np.asarray(frame, np.float32)
    if frame.size < 2:
        return False
    db = 10 * np.log10(np.mean(frame.astype(np.float64) ** 2) + 1e-12)
    zcr = np.mean(np.signbit(frame[1:]) != np.signbit(frame[:-1]))
    return bool(db > energy_db and zcr < max_zcr)


def speech_fraction(window: np.ndarray, frame: int = BLOCK, **kw) -> float:
    """Fraction of 20 ms frames in `window` judged speech (no hangover)."""
    window = np.asarray(window, np.float32).reshape(-1)
    n = len(window) // frame
    if n == 0:
        return 0.0
    return sum(frame_is_speech(window[i * frame:(i + 1) * frame], **kw) for i in range(n)) / n


class Vad:
    """Streaming per-block decision; stays 'speech' for `hangover` blocks after the last speech frame so word gaps
    don't chop a window in half."""

    def __init__(self, energy_db: float = -45.0, max_zcr: float = 0.35, hangover: int = 15):
        self.kw = dict(energy_db=energy_db, max_zcr=max_zcr)
        self.hangover = hangover
        self._left = 0

    def __call__(self, block: np.ndarray) -> bool:
        if frame_is_speech(block, **self.kw):
            self._left = self.hangover
            return True
        if self._left > 0:
            self._left -= 1
            return True
        return False

    def reset(self) -> None:
        self._left = 0
