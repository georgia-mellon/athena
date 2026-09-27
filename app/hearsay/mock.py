"""Hearsay placeholder: a deterministic stand-in for the voice-authenticity model (no weights, no upstream repo)."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from app.source.types import SR, VoiceScore


class MockVoice:
    """p_synthetic from a scripted schedule (cycled per call), else from spectral flatness: tonal/"too clean" audio
    scores high, noise-like audio low. Arbitrary but deterministic. `latency_ms` sleeps to mimic model timing."""
    name = "mock_voice"
    sample_rate = SR

    def __init__(self, schedule: Sequence[float] | None = None, latency_ms: float = 0.0):
        self.schedule, self.latency_ms, self._i = list(schedule or []), latency_ms, 0

    def score(self, audio: np.ndarray) -> VoiceScore:
        t0 = time.perf_counter()
        if self.schedule:
            p = float(self.schedule[self._i % len(self.schedule)])
            self._i += 1
        else:
            spec = np.abs(np.fft.rfft(np.asarray(audio, np.float64))) ** 2 + 1e-12
            flatness = np.exp(np.mean(np.log(spec))) / np.mean(spec)
            p = float(np.clip(1.0 - flatness, 0.0, 1.0))
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        pc = min(max(p, 1e-6), 1 - 1e-6)
        return VoiceScore(p_synthetic=p, margin=float(np.log(pc / (1 - pc))), threshold=0.0,
                          latency_ms=(time.perf_counter() - t0) * 1000, detail={"mock": True})
