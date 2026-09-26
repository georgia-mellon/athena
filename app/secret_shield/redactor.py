"""Delay line + redaction envelope for the spoken-secret shield (plan 06 §3).

The outbound stream is delayed by a constant `delay` samples whenever the feature is enabled, so arming never makes
the stream jump. A worker (the spotter) marks absolute sample ranges to redact; the audio thread applies them when
those samples reach the delay line's output. A mark that arrives after its samples already left is counted in
`leaked_samples`: that's the honest metric. The audio thread does O(block) work: no model, no allocation beyond
the block.
"""
from __future__ import annotations

import threading

import numpy as np

from app.source.types import SR

RAMP = int(0.005 * SR)      # 5 ms fades so a mute doesn't click
TONE_HZ = 1_000.0


class Redactor:
    def __init__(self, delay: int = SR // 2, style: str = "tone"):
        if style not in ("mute", "tone", "noise"):
            raise ValueError("style must be mute | tone | noise")
        self.delay, self.style = int(delay), style
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0)
        self.reset()

    @property
    def latency(self) -> int:
        return self.delay

    def reset(self) -> None:
        with self._lock:
            self._buf = np.zeros(self.delay, np.float32)
            self._out = -self.delay         # absolute index (input clock) of the next sample to leave
            self._marks: list[tuple[int, int]] = []
            self.leaked_samples = 0
            self.redacted_samples = 0

    def mark(self, start: int, end: int) -> int:
        """Redact input samples [start, end). Returns how many of them had already left (leaked)."""
        start, end = int(start), int(end)
        if end <= start:
            return 0
        with self._lock:
            leaked = max(0, min(end, self._out) - start)
            self.leaked_samples += leaked
            if end + RAMP > self._out:
                self._marks.append((max(start, self._out), end))
        return leaked

    def process(self, block: np.ndarray, start: int) -> np.ndarray:
        """Push `block` (input samples [start, start+n)); return the n delayed samples, redacted where marked."""
        n = len(block)
        x = np.concatenate([self._buf, np.asarray(block, np.float32)])
        out, self._buf = x[:n].copy(), x[n:]
        with self._lock:
            lo = start - self.delay                          # absolute index of out[0]
            self._out = lo + n
            marks = [m for m in self._marks if m[1] + RAMP > lo]
            self._marks = [m for m in marks if m[1] + RAMP > self._out]
        if not marks:
            return out
        gain = np.ones(n, np.float32)
        for a, b in marks:
            # fade in/out over RAMP samples around [a, b), clipped to this block
            idx = np.arange(lo, lo + n)
            g = np.clip(np.minimum(idx - (a - RAMP), (b + RAMP) - idx) / RAMP, 0.0, 1.0)
            gain = np.minimum(gain, 1.0 - g)
        self.redacted_samples += int(np.sum(gain < 0.5))
        fill = self._fill(n, lo) * (1.0 - gain)
        return (out * gain + fill).astype(np.float32)

    def _fill(self, n: int, lo: int) -> np.ndarray:
        if self.style == "tone":
            return (0.05 * np.sin(2 * np.pi * TONE_HZ * np.arange(lo, lo + n) / SR)).astype(np.float32)
        if self.style == "noise":
            return (0.02 * self._rng.standard_normal(n)).astype(np.float32)
        return np.zeros(n, np.float32)
