"""Ring buffer with absolute sample counters.

The audio thread writes 20 ms blocks; workers read "the last N seconds" or an exact absolute range (a key event's
sample index from KeyClock), so every consumer agrees on one timeline without copying state between threads.
"""
from __future__ import annotations

import threading

import numpy as np

from app.source.types import SR


class Ring:
    def __init__(self, seconds: float = 30.0, sr: int = SR):
        self.sr = sr
        self.cap = int(seconds * sr)
        self._buf = np.zeros(self.cap, np.float32)
        self.total = 0  # absolute count of samples ever written; sample i lives at _buf[i % cap] while i >= total - cap
        self._lock = threading.Lock()

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32).reshape(-1)
        skip = max(0, len(x) - self.cap)  # a write longer than cap keeps only its tail
        x = x[skip:]
        with self._lock:
            self.total += skip
            i = self.total % self.cap
            n = min(len(x), self.cap - i)
            self._buf[i:i + n] = x[:n]
            self._buf[:len(x) - n] = x[n:]
            self.total += len(x)

    def read_range(self, start: int, stop: int) -> np.ndarray | None:
        """Samples [start, stop) by absolute index, or None if any of it is overwritten or not written yet."""
        with self._lock:
            if start < max(0, self.total - self.cap) or stop > self.total or start > stop:
                return None
            idx = np.arange(start, stop) % self.cap
            return self._buf[idx].copy()

    def read_last(self, n: int) -> np.ndarray:
        """The newest n samples (fewer if less has been written)."""
        with self._lock:
            stop = self.total
        return self.read_range(max(0, stop - min(n, self.cap)), stop)

    def read_last_seconds(self, seconds: float) -> np.ndarray:
        return self.read_last(int(seconds * self.sr))
