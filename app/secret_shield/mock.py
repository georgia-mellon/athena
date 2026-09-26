"""Secret-shield placeholder: a scripted spotter for tests and replay (no recognizer model)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from app.source.types import SR, SecretSpan


class MockSpotter:
    """Scripted spoken-secret spotter: `spans` = [(t_start_s, t_end_s, category, length)] on the fed stream's clock.
    Each is returned once the stream has been fed `lag_s` past its start, which mimics a streaming recognizer's
    partial-result latency (so late marks leak through the delay line, as they would for real)."""

    def __init__(self, spans: Sequence[tuple[float, float, str, int]] = (), mode: str = "outbound",
                 lag_s: float = 0.3):
        self.name = f"mock_spotter_{mode}"
        self.spans, self.lag = sorted(spans), round(lag_s * SR)
        self.reset()

    def reset(self) -> None:
        self._next = 0

    def feed(self, block: np.ndarray, start: int) -> list[SecretSpan]:
        pos, out = start + len(block), []
        while self._next < len(self.spans) and round(self.spans[self._next][0] * SR) + self.lag <= pos:
            a, b, cat, n = self.spans[self._next]
            out.append(SecretSpan(round(a * SR), round(b * SR), cat, n))
            self._next += 1
        return out
