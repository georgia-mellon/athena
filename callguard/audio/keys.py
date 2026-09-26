"""KeyClock: OS key presses mapped onto the mic stream's absolute sample index.

The attacker and shield both need to know *where in the mic audio* each keystroke sits. The mic callback calls
`anchor(sample_index, monotonic_time)` every block; a press at monotonic time t maps to
`anchor_sample + (t - anchor_time + offset_s) * sr`. Re-anchoring every block absorbs clock drift.
Key identity is kept in memory only (for the demo's "typed vs attacker read" readout) and never written anywhere.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import NamedTuple

from callguard.types import SR


class KeyEvent(NamedTuple):
    sample: int     # absolute index in the mic stream
    key: str        # in memory only
    t: float        # time.monotonic() of the press


class KeyClock:
    def __init__(self, sr: int = SR, offset_s: float = 0.0, keep: int = 512):
        self.sr = sr
        self.offset_s = offset_s  # calibration knob: + if key sounds land later in the audio than their OS timestamp
        self._events: deque[KeyEvent] = deque(maxlen=keep)
        self._anchor: tuple[int, float] | None = None
        self._lock = threading.Lock()
        self._listener = None

    def anchor(self, sample: int, t: float | None = None) -> None:
        """Declare that absolute `sample` was captured at monotonic time `t` (called by the mic stream)."""
        self._anchor = (int(sample), time.monotonic() if t is None else t)

    @property
    def now_sample(self) -> int | None:
        return None if self._anchor is None else self._anchor[0]

    def sample_of(self, t: float) -> int | None:
        a = self._anchor
        if a is None:
            return None
        return a[0] + round((t - a[1] + self.offset_s) * self.sr)

    def press(self, key: str, t: float | None = None) -> KeyEvent | None:
        t = time.monotonic() if t is None else t
        s = self.sample_of(t)
        if s is None:  # no audio running yet: nothing to align to
            return None
        ev = KeyEvent(s, key, t)
        with self._lock:
            self._events.append(ev)
        return ev

    def recent(self, n: int = 20) -> list[KeyEvent]:
        with self._lock:
            return list(self._events)[-n:]

    def between(self, start: int, stop: int) -> list[KeyEvent]:
        """Presses with start <= sample < stop."""
        with self._lock:
            return [e for e in self._events if start <= e.sample < stop]

    def in_range(self, start: int, stop: int) -> list[int]:
        """Absolute sample indices of presses in [start, stop)."""
        return [e.sample for e in self.between(start, stop)]

    def start(self) -> KeyClock:
        """Start the global pynput listener. Without pynput (or a display/permission) we log and run without keys."""
        try:
            from pynput import keyboard

            def on_press(k):
                name = getattr(k, "char", None) or getattr(k, "name", None) or str(k)
                self.press(str(name))

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        except Exception as e:  # noqa: BLE001
            print(f"[keys] key listener unavailable ({e}); running without key timing")
        return self

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class ScriptedKeyClock(KeyClock):
    """Replay/test clock: events are known in advance by sample index; `recent` only shows ones already played
    (sample <= the last anchored sample)."""

    def __init__(self, events: list[tuple[int, str]], sr: int = SR):
        super().__init__(sr=sr, keep=max(len(events), 1))
        for s, k in sorted(events):
            self._events.append(KeyEvent(int(s), k, s / sr))

    @classmethod
    def from_seconds(cls, events: list[tuple[float, str]], sr: int = SR) -> ScriptedKeyClock:
        return cls([(round(t * sr), k) for t, k in events], sr)

    def recent(self, n: int = 20) -> list[KeyEvent]:
        now = self.now_sample
        with self._lock:
            played = [e for e in self._events if now is not None and e.sample <= now]
        return played[-n:]

    def start(self) -> ScriptedKeyClock:
        return self
