"""Replay sources: WAV files as 20 ms blocks, a mixer, and scripted key tracks.

Replay mode feeds the same pipeline as the live devices (plan 02 §1), so a scenario is just audio + key timings.
"""
from __future__ import annotations

import csv
import time
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.source.audio.keys import ScriptedKeyClock
from app.source.types import BLOCK, SR


def load_wav(path: str | Path, sr: int = SR) -> np.ndarray:
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g).astype(np.float32)
    return x


class FileSource:
    """A WAV path or a 1-D array at 16 kHz, optionally delayed by `offset_s` and scaled by `gain`."""

    def __init__(self, audio: str | Path | np.ndarray, gain: float = 1.0, offset_s: float = 0.0, sr: int = SR):
        x = load_wav(audio, sr) if isinstance(audio, (str, Path)) else np.asarray(audio, np.float32).reshape(-1)
        self.sr = sr
        self.audio = np.concatenate([np.zeros(round(offset_s * sr), np.float32), x * np.float32(gain)])

    def __len__(self) -> int:
        return len(self.audio)

    def blocks(self, realtime: bool = False, block: int = BLOCK) -> Iterator[np.ndarray]:
        """Consecutive blocks (last one zero-padded). realtime=True paces them at wall-clock speed against a fixed
        schedule, so a slow consumer doesn't accumulate drift."""
        t0 = time.monotonic()
        for i, s in enumerate(range(0, len(self.audio), block)):
            b = self.audio[s:s + block]
            if len(b) < block:
                b = np.pad(b, (0, block - len(b)))
            if realtime:
                wait = t0 + i * block / self.sr - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            yield b


def mix(*sources: FileSource) -> FileSource:
    """Layer sources (e.g. AI-agent voice + typing + room noise) into one source; length = the longest."""
    n = max((len(s) for s in sources), default=0)
    out = np.zeros(n, np.float32)
    for s in sources:
        out[:len(s)] += s.audio
    return FileSource(out)


def load_key_track(path: str | Path, sr: int = SR) -> ScriptedKeyClock:
    """CSV with rows `t_seconds,key` (header optional) -> a clock for replay."""
    events = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                events.append((float(row[0]), row[1].strip()))
            except (ValueError, IndexError):
                continue  # header or blank line
    return ScriptedKeyClock.from_seconds(events, sr)
