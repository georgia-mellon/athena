"""Keyguard shield: remove keystroke transients while preserving speech.

Design (all knobs are what the blue arena agent tunes):
- STFT the whole signal *uniformly* (so processing itself leaks no timing).
- At each keystroke frame, inpaint magnitudes from neighbouring non-key frames
  (median over a guard-banded window). Speech is slowly-varying so it survives;
  the broadband click does not.
- randomize: whatever residue remains gets a *fresh* random per-stroke spectral
  gain, so a self-supervised attacker's clusters stop lining up with keys.
- decoys: inject fake keystroke-shaped transients at random times so inter-key
  timing can't be read off the residual either.

Runs on offline audio here; the real-time BlackHole path reuses the same
per-frame op (see realtime.py) fed by OS key-event timestamps.
"""
from __future__ import annotations
import numpy as np
import librosa
from dataclasses import dataclass
from ..config import SR, N_FFT, HOP, KEY_WIN
from .. import segment

RNG = np.random.default_rng(0)


@dataclass
class ShieldConfig:
    strength: float = 1.0        # 0..1 how hard to inpaint the key frames
    guard_frames: int = 2        # frames each side excluded from the inpaint source
    ctx_frames: int = 12         # frames each side used as inpaint source
    randomize: float = 1.0       # 0..1 residue randomization amount
    decoys: int = 0              # number of fake keystrokes to inject
    key_frames: int = 14         # frames per keystroke to treat as "click" (~110ms:
                                 # a keystroke's push+release, not just the onset tip)


class Shield:
    def __init__(self, cfg: ShieldConfig | None = None, seed: int | None = None):
        self.cfg = cfg or ShieldConfig()
        self.rng = np.random.default_rng(seed)

    def apply(self, y: np.ndarray, onsets: np.ndarray | None = None) -> np.ndarray:
        """Return shielded audio. onsets = keystroke sample indices; if None we
        detect them (real system passes OS timestamps instead)."""
        cfg = self.cfg
        if onsets is None:
            onsets = segment.onsets(y)
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
        mag, phase = np.abs(S), np.angle(S)
        n_frames = mag.shape[1]
        key_cols = self._key_columns(onsets, n_frames)

        for c in key_cols:
            lo = max(0, c - cfg.guard_frames - cfg.ctx_frames)
            hi = min(n_frames, c + cfg.guard_frames + cfg.ctx_frames + 1)
            src = [j for j in range(lo, hi)
                   if abs(j - c) > cfg.guard_frames and j not in key_cols]
            if not src:
                continue
            neigh = np.median(mag[:, src], axis=1)
            # keep speech (bins where current ~ neighbours), remove only the
            # keystroke's *excess* transient energy -- preserves STOI far better
            # than replacing the whole magnitude with the neighbour median.
            speech = np.minimum(mag[:, c], neigh)
            excess = mag[:, c] - speech
            if cfg.randomize > 0:                      # scramble residue identity
                gain = 1.0 + cfg.randomize * (self.rng.random(mag.shape[0]) - 0.5)
                excess = np.maximum(excess, 0) * gain
            mag[:, c] = speech + (1 - cfg.strength) * excess

        out = librosa.istft(mag * np.exp(1j * phase), hop_length=HOP, length=len(y))

        if cfg.decoys > 0:
            out = self._inject_decoys(out)
        return out.astype(np.float32)

    def _key_columns(self, onsets, n_frames):
        cols = set()
        for o in onsets:
            c = int(o / HOP)
            for d in range(-2, self.cfg.key_frames):   # cover the pre-onset rise too
                if 0 <= c + d < n_frames:
                    cols.add(c + d)
        return cols

    def _inject_decoys(self, y):
        y = y.copy()
        for _ in range(self.cfg.decoys):
            pos = int(self.rng.integers(0, max(1, len(y) - KEY_WIN)))
            click = self.rng.standard_normal(200).astype(np.float32)
            click *= np.exp(-np.linspace(0, 6, 200))          # decaying transient
            amp = 0.02 * self.rng.random()
            y[pos:pos + 200] += amp * click
        return y
