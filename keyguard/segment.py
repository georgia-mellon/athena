"""Keystroke onset detection.

Ported from the energy-threshold approach in the Harrison-reproduction repos
(coatnet/isolate_key_presses, shoyo/acoustic-keylogger): a keystroke is a sharp
broadband transient, so we peak-pick an onset-strength envelope. Returns onset
sample indices; features.py cuts the fixed window around each.
"""
import numpy as np
import librosa
from scipy.signal import find_peaks
from .config import SR, KEY_WIN, ONSET_MIN_GAP_S, ONSET_PROM, HOP


def onsets(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return keystroke onset sample indices, ascending."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    if env.max() <= 0:
        return np.array([], dtype=int)
    env = env / env.max()
    min_gap = int(ONSET_MIN_GAP_S * sr / HOP)
    peaks, _ = find_peaks(env, height=ONSET_PROM, distance=max(1, min_gap))
    return (peaks * HOP).astype(int)


def onsets_n(y: np.ndarray, n: int, sr: int = SR) -> np.ndarray:
    """Return exactly the n strongest onsets (used when count is known, e.g.
    labeled training files with 25 presses)."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    min_gap = int(ONSET_MIN_GAP_S * sr / HOP)
    peaks, props = find_peaks(env, distance=max(1, min_gap))
    if len(peaks) == 0:
        return np.array([], dtype=int)
    order = np.argsort(env[peaks])[::-1][:n]
    return np.sort(peaks[order]) * HOP


def windows(y: np.ndarray, onset_idx: np.ndarray) -> np.ndarray:
    """Cut fixed KEY_WIN windows (with small pre-roll) around each onset.
    Returns (n, KEY_WIN) float32, zero-padded at edges."""
    from .config import PRE_S
    pre = int(PRE_S * SR)
    out = np.zeros((len(onset_idx), KEY_WIN), dtype=np.float32)
    for i, o in enumerate(onset_idx):
        a = max(0, o - pre)
        seg = y[a:a + KEY_WIN]
        out[i, :len(seg)] = seg
    return out
