"""Build labeled keystroke datasets from the Harrison per-key WAVs.

Each MBPWavs/<KEY>.wav (or Zoom/<idx>.wav) holds ~25 presses of one key.
We segment, window, mel, and split by press index so train/test never share
the same recording of a press.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from . import audio, segment, features
from .config import CLASSES, CLS_IDX

PRESSES = 25


def _key_mels(path, n=PRESSES):
    y = audio.load(path)
    on = segment.onsets_n(y, n)
    return features.mel(segment.windows(y, on)), on, y


def build(root, keys=CLASSES, n_test=5, transform=None):
    """Return (Xtr, ytr, Xte, yte). transform(y)->y runs on raw audio first
    (used to evaluate against a shielded signal)."""
    root = Path(root)
    Xtr, ytr, Xte, yte = [], [], [], []
    for k in keys:
        p = root / f"{k}.wav"
        if not p.exists():
            continue
        y = audio.load(p)
        if transform is not None:
            y = transform(y)
        on = segment.onsets_n(y, PRESSES)
        m = features.mel(segment.windows(y, on))
        if len(m) < PRESSES:
            pad = np.zeros((PRESSES - len(m), *m.shape[1:]), np.float32)
            m = np.concatenate([m, pad]) if len(m) else pad
        idx = CLS_IDX[k]
        Xtr.append(m[:-n_test]); ytr += [idx] * (PRESSES - n_test)
        Xte.append(m[-n_test:]); yte += [idx] * n_test
    return (np.concatenate(Xtr), np.array(ytr),
            np.concatenate(Xte), np.array(yte))


def build_aug(root, keys=CLASSES, n_test=5, jitter=(-80, 0, 80), transform=None):
    """Like build() but multiplies training presses with small window jitters
    (in samples) to grow the tiny 20-shot training set. Test stays un-jittered."""
    from pathlib import Path
    Xtr, ytr, Xte, yte = [], [], [], []
    root = Path(root)
    for k in keys:
        p = root / f"{k}.wav"
        if not p.exists():
            continue
        y = audio.load(p)
        if transform is not None:
            y = transform(y)
        on = segment.onsets_n(y, PRESSES)
        tr_on, te_on = on[:-n_test], on[-n_test:]
        idx = CLS_IDX[k]
        for j in jitter:
            m = features.mel(segment.windows(y, tr_on + j))
            Xtr.append(m); ytr += [idx] * len(m)
        m = features.mel(segment.windows(y, te_on))
        Xte.append(m); yte += [idx] * len(m)
    import numpy as np
    return (np.concatenate(Xtr), np.array(ytr),
            np.concatenate(Xte), np.array(yte))
