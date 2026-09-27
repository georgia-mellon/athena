"""Invariant = channel-normalize away keyboard/mic spectral coloration before KeyNet,
then train-time augment to randomize whatever coloration normalization misses.

Hypothesis: baseline's mel() normalizes each window by one global mean/std, so the
per-frequency-band coloration a keyboard+mic pair bakes in (its near-time-invariant
convolutive channel response) survives as a roughly constant offset per mel bin across
the whole 0.3s window. KeyNet can shortcut on that instead of the key's transient click
shape.

Two changes, same KeyNet architecture + training loop:
(a) Per-mel-band CMVN -- subtract/scale each mel bin by its own mean/std across the
    window's time frames -- cancels that constant term and leaves only the within-window
    dynamics, which should transfer across unseen keyboards.
(b) Channel-randomizing augmentation on top, so the net can't shortcut on whatever
    coloration CMVN leaves behind: random smooth band-EQ curves, a short synthetic-room
    time-smear (per-band FIR), gain jitter, +-5% freq/time warp (pitch/tempo-ish), and
    SpecAugment freq+time masks. All applied per-batch, per-epoch in spectrogram space
    (cheap, no waveform re-synthesis) by overriding SupervisedAttacker's `_specaug` hook
    -- supervised.py itself is untouched.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from ..attackers.supervised import SupervisedAttacker
from ..features import mel

EPOCHS = 40
FT_EPOCHS = 30


def _cmvn(mels: np.ndarray) -> np.ndarray:
    """Per-mel-bin mean/var normalization across time (cancels channel coloration)."""
    mu = mels.mean(axis=2, keepdims=True)
    sd = mels.std(axis=2, keepdims=True) + 1e-6
    return (mels - mu) / sd


def _feats(wins: np.ndarray) -> np.ndarray:
    return _cmvn(mel(wins))


# ---- channel-randomizing augmentation (all operate on (n,1,mel,frames) tensors) ----

def _gain_jitter(xb: torch.Tensor, max_db: float = 3.0) -> torch.Tensor:
    n = xb.shape[0]
    db = (torch.rand(n, 1, 1, 1, device=xb.device) * 2 - 1) * max_db
    return xb * (10.0 ** (db / 20.0))


def _band_eq(xb: torch.Tensor, max_db: float = 6.0, ncp: int = 4) -> torch.Tensor:
    """Random smooth per-band gain curve -- simulates a different mic/keyboard channel."""
    n, _, mel_bins, _ = xb.shape
    cp = (torch.rand(n, 1, ncp, 1, device=xb.device) * 2 - 1) * max_db
    curve = F.interpolate(cp, size=(mel_bins, 1), mode="bilinear", align_corners=True)
    return xb + curve  # additive in log-mel (dB) domain


def _room_smear(xb: torch.Tensor, max_len: int = 5, p: float = 0.5) -> torch.Tensor:
    """Short per-band decaying FIR along time -- a cheap synthetic-room impulse."""
    if random.random() > p:
        return xb
    length = random.randint(2, max_len)
    decay = random.uniform(0.5, 2.0)
    kernel = torch.exp(-torch.arange(length, device=xb.device).float() * decay)
    kernel = (kernel / kernel.sum()).flip(0).view(1, 1, 1, length)
    xb_pad = F.pad(xb, (length - 1, 0))
    return F.conv2d(xb_pad, kernel)


def _warp(xb: torch.Tensor, max_pct: float = 0.05) -> torch.Tensor:
    """+-max_pct resize-and-back on freq/time axes -- pitch/tempo-ish warp."""
    n, c, mel_bins, frames = xb.shape
    sf = 1 + (random.random() * 2 - 1) * max_pct
    st = 1 + (random.random() * 2 - 1) * max_pct
    new_mel = max(1, round(mel_bins * sf))
    new_frames = max(1, round(frames * st))
    xb = F.interpolate(xb, size=(new_mel, new_frames), mode="bilinear", align_corners=False)
    return F.interpolate(xb, size=(mel_bins, frames), mode="bilinear", align_corners=False)


def _spec_mask(xb: torch.Tensor, freq_mask: int = 8, time_mask: int = 8, n_masks: int = 2) -> torch.Tensor:
    """SpecAugment: zero random freq bands and time spans (data is ~0-mean post-CMVN)."""
    xb = xb.clone()
    _, _, mel_bins, frames = xb.shape
    for _ in range(n_masks):
        fw = random.randint(0, min(freq_mask, mel_bins))
        f0 = random.randint(0, mel_bins - fw)
        xb[:, :, f0:f0 + fw, :] = 0
        tw = random.randint(0, min(time_mask, frames))
        t0 = random.randint(0, frames - tw)
        xb[:, :, :, t0:t0 + tw] = 0
    return xb


class ChannelRobustAttacker(SupervisedAttacker):
    """Same KeyNet; swaps the batch augmentation for channel-randomizing ops."""

    def _specaug(self, xb: torch.Tensor) -> torch.Tensor:
        xb = _gain_jitter(xb)
        xb = _band_eq(xb)
        xb = _room_smear(xb)
        xb = _warp(xb)
        xb = _spec_mask(xb)
        return xb + 0.02 * torch.randn_like(xb)


class Invariant:
    def __init__(self) -> None:
        self.atk = ChannelRobustAttacker()

    def fit(self, wins: np.ndarray, y: np.ndarray, dom: np.ndarray) -> None:
        self.atk.fit(_feats(wins), y, epochs=EPOCHS)

    def finetune(self, wins: np.ndarray, y: np.ndarray) -> None:
        self.atk.fit(_feats(wins), y, epochs=FT_EPOCHS, lr=3e-4, bs=32)

    def predict_proba(self, wins: np.ndarray) -> np.ndarray:
        return self.atk.predict_proba(_feats(wins))


def make() -> Invariant:
    return Invariant()


def _demo() -> None:
    """ponytail self-check: cmvn cancels channel offset; augment ops preserve shape
    and actually perturb the input (so they're not silently no-ops)."""
    rng = np.random.default_rng(0)
    m = rng.normal(size=(3, 8, 20)).astype(np.float32)
    coloration = rng.normal(size=(1, 8, 1)).astype(np.float32) * 5  # per-band, time-const
    a, b = _cmvn(m), _cmvn(m + coloration)
    assert np.allclose(a, b, atol=1e-4), "cmvn should be invariant to per-band channel offset"
    assert a.shape == m.shape

    torch.manual_seed(0)
    random.seed(0)
    xb = torch.randn(4, 1, 64, 38)
    fns = (_gain_jitter, _band_eq, lambda x: _room_smear(x, p=1.0), _warp, _spec_mask)
    names = ("_gain_jitter", "_band_eq", "_room_smear", "_warp", "_spec_mask")
    for fn, name in zip(fns, names):
        out = fn(xb)
        assert out.shape == xb.shape, f"{name} changed shape"
        assert not torch.allclose(out, xb), f"{name} was a no-op"
    print("ok: cmvn cancels per-band channel coloration; augment ops perturb + preserve shape")


if __name__ == "__main__":
    _demo()
