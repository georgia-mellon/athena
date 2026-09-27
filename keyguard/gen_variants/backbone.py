"""Backbone = learnable SincNet-style front end straight on the raw 4800-sample
waveform, replacing baseline's log-mel image + 2D CNN.

Hypothesis: log-mel with a single global per-window mean/std norm smooths away the
sub-millisecond push/release transient shape -- the part of a keystroke's acoustic
signature that should be closer to keyboard-invariant -- while still letting KeyNet's
free-form 2D conv filters shortcut on keyboard/mic spectral coloration baked into the
mel image. A first layer of LEARNED band-pass filters applied directly to the audio
(SincNet, Ravanelli & Bengio 2018) is constrained to physically meaningful filters
(each is a difference of two low-pass sincs, so training only moves two cutoff
frequencies per filter instead of every tap), which should track the transient's
band-pass envelope with less freedom to memorize a specific channel response. Same
training loop (AdamW + cosine LR + light aug) as baseline; only the representation
and first layer change.
"""
from __future__ import annotations
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import N_CLASSES, SR

DEVICE = os.environ.get("KEYGUARD_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu")

EPOCHS = 30
FT_EPOCHS = 20
SINC_CH = 24
SINC_KERNEL = 65
SINC_STRIDE = 16
BATCH = 128


def _to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def _to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


class SincConv1d(nn.Module):
    """Learnable band-pass filterbank (Ravanelli & Bengio 2018), mel-initialized.
    Each filter = difference of two low-pass sincs, windowed; only two cutoff
    frequencies per filter are learned instead of every raw tap."""

    def __init__(self, out_channels=SINC_CH, kernel_size=SINC_KERNEL, sr=SR,
                 stride=SINC_STRIDE, min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.sr = sr
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz, high_hz = min_low_hz, sr / 2 - (min_low_hz + min_band_hz)
        mel = np.linspace(_to_mel(low_hz), _to_mel(high_hz), out_channels + 1)
        hz = _to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

        n_lin = torch.linspace(0, kernel_size / 2 - 1, steps=int(kernel_size / 2))
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sr)

    def forward(self, x):                                    # x: (n, 1, samples)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                            self.min_low_hz, self.sr / 2)
        band = (high - low)[:, 0]

        f_t_low = low * self.n_
        f_t_high = high * self.n_
        bp_left = ((torch.sin(f_t_high) - torch.sin(f_t_low)) / (self.n_ / 2)) * self.window_
        bp_center = 2 * band.view(-1, 1)
        bp_right = torch.flip(bp_left, dims=[1])

        bp = torch.cat([bp_left, bp_center, bp_right], dim=1)
        bp = bp / (2 * band[:, None] + 1e-8)
        filt = bp.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filt, stride=self.stride, padding=self.kernel_size // 2)


class ConvBlock1d(nn.Module):
    """ponytail: plain conv-bn-relu-pool, no SE -- keeps the backend cheap enough
    to fit the CPU time budget; the SincConv front end carries the hypothesis."""

    def __init__(self, cin, cout, k=7, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return F.max_pool1d(x, 2) if x.shape[-1] >= 2 else x


class RawNet(nn.Module):
    """SincConv front end -> energy envelope -> compact 1D CNN -> 36-way softmax."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.sinc = SincConv1d()
        self.bn0 = nn.BatchNorm1d(SINC_CH)
        self.b1 = ConvBlock1d(SINC_CH, 32)
        self.b2 = ConvBlock1d(32, 64)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):                                    # x: (n, 1, samples)
        x = torch.abs(self.sinc(x))
        x = F.max_pool1d(x, 4)
        x = F.leaky_relu(self.bn0(x), 0.2)
        x = self.b2(self.b1(x))
        x = F.adaptive_avg_pool1d(x, 1).flatten(1)
        return self.head(self.drop(x))


def _norm(wins: np.ndarray) -> np.ndarray:
    """Per-window DC-remove + peak-normalize (raw-waveform analog of mel's per-window
    mean/std norm; keeps level/gain from being a shortcut feature)."""
    w = wins - wins.mean(axis=1, keepdims=True)
    peak = np.max(np.abs(w), axis=1, keepdims=True) + 1e-6
    return (w / peak).astype(np.float32)


class RawAttacker:
    def __init__(self, n_classes=N_CLASSES):
        self.net = RawNet(n_classes).to(DEVICE)

    def _batch(self, w):
        return torch.from_numpy(w[:, None, :]).float().to(DEVICE)

    def fit(self, w, labels, epochs=EPOCHS, lr=1e-3, bs=BATCH, aug=True):
        X = self._batch(w)
        y = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        n = len(X)
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = X[idx], y[idx]
                if aug:
                    xb = xb + 0.01 * torch.randn_like(xb)
                    xb = torch.roll(xb, shifts=int(torch.randint(-40, 41, (1,))), dims=-1)
                opt.zero_grad()
                loss = F.cross_entropy(self.net(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def predict_proba(self, w) -> np.ndarray:
        if len(w) == 0:
            return np.zeros((0, N_CLASSES), np.float32)
        self.net.eval()
        return F.softmax(self.net(self._batch(w)), dim=1).cpu().numpy()


CAP_PER_DOM = 250     # ponytail: bounds per-fit compute as the pool grows keyboard by
                      # keyboard; raise if CPU budget allows more per-domain data.


def _subsample(wins, y, dom, cap=CAP_PER_DOM, seed=0):
    """Cap samples per domain so training cost stays bounded regardless of how many
    presses a given keyboard contributed, while every domain keeps a voice."""
    rng = np.random.default_rng(seed)
    keep = []
    for d in np.unique(dom):
        idx = np.flatnonzero(dom == d)
        keep.append(rng.choice(idx, cap, replace=False) if len(idx) > cap else idx)
    keep = np.concatenate(keep)
    return wins[keep], y[keep], dom[keep]


class Backbone:
    def __init__(self):
        self.atk = RawAttacker()

    def fit(self, wins, y, dom):
        wins, y, dom = _subsample(wins, y, dom)
        self.atk.fit(_norm(wins), y, epochs=EPOCHS)

    def finetune(self, wins, y):
        self.atk.fit(_norm(wins), y, epochs=FT_EPOCHS, lr=3e-4, bs=32)

    def predict_proba(self, wins):
        return self.atk.predict_proba(_norm(wins))


def make():
    return Backbone()


def _demo():
    """ponytail self-check: shapes + normalization sanity, no training."""
    from ..config import KEY_WIN
    torch.manual_seed(0)
    x = torch.randn(4, 1, KEY_WIN)
    sc = SincConv1d()
    out = sc(x)
    assert out.shape[0] == 4 and out.shape[1] == SINC_CH, out.shape

    net = RawNet()
    y = net(x)
    assert y.shape == (4, N_CLASSES), y.shape

    w = (np.random.randn(3, KEY_WIN).astype(np.float32) * 5) + 3.0
    wn = _norm(w)
    assert np.abs(wn.mean(axis=1)).max() < 1e-3, "should be ~DC-free"
    assert np.allclose(np.abs(wn).max(axis=1), 1.0, atol=1e-3), "should be peak-normalized"
    print("ok: sincnet raw-waveform backbone shapes + normalization check out")


if __name__ == "__main__":
    _demo()
