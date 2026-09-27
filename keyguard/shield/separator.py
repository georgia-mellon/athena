"""Causal neural keystroke separator, conditioned on key-event timestamps.

The DSP shield (shield.py) can't cleanly pull a broadband click out of speech
that occupies the same time-frequency cells, so it damages speech. This is the
learned replacement: a causal, FiLM-conditioned temporal U-Net over the STFT
magnitude that predicts a suppression mask. It sees only past frames (real-time
safe) and is *told where the keystrokes are* via a per-frame key mask derived
from OS key timestamps -- so it can spend capacity separating exactly at those
frames while leaving clean speech untouched.

Architecture:
  mix log-magnitude (F=n_fft/2+1 channels, T frames)
    -> input 1x1 conv projection
    -> encoder: dilated causal Conv1d + GLU + FiLM(key mask) residual blocks
       (dilations 1,2,4,8 give a large causal receptive field)
    -> decoder: mirrored blocks with U-Net skip connections
    -> 1x1 conv + sigmoid -> mask in [0,1]
  output audio = ISTFT(mask * mix_mag, mix_phase), then optional per-stroke
  residue randomization (spec: randomize each stroke's residue).

FiLM conditioning lets the key mask modulate every block (per-channel scale +
shift), which is what makes it *conditioned on timestamps* rather than a blind
denoiser. Trained to reconstruct clean speech magnitude while zeroing keystroke
energy (train_separator.py).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from ..config import SR, N_FFT, HOP
from .. import segment

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FREQ = N_FFT // 2 + 1           # 513


def _causal_pad(x, pad):
    """Left-pad time so a conv sees only past frames."""
    return F.pad(x, (pad, 0))


class FiLM(nn.Module):
    """Per-channel scale+shift from the key-mask conditioning vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.to_gamma = nn.Conv1d(cond_dim, channels, 1)
        self.to_beta = nn.Conv1d(cond_dim, channels, 1)

    def forward(self, x, cond):     # x:(B,C,T)  cond:(B,cond_dim,T)
        return x * (1 + self.to_gamma(cond)) + self.to_beta(cond)


class ChannelNorm(nn.Module):
    """Per-frame LayerNorm over channels only. Causal: each time step is
    normalized independently, so no future frame ever touches the past (unlike
    GroupNorm/InstanceNorm, which normalize over time and break causality)."""

    def __init__(self, ch):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, ch, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x):            # x:(B,C,T)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.g + self.b


class CausalBlock(nn.Module):
    def __init__(self, ch, cond_dim, dilation, k=3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(ch, 2 * ch, k, dilation=dilation)   # GLU -> 2*ch
        self.film = FiLM(cond_dim, ch)
        self.norm = ChannelNorm(ch)   # causal-safe: stabilizes without time mixing

    def forward(self, x, cond):
        y = _causal_pad(x, self.pad)
        y = F.glu(self.conv(y), dim=1)
        y = self.film(y, cond)
        return self.norm(x + y)      # residual


class SepUNet(nn.Module):
    def __init__(self, freq=FREQ, ch=192, cond_dim=32, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv1d(freq, ch, 1)
        self.cond = nn.Sequential(nn.Conv1d(1, cond_dim, 1), nn.ReLU(),
                                  nn.Conv1d(cond_dim, cond_dim, 1))
        self.enc = nn.ModuleList([CausalBlock(ch, cond_dim, d) for d in dilations])
        self.dec = nn.ModuleList([CausalBlock(ch, cond_dim, d)
                                  for d in reversed(dilations)])
        self.skip = nn.ModuleList([nn.Conv1d(2 * ch, ch, 1) for _ in dilations])
        self.out = nn.Conv1d(ch, freq, 1)
        # start the mask near 0.5 (pre-sigmoid ~0): a large-magnitude init saturates
        # the sigmoid, its gradient dies, and the net freezes at pass-through.
        nn.init.normal_(self.out.weight, 0.0, 0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, logmag, keymask):     # logmag:(B,F,T) keymask:(B,1,T)
        cond = self.cond(keymask)
        x = self.inp(logmag)
        skips = []
        for blk in self.enc:
            x = blk(x, cond)
            skips.append(x)
        for blk, sk, merge in zip(self.dec, reversed(skips), self.skip):
            x = merge(torch.cat([x, sk], dim=1))
            x = blk(x, cond)
        return torch.sigmoid(self.out(x))    # mask (B,F,T)


class NeuralShield:
    """Drop-in replacement for shield.Shield: .apply(y, onsets)->y."""

    def __init__(self, weights="runs/separator.pt", randomize=0.0, key_frames=6,
                 seed=None):
        self.net = SepUNet().to(DEVICE)
        self.randomize = randomize
        self.key_frames = key_frames
        self.rng = np.random.default_rng(seed)
        self._loaded = False
        if weights:
            try:
                self.net.load_state_dict(torch.load(weights, map_location=DEVICE))
                self.net.eval()
                self._loaded = True
            except Exception:
                pass

    def key_mask(self, onsets, n_frames):
        m = np.zeros(n_frames, np.float32)
        for o in onsets:
            c = int(o / HOP)
            m[max(0, c - 2):min(n_frames, c + self.key_frames)] = 1.0
        return m

    @torch.no_grad()
    def apply(self, y, onsets=None):
        if onsets is None:
            onsets = segment.onsets(y)
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
        mag, phase = np.abs(S), np.angle(S)
        logmag = np.log1p(mag)
        km = self.key_mask(onsets, mag.shape[1])
        lm = torch.from_numpy(logmag[None]).float().to(DEVICE)
        kt = torch.from_numpy(km[None, None]).float().to(DEVICE)
        mask = self.net(lm, kt)[0].cpu().numpy() if self._loaded \
            else np.ones_like(mag)
        out_mag = mag * mask
        if self.randomize > 0:
            for c in np.where(km > 0)[0]:
                g = 1.0 + self.randomize * (self.rng.random(mag.shape[0]) - 0.5)
                out_mag[:, c] *= np.maximum(g, 0)
        out = librosa.istft(out_mag * np.exp(1j * phase), hop_length=HOP,
                            length=len(y))
        return out.astype(np.float32)


def demo():
    """Self-check: forward pass is causal (future frames don't change the past)."""
    net = SepUNet().eval()
    T = 40
    lm = torch.randn(1, FREQ, T)
    km = torch.zeros(1, 1, T)
    with torch.no_grad():
        a = net(lm, km)
        lm2 = lm.clone(); lm2[..., T // 2:] += 5.0     # perturb the future only
        b = net(lm2, km)
    diff_past = (a[..., :T // 2] - b[..., :T // 2]).abs().max().item()
    assert diff_past < 1e-4, f"not causal: past changed by {diff_past}"
    assert a.shape == (1, FREQ, T)
    print(f"separator demo ok: causal (past unchanged by future: {diff_past:.2e}), "
          f"mask shape {tuple(a.shape)}, params "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M")


if __name__ == "__main__":
    demo()
