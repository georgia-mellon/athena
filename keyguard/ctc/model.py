"""CRNN + CTC keystroke transcriber (handles overlapping free typing).

log-mel (small hop for time resolution) -> CNN stack (downsamples frequency,
keeps time) -> BiGRU -> linear over 37 symbols (blank + 36 keys). CTC learns the
alignment, so overlapping/adjacent clicks need no hard segmentation.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from ..config import SR
from .data import VOCAB, BLANK

import os
DEVICE = os.environ.get("KEYGUARD_DEVICE") or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")
N_MELS = 64
N_FFT = 512
HOP = 128                    # 8ms hop (Slater used 10ms); halves T vs 64 for MPS speed


def logmel(y: np.ndarray) -> np.ndarray:
    """(samples,) -> (frames, N_MELS) normalized log-mel."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, power=2.0)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m.T.astype(np.float32)             # (T, mel)


class CRNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_sym=len(VOCAB), hidden=192):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                       # halve freq, keep time
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.feat = 128 * (n_mels // 8)
        self.rnn = nn.GRU(self.feat, hidden, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, n_sym)

    def forward(self, x):                    # x: (B, T, mel)
        x = x.unsqueeze(1).transpose(2, 3)   # (B,1,mel,T)
        x = self.cnn(x)                      # (B,128,mel/8,T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)   # (B,T,feat)
        x, _ = self.rnn(x)
        return self.head(x)                  # (B,T,n_sym) logits


class _DilBlock(nn.Module):
    """Dilated Conv1d + GLU + residual (non-causal; offline transcription).
    Conv-only so it runs natively/fast on MPS, unlike GRU."""

    def __init__(self, ch, dilation, k=3):
        super().__init__()
        self.pad = (k - 1) * dilation // 2
        self.conv = nn.Conv1d(ch, 2 * ch, k, padding=self.pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        y = F.glu(self.conv(x), dim=1)
        return F.relu(self.bn(x + y))


class ConvCTC(nn.Module):
    """Conv-only CTC encoder (QuartzNet/Jasper-style): MPS-native, no RNN.
    Same (B,T,mel)->(B,T,n_sym) interface as CRNN."""

    def __init__(self, n_mels=N_MELS, n_sym=len(VOCAB), ch=160,
                 dilations=(1, 2, 4, 8, 16, 1, 2, 4)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.proj = nn.Conv1d(128 * (n_mels // 8), ch, 1)
        self.blocks = nn.ModuleList([_DilBlock(ch, d) for d in dilations])
        self.head = nn.Conv1d(ch, n_sym, 1)
        with torch.no_grad():                # discourage the all-blank CTC collapse
            self.head.bias[BLANK] = -1.0

    def forward(self, x):                    # x: (B,T,mel)
        x = x.unsqueeze(1).transpose(2, 3)   # (B,1,mel,T)
        x = self.cnn(x)                      # (B,128,mel/8,T)
        b, c, f, t = x.shape
        x = x.reshape(b, c * f, t)           # (B, feat, T)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).transpose(1, 2)  # (B,T,n_sym)


def greedy_decode(logits: torch.Tensor) -> list[list[int]]:
    """CTC greedy: argmax, collapse repeats, drop blanks. logits (B,T,S)."""
    ids = logits.argmax(-1).cpu().numpy()
    out = []
    for row in ids:
        seq, prev = [], BLANK
        for s in row:
            if s != prev and s != BLANK:
                seq.append(int(s))
            prev = s
        out.append(seq)
    return out


def ids_to_str(ids):
    return "".join(VOCAB[i] for i in ids)


def cer(ref_ids, hyp_ids) -> float:
    """Character error rate via edit distance on symbol id sequences."""
    from rapidfuzz.distance import Levenshtein
    r, h = ids_to_str(ref_ids), ids_to_str(hyp_ids)
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)


def demo():
    net = CRNN().eval()
    y = np.random.randn(SR).astype(np.float32)
    x = torch.from_numpy(logmel(y))[None]
    with torch.no_grad():
        out = net(x)
    dec = greedy_decode(out)
    assert out.shape[0] == 1 and out.shape[2] == len(VOCAB)
    assert cer([1, 2, 3], [1, 2, 3]) == 0.0 and cer([1, 2, 3], [1, 3]) > 0
    print(f"CRNN ok: mel {tuple(x.shape)} -> logits {tuple(out.shape)}, "
          f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M, "
          f"decoded len {len(dec[0])}")


if __name__ == "__main__":
    demo()
