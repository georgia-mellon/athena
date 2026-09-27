"""Non-Keyguard protection baselines for a fair grid comparison.

- none     : identity
- spectral : RNNoise-style stationary-noise spectral gate (stand-in for the call
  app's built-in suppression / RNNoise; a keystroke is broadband so a stationary
  gate barely dents it -- that's the point of the comparison).
"""
import numpy as np
import librosa
from ..config import N_FFT, HOP


def none(y):
    return y


def spectral_gate(y, reduction_db=12.0):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag, phase = np.abs(S), np.angle(S)
    noise = np.median(mag, axis=1, keepdims=True)
    gain = np.maximum(1 - (noise / (mag + 1e-9)), 10 ** (-reduction_db / 20))
    out = librosa.istft(mag * gain * np.exp(1j * phase), hop_length=HOP, length=len(y))
    return out.astype(np.float32)


PROTECTIONS = {"none": none, "spectral": spectral_gate}
