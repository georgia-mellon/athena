"""Mel-spectrogram features for keystroke windows (the attacker's 'image')."""
import numpy as np
import librosa
from .config import SR, N_MELS, N_FFT, HOP


def mel(windows: np.ndarray, sr: int = SR) -> np.ndarray:
    """(n, samples) -> (n, N_MELS, frames) log-mel, per-window normalized."""
    out = []
    for w in windows:
        m = librosa.feature.melspectrogram(
            y=w, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
        )
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.mean()) / (m.std() + 1e-6)
        out.append(m.astype(np.float32))
    return np.stack(out) if out else np.zeros((0, N_MELS, 1), np.float32)
