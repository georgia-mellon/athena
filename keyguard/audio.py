"""Load audio to mono float32 at config.SR."""
import numpy as np
import librosa
from .config import SR


def load(path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)
