"""On-the-fly acoustic augmentation for cross-device keystroke robustness.

Research lever: DECKER-style Acoustic Style Randomization (random IIR EQ +
envelope perturbation) cheaply simulates "a different keyboard/mic" so the
recognizer generalizes across devices without a bigger model. Each aug is
independently toggleable and seeded by an np.random.Generator (reproducible).

Training pipeline:
    mel = augment_logmel(logmel(augment_waveform(y, rng)), rng)

All functions return NEW arrays; inputs are never mutated. numpy + scipy only.
"""
from __future__ import annotations
import glob
import numpy as np
from scipy import signal
from scipy.io import wavfile
from ..config import SR, DATA

# --- knobs (named, no magic numbers) ---
STYLE_N_FILTERS = (2, 4)          # random peaking/shelving stages per pass
STYLE_GAIN_DB = (-9.0, 9.0)       # per-filter EQ gain range
STYLE_Q = (0.5, 3.0)              # filter sharpness
STYLE_ENV_GAMMA = (0.85, 1.15)    # envelope power perturbation
SPEC_TIME_SHIFT = 0.40            # +/- fraction of frames
SPEC_MASK_FRAC = 0.10             # per-mask width as fraction of axis
SPEC_MAX_AREA = 0.20              # cap on total masked area fraction
_NOISE_CACHE: list[np.ndarray] | None = None


def _biquad_peaking(f0: float, q: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
    """RBJ peaking-EQ biquad. f0 normalized to Nyquist (0..1)."""
    a = 10 ** (gain_db / 40.0)
    w0 = np.pi * np.clip(f0, 1e-3, 0.999)
    alpha = np.sin(w0) / (2 * max(q, 1e-3))
    cw = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cw, 1 - alpha * a])
    aa = np.array([1 + alpha / a, -2 * cw, 1 - alpha / a])
    return b / aa[0], aa / aa[0]


def style_randomize(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random IIR EQ (a few peaking biquads) + mild envelope power perturbation.

    Core cross-device trick: reshapes the spectral tilt / resonances the way a
    different keyboard body or microphone would, without changing timing.
    """
    out = y.astype(np.float32, copy=True)
    n = int(rng.integers(STYLE_N_FILTERS[0], STYLE_N_FILTERS[1] + 1))
    for _ in range(n):
        f0 = float(rng.uniform(0.02, 0.9))
        q = float(rng.uniform(*STYLE_Q))
        gain = float(rng.uniform(*STYLE_GAIN_DB))
        b, a = _biquad_peaking(f0, q, gain)
        out = signal.lfilter(b, a, out).astype(np.float32)
    # envelope perturbation: raise the analytic amplitude to a random power
    env = np.abs(signal.hilbert(out)) + 1e-6
    gamma = float(rng.uniform(*STYLE_ENV_GAMMA))
    out = out * (env ** (gamma - 1.0)).astype(np.float32)
    return _renorm(out, y)


def _renorm(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match peak level of ref so downstream gain/SNR stay meaningful."""
    peak = np.max(np.abs(out))
    if peak < 1e-9:
        return out.astype(np.float32)
    return (out * (np.max(np.abs(ref)) + 1e-9) / peak).astype(np.float32)


def _load_noise() -> list[np.ndarray]:
    global _NOISE_CACHE
    if _NOISE_CACHE is None:
        _NOISE_CACHE = []
        for p in sorted(glob.glob(str(DATA / "noise" / "*.wav"))):
            try:
                _, w = wavfile.read(p)
                w = w.astype(np.float32)
                if w.ndim > 1:
                    w = w.mean(axis=1)
                _NOISE_CACHE.append(w / (np.max(np.abs(w)) + 1e-9))
            except Exception:
                continue  # bad file -> skip, Gaussian still covers us
    return _NOISE_CACHE


def add_noise(y: np.ndarray, rng: np.random.Generator,
              snr_db: tuple[float, float] = (5.0, 20.0)) -> np.ndarray:
    """Additive noise at a random SNR. Gaussian by default; samples a real clip
    from data/noise/*.wav when present (external datasets NOT required)."""
    sig_p = float(np.mean(y ** 2)) + 1e-12
    snr = float(rng.uniform(*snr_db))
    target_p = sig_p / (10 ** (snr / 10.0))
    clips = _load_noise()
    if clips and rng.random() < 0.5:
        c = clips[int(rng.integers(len(clips)))]
        if len(c) < len(y):
            c = np.tile(c, int(np.ceil(len(y) / len(c))))
        start = int(rng.integers(0, len(c) - len(y) + 1))
        noise = c[start:start + len(y)]
    else:
        noise = rng.standard_normal(len(y)).astype(np.float32)
    noise = noise * np.sqrt(target_p / (np.mean(noise ** 2) + 1e-12))
    return (y + noise).astype(np.float32)


def random_gain(y: np.ndarray, rng: np.random.Generator,
                db: tuple[float, float] = (-12.0, 12.0)) -> np.ndarray:
    """Scale by a random dB gain (mic/level mismatch)."""
    g = 10 ** (float(rng.uniform(*db)) / 20.0)
    return (y * g).astype(np.float32)


def spec_augment(mel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SpecAugment on (T, mel) log-mel: time-shift + time/freq masks (mean-fill),
    total masked area bounded by SPEC_MAX_AREA. Returns a new array, same shape."""
    out = mel.copy()
    t, f = out.shape
    fill = float(out.mean())
    shift = int(rng.integers(-int(t * SPEC_TIME_SHIFT), int(t * SPEC_TIME_SHIFT) + 1))
    out = np.roll(out, shift, axis=0)
    area = 0.0
    for axis, size in ((0, t), (1, f)):
        for _ in range(int(rng.integers(1, 3))):     # 1-2 masks per axis
            w = int(rng.integers(1, max(2, int(size * SPEC_MASK_FRAC)) + 1))
            if (area + w / size) > SPEC_MAX_AREA:
                break
            start = int(rng.integers(0, size - w + 1))
            if axis == 0:
                out[start:start + w, :] = fill
            else:
                out[:, start:start + w] = fill
            area += w / size
    return out.astype(np.float32)


def augment_waveform(y: np.ndarray, rng: np.random.Generator,
                     p: dict[str, float] | None = None) -> np.ndarray:
    """Compose style_randomize -> (RIR) -> add_noise -> random_gain, each applied
    with its own probability. Length-preserving. Defaults are ON for all."""
    prob = {"style": 0.8, "noise": 0.8, "gain": 0.7, **(p or {})}
    out = y.astype(np.float32, copy=True)
    if rng.random() < prob["style"]:
        out = style_randomize(out, rng)
    # TODO: convolutional room impulse response (RIR) reverb goes here.
    #       Plug OpenSLR SLR28 (RIRs) via scipy.signal.fftconvolve when available.
    if rng.random() < prob["noise"]:
        out = add_noise(out, rng)          # TODO: MUSAN (OpenSLR SLR17) noise bank
    if rng.random() < prob["gain"]:
        out = random_gain(out, rng)
    return out


def augment_logmel(mel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Feature-domain augmentation on the (T, mel) log-mel."""
    return spec_augment(mel, rng)


def demo() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(SR).astype(np.float32)
    ya = augment_waveform(y, rng)
    assert ya.shape == y.shape, "waveform length changed"
    assert np.all(np.isfinite(ya)), "non-finite samples"
    assert float(np.mean(np.abs(ya - y))) > 0, "waveform unchanged"

    mel = rng.standard_normal((40, 64)).astype(np.float32)
    ma = spec_augment(mel, rng)
    assert ma.shape == mel.shape, "mel shape changed"
    masked = float(np.mean(ma == float(mel.mean())))
    assert masked <= SPEC_MAX_AREA + 0.05, f"masked {masked:.2f} exceeds cap"

    print(f"augment OK: |dy|={np.mean(np.abs(ya - y)):.4f} "
          f"rms {np.sqrt(np.mean(y**2)):.3f}->{np.sqrt(np.mean(ya**2)):.3f}, "
          f"mel masked~{masked*100:.1f}% shape {ma.shape}, "
          f"noise clips {len(_load_noise())}")


if __name__ == "__main__":
    demo()
