"""Synthetic speech + keystroke mixtures for training/evaluating the shield.

Clean speech (librosa's bundled LibriSpeech clips) is the PESQ/STOI reference;
we drop recorded keystrokes onto it at a target SNR and record exact onset
sample indices + the typed string. Reuses audio/segment/config, no new deps.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from . import audio, segment
from .config import SR, DATA, PRE_S, CLASSES

KEY_ROOT = DATA / "harrison" / "MBPWavs"
SPEECH_DIR = DATA / "speech"
SPEECH_EXAMPLES = ["libri1", "libri2", "libri3"]  # all librosa ships
CLIP_S = 4.0            # slice long libri clips into ~4s pieces -> ~10 clips
N_PRESSES = 25          # presses per Harrison per-key WAV
_PRE = int(PRE_S * SR)  # window pre-roll: onset sits _PRE samples into a window


def fetch_speech(out_dir: Path = SPEECH_DIR) -> list[Path]:
    """Download the 3 bundled libri clips, slice into ~CLIP_S pieces, cache as
    WAV. Idempotent: returns cached clips if already present."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(out_dir.glob("*.wav"))
    if cached:
        return cached
    step = int(CLIP_S * SR)
    written: list[Path] = []
    for name in SPEECH_EXAMPLES:
        y = audio.load(librosa.example(name))  # auto-downloads to librosa cache
        for j in range(len(y) // step):
            fp = out_dir / f"{name}_{j}.wav"
            sf.write(fp, y[j * step:(j + 1) * step], SR)
            written.append(fp)
    return written


def _one_press(key: str, root: Path, rng: np.random.Generator) -> np.ndarray:
    """Grab one random press window of `key` from the Harrison recording."""
    y = audio.load(root / f"{key}.wav")
    on = segment.onsets_n(y, N_PRESSES)
    if len(on) == 0:
        raise ValueError(f"no onsets detected in {key}.wav")
    pick = on[rng.integers(len(on)):][:1]
    return segment.windows(y, pick)[0]  # (KEY_WIN,) float32


def mix(speech: np.ndarray, keystroke_root=KEY_ROOT, n_keys: int = 5,
        snr_db: float = 5.0, seed: int = 0) -> dict:
    """Overlay n_keys random keystrokes onto `speech` at `snr_db` (keystroke
    power relative to speech power). Returns mix/clean/onsets/text, with onsets
    and text ordered by placement time (i.e. realistic typing order)."""
    root = Path(keystroke_root)
    if not 1 <= n_keys <= len(CLASSES):
        raise ValueError(f"n_keys must be in 1..{len(CLASSES)}, got {n_keys}")
    rng = np.random.default_rng(seed)
    clean = np.asarray(speech, dtype=np.float32).copy()
    mixed = clean.copy()
    p_speech = float(np.mean(clean ** 2)) + 1e-12

    keys = rng.choice(CLASSES, size=n_keys, replace=False)
    placed = []  # (onset_idx, key)
    for k in keys:
        w = _one_press(k, root, rng)
        p_key = float(np.mean(w ** 2)) + 1e-12
        scale = np.sqrt((p_speech / (10 ** (snr_db / 10))) / p_key)
        w = (w * scale).astype(np.float32)
        room = len(clean) - len(w)
        if room <= 0:
            raise ValueError("speech clip shorter than a keystroke window")
        start = int(rng.integers(0, room))
        mixed[start:start + len(w)] += w
        placed.append((start + _PRE, k))  # onset is _PRE into the window

    placed.sort()
    onsets = np.array([o for o, _ in placed], dtype=int)
    text = "".join(k for _, k in placed)
    return {"mix": mixed, "clean": clean, "onsets": onsets, "text": text}


def make_dataset(n_clips: int = 20, out_dir="data/synth", seed: int = 0) -> list[dict]:
    """Write mixture + clean WAVs and a JSON manifest. Each manifest entry:
    {mix_wav, clean_wav, onsets:[int...], text:str, snr_db:float, n_keys:int}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    speech = [audio.load(p) for p in fetch_speech()]
    if not speech:
        raise RuntimeError("no speech clips available")
    rng = np.random.default_rng(seed)
    manifest = []
    for i in range(n_clips):
        sp = speech[i % len(speech)]
        n_keys = int(rng.integers(3, 9))          # 3..8 keystrokes
        snr_db = float(rng.uniform(-5.0, 15.0))
        r = mix(sp, KEY_ROOT, n_keys, snr_db, seed=seed + i)
        mix_wav = out / f"mix_{i:03d}.wav"
        clean_wav = out / f"clean_{i:03d}.wav"
        sf.write(mix_wav, r["mix"], SR)
        sf.write(clean_wav, r["clean"], SR)
        manifest.append({
            "mix_wav": str(mix_wav),
            "clean_wav": str(clean_wav),
            "onsets": r["onsets"].tolist(),
            "text": r["text"],
            "snr_db": round(snr_db, 2),
            "n_keys": n_keys,
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def demo() -> None:
    clips = fetch_speech()
    print(f"speech clips cached in {SPEECH_DIR}: {len(clips)}")
    sp = audio.load(clips[0])
    n_keys = 5
    r = mix(sp, KEY_ROOT, n_keys=n_keys, snr_db=5.0, seed=0)
    assert len(r["onsets"]) == n_keys, (len(r["onsets"]), n_keys)
    assert len(r["mix"]) == len(r["clean"]) == len(sp), "length mismatch"
    assert (np.diff(r["onsets"]) >= 0).all(), "onsets not time-ordered"
    print(f"mix {r['mix'].shape} clean {r['clean'].shape} "
          f"onsets {r['onsets']} text {r['text']!r}")
    print("self-check OK")


if __name__ == "__main__":
    demo()
