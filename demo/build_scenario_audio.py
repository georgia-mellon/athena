"""Build demo/audio/ai_caller/{far_end.wav, mic.wav, keys.csv} for demo/scenarios/ai_caller.toml (deterministic).

Sources (all read-only):
- far end, real colleague: Hearsay test_internal bonafide LibriSpeech, one speaker (COLLEAGUE).
- far end, AI agent: Hearsay test_internal DiffSSD ElevenLabs clone of that same LibriSpeech speaker (the story:
  the agent clones a colleague's voice). Both loudness-normalised to about -23 dBFS RMS so level gives nothing away.
- mic keystrokes: Keyguard harrison presses from the TEST split of harrison_split() only (the provisional attacker
  trained on the train split). Window start = onset - PRE_S so the onset lands on the logged key time.
- mic speech: another LibriSpeech speaker (LOCAL), +10 dB over the key-window power, with gaps; -60 dBFS noise floor.

Usage: .venv\\Scripts\\python demo\\build_scenario_audio.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # Hearsay / Keyguard are read-only: no __pycache__ there

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from callguard.audio.vad import speech_fraction  # noqa: E402
from callguard.drivers.keyguard_real import harrison_split  # noqa: E402
from callguard.types import SR  # noqa: E402

HEARSAY_ROOT = Path(os.environ.get("HEARSAY_ROOT") or REPO.parent / "Hearsay")
OUT = REPO / "demo" / "audio" / "ai_caller"
SEED = 4821
TOTAL = 60.0
SEGMENTS = [("colleague", 0.0, 12.0, "real"), ("ai_agent", 12.0, 40.0, "synthetic"), ("colleague_returns", 40.0, 60.0, "real")]
COLLEAGUE = "100"                         # LibriSpeech speaker id (the agent clones this voice)
AGENT = ("diffssd", "elevenlabs", "librispeech:100")
LOCAL = "2803"                            # the user at the keyboard
CODE = "RESET4821"                        # fake code only
TYPING = [22.0, 29.5]                     # shield off at the first, on (28 s) at the second
PRE_S = 0.02                              # keyguard.config.PRE_S
FAR_DBFS, SPEECH_OVER_KEYS_DB, NOISE_DBFS, GAP = -23.0, 10.0, -60.0, 0.3
LOCAL_DBFS = -28.0                        # user's speech at the mic; keys scaled to sit 10 dB under it
# (harrison presses are loud, ~-22 dBFS window RMS; KeyNet's log-mel is max-referenced and standardised, so gain is
# irrelevant to the attacker, only the speech/key ratio and the noise floor matter)


def db_gain(x: np.ndarray, dbfs: float) -> np.ndarray:
    return x * (10 ** (dbfs / 20) / (np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def trim(x: np.ndarray, below_db: float = 35.0) -> np.ndarray:
    """Drop leading/trailing 20 ms frames more than `below_db` under the loudest frame."""
    n = len(x) // 320
    e = 10 * np.log10(np.mean(x[:n * 320].reshape(n, 320).astype(np.float64) ** 2, axis=1) + 1e-12)
    on = np.flatnonzero(e > e.max() - below_db)
    return x[on[0] * 320:(on[-1] + 1) * 320]


def clips(rows: pd.DataFrame, rng) -> list[np.ndarray]:
    out = []
    for p in rows.path.iloc[rng.permutation(len(rows))]:
        x, sr = sf.read(HEARSAY_ROOT / p, dtype="float32", always_2d=True)
        assert sr == SR, (p, sr)
        out.append(trim(x.mean(1)))
    return out


def fill(pool: list[np.ndarray], seconds: float, dbfs: float, gaps=(GAP, GAP), rng=None) -> tuple[np.ndarray, int]:
    """Concatenate clips (each normalised to `dbfs`) with gaps until `seconds` is full; cut the last clip.
    Returns (audio, clips used)."""
    n, parts, have = round(seconds * SR), [], 0
    for x in pool:
        if have >= n:
            break
        g = np.zeros(round((rng.uniform(*gaps) if rng is not None else gaps[0]) * SR), np.float32)
        parts += [db_gain(x, dbfs).astype(np.float32), g]
        have += len(x) + len(g)
    assert have >= n, f"not enough audio for {seconds} s"
    return np.concatenate(parts)[:n], len(parts) // 2


def main() -> None:
    rng = np.random.default_rng(SEED)
    m = pd.read_parquet(HEARSAY_ROOT / "data" / "processed" / "manifest.parquet")
    t = m[m.split == "test_internal"]
    libri = t[(t.label == "bonafide") & (t.source == "librispeech")]
    agent_rows = t[(t.label == "spoof") & (t.source == AGENT[0]) & (t.generator == AGENT[1]) & (t.speaker == AGENT[2])]

    # far end: colleague clips shared between both real segments (no repeats), agent clips from one cloned voice
    col = clips(libri[libri.speaker.astype(str) == COLLEAGUE], rng)
    agent = clips(agent_rows, rng)
    far = []
    for name, a, b, voice in SEGMENTS:
        if voice == "real":
            seg, used = fill(col, b - a, FAR_DBFS)
            col = col[used:]                            # the colleague's second visit uses fresh clips
        else:
            seg, _ = fill(agent, b - a, FAR_DBFS)
        far.append(seg)
    far = np.concatenate(far)

    # keystrokes: test-split presses only, each press used once
    _, _, Xte, yte = harrison_split()
    from keyguard.config import CLS_IDX
    pools = {k: list(rng.permutation(np.flatnonzero(yte == CLS_IDX[k]))) for k in sorted(set(CODE))}  # sorted: set order is per-process
    n = round(TOTAL * SR)
    keys_track, key_log, used_wins = np.zeros(n, np.float32), [], []
    for t0 in TYPING:
        s = round(t0 * SR)
        for ch in CODE:
            w = Xte[pools[ch].pop()]
            a = s - round(PRE_S * SR)
            keys_track[a:a + len(w)] += w
            key_log.append((s / SR, ch))
            used_wins.append(w)
            s += round(rng.uniform(0.45, 0.6) * SR)
    key_pow = float(np.mean([np.mean(w.astype(np.float64) ** 2) for w in used_wins]))
    key_dbfs = LOCAL_DBFS - SPEECH_OVER_KEYS_DB
    keys_track *= np.float32(10 ** (key_dbfs / 20) / np.sqrt(key_pow))

    # local speech: another speaker, +10 dB over the key windows, with conversational gaps
    local = clips(libri[libri.speaker.astype(str) == LOCAL], rng)
    speech, _ = fill(local, TOTAL, key_dbfs + SPEECH_OVER_KEYS_DB, gaps=(0.6, 2.5), rng=rng)
    noise = db_gain(rng.standard_normal(n), NOISE_DBFS).astype(np.float32)
    mic = speech + keys_track + noise

    OUT.mkdir(parents=True, exist_ok=True)
    for name, x in (("far_end", far), ("mic", mic)):
        peak = float(np.abs(x).max())
        if peak > 0.99:
            print(f"WARNING {name} peak {peak:.2f}; scaled down")
            x = x * (0.99 / peak)
        sf.write(OUT / f"{name}.wav", x.astype(np.float32), SR, subtype="FLOAT")
    with open(OUT / "keys.csv", "w", newline="") as f:
        f.write("t_seconds,key\n" + "".join(f"{t:.6f},{k}\n" for t, k in key_log))

    # report
    print(f"colleague : librispeech speaker {COLLEAGUE} (bonafide, test_internal)")
    print(f"agent     : source={AGENT[0]} generator={AGENT[1]} speaker={AGENT[2]} (spoof, test_internal; "
          f"{len(agent_rows)} clips)")
    print(f"local mic : librispeech speaker {LOCAL}; keys = harrison TEST split, {len(key_log)} presses of {CODE} x2")
    print(f"key-window power {key_dbfs:.1f} dBFS, local speech {key_dbfs + SPEECH_OVER_KEYS_DB:.1f} dBFS "
          f"(+{SPEECH_OVER_KEYS_DB:.0f} dB), noise {NOISE_DBFS:.0f} dBFS, mic peak {np.abs(mic).max():.2f}")
    ends = np.arange(4 * SR, len(far) + 1, 2 * SR)          # the pipeline's scored windows: 4 s, hop 2 s
    frac = {e: speech_fraction(far[e - 4 * SR:e]) for e in ends}
    print(f"{'segment':15s} {'dur':>5s} {'RMS dBFS':>8s} {'VAD min':>7s} {'VAD med':>7s}  windows (end s: frac)")
    for name, a, b, _ in SEGMENTS:
        x = far[round(a * SR):round(b * SR)]
        inside = [e for e in ends if e - 4 * SR >= a * SR and e <= b * SR]
        fr = [frac[e] for e in inside]
        rms = 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        print(f"{name:15s} {b - a:5.1f} {rms:8.1f} {min(fr):7.2f} {np.median(fr):7.2f}  "
              + " ".join(f"{e / SR:.0f}:{frac[e]:.2f}" for e in inside))
    low = [e / SR for e in ends if frac[e] < 0.5]
    print("windows below 0.5 speech:", low or "none")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
