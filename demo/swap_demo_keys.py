"""Swap the ai_caller demo's keystrokes for Keyguard-bank presses, in place, without Hearsay's dataset.

build_scenario_audio.py now takes keystrokes from Keyguard's per-key bank (the CTC attacker's domain), but rebuilding
needs Hearsay's dataset. A demo built earlier has harrison presses, which the CTC attacker reads at chance. While the
code is typed the builder silences the user's speech (from 0.3 s before the burst to 0.5 s after it), so there the mic
is keys + noise floor only: this rewrites that stretch with bank presses at the same times, keys and key-window level
(KEY_DBFS) plus a fresh noise floor (NOISE_DBFS). far_end.wav and keys.csv are untouched; the original mic is kept as
mic.harrison.wav.

Usage: uv run python demo/swap_demo_keys.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from app.keystroke_guard.driver import keyguard_bank  # noqa: E402
from app.source.types import SR  # noqa: E402

DIR = REPO / "demo" / "audio" / "ai_caller"
KEY_DBFS, NOISE_DBFS, PRE_S = -24.0, -65.0, 0.02      # build_scenario_audio.py's values
MARGIN = (0.25, 0.45)    # inside the builder's silenced span (0.3 s before, 0.5 s after), clear of its 50 ms fades
BURST_GAP = 2.0          # keys further apart than this start a new typing burst
SEED = 4821


def main() -> int:
    orig = DIR / "mic.harrison.wav"
    if not orig.exists():
        (DIR / "mic.wav").rename(orig)
    mic, sr = sf.read(orig, dtype="float32")
    assert sr == SR, sr
    keys = pd.read_csv(DIR / "keys.csv")
    t, k = keys.t_seconds.to_numpy(), keys.key.astype(str).str.upper().to_list()
    bank, rng = keyguard_bank(), np.random.default_rng(SEED)
    pools = {c: list(rng.permutation(len(bank[c]))) for c in sorted(set(k))}
    clips = [bank[c][pools[c].pop()] for c in k]
    gain = 10 ** (KEY_DBFS / 20) / np.sqrt(np.mean([np.mean(c.astype(np.float64) ** 2) for c in clips]))

    starts = np.r_[0, np.flatnonzero(np.diff(t) > BURST_GAP) + 1]
    ends = np.r_[starts[1:], len(t)]
    out = mic.copy()
    for s, e in zip(starts, ends):
        a, b = round((t[s] - MARGIN[0]) * SR), round((t[e - 1] + MARGIN[1]) * SR)
        seg = (10 ** (NOISE_DBFS / 20) * rng.standard_normal(b - a)).astype(np.float32)
        for i in range(s, e):
            o = round((t[i] - PRE_S) * SR) - a
            c = clips[i] * np.float32(gain)
            seg[o:o + len(c)] += c[:len(seg) - o]
        out[a:b] = seg
        print(f"burst {t[s]:.1f}-{t[e - 1]:.1f} s: {''.join(k[s:e])}, replaced {a / SR:.2f}-{b / SR:.2f} s")
    sf.write(DIR / "mic.wav", out, SR, subtype="FLOAT")
    print(f"wrote {DIR / 'mic.wav'} (original: {orig.name}); peak {np.abs(out).max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
