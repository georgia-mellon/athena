"""Convert continuous/free-typing datasets (audio + keystroke-log timestamps,
NOT synced sample-for-sample) into data/continuous/<name>/{*.wav, labels.jsonl}.

ponytail: one source shape today (SKAID: per-participant dir holding one m4a
per task phase, plus a per-participant keylog csv of epoch-ms press/release
rows tagged "Phase 1"/"Phase 2"). Add a second convert_* function only when a
second dataset needs a different shape.

The keylog clock and the audio clock are not the same clock, so we can't just
subtract timestamps. We estimate a single (per-session) offset by brute-force
scoring candidate offsets against the detected click onsets, then snap each
mapped press time to the nearest real onset within SNAP_TOL_S if one is close
enough (task spec: "align to nearest detected onset within ~150ms").

Run: uv run python3 -m keyguard.continuous_convert
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import load
from .config import DATA, SR
from .segment import onsets as detect_onsets


def _load_any(path: Path) -> np.ndarray:
    """librosa/soundfile can't decode m4a directly on this stack; ponytail:
    shell out to ffmpeg (already a runtime dep via audio pipelines) into a
    scratch wav, then reuse the normal 16k-mono loader."""
    if path.suffix.lower() != ".m4a":
        return load(path)
    # Windows: a NamedTemporaryFile stays open, so ffmpeg can't write it (sharing
    # violation). Allocate a closed temp path, let ffmpeg own it, then clean up.
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), tmp_name],
            check=True, capture_output=True,
        )
        return load(Path(tmp_name))
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass

CONT = DATA / "continuous"
MIN_HIT_RATIO = float(os.environ.get("KEYGUARD_MIN_HIT_RATIO", "0.55"))  # drop poorly-aligned sessions
SNAP_TOL_S = 0.15
COARSE_STEP_MS = 10
FINE_STEP_MS = 1
HIT_TOL_MS = 60           # "counts as aligned" tolerance while scoring offsets


def _alnum_key(raw: str) -> str | None:
    return raw.upper() if len(raw) == 1 and raw.isalnum() else None


def _press_events(csv_path: Path, phase: str) -> list[tuple[int, str]]:
    """[(timestamp_ms, KEY)] for A-Z0-9 presses in one phase, typed order."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["Phase"] != phase or r["Event"] != "press":
            continue
        key = _alnum_key(r["Key"])
        if key is not None:
            # some CSVs got Excel-mangled to scientific notation ("1.75347E+12");
            # float() parses both that and plain integer-ms strings.
            out.append((int(float(r["Timestamp (ms)"])), key))
    return out


def _score_offset(offset_ms: float, press_ms: np.ndarray, onset_ms_sorted: np.ndarray) -> int:
    audio_t = press_ms - offset_ms
    idx = np.clip(np.searchsorted(onset_ms_sorted, audio_t), 1, len(onset_ms_sorted) - 1)
    d = np.minimum(np.abs(onset_ms_sorted[idx] - audio_t), np.abs(onset_ms_sorted[idx - 1] - audio_t))
    return int((d < HIT_TOL_MS).sum())


def _best_offset_ms(press_ms: np.ndarray, onset_ms_sorted: np.ndarray, audio_dur_ms: float) -> float:
    """Brute-force search: recording starts a little before/around the first
    press and roughly spans the phase, so search a generous window around 0."""
    if len(onset_ms_sorted) == 0:
        return float(press_ms.min())
    lo, hi = press_ms.min() - audio_dur_ms, press_ms.min() + audio_dur_ms
    coarse = np.arange(lo, hi, COARSE_STEP_MS)
    scores = [_score_offset(o, press_ms, onset_ms_sorted) for o in coarse]
    best = coarse[int(np.argmax(scores))]
    fine = np.arange(best - COARSE_STEP_MS, best + COARSE_STEP_MS, FINE_STEP_MS)
    fine_scores = [_score_offset(o, press_ms, onset_ms_sorted) for o in fine]
    return float(fine[int(np.argmax(fine_scores))])


def _map_to_samples(press_ms: np.ndarray, offset_ms: float, onset_samp: np.ndarray) -> np.ndarray:
    audio_t_ms = press_ms - offset_ms
    computed = np.round(audio_t_ms / 1000 * SR).astype(int)
    if len(onset_samp) == 0:
        return computed
    tol = int(SNAP_TOL_S * SR)
    idx = np.clip(np.searchsorted(onset_samp, computed), 1, len(onset_samp) - 1)
    d_hi = np.abs(onset_samp[idx] - computed)
    d_lo = np.abs(onset_samp[idx - 1] - computed)
    use_hi = d_hi < d_lo
    nearest = np.where(use_hi, onset_samp[idx], onset_samp[idx - 1])
    nearest_d = np.where(use_hi, d_hi, d_lo)
    return np.where(nearest_d <= tol, nearest, computed)


def convert_skaid(src: Path = DATA / "external/skaid", name: str = "skaid") -> Path:
    rec_root = src / "recordings" / "Participant Recordings"
    log_root = src / "keylogs" / "Keystroke Logs"
    out_dir = CONT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_task = {"Phase 1": "email_1", "Phase 2": "free_form"}
    n_sessions, n_keys = 0, 0
    with open(out_dir / "labels.jsonl", "w") as jf:
        for pdir in sorted(rec_root.iterdir()):
            if not pdir.is_dir():
                continue
            sid = pdir.name
            log_csv = log_root / f"{sid}.csv"
            if not log_csv.exists():
                continue
            for phase, task in phase_task.items():
                m4a = pdir / f"{sid}_{task}.m4a"
                if not m4a.exists():
                    continue
                events = _press_events(log_csv, phase)
                if len(events) < 10:
                    continue
                y = _load_any(m4a)
                press_ms = np.array([t for t, _ in events], dtype=float)
                onset_samp = detect_onsets(y)
                onset_ms_sorted = np.sort(onset_samp / SR * 1000)
                offset = _best_offset_ms(press_ms, onset_ms_sorted, len(y) / SR * 1000)
                samples = _map_to_samples(press_ms, offset, np.sort(onset_samp))
                samples = np.clip(samples, 0, len(y) - 1)
                hits = _score_offset(offset, press_ms, onset_ms_sorted)
                hit_ratio = hits / len(events)
                # Quality gate: a session whose keylog clock won't align to the
                # detected clicks (e.g. Excel-truncated ms timestamps) yields noisy
                # per-key labels -> skip it rather than poison training.
                if hit_ratio < MIN_HIT_RATIO:
                    print(f"{sid}/{task}: SKIP (onset_hits={hits}/{len(events)} "
                          f"= {hit_ratio:.0%} < {MIN_HIT_RATIO:.0%})")
                    continue
                wav_rel = f"{sid}_{task}.wav"
                sf.write(out_dir / wav_rel, y, SR, subtype="PCM_16")
                jf.write(json.dumps({
                    "wav": wav_rel,
                    "keys": "".join(k for _, k in events),
                    "onset_samples": [int(s) for s in samples],
                }) + "\n")
                n_sessions += 1
                n_keys += len(events)
                print(f"{sid}/{task}: {len(events)} keys, offset={offset:.0f}ms, "
                      f"onset_hits={hits}/{len(events)} ({hit_ratio:.0%})")
    print(f"WROTE {out_dir}: {n_sessions} sessions, {n_keys} keystrokes")
    return out_dir


def demo() -> None:
    """ponytail self-check: synthetic clicks + a keylog with a known clock
    offset, verify the recovered onset_samples land within SNAP_TOL_S."""
    rng = np.random.default_rng(0)
    dur_s, n = 20.0, 15
    true_click_s = np.sort(rng.uniform(1, dur_s - 1, n))
    y = np.zeros(int(dur_s * SR), dtype=np.float32)
    for t in true_click_s:
        i = int(t * SR)
        y[i:i + 50] += rng.standard_normal(50).astype(np.float32) * 0.9
    onset_samp = detect_onsets(y)
    assert len(onset_samp) >= n - 2, f"expected ~{n} onsets, detected {len(onset_samp)}"

    true_offset_ms = 3372.0  # keylog clock is ahead of audio-start clock by this much
    press_ms = true_click_s * 1000 + true_offset_ms
    onset_ms_sorted = np.sort(onset_samp / SR * 1000)
    offset = _best_offset_ms(press_ms, onset_ms_sorted, dur_s * 1000)
    assert abs(offset - true_offset_ms) < HIT_TOL_MS, (offset, true_offset_ms)

    samples = _map_to_samples(press_ms, offset, np.sort(onset_samp))
    true_samples = (true_click_s * SR).astype(int)
    err_s = np.abs(samples - true_samples) / SR
    assert (err_s < SNAP_TOL_S).all(), err_s.max()
    print("demo OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
        raise SystemExit(0)
    convert_skaid()
