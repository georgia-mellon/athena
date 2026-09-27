"""Overlap-robust segment-and-classify attacker (the "reads fast free typing" agent).

Why: the isolated-key KeyNet (runs/supervised_mbp.pt) was trained on clean 0.30 s
windows around a lone keypress, so on CONTINUOUS typing every window is polluted by
2-6 neighbouring clicks and it collapses (~85-110% CER, see overlap_eval). The CTC
transcriber also mode-collapses. Fix, from first principles:

  1. Train the per-key CNN on SHORT windows cut at onsets of OVERLAPPING synth typing
     (train distribution == test distribution). A short window (~0.14 s) is dominated
     by the target key's onset transient; trailing neighbours become learnable noise.
  2. Decode continuous audio: tuned onset detection -> short windows -> KeyNet per
     onset -> key string. (LM correction is a separate downstream step; here we report
     the raw acoustic CER so improvements are attributable to the acoustic model.)

Scored by sequence CER against ground truth from the overlap engine, across overlap
levels, alongside the isolated-KeyNet and CTC baselines.

Train+eval:  KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.ctc.seg_attack
Eval only:   KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.ctc.seg_attack eval
Self-check:  KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.ctc.seg_attack demo
"""
from __future__ import annotations
import os
import sys

import numpy as np
import librosa
from scipy.signal import find_peaks

from ..config import SR, HOP, RUNS, ONSET_PROM
from ..features import mel
from ..attackers.supervised import SupervisedAttacker
from .data import synth_line, random_text, CLIP
from .overlap_eval import cer_str, _match, _ids_to_keys, SPEED_BINS

BANK = os.environ.get("KEYGUARD_BANK", "data/harrison/MBPWavs")
CKPT = os.environ.get("KEYGUARD_SEG_CKPT", "runs/seg_overlap.pt")
WIN_S = float(os.environ.get("KEYGUARD_SEG_WIN", "0.14"))   # short: onset-transient dominated
WIN = int(WIN_S * SR)
PRE = int(0.01 * SR)                                        # 10 ms pre-roll
MIN_GAP_S = float(os.environ.get("KEYGUARD_SEG_GAP", "0.03"))  # tight: dense typing
TRAIN_LINES = int(os.environ.get("KEYGUARD_SEG_LINES", "1200"))
EPOCHS = int(os.environ.get("KEYGUARD_SEG_EPOCHS", "40"))
EVAL_LINES = 16


def _cut(y: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Fixed WIN windows starting PRE before each onset, zero-padded."""
    out = np.zeros((len(onsets), WIN), np.float32)
    for i, o in enumerate(onsets):
        a = max(0, o - PRE)
        seg = y[a:a + WIN]
        out[i, :len(seg)] = seg
    return out


def detect_onsets(y: np.ndarray, min_gap_s: float = MIN_GAP_S, prom: float = ONSET_PROM) -> np.ndarray:
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    if env.max() <= 0:
        return np.array([], int)
    env = env / env.max()
    peaks, _ = find_peaks(env, height=prom, distance=max(1, int(min_gap_s * SR / HOP)))
    return (peaks * HOP).astype(int)


def build_train(n_lines: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Overlap synth -> (window, key-idx) pairs cut at TRUE onsets."""
    rng = np.random.default_rng(seed)
    wins, labs = [], []
    for _ in range(n_lines):
        y, lab_ids, on = synth_line(random_text(rng), rng, wpm=(150, 580), root=BANK)
        if len(lab_ids) == 0:
            continue
        w = _cut(y, on)
        w = w * rng.uniform(0.8, 1.2)                       # gain jitter
        w = w + 0.003 * rng.standard_normal(w.shape).astype(np.float32)
        wins.append(w)
        labs.extend(int(i) - 1 for i in lab_ids)            # CTC sym (1..36) -> class (0..35)
    return np.concatenate(wins), np.array(labs)


def train() -> SupervisedAttacker:
    print(f"building overlap train set ({TRAIN_LINES} lines, win={WIN_S*1000:.0f}ms)...", flush=True)
    W, y = build_train(TRAIN_LINES)
    print(f"  {len(y)} keystroke windows, {len(set(y.tolist()))} classes; mel + train {EPOCHS} ep", flush=True)
    atk = SupervisedAttacker()
    atk.fit(mel(W), y, epochs=EPOCHS, log=lambda e, l: print(f"  ep {e} loss {l:.3f}", flush=True))
    atk.save(CKPT)
    print(f"saved {CKPT}", flush=True)
    return atk


def decode(y: np.ndarray, atk: SupervisedAttacker, min_gap_s: float = MIN_GAP_S) -> tuple[str, np.ndarray]:
    det = detect_onsets(y, min_gap_s)
    if len(det) == 0:
        return "", det
    return "".join(atk.predict(mel(_cut(y, det)))), det


def evaluate(atk: SupervisedAttacker, seed: int = 100, lines: int = EVAL_LINES) -> dict:
    rows = []
    for lo, hi in SPEED_BINS:
        rng = np.random.default_rng(seed + lo)
        ov, ce, pr, rc = [], [], [], []
        for _ in range(lines):
            y, lab_ids, on = synth_line(random_text(rng), rng, wpm=(lo, hi), root=BANK)
            if len(lab_ids) == 0:
                continue
            gaps = np.diff(on)
            ov.append(float(np.mean(gaps < CLIP)) if len(gaps) else 0.0)
            hyp, det = decode(y, atk)
            ce.append(cer_str(_ids_to_keys(list(lab_ids)), hyp))
            p, r = _match(det, on)
            pr.append(p); rc.append(r)
        rows.append({"wpm": (lo + hi) // 2, "overlap": float(np.mean(ov)),
                     "seg_cer": float(np.mean(ce)),
                     "onset_precision": float(np.mean(pr)), "onset_recall": float(np.mean(rc))})
    return {"bank": BANK, "win_s": WIN_S, "min_gap_s": MIN_GAP_S, "rows": rows}


def _print(r: dict, title: str) -> None:
    print(f"\n{title}  bank={r['bank']}  win={r['win_s']*1000:.0f}ms  gap={r['min_gap_s']*1000:.0f}ms"
          f"  (CER lower=better, chance~1.0)")
    print(f"{'wpm':>5}{'overlap%':>10}{'SEG CER':>9}{'onset_P':>9}{'onset_R':>9}")
    for x in r["rows"]:
        print(f"{x['wpm']:>5}{x['overlap']:>9.0%}{x['seg_cer']:>9.1%}"
              f"{x['onset_precision']:>9.1%}{x['onset_recall']:>9.1%}")
    print(f"  MEAN CER {np.mean([x['seg_cer'] for x in r['rows']]):.1%}  "
          f"MEAN onset_R {np.mean([x['onset_recall'] for x in r['rows']]):.1%}")


def run() -> dict:
    import json
    atk = train()
    r = evaluate(atk)
    _print(r, "OVERLAP-TRAINED SEG ATTACKER")
    (RUNS / "seg_overlap_eval.json").write_text(json.dumps(r, indent=2))
    print(f"\nwrote {RUNS/'seg_overlap_eval.json'}")
    return r


def demo() -> None:
    W, y = build_train(6)
    assert W.shape[1] == WIN and len(W) == len(y) and set(y.tolist()) <= set(range(36))
    m = mel(W)
    assert m.ndim == 3 and m.shape[0] == len(y)
    print(f"seg_attack demo ok: {len(y)} windows shape {W.shape}, mel {m.shape}, "
          f"win={WIN_S*1000:.0f}ms -- full run trains + evals.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "demo":
        demo()
    elif arg == "eval":
        _print(evaluate(SupervisedAttacker().load(CKPT)), "OVERLAP-TRAINED SEG ATTACKER (eval)")
    else:
        run()
