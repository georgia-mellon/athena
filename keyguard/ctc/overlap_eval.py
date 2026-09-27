"""Rigorous overlap-typing evaluation: can the attacker read CONTINUOUS typing
where keystrokes physically overlap (fast typing over a call)?

Two decoders scored by the SAME metric (sequence CER, edit distance on the key
string) across overlap levels:

  SEG  : blind onset-detect -> cut KEY_WIN windows -> KeyNet classify each
         (reuses the single-key attacker that already works ~87% on Harrison).
         Also reports onset precision/recall so we see WHERE it breaks.
  CTC  : the ConvCTC transcriber (runs/ctc.pt), greedy decode.

Synth lines come from the overlap engine (ctc/data.synth_line), which returns the
audio, the true key-id sequence, and the true onset samples -- exact ground truth.
Chance CER ~= 1.0 (random string). Lower is better.

Run:  KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.ctc.overlap_eval
Fast: KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.ctc.overlap_eval demo
"""
from __future__ import annotations
import os
import sys

import numpy as np

from ..config import SR, CLASSES, HOP, ONSET_MIN_GAP_S, ONSET_PROM
from .. import segment
from ..features import mel
from ..attackers.supervised import SupervisedAttacker
from .data import synth_line, random_text, VOCAB, CLIP
from .model import ConvCTC, logmel, greedy_decode, DEVICE

BANK = os.environ.get("KEYGUARD_BANK", "data/harrison/MBPWavs")
KEYNET_CKPT = os.environ.get("KEYGUARD_KEYNET", "runs/supervised_mbp.pt")
CTC_CKPT = os.environ.get("KEYGUARD_CKPT", "runs/ctc.pt")
SPEED_BINS = [(150, 220), (250, 340), (360, 460), (470, 580)]
LINES = 12                      # synth lines per speed bin
TOL = int(0.04 * SR)           # onset match tolerance (+/-40 ms)


def _ids_to_keys(ids: list[int]) -> str:
    """CTC symbol ids (1..36) -> key string."""
    return "".join(VOCAB[i] for i in ids)


def _onsets(y: np.ndarray, min_gap_s: float, prom: float) -> np.ndarray:
    """Blind onset peaks with tunable spacing/prominence (overlap needs tighter gap)."""
    import librosa
    from scipy.signal import find_peaks
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    if env.max() <= 0:
        return np.array([], int)
    env = env / env.max()
    peaks, _ = find_peaks(env, height=prom, distance=max(1, int(min_gap_s * SR / HOP)))
    return (peaks * HOP).astype(int)


def _match(det: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Greedy nearest-match onset precision/recall within TOL."""
    if len(true) == 0:
        return (1.0 if len(det) == 0 else 0.0), 1.0
    used = np.zeros(len(det), bool)
    hits = 0
    for t in true:
        if len(det):
            j = int(np.argmin(np.abs(det - t)))
            if not used[j] and abs(det[j] - t) <= TOL:
                used[j] = True
                hits += 1
    recall = hits / len(true)
    precision = hits / len(det) if len(det) else 0.0
    return precision, recall


def _seg_decode(y: np.ndarray, keynet: SupervisedAttacker, min_gap_s: float) -> tuple[str, np.ndarray]:
    det = _onsets(y, min_gap_s, ONSET_PROM)
    if len(det) == 0:
        return "", det
    wins = segment.windows(y, det)
    keys = keynet.predict(mel(wins))
    return "".join(keys), det


def cer_str(ref: str, hyp: str) -> float:
    from rapidfuzz.distance import Levenshtein
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def evaluate(min_gap_s: float = 0.05, seed: int = 0, lines: int = LINES,
             use_ctc: bool = True) -> dict:
    keynet = SupervisedAttacker().load(KEYNET_CKPT)
    ctc = None
    if use_ctc and os.path.exists(CTC_CKPT):
        import torch
        ctc = ConvCTC().to(DEVICE)
        ctc.load_state_dict(torch.load(CTC_CKPT, map_location=DEVICE))
        ctc.eval()

    rows = []
    for lo, hi in SPEED_BINS:
        rng = np.random.default_rng(seed + lo)
        ov, seg_cer, ctc_cer, prec, rec = [], [], [], [], []
        for _ in range(lines):
            y, lab_ids, on = synth_line(random_text(rng), rng, wpm=(lo, hi), root=BANK)
            if len(lab_ids) == 0:
                continue
            true = _ids_to_keys(list(lab_ids))
            gaps = np.diff(on)
            ov.append(float(np.mean(gaps < CLIP)) if len(gaps) else 0.0)
            hyp, det = _seg_decode(y, keynet, min_gap_s)
            seg_cer.append(cer_str(true, hyp))
            p, r = _match(det, on)
            prec.append(p); rec.append(r)
            if ctc is not None:
                import torch
                with torch.no_grad():
                    logits = ctc(torch.from_numpy(logmel(y))[None].to(DEVICE))
                ctc_cer.append(cer_str(true, _ids_to_keys(greedy_decode(logits)[0])))
        rows.append({
            "wpm": (lo + hi) // 2, "overlap": float(np.mean(ov)),
            "seg_cer": float(np.mean(seg_cer)),
            "ctc_cer": float(np.mean(ctc_cer)) if ctc_cer else None,
            "onset_precision": float(np.mean(prec)), "onset_recall": float(np.mean(rec)),
        })
    return {"min_gap_s": min_gap_s, "bank": BANK, "rows": rows}


def _print(r: dict) -> None:
    print(f"\noverlap eval  bank={r['bank']}  min_gap={r['min_gap_s']*1000:.0f}ms  "
          f"KeyNet={KEYNET_CKPT}  (CER lower=better, chance~1.0)")
    print(f"{'wpm':>5}{'overlap%':>10}{'SEG CER':>9}{'CTC CER':>9}"
          f"{'onset_P':>9}{'onset_R':>9}")
    for x in r["rows"]:
        c = f"{x['ctc_cer']:.1%}" if x["ctc_cer"] is not None else "  --"
        print(f"{x['wpm']:>5}{x['overlap']:>9.0%}{x['seg_cer']:>9.1%}{c:>9}"
              f"{x['onset_precision']:>9.1%}{x['onset_recall']:>9.1%}")


def run() -> dict:
    import json
    from ..config import RUNS
    out = {"gaps": {}}
    for g in (ONSET_MIN_GAP_S, 0.05, 0.03):
        r = evaluate(min_gap_s=g)
        out["gaps"][f"{g:.3f}"] = r
        _print(r)
    (RUNS / "overlap_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RUNS/'overlap_eval.json'}")
    return out


def demo() -> None:
    assert _ids_to_keys([1, 2, 3]) == CLASSES[0] + CLASSES[1] + CLASSES[2]
    assert cer_str("abc", "abc") == 0.0 and 0 < cer_str("abc", "axc") <= 1.0
    p, rc = _match(np.array([100, 5000]), np.array([120, 5000]))
    assert rc == 1.0, rc
    r = evaluate(min_gap_s=0.05, lines=2, use_ctc=False)
    assert len(r["rows"]) == len(SPEED_BINS)
    for x in r["rows"]:
        assert 0.0 <= x["seg_cer"] <= 2.0 and 0.0 <= x["onset_recall"] <= 1.0
    _print(r)
    print("overlap_eval demo ok")


if __name__ == "__main__":
    demo() if len(sys.argv) > 1 and sys.argv[1] == "demo" else run()
