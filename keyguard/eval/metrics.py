"""Hard metrics that decide every arena round.

- text_similarity : 1 - normalized edit distance (attacker's reconstruction)
- mutual_info_bits: I(true; predicted) in bits/keystroke. Data-processing
  inequality makes this a valid *lower bound* on bits the audio leaks.
- cluster_purity  : how well unsupervised clusters line up with true keys
- speech quality  : PESQ + STOI vs a clean reference (synthetic mixtures only)
"""
from __future__ import annotations
import numpy as np
from rapidfuzz.distance import Levenshtein


def text_similarity(true: str, pred: str) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    return 1.0 - Levenshtein.distance(true, pred) / max(len(true), len(pred))


def top1_accuracy(true_labels, pred_labels) -> float:
    if not len(true_labels):
        return 0.0
    return float(np.mean([t == p for t, p in zip(true_labels, pred_labels)]))


def mutual_info_bits(true_labels, pred_labels, n_classes: int) -> float:
    """I(Y; Yhat) in bits from the confusion matrix. Lower bound on leakage."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    if len(true_labels) == 0:
        return 0.0
    C = np.zeros((n_classes, n_classes))
    for t, p in zip(true_labels, pred_labels):
        C[t, p] += 1
    P = C / C.sum()
    py = P.sum(0, keepdims=True)
    px = P.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = P * np.log2(P / (px * py))
    return float(np.nansum(term[P > 0]))


def cluster_purity(true_labels, cluster_ids) -> float:
    true_labels = np.asarray(true_labels)
    cluster_ids = np.asarray(cluster_ids)
    if len(true_labels) == 0:
        return 0.0
    total = 0
    for c in np.unique(cluster_ids):
        members = true_labels[cluster_ids == c]
        if len(members):
            total += np.bincount(members).max()
    return total / len(true_labels)


def speech_quality(ref: np.ndarray, deg: np.ndarray, sr: int) -> dict:
    """PESQ (wideband) + STOI. Returns {} if libs unavailable or too short."""
    out = {}
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    try:
        from pystoi import stoi
        out["stoi"] = float(stoi(ref, deg, sr, extended=False))
    except Exception:
        pass
    try:
        from pesq import pesq
        sr_p = 16000
        if sr != sr_p:
            import librosa
            r = librosa.resample(ref, orig_sr=sr, target_sr=sr_p)
            d = librosa.resample(deg, orig_sr=sr, target_sr=sr_p)
        else:
            r, d = ref, deg
        out["pesq"] = float(pesq(sr_p, r, d, "wb"))
    except Exception:
        pass
    return out


def demo():
    assert abs(text_similarity("hello", "hello") - 1.0) < 1e-9
    assert text_similarity("hello", "hallo") == 0.8
    assert mutual_info_bits([0, 1, 2], [0, 1, 2], 3) > 1.5   # perfect -> log2(3)
    assert mutual_info_bits([0, 1, 2], [0, 0, 0], 3) == 0.0  # no info
    assert cluster_purity([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    print("metrics demo ok")


if __name__ == "__main__":
    demo()
