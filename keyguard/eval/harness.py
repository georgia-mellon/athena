"""Evaluation harness: the hard metric that decides every arena round.

For a protection level it (optionally) *retrains* the attacker on the protected
audio (adaptive attacker) and reports how well it still reads keys. Adaptive is
the honest headline the spec demands: the attacker has adapted to the shield.
"""
from __future__ import annotations
import numpy as np
from ..dataset import build_aug
from ..attackers.supervised import SupervisedAttacker
from ..config import N_CLASSES
from . import metrics


def evaluate(protect=None, root="data/harrison/MBPWavs", adaptive=True,
             epochs=120, log=None):
    """Return metrics for one (protection, attacker) cell.

    protect  : fn(y)->y applied to audio before segmentation, or None.
    adaptive : True trains the attacker ON protected audio (attacker adapts);
               False trains on clean and tests on protected (naive attacker).
    """
    tf = protect if adaptive else None
    Xtr, ytr, _, _ = build_aug(root, transform=tf)
    _, _, Xte, yte = build_aug(root, transform=protect)   # test always protected
    atk = SupervisedAttacker()
    atk.fit(Xtr, ytr, epochs=epochs, log=log)
    proba = atk.predict_proba(Xte)
    pred = proba.argmax(1) if len(proba) else np.array([], int)
    acc = metrics.top1_accuracy([atk.classes[i] for i in yte],
                                [atk.classes[i] for i in pred])
    mi = metrics.mutual_info_bits(yte, pred, N_CLASSES)
    return {"attack_acc": acc, "mi_bits": mi, "n_test": int(len(yte)),
            "chance": 1.0 / N_CLASSES, "max_bits": float(np.log2(N_CLASSES))}


def speech_quality(protect, manifest="data/synth/manifest.json", limit=8):
    """PESQ/STOI of protected vs clean on synthetic mixtures. {} if no manifest."""
    import json, os
    from .. import audio
    if not os.path.exists(manifest):
        return {}
    items = json.load(open(manifest))
    ps, ss = [], []
    for it in items[:limit]:
        clean = audio.load(it["clean_wav"])
        mix = audio.load(it["mix_wav"])
        deg = protect(mix) if protect else mix
        q = metrics.speech_quality(clean, deg, 16000)
        if "pesq" in q:
            ps.append(q["pesq"])
        if "stoi" in q:
            ss.append(q["stoi"])
    out = {}
    if ps:
        out["pesq"] = float(np.mean(ps))
    if ss:
        out["stoi"] = float(np.mean(ss))
    return out
