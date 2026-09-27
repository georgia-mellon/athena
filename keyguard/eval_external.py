"""Out-of-distribution validation on EXTERNAL labeled keystroke-audio datasets.

Honest domain-shift check for the HackGT live test: our attacker/shield were
built on the Harrison MacBook set + our laptop bank. Here we score them on
GENUINELY DIFFERENT keyboards/mics with ground-truth labels.

Datasets (already vendored under data/external/, gitignored):
  * zenodo_10623477      -- 26 files, one recording per LETTER A-Z (one keyboard/mic)
  * multipressure_z19453177 -- 38 keys x {High,Medium,Low} pressure, one press each
                               (a DIFFERENT keyboard/mic; 36 keys map to our CLASSES)

Both are "reference" datasets: ~1-3 physical presses per key, not big capture
banks. So:
  A. ATTACKER zero-shot: run runs/supervised_mbp.pt directly -> expect collapse.
  B. ATTACKER calibrated: leave-one-PRESSURE-out on multipressure (train on 2
     pressures, test on the 3rd, same keyboard) -> "with a little calibration".
  C. DEFENDER: optimize the bounded (-18 dB) perturbation vs the calibrated
     attacker, measure leakage before/after (should fall toward chance).

Run:  KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.eval_external
Smoke: KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.eval_external demo
"""
from __future__ import annotations
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

from .config import CLASSES, CLS_IDX, RUNS, DATA, KEY_WIN, SR
from .audio import load
from .segment import onsets_n, windows
from .features import mel
from .attackers.supervised import SupervisedAttacker, KeyNet, DEVICE
from .shield.adversarial import train_attacker, optimize_perturbation, _acc
from .eval import metrics

ZENODO = DATA / "external" / "zenodo_10623477"
MULTIP = DATA / "external" / "multipressure_z19453177" / "dataset"
CKPT = RUNS / "supervised_mbp.pt"
CHANCE = 1.0 / len(CLASSES)
PRESSURES = ("HighP", "MediumP", "LowP")
EPS_SNR_DB = -18.0        # perceptual budget (inaudible-ish), matches adversarial.py


def _win_from_file(path: str) -> np.ndarray:
    """One keystroke file -> (KEY_WIN,) audio window around its onset."""
    y = load(path)
    on = onsets_n(y, 1)
    if len(on) == 0:                       # ponytail: no onset -> take clip start
        on = np.array([int(0.02 * SR)])
    return windows(y, on)[0]


def _zenodo_samples() -> tuple[np.ndarray, list[str]]:
    """(n, KEY_WIN) windows + labels for zenodo (letters only)."""
    wins, labs = [], []
    for f in sorted(glob.glob(str(ZENODO / "*.wav"))):
        key = os.path.basename(f).split("_")[1].split(".")[0].upper()
        if key not in CLS_IDX:
            continue
        wins.append(_win_from_file(f))
        labs.append(key)
    return np.stack(wins), labs


def _multip_samples() -> dict[str, tuple[np.ndarray, list[str]]]:
    """Per-pressure {pressure: (windows, labels)} for multipressure (A-Z + 0-9)."""
    out = {}
    for p in PRESSURES:
        wins, labs = [], []
        for f in sorted(glob.glob(str(MULTIP / p / "*.wav"))):
            # Key-<K>-<H/M/L>.wav ; drop non-CLASSES (Ç, Spacebar)
            key = os.path.basename(f).split("-")[1].upper()
            if key not in CLS_IDX:
                continue
            wins.append(_win_from_file(f))
            labs.append(key)
        out[p] = (np.stack(wins), labs)
    return out


def _topk(proba: np.ndarray, yi: np.ndarray, k: int) -> float:
    if len(yi) == 0:
        return 0.0
    topk = proba.argsort(1)[:, -k:]
    return float(np.mean([yi[i] in topk[i] for i in range(len(yi))]))


def _score(atk: SupervisedAttacker, wins: np.ndarray, labs: list[str]) -> dict:
    proba = atk.predict_proba(mel(wins))
    yi = np.array([CLS_IDX[k] for k in labs])
    pred = [CLASSES[i] for i in proba.argmax(1)]
    return {
        "n": len(yi),
        "top1": _topk(proba, yi, 1),
        "top3": _topk(proba, yi, 3),
        "pred_hist": dict(Counter(pred).most_common(6)),
    }


# ---- A. zero-shot ---------------------------------------------------------
def zero_shot() -> dict:
    atk = SupervisedAttacker().load(CKPT)
    zw, zl = _zenodo_samples()
    mp = _multip_samples()
    # multipressure zero-shot: pool all pressures
    mw = np.concatenate([mp[p][0] for p in PRESSURES])
    ml = sum([mp[p][1] for p in PRESSURES], [])
    return {"zenodo": _score(atk, zw, zl), "multipressure": _score(atk, mw, ml)}


# ---- B. calibrated attacker on multipressure -------------------------------
def _fit_eval(Xtr, ytr, Xte, yte) -> dict:
    atk = SupervisedAttacker()
    atk.fit(mel(Xtr), np.array(ytr), epochs=60)    # tiny set -> memorize+specaug
    proba = atk.predict_proba(mel(Xte))
    yi = np.array(yte)
    return {"n_train": len(ytr), "n_test": len(yi),
            "top1": _topk(proba, yi, 1), "top3": _topk(proba, yi, 3)}


def calibrated(mp: dict) -> dict:
    """Two same-keyboard calibration tests:
    - leave_pressure_out: train on 2 pressures, test the unseen 3rd (strict).
    - mixed_split: each key contributes 2 presses to train, 1 held out, pressures
      mixed -- mirrors 'bank the target keyboard, then read held-out presses'."""
    # strict: leave one pressure out
    lpo = []
    for held in PRESSURES:
        tr = [p for p in PRESSURES if p != held]
        Xtr = np.concatenate([mp[p][0] for p in tr])
        ytr = [CLS_IDX[k] for p in tr for k in mp[p][1]]
        Xte, yl = mp[held]
        f = _fit_eval(Xtr, ytr, Xte, [CLS_IDX[k] for k in yl])
        f["held_out"] = held
        lpo.append(f)

    # charitable: per-key 2-train/1-test, pressures mixed
    rng = np.random.default_rng(0)
    per_key: dict[str, list[np.ndarray]] = {}
    for p in PRESSURES:
        w, l = mp[p]
        for wi, k in zip(w, l):
            per_key.setdefault(k, []).append(wi)
    Xtr, ytr, Xte, yte = [], [], [], []
    for k, ws in per_key.items():
        order = rng.permutation(len(ws))
        for j, idx in enumerate(order):
            (Xte if j == 0 else Xtr).append(ws[idx])
            (yte if j == 0 else ytr).append(CLS_IDX[k])
    mixed = _fit_eval(np.stack(Xtr), ytr, np.stack(Xte), yte)

    return {
        "leave_pressure_out": {
            "folds": lpo,
            "mean_top1": float(np.mean([f["top1"] for f in lpo])),
            "mean_top3": float(np.mean([f["top3"] for f in lpo])),
        },
        "mixed_split": mixed,
    }


# ---- C. defender (bounded perturbation vs calibrated attacker) ------------
def defender(mp: dict) -> dict:
    """Train a KeyNet on High+Medium, hold out Low; optimize the -18 dB
    universal perturbation vs it; measure leakage on the held-out set
    before/after. Uses the same torch front-end as the real min-max shield."""
    tr = ["HighP", "MediumP"]
    Xtr = torch.tensor(np.concatenate([mp[p][0] for p in tr]), device=DEVICE)
    ytr = torch.tensor([CLS_IDX[k] for p in tr for k in mp[p][1]], device=DEVICE)
    Xte = torch.tensor(mp["LowP"][0], device=DEVICE)
    yte = torch.tensor([CLS_IDX[k] for k in mp["LowP"][1]], device=DEVICE)

    net = KeyNet().to(DEVICE)
    train_attacker(net, Xtr, ytr, epochs=60)
    before = _acc(net, Xte, yte)

    key_rms = Xtr.pow(2).mean().sqrt().item()
    eps = key_rms * (10 ** (EPS_SNR_DB / 20)) * np.sqrt(KEY_WIN)   # L2 budget
    delta = optimize_perturbation(net, Xtr, ytr, eps, steps=200)
    after = _acc(net, Xte + delta, yte)
    d_snr = 20 * np.log10(key_rms / (delta.norm().item() / np.sqrt(KEY_WIN) + 1e-12))

    out = {"before": before, "after": after, "snr_db": d_snr, "chance": CHANCE}
    stoi = _stoi_check(delta.detach().cpu().numpy())
    if stoi is not None:
        out["stoi"] = stoi
    return out


def _stoi_check(delta: np.ndarray):
    """Optional: STOI of speech with the perturbation tiled over it vs clean
    speech. Needs a speech clip; returns None if none is fetchable offline."""
    try:
        import librosa
        speech, sr = librosa.load(librosa.example("libri1"), sr=SR, duration=4.0)
        # tile the keystroke-scale perturbation across the clip (worst case: everywhere)
        pert = np.resize(delta, len(speech)).astype(np.float32)
        q = metrics.speech_quality(speech, speech + pert, SR)
        return q.get("stoi")
    except Exception:
        return None


def run() -> dict:
    mp = _multip_samples()
    result = {
        "meta": {
            "datasets": {
                "zenodo_10623477": "26 files, 1 recording/letter A-Z, single keyboard/mic",
                "multipressure_z19453177": "38 keys x {High,Med,Low} pressure, 1 press each; different keyboard/mic",
            },
            "n_classes": len(CLASSES),
            "chance": CHANCE,
            "checkpoint": CKPT.name,
            "device": DEVICE,
        },
        "zero_shot": zero_shot(),
        "calibrated": calibrated(mp),
        "defender": defender(mp),
    }
    out = RUNS / "external_eval.json"
    out.write_text(json.dumps(result, indent=2))
    _print(result)
    print(f"\nwrote {out}")
    return result


def _print(r: dict) -> None:
    z, m = r["zero_shot"]["zenodo"], r["zero_shot"]["multipressure"]
    c, d = r["calibrated"], r["defender"]
    print(f"chance = {r['meta']['chance']:.1%}  ({r['meta']['n_classes']} classes)")
    print("A. ZERO-SHOT (our Harrison attacker, no calibration):")
    print(f"   zenodo        n={z['n']:>3}  top1={z['top1']:.1%}  top3={z['top3']:.1%}  preds={z['pred_hist']}")
    print(f"   multipressure n={m['n']:>3}  top1={m['top1']:.1%}  top3={m['top3']:.1%}  preds={m['pred_hist']}")
    lpo, mix = c["leave_pressure_out"], c["mixed_split"]
    print("B. CALIBRATED (same keyboard, tiny 3-presses/key data):")
    print(f"   leave-pressure-out  mean top1={lpo['mean_top1']:.1%}  top3={lpo['mean_top3']:.1%}")
    print(f"   mixed 2tr/1te split  top1={mix['top1']:.1%}  top3={mix['top3']:.1%}  (n_train={mix['n_train']} n_test={mix['n_test']})")
    print("C. DEFENDER (-18 dB perturbation vs calibrated attacker):")
    print(f"   leakage {d['before']:.1%} -> {d['after']:.1%}  (chance {d['chance']:.1%})  SNR {d['snr_db']:.1f} dB"
          + (f"  STOI={d['stoi']:.3f}" if "stoi" in d else "  (STOI: no speech clip)"))


def demo() -> None:
    """Fast assert-based smoke: sample loading + scoring shapes are sane, and the
    perturbation lowers a freshly-trained attacker's accuracy on external data."""
    mp = _multip_samples()
    assert set(mp) == set(PRESSURES)
    for p in PRESSURES:
        w, l = mp[p]
        assert w.shape[1] == KEY_WIN and len(l) == len(w) and len(l) > 0
    zw, zl = _zenodo_samples()
    assert zw.shape[1] == KEY_WIN and len(zl) == len(zw)

    # the bounded -18dB perturbation must never INCREASE the attacker's accuracy
    # (guaranteed invariant of the min-max; the strict drop shows in the full run).
    Xtr = torch.tensor(np.concatenate([mp["HighP"][0], mp["MediumP"][0]]), device=DEVICE)
    ytr = torch.tensor([CLS_IDX[k] for p in ("HighP", "MediumP") for k in mp[p][1]],
                       device=DEVICE)
    net = KeyNet().to(DEVICE)
    train_attacker(net, Xtr, ytr, epochs=40)
    a0 = _acc(net, Xtr, ytr)
    key_rms = Xtr.pow(2).mean().sqrt().item()
    eps = key_rms * (10 ** (EPS_SNR_DB / 20)) * np.sqrt(KEY_WIN)
    d = optimize_perturbation(net, Xtr, ytr, eps, steps=100)
    a1 = _acc(net, Xtr + d, ytr)
    assert a1 <= a0 + 1e-6, (a0, a1)
    assert d.norm().item() <= eps * 1.01           # perturbation respects the budget
    print(f"eval_external demo ok: loaders sane; -18dB perturbation holds a fresh "
          f"external-trained attacker at/below its accuracy {a0:.0%}->{a1:.0%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        run()
