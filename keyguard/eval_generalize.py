"""Cross-keyboard domain-GENERALIZATION experiment for the keystroke attacker.

Question (the honest go/no-go): if we train the supervised attacker on a DIVERSE
POOL of many keyboards' labeled audio instead of just Harrison, does it GENERALIZE
zero-shot to an UNSEEN keyboard -- i.e. can we ship a keyboard-agnostic attacker
that needs no per-keyboard calibration?

Method:
  * Pool every usable labeled source into one 36-class dataset (A-Z, 0-9). Each
    source is a "keyboard" domain (a distinct keyboard/mic rig). We reuse the
    project front-end: onset-segment -> fixed KEY_WIN window -> log-mel.
  * TRUE test = LEAVE-ONE-KEYBOARD-OUT: for each held-out keyboard, train on ALL
    other keyboards, test zero-shot on the held-out one. Report per-held-out and
    mean top-1/top-3.
  * Ceiling = within-distribution split that MIXES keyboards (shows what the model
    can do when it has seen every keyboard). Chance = 1/36 = 2.8%.

Decision rule (see REALWORLD.md): if zero-shot LOKO top-3 is meaningfully above
chance (rule of thumb >=~15%, >5x chance), KEEP it; else FADE it (diverse-training
did not generalize -> keep per-keyboard calibration).

Run:   KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.eval_generalize
Smoke: KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.eval_generalize demo
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

from .config import CLASSES, CLS_IDX, DATA, RUNS, KEY_WIN
from .audio import load
from .segment import onsets_n, windows
from .features import mel
from .attackers.supervised import SupervisedAttacker, DEVICE
from .eval_external import _multip_samples, _zenodo_samples

CHANCE = 1.0 / len(CLASSES)
HARRISON = DATA / "harrison"
MKA_ROOT = DATA / "external" / "mka" / "extracted" / "MKA datasets"
PRESSES = 25                 # Harrison files hold ~25 presses/key
MAX_PER_KEY = 25             # cap presses/key/domain so no source dominates + keep compute modest

# MKA platforms -> physical-keyboard domain. Mac/Messenger/Zoom are the SAME Mac
# keyboard over different channels, so they group into one keyboard (else holding
# out "Mac" would leak via Messenger/Zoom and not be a true keyboard-out test).
_MKA_KEYBOARD = {"hp": "mka_hp", "lenovo": "mka_lenovo", "msi": "mka_msi",
                 "mac": "mka_mac", "messenger": "mka_mac", "zoom": "mka_mac"}

Domain = tuple[np.ndarray, list[str]]     # (windows (n, KEY_WIN), labels)


# ---- per-source loaders (each returns one keyboard domain) -----------------
def _cap(wins: list[np.ndarray], labs: list[str]) -> Domain:
    """Stack + cap to MAX_PER_KEY presses per class (deterministic head)."""
    if not wins:
        return np.zeros((0, KEY_WIN), np.float32), []
    W = np.stack(wins) if wins[0].ndim == 1 else np.concatenate(wins)
    seen: dict[str, int] = {}
    keep = []
    for i, k in enumerate(labs):
        if seen.get(k, 0) < MAX_PER_KEY:
            keep.append(i)
            seen[k] = seen.get(k, 0) + 1
    return W[keep], [labs[i] for i in keep]


def _harrison_domain() -> Domain:
    """MacBook Pro keyboard, phone mic (MBPWavs) + Zoom channel -- same physical
    keyboard, so grouped as ONE keyboard domain."""
    wins, labs = [], []
    for sub in ("MBPWavs", "Zoom"):
        for f in sorted(glob.glob(str(HARRISON / sub / "*.wav"))):
            key = os.path.basename(f).split(".")[0].upper()
            if key not in CLS_IDX:
                continue
            y = load(f)
            on = onsets_n(y, PRESSES)
            for w in windows(y, on):
                wins.append(w)
                labs.append(key)
    return _cap(wins, labs)


def _live_domain() -> Domain:
    """Our laptop bank: per-key windows already onset-cut (shorter than KEY_WIN);
    tail-pad to KEY_WIN so mel frame count matches the pool."""
    npz = DATA / "live_bank.npz"
    if not npz.exists():
        return np.zeros((0, KEY_WIN), np.float32), []
    d = np.load(npz)
    wins, labs = [], []
    for key in d.files:
        k = key.upper()
        if k not in CLS_IDX:
            continue
        arr = d[key]
        L = min(arr.shape[1], KEY_WIN)
        for row in arr:
            w = np.zeros(KEY_WIN, np.float32)
            w[:L] = row[:L]
            wins.append(w)
            labs.append(k)
    return _cap(wins, labs)


def _multipressure_domain() -> Domain:
    mp = _multip_samples()
    wins = [w for p in mp for w in mp[p][0]]
    labs = [k for p in mp for k in mp[p][1]]
    return _cap(wins, labs)


def _zenodo_domain() -> Domain:
    w, l = _zenodo_samples()
    return _cap(list(w), l)


def _mka_domains() -> dict[str, Domain]:
    """Multi-Keyboard Acoustic (Mendeley bpt2hvf8n3): distinct physical keyboards
    HP / Lenovo / MSI / Mac, the Mac also captured over Messenger + Zoom channels.
    Uses the SEGMENTED per-press wavs; key = the parent folder name (only the 36
    A-Z/0-9 folders count -- 'enter', 'caps', etc. are dropped)."""
    if not MKA_ROOT.exists():
        return {}
    grouped: dict[str, list] = {}   # keyboard-domain -> [(file, key)]
    for f in glob.glob(str(MKA_ROOT / "*" / "Sound Segment(wav)" / "*" / "*.wav")):
        parts = f.split(os.sep)
        platform = parts[-4].lower()
        kbd = _MKA_KEYBOARD.get(platform)
        key = parts[-2].upper()                      # parent folder is the true label
        if kbd is None or len(key) != 1 or key not in CLS_IDX:
            continue
        grouped.setdefault(kbd, []).append((f, key))
    out: dict[str, Domain] = {}
    for kbd, items in grouped.items():
        wins, labs = [], []
        for f, key in sorted(items):
            y = load(f)
            on = onsets_n(y, 1)
            if len(on) == 0:
                on = np.array([int(0.02 * 16000)])
            wins.append(windows(y, on)[0])
            labs.append(key)
        w, l = _cap(wins, labs)
        if len(set(l)) >= len(CLASSES) // 2:         # keep only reasonably complete keyboards
            out[kbd] = (w, l)
    return out


def load_domains() -> dict[str, Domain]:
    """All usable labeled keyboard domains -> {name: (windows, labels)}."""
    doms: dict[str, Domain] = {
        "harrison": _harrison_domain(),
        "live": _live_domain(),
        "multipressure": _multipressure_domain(),
        "zenodo": _zenodo_domain(),
    }
    doms.update(_mka_domains())
    return {k: v for k, v in doms.items() if len(v[1]) > 0}


# ---- experiment -----------------------------------------------------------
def _prep(doms: dict[str, Domain]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """mel each domain once (reused across every fold)."""
    return {n: (mel(W), np.array([CLS_IDX[k] for k in L])) for n, (W, L) in doms.items()}


def _topk(proba: np.ndarray, yi: np.ndarray, k: int) -> float:
    if len(yi) == 0:
        return 0.0
    topk = proba.argsort(1)[:, -k:]
    return float(np.mean([yi[i] in topk[i] for i in range(len(yi))]))


def _fit_score(Xtr, ytr, Xte, yte, epochs: int) -> tuple[float, float]:
    atk = SupervisedAttacker()
    atk.fit(Xtr, ytr, epochs=epochs)          # KeyNet + specaug built in
    proba = atk.predict_proba(Xte)
    return _topk(proba, yte, 1), _topk(proba, yte, 3)


def ceiling(prep, epochs: int) -> dict:
    """Within-distribution ceiling: pool ALL keyboards, random 80/20 split
    (mixing keyboards). Shows what the model can do having seen every keyboard."""
    X = np.concatenate([prep[n][0] for n in prep])
    y = np.concatenate([prep[n][1] for n in prep])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    t1, t3 = _fit_score(X[tr], y[tr], X[te], y[te], epochs)
    return {"n_train": int(len(tr)), "n_test": int(len(te)), "top1": t1, "top3": t3}


def leave_one_keyboard_out(prep, epochs: int) -> dict:
    """TRUE generalization: train on all other keyboards, test zero-shot on each
    held-out keyboard."""
    names = list(prep)
    folds = {}
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([prep[n][0] for n in tr])
        ytr = np.concatenate([prep[n][1] for n in tr])
        Xte, yte = prep[held]
        t1, t3 = _fit_score(Xtr, ytr, Xte, yte, epochs)
        folds[held] = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
                       "top1": t1, "top3": t3}
    return {
        "folds": folds,
        "mean_top1": float(np.mean([f["top1"] for f in folds.values()])),
        "mean_top3": float(np.mean([f["top3"] for f in folds.values()])),
    }


def _verdict(loko: dict) -> str:
    """KEEP if zero-shot LOKO top-3 meaningfully beats chance (>=~15%, >5x)."""
    keep = loko["mean_top3"] >= 5 * CHANCE and loko["mean_top1"] > CHANCE * 1.5
    return "KEEP" if keep else "FADED"


def run(epochs: int = 60) -> dict:
    doms = load_domains()
    prep = _prep(doms)
    n_kbd = len(prep)
    if n_kbd < 2:
        result = {"meta": {"error": "need >=2 keyboard domains for LOKO",
                           "domains": {n: len(prep[n][1]) for n in prep}}}
        (RUNS / "generalize.json").write_text(json.dumps(result, indent=2))
        print("Only", n_kbd, "domain -> cannot test generalization. FADED (data).")
        return result

    cap = ceiling(prep, epochs)
    loko = leave_one_keyboard_out(prep, epochs)
    result = {
        "meta": {
            "n_classes": len(CLASSES),
            "chance": CHANCE,
            "device": DEVICE,
            "epochs": epochs,
            "max_per_key_per_domain": MAX_PER_KEY,
            "domains": {n: {"n": int(len(prep[n][1])),
                            "n_keys": int(len(set(prep[n][1].tolist())))} for n in prep},
        },
        "ceiling_within_distribution": cap,
        "leave_one_keyboard_out": loko,
        "verdict": _verdict(loko),
    }
    (RUNS / "generalize.json").write_text(json.dumps(result, indent=2))
    _print(result)
    print(f"\nwrote {RUNS / 'generalize.json'}")
    return result


def _print(r: dict) -> None:
    m = r["meta"]
    print(f"chance = {m['chance']:.1%}  ({m['n_classes']} classes)  device={m['device']}")
    print("domains (keyboard/mic rigs) in the pool:")
    for n, d in m["domains"].items():
        print(f"   {n:<16} n={d['n']:>4}  keys={d['n_keys']}")
    c = r["ceiling_within_distribution"]
    print(f"\nCEILING (mix keyboards, 80/20): top1={c['top1']:.1%}  top3={c['top3']:.1%}"
          f"  (n_tr={c['n_train']} n_te={c['n_test']})")
    print("LEAVE-ONE-KEYBOARD-OUT (train others, test held-out zero-shot):")
    for n, f in r["leave_one_keyboard_out"]["folds"].items():
        print(f"   held-out {n:<16} top1={f['top1']:.1%}  top3={f['top3']:.1%}"
              f"  (n_te={f['n_test']})")
    lk = r["leave_one_keyboard_out"]
    print(f"   MEAN            top1={lk['mean_top1']:.1%}  top3={lk['mean_top3']:.1%}")
    print(f"\nVERDICT: {r['verdict']}  (KEEP if LOKO top3 >= {5*m['chance']:.1%})")


def demo() -> None:
    """Fast assert-based smoke: domains load with consistent mel dims, and a
    tiny-epoch LOKO produces finite, in-range numbers on >=2 domains."""
    doms = load_domains()
    assert len(doms) >= 2, f"need >=2 keyboard domains, got {list(doms)}"
    prep = _prep(doms)
    frames = {p[0].shape[-1] for p in prep.values() if len(p[0])}
    assert len(frames) == 1, f"inconsistent mel frame count across domains: {frames}"
    for n, (X, y) in prep.items():
        assert X.ndim == 4 or (X.ndim == 3), X.shape
        assert len(X) == len(y) and len(y) > 0, n
        assert set(int(v) for v in y).issubset(range(len(CLASSES)))
    loko = leave_one_keyboard_out(prep, epochs=3)
    assert 0.0 <= loko["mean_top1"] <= 1.0 and 0.0 <= loko["mean_top3"] <= 1.0
    assert loko["mean_top3"] >= loko["mean_top1"] - 1e-9
    print(f"eval_generalize demo ok: {len(doms)} domains {list(doms)}; "
          f"3-epoch LOKO mean top1={loko['mean_top1']:.1%} top3={loko['mean_top3']:.1%} "
          f"(chance {CHANCE:.1%}) -- full numbers need the real run.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        ep = int(sys.argv[1]) if len(sys.argv) > 1 else 60
        run(ep)
