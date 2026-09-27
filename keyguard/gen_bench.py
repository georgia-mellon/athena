"""Shared cross-keyboard benchmark for attacker variants.

Every keyboard domain lives as data/pool/<name>.npz {wins (n, KEY_WIN) float32,
labels (n,) str}. A variant is a module exposing `make() -> model` where model has
    fit(wins, y, dom)            y = class idx, dom = domain idx (for DANN/per-domain norm)
    predict_proba(wins) -> (n, 36)
    finetune(wins, y)  [optional; else few-shot refits on pool + target shots]
Metrics (chance 2.8% top-1 / 8.3% top-3):
    loko    : leave-one-keyboard-out zero-shot, mean over testable domains
    fewshot : pretrain on all other keyboards, then k presses/key of the target
              (k=5,10), tested on the target's remaining presses. `live` = user's keyboard.
Run:  KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.gen_bench keyguard.gen_variants.baseline
      uv run python3 -m keyguard.gen_bench export     # (re)dump existing domains to data/pool
"""
from __future__ import annotations
import copy
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

from .config import CLS_IDX, DATA, RUNS

POOL = DATA / "pool"
OUT = RUNS / "gen_bench"
MIN_TEST = 100                       # domains smaller than this are train-only
FEWSHOT_TARGETS = ("live", "mka_hp", "harrison")
SHOTS = (5, 10)
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))


def export_existing() -> None:
    from .eval_generalize import load_domains
    POOL.mkdir(parents=True, exist_ok=True)
    for name, (w, l) in load_domains().items():
        np.savez_compressed(POOL / f"{name}.npz", wins=w.astype(np.float32), labels=np.array(l))
        print(name, len(l))


def load_pool() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    doms = {}
    for f in sorted(POOL.glob("*.npz")):
        d = np.load(f)
        labs = np.array([str(s).upper() for s in d["labels"]])
        keep = np.array([s in CLS_IDX for s in labs], bool)
        if keep.sum():
            doms[f.stem] = (d["wins"][keep].astype(np.float32),
                            np.array([CLS_IDX[s] for s in labs[keep]]))
    return doms


def topk(p: np.ndarray, y: np.ndarray, k: int) -> float:
    return float(np.mean([y[i] in p[i].argsort()[-k:] for i in range(len(y))])) if len(y) else 0.0


def _stack(doms, names):
    W = np.concatenate([doms[n][0] for n in names])
    y = np.concatenate([doms[n][1] for n in names])
    d = np.concatenate([np.full(len(doms[n][1]), i) for i, n in enumerate(names)])
    return W, y, d


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def loko(make, doms) -> dict:
    folds = {}
    for held in [n for n in doms if len(doms[n][1]) >= MIN_TEST]:
        W, y, d = _stack(doms, [n for n in doms if n != held])
        _seed()
        m = make()
        m.fit(W, y, d)
        p = m.predict_proba(doms[held][0])
        yt = doms[held][1]
        folds[held] = {"top1": topk(p, yt, 1), "top3": topk(p, yt, 3), "n": int(len(yt))}
        print(f"  loko {held:<18} top1={folds[held]['top1']:.1%} top3={folds[held]['top3']:.1%}", flush=True)
    return {"folds": folds,
            "mean_top1": float(np.mean([f["top1"] for f in folds.values()])),
            "mean_top3": float(np.mean([f["top3"] for f in folds.values()]))}


def _split_shots(y, k, rng):
    tr = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > k:                      # need >=1 left to test
            tr.extend(rng.choice(idx, k, replace=False))
    tr = np.array(sorted(tr), int)
    te = np.setdiff1d(np.arange(len(y)), tr)
    return tr, te


def fewshot(make, doms) -> dict:
    res = {}
    for tgt in [t for t in FEWSHOT_TARGETS if t in doms]:
        others = [n for n in doms if n != tgt]
        W, y, d = _stack(doms, others)
        _seed()
        pre = make()
        pre.fit(W, y, d)
        Wt, yt = doms[tgt]
        for k in SHOTS:
            tr, te = _split_shots(yt, k, np.random.default_rng(k))
            _seed()
            if hasattr(pre, "finetune"):
                m = copy.deepcopy(pre)
                m.finetune(Wt[tr], yt[tr])
            else:                              # refit on pool + upweighted target shots
                rep = max(1, len(y) // (4 * len(tr)))
                m = make()
                m.fit(np.concatenate([W] + [Wt[tr]] * rep), np.concatenate([y] + [yt[tr]] * rep),
                      np.concatenate([d] + [np.full(len(tr), d.max() + 1)] * rep))
            p = m.predict_proba(Wt[te])
            key = f"{tgt}@{k}"
            res[key] = {"top1": topk(p, yt[te], 1), "top3": topk(p, yt[te], 3), "n_test": int(len(te))}
            print(f"  fewshot {key:<14} top1={res[key]['top1']:.1%} top3={res[key]['top3']:.1%}", flush=True)
    return res


def run(module: str, which: str = "all") -> dict:
    make = importlib.import_module(module).make
    doms = load_pool()
    print(f"{module}: {len(doms)} domains " + ", ".join(f"{n}:{len(v[1])}" for n, v in doms.items()), flush=True)
    t = time.time()
    r = {"variant": module, "domains": {n: int(len(v[1])) for n, v in doms.items()}}
    if which in ("all", "loko"):
        r["loko"] = loko(make, doms)
    if which in ("all", "fewshot"):
        r["fewshot"] = fewshot(make, doms)
    r["seconds"] = round(time.time() - t)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{module.rsplit('.', 1)[-1]}.json").write_text(json.dumps(r, indent=2))
    if "loko" in r:
        print(f"LOKO mean top1={r['loko']['mean_top1']:.1%} top3={r['loko']['mean_top3']:.1%}")
    return r


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export_existing()
    else:
        run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "all")
