"""Adaptive attacker: retrain the MtlCRNN CTC transcriber on SHIELDED audio.

The min-max step. Reuses the exact training recipe from ctc.train_overlap
(class-weighted dense frame-CE + onset-BCE, no curriculum) but feeds pre-built
SHIELDED train chunks (dumped by ctc_arena --dump-train) instead of clean SKAID.
Evaluates on shielded held-out chunks each N steps and saves the best.

Usage:
  KEYGUARD_DEVICE=cuda PYTHONPATH=$(pwd) python -m keyguard.arena.adaptive_train \
      --train scratch/shielded_train.pt --test scratch/shielded_test.pt \
      --out runs/ctc_shielded_adaptive.pt --steps 2500 [--warm runs/ctc_skaid_crnn.pt]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from ..ctc.data import VOCAB, BLANK
from ..ctc.model import DEVICE
from ..ctc.train_overlap import MtlCRNN, _collate
from .ctc_arena import eval_cer


def train(train_chunks, test_chunks, out, steps=2500, batch=16, lr=3e-4,
          warm="", lambda_onset=0.5, blank_w=0.05, seed=0):
    torch.manual_seed(seed)
    net = MtlCRNN().to(DEVICE)
    if warm:
        net.load_state_dict(torch.load(warm, map_location=DEVICE))
        print(f"warm-started from {warm}")
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    fce_w = torch.ones(len(VOCAB), device=DEVICE); fce_w[BLANK] = blank_w
    fce = nn.CrossEntropyLoss(weight=fce_w, ignore_index=-100)
    brng = np.random.default_rng(1234)
    warmup = 200
    best = 1e9
    t0 = time.time()
    pool = list(train_chunks)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        idx = brng.integers(0, len(pool), size=batch)
        mel, onset, fkey, in_len, tgt, tlen = _collate([pool[i] for i in idx], brng)
        logits, onset_logit = net(mel)
        mask = (torch.arange(mel.shape[1], device=DEVICE)[None] < in_len.to(DEVICE)[:, None]).float()
        loss = lambda_onset * (bce(onset_logit, onset) * mask).sum() / mask.sum()
        loss = loss + fce(logits.reshape(-1, logits.shape[-1]), fkey.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            cer, n = eval_cer(net, test_chunks)
            tag = ""
            if cer < best:
                best = cer; torch.save(net.state_dict(), out); tag = "  <-best"
            print(f"step {step:4d} loss {float(loss):.3f} "
                  f"({(time.time()-t0)/max(1,step+1)*1000:.0f}ms/step) "
                  f"shielded-CER {cer:.1%} (n={n}){tag}", flush=True)
            net.train()
    print(f"done. best shielded held-out CER {best:.1%}; weights {out}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--warm", default="")
    args = ap.parse_args()
    tr = torch.load(args.train, weights_only=False)
    te = torch.load(args.test, weights_only=False)
    print(f"device={DEVICE} train_chunks={len(tr)} test_chunks={len(te)}")
    train(tr, te, args.out, steps=args.steps, warm=args.warm)


if __name__ == "__main__":
    main()
