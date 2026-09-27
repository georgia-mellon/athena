"""Adapt the attacker to the user's REAL typing (few-shot), honestly evaluated.

Given the target user's real continuous-typing sessions (data/continuous/live/),
warm-start from a base checkpoint (SKAID real model, or bank model, or scratch) and
fine-tune on the real audio, then evaluate by LEAVE-ONE-SESSION-OUT so the number is
on typing the model never saw. Handles the keylog->audio latency by aligning each
session's keylog times to detected onsets (reusing continuous_convert). Timing-safe
augmentation only (gain/EQ/noise; never time-shift, which would desync labels).

  python -m keyguard.agents.finetune_live --base runs/ctc_skaid_crnn.pt --steps 600
  KEYGUARD_LIVE_MIX_BANK=1 ... also fold in synth from the per-key bank.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .. import config, audio, segment
from ..continuous_convert import _best_offset_ms, _map_to_samples, _score_offset
from ..ctc import train_overlap as T
from ..ctc.data import VOCAB, SYM_OF_KEY, BLANK, CLIP
from ..ctc.model import logmel, DEVICE, HOP
from ..ctc.overlap_eval import cer_str
from ..config import SR
from ..ctc import augment as AUG
from .defense_audio import decode

LIVE_DIR = "data/continuous/live"


def load_live(align=True):
    """-> list of dict(y, keys(str), onset_samples(np), wav, hit_ratio)."""
    rows = [json.loads(l) for l in open(os.path.join(LIVE_DIR, "labels.jsonl"))]
    out = []
    for r in rows:
        y = audio.load(os.path.join(LIVE_DIR, r["wav"]))
        keys = "".join(k for k in r["keys"].upper() if k in SYM_OF_KEY)
        klog = np.array([s for k, s in zip(r["keys"].upper(), r["onset_samples"])
                         if k in SYM_OF_KEY], dtype=float)   # keylog sample positions
        hit = 1.0
        samp = klog.astype(int)
        if align and len(klog):
            press_ms = klog / SR * 1000.0
            det = segment.onsets(y)
            det_ms = np.sort(det / SR * 1000.0)
            off = _best_offset_ms(press_ms, det_ms, len(y) / SR * 1000.0)
            samp = np.clip(_map_to_samples(press_ms, off, np.sort(det)), 0, len(y) - 1)
            hit = _score_offset(off, press_ms, det_ms) / max(1, len(klog))
        out.append(dict(y=y, keys=keys, onset_samples=samp, wav=r["wav"], hit=hit))
    return out


def _chunks(sess, chunk_s=5.0, aug=False, rng=None):
    """(logmel, frame_key) chunks from one real session (dense frame-CE targets)."""
    y, keys, on = sess["y"], sess["keys"], np.asarray(sess["onset_samples"])
    step = int(chunk_s * SR)
    out = []
    for a in range(0, len(y), step):
        b = a + step
        sel = np.where((on >= a) & (on < b))[0]
        if len(sel) < 2:
            continue
        seg = y[a:b].astype(np.float32)
        if aug and rng is not None:
            seg = AUG.augment_waveform(seg, rng)          # gain/EQ/noise, timing-safe
        m = logmel(seg)
        ids = np.array([SYM_OF_KEY[keys[i]] for i in sel], np.int64)
        rel = on[sel] - a
        out.append((m, T.frame_key_target(m.shape[0], rel, ids)))
    return out


def eval_session(net, sess):
    return cer_str(sess["keys"], decode(net, sess["y"]))


def _skaid_replay(n):
    """A few real SKAID chunks (m, frame_key) for anti-forgetting replay."""
    path = "data/continuous/skaid/labels.jsonl"
    if not (n and os.path.exists(path)):
        return []
    tr, _ = T.load_skaid(path)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(tr), size=min(n, len(tr)))
    return [(tr[i][0], tr[i][3]) for i in idx]     # (logmel, frame_key)


def finetune(base, train_sessions, steps=600, lr=1e-4, freeze_cnn=True,
             blank_w=0.05, aug=True, mix_bank=0, replay=0, adabn=True, adapt_rnn=True,
             label_smooth=0.1):
    net = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    if base and os.path.exists(base):
        net.load_state_dict(torch.load(base, map_location=DEVICE))
        print(f"  warm-start from {base}")

    rng = np.random.default_rng(0)
    real_pool = []
    for s in train_sessions:
        real_pool += _chunks(s, aug=aug, rng=rng)
    aux_pool = list(_skaid_replay(replay))
    if mix_bank and os.path.exists("data/live_bank.npz"):
        aux_pool += [(m, fk) for (m, lab, on, fk, gap) in T.build_pool(mix_bank, aug=True)]
    assert real_pool, "no real training chunks"

    # AdaBN: recalibrate BatchNorm running stats to the target keyboard (free win)
    if adabn:
        net.train()
        with torch.no_grad():
            for m, _ in real_pool:
                net(torch.from_numpy(np.ascontiguousarray(m))[None].to(DEVICE))

    net.train()
    for p in net.parameters():
        p.requires_grad = True
    if freeze_cnn:
        for p in net.cnn.parameters():
            p.requires_grad = False
    if not adapt_rnn:
        for p in net.rnn.parameters():
            p.requires_grad = False
    # discriminative LR: head/onset fast, BiGRU slow
    head_params = list(net.head.parameters()) + list(net.onset_head.parameters())
    rnn_params = [p for p in net.rnn.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": lr}]
    if rnn_params:
        groups.append({"params": rnn_params, "lr": lr * 0.1})
    w = torch.ones(len(VOCAB), device=DEVICE); w[BLANK] = blank_w
    fce = nn.CrossEntropyLoss(weight=w, ignore_index=-100, label_smoothing=label_smooth)
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    for step in range(steps):
        # ~50% real, ~50% replay/aux (anti-forgetting) when aux exists
        use_real = (not aux_pool) or (rng.random() < 0.5)
        m, fk = (real_pool if use_real else aux_pool)[
            int(rng.integers(len(real_pool if use_real else aux_pool)))]
        logits, _ = net(torch.from_numpy(np.ascontiguousarray(m))[None].to(DEVICE))
        loss = fce(logits.reshape(-1, logits.shape[-1]),
                   torch.from_numpy(fk).to(DEVICE).reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/ctc_skaid_crnn.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--mix-bank", type=int, default=0, help="# synth-from-bank chunks to fold in")
    ap.add_argument("--replay", type=int, default=200, help="# SKAID chunks for anti-forgetting replay")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-adabn", action="store_true")
    ap.add_argument("--no-adapt-rnn", action="store_true")
    a = ap.parse_args()

    def _ft(base, train):
        return finetune(base, train, a.steps, a.lr, not a.no_freeze,
                        aug=not a.no_aug, mix_bank=a.mix_bank, replay=a.replay,
                        adabn=not a.no_adabn, adapt_rnn=not a.no_adapt_rnn)

    sess = load_live()
    print(f"live sessions: {len(sess)}  (alignment hit-ratios: "
          f"{[round(s['hit'],2) for s in sess]})")
    print(f"base={a.base}  steps={a.steps} lr={a.lr} freeze_cnn={not a.no_freeze} "
          f"mix_bank={a.mix_bank} aug={not a.no_aug}\n")

    # zero-shot base for reference
    net0 = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    if os.path.exists(a.base):
        net0.load_state_dict(torch.load(a.base, map_location=DEVICE)); net0.eval()
        for s in sess:
            print(f"  zero-shot {a.base} on {s['wav']}: CER {eval_session(net0, s):.0%}")

    if len(sess) < 2:
        print("need >=2 sessions for leave-one-out; fine-tuning on all, no held-out.")
        net = _ft(a.base, sess)
        for s in sess:
            print(f"  (train) {s['wav']}: CER {eval_session(net, s):.0%}")
        return

    loo = []
    for i in range(len(sess)):
        heldout = sess[i]; train = [s for j, s in enumerate(sess) if j != i]
        print(f"\n== held out {heldout['wav']} (train on {len(train)} other) ==")
        net = _ft(a.base, train)
        cer = eval_session(net, heldout)
        loo.append(cer)
        print(f"  HELD-OUT real CER: {cer:.0%}  (ref {heldout['keys'][:40]})")
        print(f"                     hyp {decode(net, heldout['y'])[:40]}")
    print(f"\nLEAVE-ONE-OUT mean real CER: {np.mean(loo):.0%}")


if __name__ == "__main__":
    main()
