"""Attacker-vs-defense arena for the CONTINUOUS CTC keystroke transcriber.

Measures the trained MtlCRNN attacker's MEAN CER on held-out SKAID audio:
  (a) CLEAN   : the real recording, chunked and decoded (baseline attacker win).
  (b) SHIELDED: the DSP shield (keyguard/shield/shield.py) applied to each
                held-out session's *raw* audio using the TRUE OS onset timestamps,
                then re-chunked and decoded.

Also reports speech-quality of the shield (STOI/PESQ, shielded vs original) so we
only credit a defense that keeps the audio perceptually intact.

Held-out = every 5th session (matches load_skaid), so the typist/recording is unseen.

Usage:
  KEYGUARD_DEVICE=cuda PYTHONPATH=$(pwd) python -m keyguard.arena.ctc_arena \
      --ckpt runs/ctc_skaid_crnn.pt --strength 1.0 --randomize 1.0 --decoys 0

  # dump shielded TRAIN chunks for the adaptive attacker:
  ... --dump-train scratch/shielded_train.pt
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from .. import audio
from ..config import SR
from ..ctc.data import SYM_OF_KEY, VOCAB
from ..ctc.model import greedy_decode, DEVICE, logmel
from ..ctc.overlap_eval import _ids_to_keys, cer_str
from ..ctc.train_overlap import MtlCRNN, _chunk_session
from ..shield.shield import Shield, ShieldConfig
from ..eval.metrics import speech_quality


def _sessions(path):
    """Yield (idx, wav_path, keys[list], onsets[np.ndarray]) per session, sorted by
    wav (same order load_skaid uses so 'every 5th' is the same held-out split)."""
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("keys") and r.get("onset_samples")]
    for i, r in enumerate(sorted(rows, key=lambda r: r["wav"])):
        wav = os.path.join(os.path.dirname(path), r["wav"])
        if not os.path.exists(wav):
            continue
        keys = [k for k in r["keys"].upper() if k in SYM_OF_KEY]
        on = [s for k, s in zip(r["keys"].upper(), r["onset_samples"]) if k in SYM_OF_KEY]
        yield i, wav, keys, np.asarray(on, dtype=np.int64)


def build_chunks(path, cfg: ShieldConfig | None, held_out=True, seed=0,
                 collect_speech=False):
    """Return (chunks, speech_rows). If cfg is None -> clean audio; else shield each
    session's raw audio with its true onsets before chunking. chunks are
    (logmel, ids, onset_target, frame_key, gap) tuples ready for the attacker."""
    rng = np.random.default_rng(seed)
    chunks, speech = [], []
    for i, wav, keys, on in _sessions(path):
        is_test = (i % 5 == 0)
        if held_out != is_test:
            continue
        if len(keys) < 2:
            continue
        y = audio.load(wav)
        if cfg is not None:
            sh = Shield(cfg, seed=i)          # deterministic per session
            y_use = sh.apply(y, onsets=on)
            if collect_speech:
                speech.append(speech_quality(y, y_use, SR))
        else:
            y_use = y
        # aug=False: honest eval, no train-time augmentation on the eval audio
        chunks.extend(_chunk_session(y_use, keys, on, aug=False, rng=rng))
    return chunks, speech


@torch.no_grad()
def eval_cer(net, chunks, cap=None):
    net.eval()
    cers = []
    for m, ids, _on, _fk, _gap in (chunks if cap is None else chunks[:cap]):
        logits, _ = net(torch.from_numpy(m)[None].to(DEVICE))
        hyp = _ids_to_keys(greedy_decode(logits)[0])
        ref = "".join(VOCAB[i] for i in ids)
        cers.append(cer_str(ref, hyp))
    return float(np.mean(cers)) if cers else 1.0, len(cers)


def load_net(ckpt):
    net = MtlCRNN().to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net.eval()
    return net


def _agg_speech(rows):
    out = {}
    for k in ("stoi", "pesq"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/continuous/skaid/labels.jsonl")
    ap.add_argument("--ckpt", default="runs/ctc_skaid_crnn.pt")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--randomize", type=float, default=1.0)
    ap.add_argument("--decoys", type=int, default=0)
    ap.add_argument("--key-frames", type=int, default=14)
    ap.add_argument("--ctx-frames", type=int, default=12)
    ap.add_argument("--dump-train", default="")   # path to save shielded TRAIN chunks
    ap.add_argument("--dump-clean-train", default="")  # save clean TRAIN chunks
    ap.add_argument("--dump-shielded-test", default="")  # save shielded TEST chunks
    args = ap.parse_args()

    cfg = ShieldConfig(strength=args.strength, randomize=args.randomize,
                       decoys=args.decoys, key_frames=args.key_frames,
                       ctx_frames=args.ctx_frames)
    print(f"device={DEVICE} ckpt={args.ckpt}")
    print(f"shield cfg: {cfg}")

    net = load_net(args.ckpt)

    clean_chunks, _ = build_chunks(args.path, None, held_out=True)
    shield_chunks, speech = build_chunks(args.path, cfg, held_out=True,
                                         collect_speech=True)
    print(f"held-out chunks: clean={len(clean_chunks)} shielded={len(shield_chunks)}")

    clean_cer, nc = eval_cer(net, clean_chunks)
    shield_cer, ns = eval_cer(net, shield_chunks)
    sq = _agg_speech(speech)
    print(f"\n=== ATTACKER CER (held-out unseen typists) ===")
    print(f"  CLEAN    : {clean_cer:.1%}  (n={nc})")
    print(f"  SHIELDED : {shield_cer:.1%}  (n={ns})")
    print(f"  delta    : {shield_cer - clean_cer:+.1%}")
    print(f"  speech   : {sq}  (shielded vs original; STOI 1=perfect, PESQ ->4.5)")

    result = {"clean_cer": clean_cer, "shielded_cer": shield_cer,
              "speech": sq, "shield_cfg": vars(cfg), "n_clean": nc, "n_shield": ns}

    if args.dump_train:
        tr, _ = build_chunks(args.path, cfg, held_out=False)
        torch.save(tr, args.dump_train)
        print(f"dumped {len(tr)} shielded TRAIN chunks -> {args.dump_train}")
    if args.dump_clean_train:
        tr, _ = build_chunks(args.path, None, held_out=False)
        torch.save(tr, args.dump_clean_train)
        print(f"dumped {len(tr)} clean TRAIN chunks -> {args.dump_clean_train}")
    if args.dump_shielded_test:
        torch.save(shield_chunks, args.dump_shielded_test)
        print(f"dumped {len(shield_chunks)} shielded TEST chunks -> {args.dump_shielded_test}")

    print("\nJSON " + json.dumps(result))
    return result


if __name__ == "__main__":
    main()
