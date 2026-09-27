"""See the attacker read REAL typing — a self-test you can run.

Loads the trained continuous attacker and runs it on held-out SKAID sessions
(typists the model never saw during training), printing what the person actually
typed (REF) vs what the attacker reconstructed from AUDIO ALONE (RAW greedy, and
LM-corrected). Also prints the aggregate character error rate (CER).

Run (CPU by default so it won't fight a GPU job; ~1-2 min for the samples):
    uv run python3 -m keyguard.ctc.demo_skaid
    uv run python3 -m keyguard.ctc.demo_skaid --n 8 --full   # + aggregate over all held-out
    KEYGUARD_DEVICE=cuda uv run python3 -m keyguard.ctc.demo_skaid --full   # faster, needs free GPU

Options:
    --ckpt PATH   attacker checkpoint (default runs/ctc_skaid_crnn.pt)
    --n N         how many sample sessions to print (default 6)
    --full        also compute aggregate CER over ALL held-out chunks
"""
from __future__ import annotations
import argparse
import os

os.environ.setdefault("KEYGUARD_DEVICE", "cpu")     # default CPU: safe alongside GPU jobs
os.environ.setdefault("KEYGUARD_SKAID", "data/continuous/skaid/labels.jsonl")
os.environ.setdefault("KEYGUARD_CURRICULUM", "0")
os.environ.setdefault("KEYGUARD_SPEC_AUG", "0")

import numpy as np
import torch

from . import train_overlap as T
from .data import VOCAB
from .model import greedy_decode, DEVICE
from .overlap_eval import cer_str
from .decode_onset import onset_gated_decode
try:
    from .lm_correct import correct
except Exception:
    correct = None


def _decode(net, m):
    """Onset-gated decode (best): one key per detected keystroke."""
    with torch.no_grad():
        logits, onl = net(torch.from_numpy(m)[None].to(DEVICE))
    return onset_gated_decode(logits, onl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ctc_skaid_crnn.pt")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        raise SystemExit(f"checkpoint not found: {a.ckpt}\n"
                         f"Train one first:  KEYGUARD_SKAID={os.environ['KEYGUARD_SKAID']} "
                         f"uv run python3 -m keyguard.ctc.train_overlap 6000")
    if not os.path.exists(os.environ["KEYGUARD_SKAID"]):
        raise SystemExit("SKAID labels not found — run: uv run python3 -m keyguard.continuous_convert")

    _, test = T.load_skaid(os.environ["KEYGUARD_SKAID"])
    net = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEVICE))
    net.eval()
    print(f"attacker={a.ckpt} model={T.MODEL} device={DEVICE} "
          f"held-out chunks={len(test)} (unseen typists)\n")

    # representative samples: real continuous typing at a plausible key rate
    # (some SKAID sessions have alignment artifacts — implausibly many keys per
    # chunk; those are label noise, not the model, so we don't showcase them).
    def _plausible(c):
        m, ids, on, fk, gap = c
        secs = m.shape[0] * 128 / 16000.0
        rate = len(ids) / max(secs, 1e-6)
        return 10 <= len(ids) <= 40 and 0.8 <= rate <= 8.0
    pool = [c for c in test if _plausible(c)] or test
    samples = pool[:: max(1, len(pool) // max(a.n, 1))][: a.n]
    for m, ids, on, fk, gap in samples:
        ref = "".join(VOCAB[i] for i in ids)
        raw = _decode(net, m)
        line = [f"  TYPED   : {ref[:72]}", f"  ATTACKER: {raw[:72]}   (CER {cer_str(ref, raw):.0%})"]
        if correct is not None:
            line.append(f"  +LM     : {correct(raw)[:72]}")
        print("\n".join(line) + "\n")

    if a.full:
        cers = []
        for m, ids, on, fk, gap in test:
            ref = "".join(VOCAB[i] for i in ids)
            cers.append(cer_str(ref, _decode(net, m)))
        print(f"AGGREGATE over {len(cers)} held-out chunks: "
              f"MEAN CER {np.mean(cers):.1%}  median {np.median(cers):.1%}  "
              f"(chance ~95-100%)")


if __name__ == "__main__":
    main()
