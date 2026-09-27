"""Arms race (min-max) between the audio defender and a RETRAINING attacker.

Proves adaptivity - i.e. AI - is essential: a FIXED perturbation is clawed back by
an attacker that retrains on defended audio, but RE-OPTIMIZING the defender each
round re-blinds it. Per round we report the protected-span CER right after the
defender crafts (high = protected) and again after the attacker adapts (drop =
clawed back); the defender then re-crafts and re-protects.

Run (GPU recommended):
    KEYGUARD_DEVICE=cuda uv run python3 -m keyguard.agents.arms_race --clips 16 --rounds 3
Writes runs/arena/arms_race.json for the demo visualization.
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from .. import config, audio
from ..ctc import train_overlap as T
from ..ctc.model import logmel, DEVICE
from ..ctc.data import VOCAB, SYM_OF_KEY, BLANK
from ..ctc.overlap_eval import cer_str
from ..config import SR, CLS_IDX, RUNS, HOP
from . import defense_audio as DA

SPAN = 7


def load_clips(n, min_keys=16):
    rows = [json.loads(l) for l in open("data/continuous/skaid/labels.jsonl")]
    clips = []
    for sess in rows:
        y = audio.load("data/continuous/skaid/" + sess["wav"])
        on = np.array(sess["onset_samples"]); keys = sess["keys"].upper()
        for start in range(2, len(on) - min_keys, min_keys):
            t0 = int(on[start]); seg = y[t0:t0 + 5 * SR].astype(np.float32)
            idx = [i for i, s in enumerate(on) if t0 <= s < t0 + 5 * SR and keys[i] in CLS_IDX]
            if len(idx) < min_keys:
                continue
            ktext = "".join(keys[i] for i in idx)
            konset = [int(on[i] - t0) for i in idx]
            s = len(ktext) // 3
            clips.append(dict(seg=seg, ktext=ktext, konset=konset, s=s, e=s + SPAN))
            if len(clips) >= n:
                return clips
    return clips


def _span_bounds(c):
    lo = max(0, c["konset"][c["s"]] - int(0.03 * SR))
    hi = min(len(c["seg"]), c["konset"][c["e"] - 1] + int(0.12 * SR))
    return lo, hi


def span_cer(net, y, c):
    lo, hi = _span_bounds(c)
    ref = c["ktext"][c["s"]:c["e"]]
    return cer_str(ref, DA.span_read(net, y, lo, hi))


def adapt_attacker(net, defended, steps=250, lr=3e-4):
    """Fine-tune the attacker to read through the current defended audio."""
    net.train()
    fce_w = torch.ones(len(VOCAB), device=DEVICE); fce_w[BLANK] = 0.05
    fce = nn.CrossEntropyLoss(weight=fce_w, ignore_index=-100)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    prepped = []
    for yp, c in defended:
        m = logmel(yp)
        ids = np.array([SYM_OF_KEY[ch] for ch in c["ktext"]], np.int64)
        fk = T.frame_key_target(m.shape[0], c["konset"], ids)
        prepped.append((m, fk))
    rng = np.random.default_rng(0)
    for _ in range(steps):
        i = int(rng.integers(len(prepped)))
        m, fk = prepped[i]
        mel = torch.from_numpy(m)[None].to(DEVICE)
        logits, _ = net(mel)
        loss = fce(logits.reshape(-1, logits.shape[-1]),
                   torch.from_numpy(fk).to(DEVICE).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--snr", type=float, default=18.0)
    ap.add_argument("--ckpt", default="runs/ctc_skaid_crnn.pt")
    a = ap.parse_args()

    net = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEVICE)); net.eval()
    clips = load_clips(a.clips)
    print(f"arms race: {len(clips)} clips, {a.rounds} rounds, span={SPAN} keys, "
          f"budget SNR {a.snr} dB, device={DEVICE}")

    base_clean = float(np.mean([span_cer(net, c["seg"], c) for c in clips]))
    print(f"round 0  attacker on CLEAN audio: protected-span CER {base_clean:.0%} (reads it)")

    history = [{"round": 0, "clean_cer": base_clean}]
    for r in range(1, a.rounds + 1):
        # DEFENDER crafts vs the CURRENT attacker
        defended, protected = [], []
        for c in clips:
            lo = max(0, c["konset"][c["s"]] - int(0.03 * SR))
            hi = min(len(c["seg"]), c["konset"][c["e"] - 1] + int(0.12 * SR))
            yp, info = DA.craft(net, c["seg"], lo, hi, mode="protect", snr_db=a.snr, steps=120)
            defended.append((yp, c)); protected.append(span_cer(net, yp, c))
        prot = float(np.mean(protected))
        print(f"round {r}  DEFENDER crafts -> protected-span CER {prot:.0%} (blinded)")

        # ATTACKER retrains on the defended audio to claw back
        adapt_attacker(net, defended)
        clawed = float(np.mean([span_cer(net, yp, c) for yp, c in defended]))
        print(f"round {r}  ATTACKER retrains -> protected-span CER {clawed:.0%} (clawed back)")
        history.append({"round": r, "protected_cer": prot, "clawed_back_cer": clawed})

    out = RUNS / "arena"; out.mkdir(parents=True, exist_ok=True)
    (out / "arms_race.json").write_text(json.dumps(
        {"span_keys": SPAN, "snr_db": a.snr, "clips": len(clips), "history": history}, indent=2))
    print(f"\nInterpretation: each round the fixed perturbation is clawed back by the "
          f"retraining attacker, and the defender must RE-OPTIMIZE to re-protect - a "
          f"moving target no fixed/garble defense can win. -> {out/'arms_race.json'}")


if __name__ == "__main__":
    main()
