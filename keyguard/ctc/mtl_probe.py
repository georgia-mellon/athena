"""Prototype: does a multi-task per-frame ONSET head fix the CTC mode-collapse?

Plain free-typing CTC collapses on real/held-out data (emits the same common
letters / blanks regardless of input) though it overfits synth to ~4% CER. The
hypothesis: adding a per-frame onset-probability head (BCE vs weak onset targets
from known keystroke positions) forces the encoder to mark WHERE keystrokes are,
stabilising CTC on sparse-event audio.

This trains TWO otherwise-identical models on the same Harrison synth pool/steps:
  (a) plain CTC        (lambda_bce = 0, onset head gets no gradient)
  (b) CTC + onset BCE  (total = ctc + lambda*bce)
then measures held-out greedy CER at slow and fast (overlapping) speeds, and
prints TRUE/PRED samples + collapse diagnostics. Standalone: does NOT touch the
main pipeline (keyguard/ctc/*.py). ponytail: one class for both arms — the only
difference is lambda, so encoder init/data/steps are provably identical.

Run: uv run python3 -m keyguard.ctc.mtl_probe
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn

from ..config import SR
from .data import synth_line, random_text, VOCAB, BLANK, CLIP
from .model import (ConvCTC, logmel, greedy_decode, cer, ids_to_str,
                    DEVICE, HOP, N_MELS)

# ---- knobs (ponytail: the tuning surface lives here) ----
POOL_SIZE = 1500          # cached synth lines (logmels precomputed once)
BATCH = 16
STEPS = 1000
LR = 1e-4
WARMUP = 50
LAMBDA_BCE = 0.5          # onset-head weight (research: 0.3-1.0)
ONSET_TOL = 1             # +/- frames counted as onset (weak-label smoothing)
EVAL_N = 40               # held-out lines per speed
WPM_SLOW = (150, 220)     # little overlap
WPM_FAST = (420, 540)     # heavy overlap
EMPTY_EVERY = 25          # torch.mps.empty_cache() cadence (avoid MPS OOM)
ROOT = "data/harrison/MBPWavs"


class MtlCTC(ConvCTC):
    """ConvCTC encoder + a second Conv1d head predicting per-frame onset prob.

    Reuses every ConvCTC layer (cnn/proj/blocks/CTC head) verbatim so arm (a)
    and arm (b) share an identical encoder; adds one 1x1 Conv1d onset head."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.onset_head = nn.Conv1d(self.proj.out_channels, 1, 1)

    def forward(self, x):                     # x: (B,T,mel)
        x = x.unsqueeze(1).transpose(2, 3)    # (B,1,mel,T)
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.reshape(b, c * f, t)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x).transpose(1, 2)     # (B,T,S) CTC logits
        onset = self.onset_head(x).squeeze(1)     # (B,T) onset logit per frame
        return logits, onset


def onset_target(n_frames: int, onset_samples: np.ndarray) -> np.ndarray:
    """Weak per-frame onset labels: 1 near a keystroke onset frame, else 0."""
    y = np.zeros(n_frames, np.float32)
    for s in onset_samples:
        f = int(s) // HOP
        for d in range(-ONSET_TOL, ONSET_TOL + 1):
            if 0 <= f + d < n_frames:
                y[f + d] = 1.0
    return y


def build_pool(rng: np.random.Generator, n: int):
    """Precompute (logmel, label_ids, onset_target) for n synth lines once."""
    pool = []
    for _ in range(n):
        text = random_text(rng)
        y, lab, on = synth_line(text, rng, root=ROOT)
        if len(lab) == 0:
            continue
        mel = logmel(y)                        # (T, mel)
        pool.append((mel, lab, onset_target(mel.shape[0], on)))
    return pool


def collate(items):
    """Pad a list of (mel,lab,onset) to a batch. Returns MPS-ready tensors."""
    maxT = max(m.shape[0] for m, _, _ in items)
    B = len(items)
    mel = torch.zeros(B, maxT, N_MELS)
    onset = torch.zeros(B, maxT)
    in_len = torch.zeros(B, dtype=torch.long)
    targets, tgt_len = [], []
    for i, (m, lab, on) in enumerate(items):
        t = m.shape[0]
        mel[i, :t] = torch.from_numpy(m)
        onset[i, :t] = torch.from_numpy(on)
        in_len[i] = t
        targets.append(torch.from_numpy(lab))
        tgt_len.append(len(lab))
    return (mel.to(DEVICE), onset.to(DEVICE), in_len,
            torch.cat(targets), torch.tensor(tgt_len, dtype=torch.long))


def train(pool, lambda_bce: float, seed: int = 0):
    """Train one MtlCTC arm. lambda_bce=0 -> plain CTC (onset head unused)."""
    torch.manual_seed(seed)                    # identical encoder init per arm
    net = MtlCTC().to(DEVICE).train()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    # CTC loss runs on CPU: MPS has no CTC kernel.
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    brng = np.random.default_rng(1234)         # same batch order for both arms
    N = len(pool)

    for step in range(STEPS):
        for g in opt.param_groups:             # linear warmup
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        idx = brng.integers(0, N, size=BATCH)
        mel, onset, in_len, targets, tgt_len = collate([pool[i] for i in idx])

        logits, onset_logit = net(mel)                       # (B,T,S),(B,T)
        logp = logits.log_softmax(-1).permute(1, 0, 2).cpu() # (T,B,S) on CPU
        loss = ctc(logp, targets, in_len, tgt_len)
        if lambda_bce > 0:
            mask = (torch.arange(mel.shape[1], device=DEVICE)[None, :]
                    < in_len.to(DEVICE)[:, None]).float()
            b = (bce(onset_logit, onset) * mask).sum() / mask.sum()
            loss = loss + lambda_bce * b

        opt.zero_grad()
        loss.backward()
        opt.step()
        if DEVICE == "mps" and step % EMPTY_EVERY == 0:
            torch.mps.empty_cache()
    return net.eval()


@torch.no_grad()
def evaluate(net, wpm, n, seed):
    """Held-out greedy CER + collapse diagnostics on fresh synth lines."""
    rng = np.random.default_rng(seed)
    cers, preds, refs = [], [], []
    for _ in range(n):
        text = random_text(rng)
        y, lab, _ = synth_line(text, rng, wpm=wpm, root=ROOT)
        if len(lab) == 0:
            continue
        mel = torch.from_numpy(logmel(y))[None].to(DEVICE)
        logits, _ = net(mel)
        hyp = greedy_decode(logits)[0]
        cers.append(cer(list(lab), hyp))
        preds.append(ids_to_str(hyp))
        refs.append(ids_to_str(list(lab)))
    mean = float(np.mean(cers))
    n_unique = len(set(preds))                 # collapse -> few unique outputs
    mean_len = float(np.mean([len(p) for p in preds]))
    return mean, n_unique, mean_len, refs, preds


def _samples(refs, preds, k=3):
    return "\n".join(f"    TRUE: {r!r}\n    PRED: {p!r}"
                     for r, p in zip(refs[:k], preds[:k]))


def main():
    t0 = time.time()
    print(f"device={DEVICE} pool={POOL_SIZE} batch={BATCH} steps={STEPS} "
          f"lambda={LAMBDA_BCE}")
    pool = build_pool(np.random.default_rng(0), POOL_SIZE)
    print(f"pool built: {len(pool)} lines in {time.time()-t0:.0f}s")

    arms = {}
    for name, lam in (("baseline_ctc", 0.0), ("ctc+onset", LAMBDA_BCE)):
        ts = time.time()
        net = train(pool, lam, seed=0)
        arms[name] = net
        print(f"trained {name} (lambda={lam}) in {time.time()-ts:.0f}s")

    print("\n=== HELD-OUT (greedy CER, fresh synth) ===")
    for name, net in arms.items():
        print(f"\n[{name}]")
        for label, wpm, seed in (("slow", WPM_SLOW, 777), ("fast", WPM_FAST, 778)):
            m, nu, ml, refs, preds = evaluate(net, wpm, EVAL_N, seed)
            print(f"  {label:4s} wpm={wpm}: CER={m:.3f}  "
                  f"unique_preds={nu}/{EVAL_N}  mean_pred_len={ml:.1f}")
            print(_samples(refs, preds))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
