"""GPU-ready trainer for the OVERLAP keystroke transcriber (the novel attacker).

Combines the fixes the literature says are needed to read fast/overlapping typing
(Slater ACSAC'19 end-to-end CTC; EnCTC/label-prior anti-collapse; SED onset head;
sim-to-real augmentation):

  * MtlCTC  = ConvCTC encoder + per-frame ONSET-BCE head (dense gradient that
              stops CTC blank-collapse and localises events without hard segmentation).
  * loss    = CTC(zero_infinity) + lambda_onset * BCE + beta_ent * (-mean entropy)
              (entropy bonus keeps the posterior from going peaky/overconfident).
  * curriculum = train slow (well-separated) typing first, widen to fast/overlapping.
  * augmentation = channel/EQ + noise + gain (augment.py) + a cheap VoIP band-limit,
              so a model trained on synth reads real, over-a-call audio.

Runs on CUDA automatically on a GPU box (no MPS crashes there). Score = sequence
CER vs overlap level (edit distance on the key string; chance ~1.0, lower better).

  GPU box:  uv run python3 -m keyguard.ctc.train_overlap 4000
  Smoke:    KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.ctc.train_overlap demo
  Bank:     KEYGUARD_BANK=data/harrison/MBPWavs (dir) or a .npz per-key bank
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .. import audio, config
from ..config import SR
from . import augment
from .data import synth_line, random_text, sample_bank, SYM_OF_KEY, VOCAB, BLANK, CLIP
from .model import logmel, greedy_decode, DEVICE, HOP, N_MELS, CRNN
from .mtl_probe import MtlCTC, onset_target
from .overlap_eval import cer_str, _ids_to_keys, SPEED_BINS


class MtlCRNN(CRNN):
    """CRNN (CNN -> BiGRU -> CTC) + a per-frame onset-BCE head on the RNN features.
    Slater ACSAC'19 used a recurrent encoder; BiGRU gives the temporal context to
    disambiguate overlapping clicks that the conv-only ConvCTC lacks. Same
    (B,T,mel) -> (logits (B,T,S), onset (B,T)) interface as MtlCTC."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.onset_head = nn.Linear(2 * self.rnn.hidden_size, 1)

    def forward(self, x):                     # x: (B,T,mel)
        x = x.unsqueeze(1).transpose(2, 3)    # (B,1,mel,T)
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.rnn(x)                    # (B,T,2*hidden)
        return self.head(x), self.onset_head(x).squeeze(-1)

BANK = os.environ.get("KEYGUARD_BANK", "data/harrison/MBPWavs")
CKPT = os.environ.get("KEYGUARD_CKPT", "runs/ctc_overlap.pt")
STEPS = int(os.environ.get("KEYGUARD_STEPS", "4000"))
POOL = int(os.environ.get("KEYGUARD_POOL", "2000"))
BATCH = int(os.environ.get("KEYGUARD_BATCH", "16"))
LR = float(os.environ.get("KEYGUARD_LR", "3e-4"))
WARMUP = 200
LAMBDA_ONSET = float(os.environ.get("KEYGUARD_LAMBDA_ONSET", "0.5"))  # onset-BCE weight (research: 0.3-1.0)
LAMBDA_FRAME = float(os.environ.get("KEYGUARD_LAMBDA_FRAME", "1.0"))   # dense frame-CE weight (primary loss)
LAMBDA_CTC = float(os.environ.get("KEYGUARD_LAMBDA_CTC", "0.0"))       # CTC weight (0=off; small values add timing/overlap modeling)
BLANK_W = float(os.environ.get("KEYGUARD_BLANK_W", "0.05"))           # blank class weight in frame-CE (down-weight to beat imbalance)
LABEL_SMOOTH = float(os.environ.get("KEYGUARD_LABEL_SMOOTH", "0.0"))  # frame-CE label smoothing (0.1 helps unseen typists)
FOCAL_GAMMA = float(os.environ.get("KEYGUARD_FOCAL_GAMMA", "0.0"))    # focal frame-CE gamma (0=off; ~1.5 focuses on hard substitution frames)
BETA_ENT = float(os.environ.get("KEYGUARD_BETA_ENT", "0.0"))          # entropy bonus (0 with dense frame-CE; we want confident onsets)
BETA_ENT0 = float(os.environ.get("KEYGUARD_BETA_ENT0", str(BETA_ENT)))  # entropy weight at step 0 (annealed -> BETA_ENT)
ENT_ANNEAL = int(os.environ.get("KEYGUARD_ENT_ANNEAL", "0"))          # steps to anneal BETA_ENT0 -> BETA_ENT (0=off)
MODEL = os.environ.get("KEYGUARD_MODEL", "crnn")                       # 'crnn' (BiGRU, best) or 'conv' (ConvCTC) encoder
WPM_FULL = (int(os.environ.get("KEYGUARD_WPM_LO", "140")),
            int(os.environ.get("KEYGUARD_WPM_HI", "580")))  # synth speed range (match real typist)
SKAID = os.environ.get("KEYGUARD_SKAID")   # path to real continuous-typing labels.jsonl
CHUNK_S = float(os.environ.get("KEYGUARD_CHUNK_S", "5.0"))   # split long sessions


def _voip(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap over-a-call sim: band-limit ~300-3800 Hz + mild quantization (p=0.5)."""
    if rng.random() > 0.5:
        return y
    from scipy import signal
    b, a = signal.butter(4, [300 / (SR / 2), 3800 / (SR / 2)], btype="band")
    z = signal.lfilter(b, a, y).astype(np.float32)
    return (np.round(z * 64) / 64).astype(np.float32)     # coarse quantize


FRAME_WIDTH = int(os.environ.get("KEYGUARD_FRAME_WIDTH", "1"))   # +/- frames labeled as the key


def frame_key_target(n_frames: int, onset_samples, ids) -> np.ndarray:
    """DENSE per-frame CTC-symbol target: blank(0) everywhere, the key id (1..36)
    at each onset frame (widened +/-FRAME_WIDTH). Trained with a class-weighted CE
    (blank down-weighted) this turns the CTC head into a per-frame keystroke
    classifier that emits real keys instead of collapsing to blank — the fix that
    took real-typing CER from ~100% (chance) to legible. -100 padding is added in
    _collate for frames beyond the true length."""
    y = np.zeros(n_frames, dtype=np.int64)                 # blank everywhere
    for s, k in zip(onset_samples, ids):
        f = int(s) // HOP
        for d in range(-FRAME_WIDTH, FRAME_WIDTH + 1):
            if 0 <= f + d < n_frames and y[f + d] == 0:
                y[f + d] = int(k)
    return y


def build_pool(n: int, seed: int = 0, aug: bool = True) -> list:
    """Precompute (logmel, label_ids, onset_target, frame_key, mean_gap) across speeds."""
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n):
        y, lab, on = synth_line(random_text(rng), rng, wpm=WPM_FULL, root=BANK)
        if len(lab) == 0:
            continue
        if aug:
            y = augment.augment_waveform(y, rng)
            y = _voip(y, rng)
        m = logmel(y)
        gap = float(np.mean(np.diff(on))) if len(on) > 1 else float(len(y))
        pool.append((m, lab, onset_target(m.shape[0], on),
                     frame_key_target(m.shape[0], on, lab), gap))
    pool.sort(key=lambda p: -p[4])           # easy (large gap) first for curriculum
    return pool


def _chunk_session(y, keys, onsets, aug, rng):
    """Split one real session into CHUNK_S-second (logmel, ids, onset_target, gap)
    tuples — the same format as the synth pool, so train()/eval() are unchanged."""
    out = []
    step = int(CHUNK_S * SR)
    onsets = np.asarray(onsets)
    for a in range(0, len(y), step):
        b = a + step
        sel = np.where((onsets >= a) & (onsets < b))[0]
        if len(sel) < 2:
            continue
        seg = y[a:b].astype(np.float32)
        if aug:
            seg = augment.augment_waveform(seg, rng)
        m = logmel(seg)
        ids = np.array([SYM_OF_KEY[keys[i]] for i in sel], np.int64)
        rel = onsets[sel] - a
        gap = float(np.mean(np.diff(onsets[sel]))) if len(sel) > 1 else step
        out.append((m, ids, onset_target(m.shape[0], rel),
                    frame_key_target(m.shape[0], rel, ids), gap))
    return out


# Split policy for real continuous data:
#   "session" (default): held-out = every 5th session by wav name. Fast but, when a
#      phrase is typed multiple times, the SAME text can land in both train and test
#      -> the number is optimistic (partial text memorization).
#   "phrase": held-out = whole PHRASE GROUPS (sessions sharing a normalized text).
#      Test text is then NEVER seen during training -> honest generalization to novel
#      text. This is the number to trust for the "reads my typing" claim.
SPLIT = os.environ.get("KEYGUARD_SPLIT", "session")
MIN_RMS = float(os.environ.get("KEYGUARD_MIN_RMS", "1e-3"))  # drop silent/dead-mic sessions


def _phrase_key(keys: str) -> str:
    """Normalized identity of a typed phrase (dedupe near-identical prompts)."""
    return "".join(c for c in keys.upper() if c.isalnum())[:48]


def load_skaid(path, seed=0, aug=True):
    """Real continuous typing -> (train_chunks, test_chunks). See SPLIT above."""
    import json
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("keys") and r.get("onset_samples")]
    rows = sorted(rows, key=lambda r: r["wav"])
    # Drop silent/dead-mic recordings: training on silence->text is pure
    # memorization and poisons generalization. Gate on waveform RMS.
    def _ok(r):
        wav = os.path.join(os.path.dirname(path), r["wav"])
        if not os.path.exists(wav):
            return False
        y = audio.load(wav)
        return float(np.sqrt(np.mean(np.square(y)))) >= MIN_RMS
    n0 = len(rows)
    rows = [r for r in rows if _ok(r)]
    if len(rows) < n0:
        print(f"audio gate: kept {len(rows)}/{n0} sessions (dropped {n0-len(rows)} "
              f"silent/dead-mic, rms<{MIN_RMS})", flush=True)
    # Decide which row indices are held out.
    if SPLIT == "phrase":
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(_phrase_key(r["keys"]), []).append(i)
        gkeys = sorted(groups)
        grng = np.random.default_rng(seed)
        grng.shuffle(gkeys)
        n_test = max(1, len(gkeys) // 5)
        test_rows = {i for gk in gkeys[:n_test] for i in groups[gk]}
        print(f"phrase split: {len(gkeys)} distinct phrases, {n_test} held out "
              f"(test text unseen in train)", flush=True)
    else:
        test_rows = {i for i in range(len(rows)) if i % 5 == 0}
    rng = np.random.default_rng(seed)
    train, test = [], []
    for i, r in enumerate(rows):
        wav = os.path.join(os.path.dirname(path), r["wav"])
        if not os.path.exists(wav):
            continue
        keys = [k for k in r["keys"].upper() if k in SYM_OF_KEY]
        on = [s for k, s in zip(r["keys"].upper(), r["onset_samples"]) if k in SYM_OF_KEY]
        is_test = i in test_rows
        chunks = _chunk_session(audio.load(wav), keys, on, aug and not is_test, rng)
        (test if is_test else train).append(chunks)
    tr = [c for s in train for c in s]
    te = [c for s in test for c in s]
    tr.sort(key=lambda p: -p[4])                # easy (large gap) first for curriculum
    return tr, te


# SpecAugment time-shift (np.roll) moves the mel but NOT the per-frame onset/frame-CE
# labels, which desyncs them and blocks frame-aligned learning. Off by default now that
# frame-CE is the primary loss; re-enable only with a time-shift-free spec_augment.
SPEC_AUG = os.environ.get("KEYGUARD_SPEC_AUG", "0") != "0"


FREQ_MASK = int(os.environ.get("KEYGUARD_FREQ_MASK", "0"))   # # of frequency masks (timing-safe SpecAugment)
FREQ_MASK_W = int(os.environ.get("KEYGUARD_FREQ_MASK_W", "8"))  # max mel bins per mask


def _freq_mask(m, rng):
    """Frequency-only SpecAugment: mask mel bands (NO time shift, so per-frame
    onset/frame-CE labels stay aligned). m: (T, mel)."""
    m = m.copy()
    fill = float(m.mean())
    for _ in range(FREQ_MASK):
        w = int(rng.integers(1, FREQ_MASK_W + 1))
        s = int(rng.integers(0, max(1, m.shape[1] - w)))
        m[:, s:s + w] = fill
    return m


def _collate(items, rng, spec_aug=SPEC_AUG):
    maxT = max(m.shape[0] for m, *_ in items)
    B = len(items)
    mel = torch.zeros(B, maxT, N_MELS)
    onset = torch.zeros(B, maxT)
    fkey = torch.full((B, maxT), -100, dtype=torch.long)   # frame-CE target (-100=ignore)
    in_len = torch.zeros(B, dtype=torch.long)
    tgts, tlen = [], []
    for i, (m, lab, on, fk, _) in enumerate(items):
        if spec_aug:
            m = augment.spec_augment(m, rng)
        if FREQ_MASK > 0:
            m = _freq_mask(m, rng)
        t = m.shape[0]
        mel[i, :t] = torch.from_numpy(np.ascontiguousarray(m))
        onset[i, :t] = torch.from_numpy(on)
        fkey[i, :t] = torch.from_numpy(fk)
        in_len[i] = t
        tgts.append(torch.from_numpy(lab)); tlen.append(len(lab))
    return (mel.to(DEVICE), onset.to(DEVICE), fkey.to(DEVICE), in_len,
            torch.cat(tgts), torch.tensor(tlen, dtype=torch.long))


CURRICULUM = os.environ.get("KEYGUARD_CURRICULUM", "1") != "0"


def _curriculum_slice(pool, step):
    """Fraction of the (gap-sorted) pool visible so far: 40% -> 100% over training.
    OFF (KEYGUARD_CURRICULUM=0) => full pool every step. For dense frame-CE the
    curriculum HURTS: the sparsest (largest-gap) chunks it shows first are almost
    all blank, so blank dominates the CE and the head collapses. Uniform sampling
    over the whole pool avoids that."""
    if not CURRICULUM:
        return pool
    frac = min(1.0, 0.4 + 0.6 * step / max(1, STEPS * 0.6))
    return pool[: max(BATCH, int(frac * len(pool)))]


@torch.no_grad()
def evaluate_chunks(net, chunks, cap=120) -> dict:
    """CER over real held-out chunks (SKAID). Groups by overlap level for the table."""
    net.eval()
    per = []
    for m, ids, _on, _fk, gap in chunks[:cap]:
        logits, _ = net(torch.from_numpy(m)[None].to(DEVICE))
        hyp = _ids_to_keys(greedy_decode(logits)[0])
        ref = "".join(VOCAB[i] for i in ids)
        ov = float(np.mean(np.diff(np.where(_on > 0.5)[0]) < (CLIP // HOP))) if _on.sum() > 1 else 0.0
        per.append((cer_str(ref, hyp), gap))
    if not per:
        return {"rows": [], "mean_cer": 1.0}
    cers = [c for c, _ in per]
    return {"rows": [{"wpm": "real", "overlap": 0.0, "cer": float(np.mean(cers))}],
            "mean_cer": float(np.mean(cers)), "n_chunks": len(per)}


@torch.no_grad()
def evaluate(net, seed=100, lines=12) -> dict:
    net.eval()
    rows = []
    for lo, hi in SPEED_BINS:
        rng = np.random.default_rng(seed + lo)
        ov, ce = [], []
        for _ in range(lines):
            y, lab, on = synth_line(random_text(rng), rng, wpm=(lo, hi), root=BANK)
            if len(lab) == 0:
                continue
            gaps = np.diff(on)
            ov.append(float(np.mean(gaps < CLIP)) if len(gaps) else 0.0)
            logits, _ = net(torch.from_numpy(logmel(y))[None].to(DEVICE))
            hyp = _ids_to_keys(greedy_decode(logits)[0])
            ce.append(cer_str(_ids_to_keys(list(lab)), hyp))
        rows.append({"wpm": (lo + hi) // 2, "overlap": float(np.mean(ov)),
                     "cer": float(np.mean(ce))})
    return {"rows": rows, "mean_cer": float(np.mean([r["cer"] for r in rows]))}


def _print_eval(e, tag=""):
    print(f"  eval{tag}: " + "  ".join(
        f"{r['wpm']}wpm(ov{r['overlap']:.0%}) CER {r['cer']:.1%}" for r in e["rows"])
        + f"  | MEAN {e['mean_cer']:.1%}", flush=True)


def train():
    import json
    print(f"device={DEVICE} bank={BANK} steps={STEPS} pool={POOL} "
          f"batch={BATCH} onset={LAMBDA_ONSET} ent={BETA_ENT}", flush=True)
    skaid_test = None
    if SKAID:
        pool, skaid_test = load_skaid(SKAID)
        print(f"SKAID real data: {len(pool)} train chunks, {len(skaid_test)} held-out "
              f"test chunks (chunk={CHUNK_S}s)", flush=True)
        assert len(pool) > BATCH, "not enough SKAID train chunks — check the path"
    else:
        assert len(sample_bank(BANK)) >= 20, f"bank {BANK} has too few keys"
        pool = build_pool(POOL)
        print(f"pool {len(pool)} synth lines (gap-sorted for curriculum); training...", flush=True)

    # Source mixing (combined pretrain): fold extra real SKAID chunks and/or extra
    # synth lines into the training pool so one run learns BOTH the target keyboard's
    # per-key acoustics (synth-bank) AND real continuous-typing dynamics (SKAID).
    mix_skaid = os.environ.get("KEYGUARD_MIX_SKAID")
    mix_synth = int(os.environ.get("KEYGUARD_MIX_SYNTH", "0"))
    if mix_skaid:
        extra, _ = load_skaid(mix_skaid)
        pool = pool + extra
        print(f"mix +{len(extra)} SKAID real chunks -> pool {len(pool)}", flush=True)
    if mix_synth > 0:
        extra = build_pool(mix_synth)
        pool = pool + extra
        print(f"mix +{len(extra)} synth lines -> pool {len(pool)}", flush=True)
    if mix_skaid or mix_synth:
        pool.sort(key=lambda p: -p[4])          # re-sort combined pool for curriculum

    torch.manual_seed(0)
    net = (MtlCRNN if MODEL == "crnn" else MtlCTC)().to(DEVICE).train()
    init = os.environ.get("KEYGUARD_INIT")   # warm-start weights (pretrain -> finetune)
    if init and os.path.exists(init):
        sd = torch.load(init, map_location=DEVICE)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"warm-start from {init} (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    print(f"model={MODEL} params={sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"onset={LAMBDA_ONSET} ent0={BETA_ENT0}->ent={BETA_ENT}@{ENT_ANNEAL}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    # class-weighted dense frame-CE: down-weight blank so the ~4% keystroke frames
    # dominate; -100 padding ignored. This is the primary anti-collapse loss.
    # label smoothing calibrates on unseen typists; focal (gamma>0) concentrates
    # gradient on hard, low-margin substitution frames.
    fce_w = torch.ones(len(VOCAB), device=DEVICE); fce_w[BLANK] = BLANK_W
    if FOCAL_GAMMA > 0:
        _ce_none = nn.CrossEntropyLoss(weight=fce_w, ignore_index=-100,
                                       label_smoothing=LABEL_SMOOTH, reduction="none")

        def fce(pred, tgt):
            ce = _ce_none(pred, tgt)                       # (N,) per-frame, 0 where ignored
            pt = torch.exp(-ce.clamp(max=30))             # ~ p_true
            valid = (tgt != -100)
            foc = ((1 - pt) ** FOCAL_GAMMA) * ce
            return foc[valid].mean() if valid.any() else foc.sum()
    else:
        fce = nn.CrossEntropyLoss(weight=fce_w, ignore_index=-100,
                                  label_smoothing=LABEL_SMOOTH)
    brng = np.random.default_rng(1234)
    best = 1e9
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        sub = _curriculum_slice(pool, step)
        idx = brng.integers(0, len(sub), size=BATCH)
        mel, onset, fkey, in_len, tgt, tlen = _collate([sub[i] for i in idx], brng)
        logits, onset_logit = net(mel)
        loss = logits.new_zeros(())
        if LAMBDA_CTC > 0:
            # CTC on-device for CUDA (the .cpu() is an MPS-only workaround: MPS has
            # no CTC kernel). Shipping logits to CPU every step kills throughput.
            logp = logits.log_softmax(-1).permute(1, 0, 2)
            if DEVICE != "cuda":
                logp = logp.cpu()
            loss = loss + LAMBDA_CTC * ctc(logp, tgt.to(logp.device),
                                           in_len.clamp(max=logits.shape[1]), tlen)
        mask = (torch.arange(mel.shape[1], device=DEVICE)[None] < in_len.to(DEVICE)[:, None]).float()
        loss = loss + LAMBDA_ONSET * (bce(onset_logit, onset) * mask).sum() / mask.sum()
        # frame-CE: force the CTC head to emit the true key id at each onset frame
        # (-100 elsewhere is ignored). Directly opposes blank/prior collapse.
        loss = loss + LAMBDA_FRAME * fce(logits.reshape(-1, logits.shape[-1]), fkey.reshape(-1))
        p = logits.softmax(-1).clamp_min(1e-8)
        ent = -(p * p.log()).sum(-1).mean()                 # posterior entropy
        beta = BETA_ENT if ENT_ANNEAL <= 0 else (
            BETA_ENT + (BETA_ENT0 - BETA_ENT) * max(0.0, 1 - step / ENT_ANNEAL))
        loss = loss - beta * ent                            # bonus (subtract), annealed high->low
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if DEVICE == "mps" and step % 25 == 0:
            torch.mps.empty_cache()
        if step % 250 == 0 or step == STEPS - 1:
            e = evaluate_chunks(net, skaid_test) if skaid_test is not None else evaluate(net)
            tag = ""
            if e["mean_cer"] < best:
                best = e["mean_cer"]; torch.save(net.state_dict(), CKPT); tag = "  <-best"
            print(f"step {step:4d} loss {float(loss):.3f} ent {float(ent):.2f}"
                  f" ({(time.time()-t0)/max(1,step+1)*1000:.0f}ms/step)", flush=True)
            _print_eval(e, tag)
            net.train()
    print(f"done. best mean CER {best:.1%}; weights {CKPT}", flush=True)
    final = evaluate_chunks(net, skaid_test) if skaid_test is not None else evaluate(net)
    (config.RUNS / "overlap_train_result.json").write_text(
        json.dumps({"best_mean_cer": best, "steps": STEPS, "device": DEVICE,
                    "data": "skaid" if SKAID else "synth", "final_eval": final}, indent=2))
    # Agent memory: remember this experiment so the self-evolving loop knows what
    # was tried and how good it was (Backboard when BACKBOARD_API_KEY set, else local).
    try:
        from .. import memory
        cfg = {"model": MODEL, "data": "skaid" if SKAID else "synth", "steps": STEPS,
               "lr": LR, "batch": BATCH, "blank_w": BLANK_W, "frame_width": FRAME_WIDTH,
               "label_smooth": LABEL_SMOOTH, "focal_gamma": FOCAL_GAMMA,
               "lambda_frame": LAMBDA_FRAME, "lambda_ctc": LAMBDA_CTC,
               "lambda_onset": LAMBDA_ONSET, "freq_mask": FREQ_MASK}
        memory.remember_experiment(os.environ.get("KEYGUARD_TAG", os.path.basename(CKPT)),
                                   best, cfg, note=f"weights={CKPT}")
    except Exception:
        pass


def demo():
    """CPU smoke: pool builds, one train step runs, eval returns in-range CER."""
    global STEPS, POOL
    STEPS, POOL = 2, 12
    pool = build_pool(6, aug=True)
    assert pool and pool[0][0].shape[1] == N_MELS
    net = MtlCTC().to(DEVICE).train()
    rng = np.random.default_rng(0)
    mel, onset, fkey, in_len, tgt, tlen = _collate(pool[:2], rng)
    logits, onset_logit = net(mel)
    assert logits.shape[0] == 2 and logits.shape[2] == len(VOCAB)
    assert onset_logit.shape == onset.shape
    e = evaluate(net, lines=2)
    assert e["mean_cer"] >= 0.0    # untrained CER can be large (insertion-heavy); just sane & finite
    print(f"train_overlap demo ok on {DEVICE}: pool {len(pool)}, "
          f"logits {tuple(logits.shape)}, eval mean CER {e['mean_cer']:.1%} "
          f"(untrained). Full run: uv run python3 -m keyguard.ctc.train_overlap 4000")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "demo":
        demo()
    else:
        if arg.isdigit():
            STEPS = int(arg)
        train()
