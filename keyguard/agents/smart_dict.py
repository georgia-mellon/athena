"""Smart-Dictionary attack — rank the most likely SECRETS from Ares's acoustics.

Idea borrowed (reimplemented from scratch, no GPL code) from Skype&Type-style
"smart dictionary" attacks: don't just read characters, turn the per-keystroke
acoustic posteriors over the secret's span into a RANKED list of candidate
strings. The scary metric for a judge isn't CER — it's:

  "Ares narrowed your N-character password from 36^N possibilities to its top-50
   guesses, and your real one is #k."

Athena's defense is then measured the honest way: after the shield, the true
secret's rank collapses (falls off the list) and the decoy rises to the top.

Acoustic-only ranking on purpose: a real password has no language prior, so we do
NOT fuse a char-LM here (that would only help English words) — this is the honest
hard case. `span_candidates` returns [(guess, prob), ...] most-likely first.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.signal import find_peaks

from ..config import SR
from ..ctc.data import VOCAB
from ..ctc.model import HOP
from .defense_audio import torch_logmel

KEYS = VOCAB[1:]                      # 37 real symbols (idx i -> KEYS[i])
KEYS_PW = [k for k in KEYS if k != " "]   # 36 alnum (passwords have no space)


def _span_slot_logposts(net, y, lo, hi, thr=0.4, min_gap_ms=90.0, win=1,
                        n_expected=None):
    """Per-onset log-posteriors over the alnum keys, for onsets inside [lo,hi].
    If n_expected is set, keep the n_expected STRONGEST onset peaks (then time-order)
    so the candidate length matches the secret — makes the guess-rank meaningful."""
    dev = next(net.parameters()).device
    with torch.no_grad():
        lg, onl = net(torch_logmel(torch.tensor(y, dtype=torch.float32, device=dev))[None])
    on = torch.sigmoid(onl)[0].cpu().numpy()
    lgn = lg[0].cpu().numpy()
    dist = max(1, int(min_gap_ms / 1000 * SR / HOP))
    peaks, _ = find_peaks(on, height=thr, distance=dist)
    f_lo, f_hi = lo // HOP, hi // HOP
    peaks = [p for p in peaks if f_lo <= p < f_hi]
    if n_expected and len(peaks) > n_expected:              # keep strongest, keep order
        peaks = sorted(sorted(peaks, key=lambda p: -on[p])[:n_expected])
    slots = []
    for p in peaks:
        a, b = max(0, p - win), min(len(lgn), p + win + 1)
        z = lgn[a:b, 1:-1].mean(0)                          # drop blank AND space cols
        z = z - z.max()
        slots.append(z - np.log(np.exp(z).sum()))
    return slots


def span_candidates(net, y, lo, hi, topn=50, topk=8, beam=300, n_expected=None):
    """Return (candidates, n_slots): candidates = [(guess, prob), ...] most likely
    first, from a beam over the span's per-onset acoustic posteriors."""
    slots = _span_slot_logposts(net, y, lo, hi, n_expected=n_expected)
    n = len(slots)
    if n == 0:
        return [], 0
    beams = [("", 0.0)]
    for lp in slots:
        cand = lp.argsort()[::-1][:topk]
        nxt = []
        for pref, sc in beams:
            for i in cand:
                nxt.append((pref + KEYS_PW[i], sc + float(lp[i])))
        nxt.sort(key=lambda t: t[1], reverse=True)
        beams = nxt[:beam]
    scores = np.array([s for _, s in beams[:topn]])
    p = np.exp(scores - scores.max()); p = p / p.sum()
    return [(g, float(pi)) for (g, _), pi in zip(beams[:topn], p)], n


def rank_of(cands, secret):
    """1-based rank of `secret` in the candidate list, or None if absent."""
    s = "".join(c for c in secret.upper() if c in KEYS)
    for i, (g, _) in enumerate(cands):
        if g == s:
            return i + 1
    return None


def search_space(n_slots):
    return len(KEYS_PW) ** n_slots if n_slots else 0


def summarize(net, y, lo, hi, secret, decoy=None, topn=50):
    ns = len("".join(c for c in secret.upper() if c in KEYS_PW))
    cands, n = span_candidates(net, y, lo, hi, topn=topn, n_expected=ns)
    return {
        "n_slots": n,
        "search_space": search_space(n),
        "shortlist": topn,
        "secret_rank": rank_of(cands, secret),
        "decoy_rank": rank_of(cands, decoy) if decoy else None,
        "top5": [{"guess": g, "prob": round(p, 3)} for g, p in cands[:5]],
    }


def demo():
    import os
    os.environ.setdefault("KEYGUARD_DEVICE", "cpu")
    from ..ctc.data import synth_line
    from ..ctc.train_overlap import MtlCRNN
    from ..ctc.model import DEVICE
    net = MtlCRNN(n_sym=len(VOCAB)).to(DEVICE)
    net.load_state_dict(torch.load("runs/ctc_rich_ft.pt", map_location=DEVICE)); net.eval()
    rng = np.random.default_rng(3)
    y, lab, on = synth_line("my password is hunter2", rng, wpm=(40, 75),
                            root="data/live_bank_rich.npz")
    kstr = "".join(VOCAB[i] for i in lab); i = kstr.replace(" ", "").find("HUNTER2")
    nonsp = [j for j, c in enumerate(kstr) if c != " "]
    lo = int(on[nonsp[i]]) - int(0.02 * SR); hi = int(on[nonsp[i + 6]]) + int(0.14 * SR)
    s = summarize(net, y, lo, hi, "HUNTER2")
    print("smart_dict:", s)


if __name__ == "__main__":
    demo()
