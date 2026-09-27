"""CTC decoding with language-model fusion -- the readable-text lever.

Greedy CTC gives noisy characters; a beam search with a char n-gram LM (shallow
fusion) pulls them toward plausible text, which is where the literature's big
CER gains come from (36% -> 7% in Slater et al.). An optional LLM rescoring hook
(Grok/Claude) re-ranks the beam's n-best for a final polish; it degrades to the
LM score with no API key so everything runs offline.

Vocab: index 0 = blank, 1..36 = config.CLASSES (see data.VOCAB). v1 has no space
symbol, so the LM is over letters/digits only; spaces are re-inserted from
inter-key gaps at a higher level.
"""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
from .data import VOCAB, BLANK, random_text


class CharLM:
    """Simple back-off char n-gram over the 36 keys, built from a text corpus."""

    def __init__(self, order=3, corpus=None):
        self.order = order
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        text = corpus or _default_corpus()
        toks = [c for c in text.upper() if c in VOCAB[1:]]
        for i in range(len(toks)):
            for n in range(order):
                ctx = "".join(toks[max(0, i - n):i])
                self.counts[n][ctx][toks[i]] += 1

    def logprob(self, context: str, ch: str) -> float:
        """Back-off log P(ch | context), interpolating shorter contexts."""
        p, weight = 0.0, 0.0
        for n in range(self.order):
            ctx = context[-n:] if n else ""
            table = self.counts[n].get(ctx)
            if table:
                total = sum(table.values())
                w = (n + 1)
                p += w * (table.get(ch, 0) + 0.1) / (total + 0.1 * len(VOCAB[1:]))
                weight += w
        p = p / weight if weight else 1.0 / len(VOCAB[1:])
        return math.log(max(p, 1e-9))


def _default_corpus():
    rng = np.random.default_rng(0)
    base = ("the quick brown fox jumps over the lazy dog password hunter meeting "
            "notes report login admin secret code send ready before thanks team "
            "email address account verify security private message please review")
    return base + " " + " ".join(random_text(rng) for _ in range(50))


def greedy(logprobs: np.ndarray) -> str:
    """logprobs: (T, S). Collapse repeats, drop blanks."""
    ids = logprobs.argmax(-1)
    out, prev = [], BLANK
    for s in ids:
        if s != prev and s != BLANK:
            out.append(VOCAB[s])
        prev = s
    return "".join(out)


def beam_search(logprobs: np.ndarray, lm: CharLM | None = None, beam=25,
                alpha=0.4) -> str:
    """Prefix beam search with optional LM shallow fusion. logprobs (T,S)."""
    T, S = logprobs.shape
    beams = {"": (0.0, -math.inf)}          # prefix -> (logp_blank, logp_nonblank)
    for t in range(T):
        lp = logprobs[t]
        nxt = defaultdict(lambda: (-math.inf, -math.inf))
        for prefix, (pb, pnb) in beams.items():
            ptot = np.logaddexp(pb, pnb)
            nb, nnb = nxt[prefix]
            nxt[prefix] = (np.logaddexp(nb, ptot + lp[BLANK]), nnb)   # blank
            for s in range(1, S):
                ch = VOCAB[s]
                add = lp[s]
                if lm is not None:
                    add = add + alpha * lm.logprob(prefix, ch)
                if prefix and ch == prefix[-1]:
                    np_ = prefix + ch
                    b2, nb2 = nxt[np_]
                    nxt[np_] = (b2, np.logaddexp(nb2, pb + add))
                    b3, nb3 = nxt[prefix]
                    nxt[prefix] = (b3, np.logaddexp(nb3, pnb + lp[s]))
                else:
                    np_ = prefix + ch
                    b2, nb2 = nxt[np_]
                    nxt[np_] = (b2, np.logaddexp(nb2, ptot + add))
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: np.logaddexp(*kv[1]), reverse=True)[:beam])
    best = max(beams.items(), key=lambda kv: np.logaddexp(*kv[1]))
    return best[0]


def llm_rescore(nbest: list[str], provider="xai") -> str:
    """Re-rank candidates with an LLM; falls back to the first (LM-best)."""
    import os
    key = os.environ.get("XAI_API_KEY" if provider == "xai" else "ANTHROPIC_API_KEY")
    if not key or not nbest:
        return nbest[0] if nbest else ""
    try:                                    # best-effort; never break the pipeline
        import httpx
        prompt = ("These are noisy guesses of a typed string (letters/digits, no "
                  "spaces). Return ONLY the single most plausible one, unchanged "
                  "if none fit:\n" + "\n".join(nbest[:8]))
        r = httpx.post("https://api.x.ai/v1/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": "grok-4",
                             "messages": [{"role": "user", "content": prompt}]},
                       timeout=30)
        return r.json()["choices"][0]["message"]["content"].strip().split()[0]
    except Exception:
        return nbest[0]


def demo():
    lm = CharLM()
    assert lm.logprob("PASSWOR", "D") > lm.logprob("PASSWOR", "Q")
    T, S = 30, len(VOCAB)
    lp = np.log(np.full((T, S), 1e-3))
    lp[:, BLANK] = math.log(0.5)            # mostly blanks -> short output
    out = beam_search(lp, lm, beam=8)
    assert isinstance(out, str)
    print(f"decode ok: LM favors PASSWORD over PASSWORQ; beam_search returns "
          f"'{out[:20]}' (len {len(out)}), corpus n-grams loaded")


if __name__ == "__main__":
    demo()
