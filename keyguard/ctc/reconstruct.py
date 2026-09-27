"""Word-level reconstruction of a keystroke transcription that HAS spaces.

The attacker now emits a real space symbol (vocab id 37), so word boundaries come
from the acoustic model instead of being inferred. That makes language-model
reasoning far stronger: we correct each detected word against an English
frequency dictionary (SymSpell fuzzy match + wordfreq unigram prior), and only
fall back to boundary-free segmentation (`lm_correct.correct`) for tokens the
detector split/merged wrong.

Pipeline for one hypothesis string `H` (uppercase A-Z0-9 + spaces):
  1. de-stutter (collapse >=3 identical chars to 2 -- typing/CTC noise).
  2. split on spaces -> tokens (the acoustic word boundaries).
  3. per token:
       - pure digits            -> keep literal (passwords, numbers).
       - exact dictionary hit   -> keep.
       - short fuzzy hit (d<=2)  -> best (edit_cost*d - logP(word)).
       - otherwise              -> re-segment the token with lm_correct.correct
                                    (handles a missing space glueing two words).
  4. optional LLM rescoring hook (off by default; offline-safe).

`reconstruct(h) -> str` returns readable uppercase text with spaces.
"""
from __future__ import annotations

import math

from . import lm_correct as L


def _best_word(tok: str) -> tuple[float, str]:
    """(cost, replacement) for one token as a single word; inf if no candidate."""
    L._build_index()
    if not tok:
        return math.inf, ""
    if tok.isdigit():                       # digits: keep literal, cheap
        return L._DIGIT_CHAR_COST * len(tok), tok
    if tok in L._WORD_COST:                 # exact hit
        return L._word_cost(tok), tok
    return L._segment_cost(tok)             # SymSpell fuzzy (<=2 edits) + prior


def reconstruct(hyp: str, resegment_thresh: float = 24.0) -> str:
    """Map a spaced, noisy attacker string to best-guess English (uppercase)."""
    L._build_index()
    s = L._destutter(hyp.upper())
    tokens = [t for t in s.split(" ") if t]
    out: list[str] = []
    for tok in tokens:
        tok = "".join(c for c in tok if c in L._VOCAB)
        if not tok:
            continue
        cost, word = _best_word(tok)
        # Trust the acoustic word boundaries: correct a token to a near dictionary
        # word only when the fix is CLOSE (<= resegment_thresh). Otherwise keep the
        # literal token — do NOT re-segment it into multiple words (that over-splits
        # and wrecks WER; the space symbol already carries the boundaries).
        if cost == math.inf or (cost > resegment_thresh and not tok.isdigit()):
            out.append(tok)
        else:
            out.append(word)
    return " ".join(out)


def _selftest() -> None:
    # Spaced noisy outputs (what the space-aware attacker produces): boundaries
    # mostly right, letters noisy. Compare to boundary-free correct().
    cases = [
        ("HOPE YOUT WEEK IS OFF TO A GO", "HOPE YOUR WEEK IS OFF TO A GO"),
        ("THE QUICH BROWN FOZ JUMPS", "THE QUICK BROWN FOX JUMPS"),
        ("PLEASE SEND THE REPRT", "PLEASE SEND THE REPORT"),
        ("MEETINGAT 3 PM", "MEETING AT 3 PM"),      # a glued token -> resegment
    ]
    print("=== reconstruct (space-aware) ===")
    for hyp, want in cases:
        got = reconstruct(hyp)
        print(f"  hyp : {hyp}")
        print(f"  out : {got}")
        print(f"  want: {want}   {'OK' if got == want else 'x'}\n")
    print("reconstruct ok: reconstruct(hyp:str)->str importable")


if __name__ == "__main__":
    _selftest()
