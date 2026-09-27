"""Dictionary-based correction of noisy CTC keystroke transcriptions.

Our overlap CTC attacker emits an uppercase A-Z0-9 stream with NO spaces (spaces
are silent gaps, so words run together) and greedy decoding leaves it legible but
noisy: doubled/inserted chars, dropped chars, the odd substitution. This module
pulls that stream back toward English.

Pipeline (all CPU-only, offline, re-implemented -- not vendored):

  1. Normalize    -> keep A-Z0-9, uppercase.
  2. De-stutter   -> collapse runs of >=3 identical chars to 2 (English almost
                     never triples a letter; CTC/typing noise does).
  3. Fuzzy Viterbi word-splitting over the run-together text. A DP over string
     positions finds the segmentation into dictionary words that minimises
        sum_words [ EDIT_COST * edit_distance(substr, word) - log P(word) ].
     Candidate words per substring come from a SymSpell index (Damerau-Levenshtein
     within edit distance 2) built from a wordfreq unigram list; word priors are
     wordfreq probabilities. A per-character "leave it literal" fallback keeps the
     DP total (digits, unrecoverable garbage) and never gets stuck.

The result is uppercase, with spaces inserted at inferred word boundaries. Import
`correct(noisy) -> str`; run `python -m keyguard.ctc.lm_correct` for the self-test.

Reference (for the n-gram/segmentation idea only): ggerganov/kbd-audio keytap2/3
(MIT). No code was copied.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_MAX_WORD_LEN = 18          # longest substring window the DP will consider
_MAX_EDIT = 2               # SymSpell max Damerau-Levenshtein distance
_EDIT_COST = 10.0           # penalty per character edit inside a word (swept)
_WORD_PENALTY = 0.0         # fixed cost per emitted word (discourages fragmenting)
_UNKNOWN_CHAR_COST = 18.0   # cost of emitting one char with no dictionary word
_DIGIT_CHAR_COST = 6.0      # digits are legitimately un-word-like; cheaper literal
_VOCAB = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_DICT_SIZE = 60000          # top-N English unigrams to index
_MIN_ZIPF = 1.5             # floor for unknown-but-plausible words

# ---------------------------------------------------------------------------
# Lazy singletons (SymSpell index + word-cost table)
# ---------------------------------------------------------------------------
_SYM = None
_WORD_COST: dict[str, float] = {}


def _build_index():
    """Build the SymSpell index and word-cost table once, lazily."""
    global _SYM, _WORD_COST
    if _SYM is not None:
        return
    from symspellpy import SymSpell
    from wordfreq import top_n_list, word_frequency

    sym = SymSpell(max_dictionary_edit_distance=_MAX_EDIT, prefix_length=7)
    cost: dict[str, float] = {}
    words = top_n_list("en", _DICT_SIZE)
    for w in words:
        # keep alphabetic words only; the vocab has no punctuation/spaces
        if not w.isalpha() or not w.isascii():
            continue
        wu = w.upper()
        if any(c not in _VOCAB for c in wu):
            continue
        f = word_frequency(w, "en")
        if f <= 0:
            continue
        # SymSpell count must be an int; scale the probability up.
        sym.create_dictionary_entry(wu, max(1, int(f * 1e9)))
        cost[wu] = -math.log(f)
    _SYM = sym
    _WORD_COST = cost


def _word_cost(word: str) -> float:
    """-log P(word); cheap for common words, expensive for rare ones."""
    c = _WORD_COST.get(word)
    if c is not None:
        return c
    # Word not in the cached table (e.g. a SymSpell suggestion just outside the
    # top-N): fall back to a floor probability so it is still usable.
    return -math.log(10 ** (_MIN_ZIPF - 9))


# ---------------------------------------------------------------------------
# Text prep
# ---------------------------------------------------------------------------
def normalize(s: str) -> str:
    """Uppercase and drop anything outside A-Z0-9."""
    return "".join(c for c in s.upper() if c in _VOCAB)


def _destutter(s: str) -> str:
    """Collapse runs of >=3 identical chars down to 2 (keep legit doubles)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


# ---------------------------------------------------------------------------
# Fuzzy Viterbi word-splitting
# ---------------------------------------------------------------------------
def _segment_cost(sub: str) -> tuple[float, str]:
    """Best (cost, replacement) for treating `sub` as one word.

    Tries SymSpell suggestions (fuzzy dictionary hits) and the sub itself; the
    replacement is the dictionary word (or the literal sub if nothing beats it).
    """
    best_cost = math.inf
    best_word = sub

    # Exact / near dictionary matches.
    suggestions = _SYM.lookup(
        sub, verbosity=0, max_edit_distance=_MAX_EDIT, include_unknown=False
    )  # verbosity 0 == CLOSEST
    for sug in suggestions:
        d = sug.distance
        w = sug.term.upper()
        c = _EDIT_COST * d + _word_cost(w) + _WORD_PENALTY
        if c < best_cost:
            best_cost, best_word = c, w

    return best_cost, best_word


@lru_cache(maxsize=4096)
def _segment(s: str) -> tuple[float, tuple[str, ...]]:
    """DP: cheapest list of words explaining string `s`."""
    n = len(s)
    if n == 0:
        return 0.0, ()
    # best[i] = (cost, words) to explain s[:i]
    best: list[tuple[float, tuple[str, ...]]] = [(0.0, ())] + [
        (math.inf, ()) for _ in range(n)
    ]
    for j in range(1, n + 1):
        lo = max(0, j - _MAX_WORD_LEN)
        for i in range(lo, j):
            prev_cost, prev_words = best[i]
            if prev_cost == math.inf:
                continue
            sub = s[i:j]
            # Option A: literal single char (only for length-1 windows).
            if len(sub) == 1:
                lit_cost = _DIGIT_CHAR_COST if sub.isdigit() else _UNKNOWN_CHAR_COST
                cand = prev_cost + lit_cost
                if cand < best[j][0]:
                    best[j] = (cand, prev_words + (sub,))
            # Option B: fuzzy dictionary word (skip pure-digit substrings).
            if not sub.isdigit():
                seg_cost, word = _segment_cost(sub)
                if seg_cost != math.inf:
                    cand = prev_cost + seg_cost
                    if cand < best[j][0]:
                        best[j] = (cand, prev_words + (word,))
    return best[n]


def correct(noisy: str) -> str:
    """Map a noisy uppercase A-Z0-9 CTC transcription to best-guess English.

    Returns uppercase text with spaces inserted at inferred word boundaries.
    Robust to the no-space run-together case; empty in -> empty out.
    """
    _build_index()
    s = _destutter(normalize(noisy))
    if not s:
        return ""
    _, words = _segment(s)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _cer(ref: str, hyp: str) -> float:
    """Character error rate on space-stripped uppercase strings.

    Our alphabet has no space symbol, so both sides are compared without spaces;
    this measures whether the *letters* got closer to the truth.
    """
    r = normalize(ref)
    h = normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)


def _corrupt(clean: str, rate: float, rng) -> str:
    """Insert/delete/substitute chars at ~`rate` to mimic noisy CTC output."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    out = []
    for ch in clean:
        r = rng.random()
        if r < rate / 3:                      # substitute
            out.append(rng.choice(list(alphabet)))
        elif r < 2 * rate / 3:                # delete
            continue
        elif r < rate:                        # insert then keep
            out.append(rng.choice(list(alphabet)))
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _selftest() -> None:
    import random

    print("Building index (wordfreq + SymSpell)...")
    _build_index()
    print(f"  indexed {len(_WORD_COST)} words\n")

    # 1) The three real noisy attacker outputs.
    real = [
        ("HOPEEYOUSWEEKSOWSFTTTOAGGO", "HOPEYOURWEEKISOFFTOAGO"),
        ("OOODSTARSSIMESTWANEDDTOTAALEA", "ODSTARTIJUSTWANTEDTOTAKEA"),
        ("GIAALLCIC", "HIALLI"),
    ]
    print("=== Real attacker outputs ===")
    tot_b = tot_a = 0.0
    for hyp, ref in real:
        out = correct(hyp)
        b, a = _cer(ref, hyp), _cer(ref, out)
        tot_b += b
        tot_a += a
        print(f"  hyp : {hyp}")
        print(f"  ref : {ref}")
        print(f"  out : {out}")
        print(f"  CER : {b:.3f} -> {a:.3f}\n")
    print(f"  mean CER: {tot_b/len(real):.3f} -> {tot_a/len(real):.3f}\n")

    # 2) Synthetic English corrupted at ~20%.
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "please send me the report before the meeting tomorrow",
        "i just wanted to take a quick break this afternoon",
        "hope your week is off to a good start everyone",
        "the password is stored in the config file on the server",
        "let me know if you have any questions about the project",
        "we should schedule a call to discuss the new design",
        "remember to back up your files before the update",
    ]
    rng = random.Random(0)
    print("=== Synthetic English, ~20% corruption ===")
    sb = sa = 0.0
    for sent in sentences:
        ref = normalize(sent)                 # spaceless truth
        noisy = _corrupt(ref, 0.20, rng)
        out = correct(noisy)
        b, a = _cer(ref, noisy), _cer(ref, out)
        sb += b
        sa += a
        print(f"  ref  : {sent}")
        print(f"  noisy: {noisy}")
        print(f"  out  : {out}")
        print(f"  CER  : {b:.3f} -> {a:.3f}\n")
    print(f"  mean CER: {sb/len(sentences):.3f} -> {sa/len(sentences):.3f}")
    print("\nDONE. correct(noisy: str) -> str is importable.")


if __name__ == "__main__":
    _selftest()
