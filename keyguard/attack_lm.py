"""Pillar 2: a language model recovers text from weak per-key acoustic guesses.

The acoustic attacker is confusable (top-1 ~38%, top-3 ~57% on a laptop mic). But
English carries ~1 bit/char of redundancy, so an LLM reading the per-key candidate
lattice reconstructs plausible text from guesses no single key is sure of. That is
*why* the defense matters: kill per-key acoustics (Pillar 1) and the LM has nothing
to multiply against -- with the shield on, the candidates are garbage and this
reconstruction fails too.

Gemini is called over REST (httpx, already a dep). With GEMINI_API_KEY unset it
degrades to the top-1 acoustic string, so the pipeline always runs offline.
"""
from __future__ import annotations
import os
import numpy as np

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# ponytail: gemini-flash-latest always points at the current flash model; override
# with GEMINI_MODEL if needed. (Pinned ids like gemini-2.5-flash 404 for new keys.)
DEFAULT_MODEL = "gemini-flash-latest"


def candidate_lattice(proba: np.ndarray, classes, k: int = 3) -> list[list[str]]:
    """Top-k guesses per keystroke from the attacker softmax. proba: (n_keys, C)."""
    if proba is None or len(proba) == 0:
        return []
    topk = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return [[classes[j] for j in row] for row in topk]


def _prompt(lattice: list[list[str]]) -> str:
    n = len(lattice)
    lines = [f"pos {i + 1}: {' '.join(row)}" for i, row in enumerate(lattice)]
    return (
        f"A user typed a string of exactly {n} characters (letters A-Z and digits "
        "0-9 only, no spaces). An acoustic side-channel gives the most likely "
        "guesses for each position, best first:\n"
        + "\n".join(lines)
        + f"\n\nReturn ONLY the single most plausible {n}-character string. "
        "No spaces, punctuation, or explanation."
    )


def gemini_reconstruct(lattice: list[list[str]], model: str | None = None) -> str:
    """Most plausible typed string given the per-key top-k lattice.

    Falls back to the top-1 acoustic string on any missing key, network error, or
    malformed response -- the attack still produces output, just weaker.
    """
    top1 = "".join(row[0] for row in lattice) if lattice else ""
    key = os.environ.get("GEMINI_API_KEY")
    if not key or not lattice:
        return top1
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    allowed = set().union(*lattice)
    try:  # best-effort; never break the pipeline
        import httpx
        r = httpx.post(
            GEMINI_URL.format(model=model),
            params={"key": key},
            json={"contents": [{"parts": [{"text": _prompt(lattice)}]}]},
            timeout=30,
        )
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = "".join(c for c in text.upper() if c in allowed)
        return cleaned or top1
    except Exception:
        return top1


def demo():
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    n = len(classes)
    # Build a lattice where top-1 spells nonsense but "PASS" hides in the top-3.
    want = "PASS"
    proba = np.full((len(want), n), 0.01, np.float32)
    for i, ch in enumerate(want):
        wrong = classes[(classes.index(ch) + 1) % n]  # top-1 is the wrong neighbor
        proba[i, classes.index(wrong)] = 0.5
        proba[i, classes.index(ch)] = 0.4              # true key is a strong #2
    lat = candidate_lattice(proba, classes, k=3)
    assert len(lat) == len(want) and all(len(r) == 3 for r in lat)
    assert all(want[i] in lat[i] for i in range(len(want)))     # truth is in the lattice
    top1 = "".join(r[0] for r in lat)
    assert top1 != want                                          # acoustic-only is wrong
    # No key set in the self-check -> must degrade to the top-1 string, same length.
    out = gemini_reconstruct(lat)
    assert out == top1 and len(out) == len(want)
    assert candidate_lattice(np.zeros((0, n)), classes) == []    # empty is safe
    print(f"attack_lm ok: top-1='{top1}' (wrong), truth '{want}' in top-3 lattice; "
          f"offline fallback returns top-1, Gemini path wired for GEMINI_API_KEY")


if __name__ == "__main__":
    demo()
