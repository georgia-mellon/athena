"""LM-fused beam-search CTC decoder -- the highest-ROI accuracy lever.

Three backends, best-available wins, each degrading gracefully:
  1. pyctcdecode + a char-level KenLM model  -> "kenlm"
  2. pyctcdecode with no LM                  -> "pyctcdecode-only"
  3. pure-python prefix beam + decode.CharLM -> "pure-python"  (always available)

pyctcdecode pins numpy<2 (incompatible here) and KenLM's `lmplz` binary is
often absent, so in this env the pure-python path is active. The code still
tries the faster backends so a fuller install lights them up with no changes.

Vocab: index 0 = blank, 1..36 = a-z,0-9 (see data.VOCAB). No space symbol.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import numpy as np
from .data import VOCAB, BLANK, random_text
from .model import cer
from . import decode

try:
    from pyctcdecode import build_ctcdecoder
    _HAVE_PYCTC = True
except Exception:
    _HAVE_PYCTC = False

KENLM_ORDER = 5
_CHARS = VOCAB[1:]                      # the 36 real symbols
_IDX = {c: i for i, c in enumerate(VOCAB)}


def _log_softmax(x: np.ndarray) -> np.ndarray:
    """(T,S) raw logits or logprobs -> normalized logprobs (idempotent enough)."""
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _corpus_lines(corpus_path: str | None) -> list[str]:
    """Uppercased word lines for the LM: from a file, else synthesized."""
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        rng = np.random.default_rng(0)
        text = " ".join(random_text(rng) for _ in range(400))
        words = "/usr/share/dict/words"          # general English, if present
        if os.path.exists(words):
            with open(words, encoding="utf-8", errors="ignore") as f:
                text += " " + " ".join(w.strip() for w in f.readlines()[:20000])
    return [w for w in text.upper().split() if w]


def build_kenlm(corpus_path: str | None = None, order: int = KENLM_ORDER,
                out_dir: str = "runs/lm") -> str | None:
    """Build a char-level (space-separated chars) n-gram KenLM ARPA from a corpus.

    Returns the .arpa path (pyctcdecode reads it directly), or None if the
    `lmplz` binary isn't installed -- callers then run without a KenLM model.
    """
    lmplz = shutil.which("lmplz")
    if not lmplz:                                # kenlm binaries absent -> graceful
        return None
    lines = _corpus_lines(corpus_path)
    # char-level: one word per line, its symbols space-separated, filtered to vocab
    charified = "\n".join(" ".join(c for c in w if c in _IDX) for w in lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    arpa = os.path.join(out_dir, f"char_{order}gram.arpa")
    with open(arpa, "wb") as out:
        subprocess.run([lmplz, "-o", str(order), "--discount_fallback"],
                       input=charified.encode(), stdout=out, check=True)
    return arpa


class LMDecoder:
    """CTC decoder with LM shallow fusion; picks the best available backend."""

    def __init__(self, labels: list[str] = VOCAB, kenlm_path: str | None = None,
                 alpha: float = 0.5, beta: float = 1.0):
        self.alpha = alpha
        self._pyctc = None
        self._has_kenlm = False
        if _HAVE_PYCTC:
            try:                                 # pyctcdecode blank token is ""
                pyctc_labels = [""] + list(labels[1:])
                self._pyctc = build_ctcdecoder(
                    pyctc_labels, kenlm_model_path=kenlm_path, alpha=alpha, beta=beta)
                self._has_kenlm = kenlm_path is not None
            except Exception:
                self._pyctc = None
        self._lm = None if self._pyctc else decode.CharLM()

    @property
    def backend(self) -> str:
        if self._pyctc is None:
            return "pure-python"
        return "kenlm" if self._has_kenlm else "pyctcdecode-only"

    def decode(self, logits_or_logprobs: np.ndarray) -> str:
        """(T,S) -> decoded string of vocab characters (no spaces)."""
        lp = _log_softmax(logits_or_logprobs)
        if self._pyctc is not None:
            return self._pyctc.decode(lp, beam_width=64).replace(" ", "")
        return decode.beam_search(lp, self._lm, alpha=self.alpha)

    def decode_ids(self, logits: np.ndarray) -> list[int]:
        """Decode to symbol indices so cer() can score against reference ids."""
        return [_IDX[c] for c in self.decode(logits) if c in _IDX]


def decode_ids(logits: np.ndarray, decoder: LMDecoder | None = None) -> list[int]:
    """Module-level convenience: decode to symbol ids with a default decoder."""
    return (decoder or LMDecoder()).decode_ids(logits)


def _synth_logits(text: str, rng, mass: float = 5.0, noise: float = 1.6) -> np.ndarray:
    """High logit mass on the right symbol per frame + noise (blank-separated)."""
    S = len(VOCAB)
    frames = [rng.standard_normal(S) * noise]            # leading blank
    frames[0][BLANK] += mass
    for c in text.upper():
        if c not in _IDX:
            continue
        blank = rng.standard_normal(S) * noise; blank[BLANK] += mass
        char = rng.standard_normal(S) * noise; char[_IDX[c]] += mass
        frames += [char, blank]
    return np.asarray(frames)


def demo() -> None:
    rng = np.random.default_rng(7)
    kenlm_path = build_kenlm()                  # None here (no lmplz) -> no-LM path
    dec = LMDecoder(kenlm_path=kenlm_path)
    strings = ["password", "meeting", "report", "admin", "secret", "hunter"]
    g_tot, l_tot = 0.0, 0.0
    for s in strings:
        lp = _synth_logits(s, rng)
        ref = [_IDX[c] for c in s.upper()]
        greedy_ids = [_IDX[c] for c in decode.greedy(lp)]   # CTC greedy baseline
        g_tot += cer(ref, greedy_ids)
        l_tot += cer(ref, dec.decode_ids(lp))
    n = len(strings)
    print(f"backend: {dec.backend}  |  greedy CER {g_tot/n:.3f}  "
          f"LM CER {l_tot/n:.3f}  over {n} strings")
    assert l_tot <= g_tot + 1e-9, f"LM CER {l_tot} worse than greedy {g_tot}"
    print("lm_decode ok: LM beam CER <= greedy CER")


if __name__ == "__main__":
    demo()
