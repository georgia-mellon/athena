"""Timing-only keystroke attacker.

Sees the *rhythm* of typing, never the sound of any key. Inter-keystroke gaps
alone leak structure: long gaps are word boundaries (spaces), the gap sequence
is a fingerprint of the typist and the text's shape. This attacker exists to
catch a shield that scrubs keystroke AUDIO but leaves keystroke TIMING intact --
it should still succeed there, and should collapse once the shield injects
decoys or emits keystrokes on a uniform clock.

Honest about limits: timing cannot recover *which* keys were pressed, so we
never fabricate a transcript. We report only what rhythm reveals -- key count,
word count, per-word lengths, and a bits-of-leakage estimate.
"""
from __future__ import annotations
import numpy as np
from ..config import SR

# ponytail: naive fixed heuristics; swap for a 2-component GMM/kmeans on the gap
# distribution if bimodality gets fuzzy on real typing.
SPACE_FACTOR = 2.0      # a gap > SPACE_FACTOR * median gap is a word boundary
MAX_WORD_LEN = 12       # uniform prior: word length unknown over 1..MAX_WORD_LEN


def intervals(onsets: np.ndarray, sr: int = SR) -> np.ndarray:
    """Inter-keystroke gaps in seconds (len = n_onsets - 1)."""
    onsets = np.asarray(onsets, dtype=float)
    if len(onsets) < 2:
        return np.array([], dtype=float)
    return np.diff(np.sort(onsets)) / sr


def _word_lengths(gaps: np.ndarray) -> list[int]:
    """Split the keystroke stream into words at long gaps. Returns key counts."""
    if len(gaps) == 0:
        return []
    thresh = SPACE_FACTOR * float(np.median(gaps))
    is_space = gaps > thresh
    lengths, run = [], 1                       # first key starts the first word
    for space in is_space:
        if space:
            lengths.append(run)
            run = 1
        else:
            run += 1
    lengths.append(run)
    return lengths


def _bits_leaked(word_lengths: list[int]) -> float:
    """Bits timing leaks by revealing the word-length partition.

    Prior: each word length is uniform over 1..MAX_WORD_LEN -> log2(MAX) bits of
    uncertainty per word. Timing collapses that to the observed distribution's
    entropy. Leak = summed reduction across words (a lower bound on structure
    leaked, in the spirit of metrics.mutual_info_bits).
    """
    if not word_lengths:
        return 0.0
    h_prior = np.log2(MAX_WORD_LEN)
    counts = np.bincount(word_lengths)
    p = counts[counts > 0] / len(word_lengths)
    h_obs = float(-np.sum(p * np.log2(p)))     # 0 when all words same length
    return max(0.0, h_prior - h_obs) * len(word_lengths)


class TimingAttacker:
    """Infers text structure from inter-keystroke timing alone."""

    def attack(self, onsets: np.ndarray) -> dict:
        onsets = np.asarray(onsets)
        gaps = intervals(onsets)
        lengths = _word_lengths(gaps)
        return {
            "n_keys": int(len(onsets)),
            "n_words_est": len(lengths),
            "word_lengths_est": lengths,
            "bits_leaked_est": _bits_leaked(lengths),
        }


def detects_timing(onsets_clean: np.ndarray, onsets_shielded: np.ndarray) -> float:
    """How well the shielded signal still exposes the original typing rhythm.

    Combines (a) key-count recoverability and (b) correlation of the two gap
    sequences. Both must hold: a shield that keeps the count but scrambles the
    rhythm, or keeps the rhythm but drops keys, both score low. Range [0, 1].
    This is the arena's timing-leak metric.
    """
    gap_clean = intervals(onsets_clean)
    gap_shield = intervals(onsets_shielded)

    n_clean = len(onsets_clean)
    n_shield = len(onsets_shielded)
    if n_clean == 0:
        return 0.0
    count_score = 1.0 - abs(n_clean - n_shield) / n_clean
    count_score = float(np.clip(count_score, 0.0, 1.0))

    m = min(len(gap_clean), len(gap_shield))
    if m < 2:
        corr_score = 0.0
    else:
        a, b = gap_clean[:m], gap_shield[:m]
        if a.std() == 0 or b.std() == 0:
            corr_score = 1.0 if np.allclose(a, b) else 0.0
        else:
            corr_score = max(0.0, float(np.corrcoef(a, b)[0, 1]))

    return count_score * corr_score            # need BOTH count and rhythm


def _synth_onsets(word_lengths, intra=0.15, space=0.5, sr=SR) -> np.ndarray:
    """Onset sample indices for words of the given key counts; long gap = space."""
    times, t = [], 0.0
    for w, n in enumerate(word_lengths):
        for k in range(n):
            times.append(t)
            t += intra
        if w < len(word_lengths) - 1:
            t += space - intra                 # replace last intra gap with a space
    return (np.array(times) * sr).astype(int)


def demo():
    # "the quick brown fox": word lengths 3,5,5,3
    x = _synth_onsets([3, 5, 5, 3])
    out = TimingAttacker().attack(x)
    assert out["n_keys"] == 16, out
    assert out["n_words_est"] == 4, out
    assert out["word_lengths_est"] == [3, 5, 5, 3], out
    assert out["bits_leaked_est"] > 0, out

    assert abs(detects_timing(x, x) - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    rand = np.sort(rng.integers(0, x.max(), size=len(x)))
    assert detects_timing(x, rand) < 0.5, detects_timing(x, rand)
    # shield that drops all keystrokes -> no timing leak
    assert detects_timing(x, np.array([])) == 0.0

    print("timing attacker demo ok:", out, "| self-corr",
          round(detects_timing(x, x), 3), "| vs random",
          round(detects_timing(x, rand), 3))


if __name__ == "__main__":
    demo()
