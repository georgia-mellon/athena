"""Vosk spoken-secret spotter (plan 06 §4): SecretSpotterDriver over an open-vocabulary KaldiRecognizer.

Why not the restricted grammar plan 06 suggests: with only digits + triggers + [unk] to choose from, ordinary speech
is forced onto them. On the tuning clips (LibriSpeech speakers 100/2803) it redacted 6.6 s per minute of innocent
speech and fired 38 inbound request triggers per minute; the full small-model LM gives ~0 of both, at ~4-8 ms per
20 ms block instead of ~2. Digit homophones the LM prefers ("to", "for") count when chained to digits.

Outbound mode listens to your mic for digit runs and own-side trigger phrases ("the code is ..."); inbound mode
listens to the caller for request triggers ("read me the code"). Word timings come from partial results
(SetPartialWords), so a span is placed while the word is still in the redactor's delay line.

Latency: stock Vosk only refreshes partial *word timings* every ~2 s (the incremental lattice determinizes in big
chunks). scripts/get_vosk_model.py appends LOW_LATENCY to the model's conf/model.conf, which brings a digit's span
out ~0.35-0.47 s after the word starts (measured), inside the 500 ms delay line. Without it spans come too late.

Privacy (plan 06 §8): recognized words live only inside feed(); spans carry category + length, never text, and
nothing here logs or prints what was said.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from callguard.types import SR, SecretSpan

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "runs" / "models" / "vosk-model-small-en-us-0.15"
# Decoder options for early partial word timings (appended to conf/model.conf by scripts/get_vosk_model.py).
LOW_LATENCY = ("--determinize-max-delay=3", "--determinize-min-chunk-size=1", "--frames-per-chunk=9")

DIGITS = ("zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
# Open-vocabulary output for a digit read in a run ("two to seven", "for nine eight"): counts as a digit only next to
# a real digit word (within gap_s), so "went to the shop" stays innocent.
HOMOPHONES = ("o", "to", "too", "for", "won", "ate")
# Short on purpose: the open-vocabulary LM often hears "my pin is" as "my pin he has".
OWN_TRIGGERS = {("code", "is"): "digits", ("my", "code"): "digits", ("my", "pin"): "digits",
                ("pin", "number"): "digits", ("password", "is"): "password", ("my", "password"): "password",
                ("card", "number"): "card"}
REQUEST_TRIGGERS = (("verification",), ("password",), ("passcode",), ("read", "me"), ("the", "code"),
                    ("your", "code"), ("that", "code"), ("pin", "number"), ("your", "pin"), ("security", "number"),
                    ("card", "number"), ("one", "time", "code"))
SAME = int(0.12 * SR)       # two sightings of a word whose starts differ by less than this are the same word
PRE = int(0.03 * SR)        # span starts this early: Vosk times are on a 30 ms frame grid
MIN_EXT = int(0.1 * SR)     # batch a growing word's extensions (hold_s already mutes ahead of it)
CHAIN_MAX = 10 * SR         # ponytail: a trigger chain (e.g. "my password is [unk] [unk]...") stops after 10 s even
                            # if speech runs on without a gap_s pause; a card number read slowly fits. Tune on real calls.


class VoskSpotter:
    """SecretSpotterDriver. Outbound: a digit that starts within gap_s of the previous digit's end is redacted
    (the first of a run passes: "at two" is fine); after an own-side trigger, every token chained within gap_s is
    redacted too, starting right at the trigger's end. Each redacted token's span runs to word end + max(tail_s,
    hold_s): hold_s mutes ahead so the *next* token of a run is covered before the recognizer has even seen it
    (recognition lags the word by ~0.4 s). Inbound: one "request" span per trigger phrase heard.
    Partials get revised ("six" becomes [unk] a moment later), so the spans are recomputed from the current
    hypothesis every block and only new ones (or a grown word's extension) are emitted: each span once.
    Not here: letters A-Z (the letter "a" and friends would chain ordinary words into runs; add with NATO words if
    a real call needs them), and min_digits (reporting threshold; the pipeline applies it to the run length)."""

    def __init__(self, mode: str = "outbound", model_dir: str | Path = MODEL_DIR, gap_s: float = 1.2,
                 tail_s: float = 0.3, hold_s: float = 0.6):
        if mode not in ("outbound", "inbound"):
            raise ValueError(f"mode must be outbound | inbound, got {mode!r}")
        model_dir = Path(model_dir)
        if not (model_dir / "am" / "final.mdl").exists():
            raise FileNotFoundError(f"Vosk model not found at {model_dir} (run scripts/get_vosk_model.py)")
        conf = (model_dir / "conf" / "model.conf").read_text().split()
        self.low_latency = all(o in conf for o in LOW_LATENCY)
        if not self.low_latency:
            log.warning("Vosk model conf lacks the low-latency options: spans will come ~2 s late "
                        "(run scripts/get_vosk_model.py)")
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-2)   # no Kaldi warnings on stderr (they carry no text, but they spam)
        self._Rec = KaldiRecognizer
        self.model = Model(str(model_dir))
        self.mode, self.name = mode, f"vosk-spotter-{mode}"
        self.gap, self.tail, self.hold = round(gap_s * SR), round(tail_s * SR), round(hold_s * SR)
        self.reset()

    def reset(self) -> None:
        self.rec = self._Rec(self.model, SR)
        self.rec.SetWords(True)
        self.rec.SetPartialWords(True)
        self._origin: int | None = None   # absolute index of the first sample fed since reset
        self._past: list[tuple[str, int, int]] = []   # recent words of finished utterances (runs span endpoints)
        self._sent: list[list] = []       # [kind, start, emitted end] of every span placed so far (recent)

    def feed(self, block: np.ndarray, start: int) -> list[SecretSpan]:
        if self._origin is None:
            self._origin = int(start)
        pcm = (np.clip(np.asarray(block, np.float32).ravel(), -1.0, 1.0) * 32767).astype(np.int16)
        final = self.rec.AcceptWaveform(pcm.tobytes())
        words = json.loads(self.rec.Result() if final else self.rec.PartialResult())
        cur = [(w["word"], self._origin + round(w["start"] * SR), self._origin + round(w["end"] * SR))
               for w in words.get("result" if final else "partial_result", [])]
        hyp = self._past + cur
        if final:                          # keep what a later run or trigger chain could still reach
            keep = int(start) + len(pcm) - CHAIN_MAX - 2 * self.gap
            self._past = [t for t in hyp if t[2] > keep]
        out: list[SecretSpan] = []
        for kind, s, e, cat, n in (self._decide(hyp) if self.mode == "outbound" else self._requests(hyp)):
            sent = next((x for x in self._sent if x[0] == kind and abs(x[1] - s) < SAME), None)
            if sent is None:
                out.append(SecretSpan(s, e, cat, n))
                self._sent.append([kind, s, e])
            elif e > sent[2] + MIN_EXT:    # the word grew in a later partial: send only the extension
                out.append(SecretSpan(sent[2], e, cat, n))
                sent[2] = e
        if len(self._sent) > 256:
            self._sent = self._sent[-128:]
        return out

    def _requests(self, hyp: list[tuple[str, int, int]]):
        words = [t[0] for t in hyp]
        for i, (_, s, _) in enumerate(hyp):
            for p in REQUEST_TRIGGERS:
                if tuple(words[i:i + len(p)]) == p:
                    yield "request", s, hyp[i + len(p) - 1][2], "request", 0

    def _decide(self, hyp: list[tuple[str, int, int]]):
        """Spans the current hypothesis calls for, recomputed from scratch each block, so a word the recognizer
        later relabels (partials are revised) stops counting. Yields (kind, start, end, category, length)."""
        words = [t[0] for t in hyp]
        chain = None                        # [category, until, length, trigger end, index after the trigger]
        prev_end, run = None, 0
        near = self._digit_like(hyp)
        for i, (w, s, e) in enumerate(hyp):
            for p, cat in OWN_TRIGGERS.items():
                if tuple(words[i:i + len(p)]) == p:
                    end = hyp[i + len(p) - 1][2]
                    chain = [cat, end + self.gap, 0, end, i + len(p)]
                    yield "trigger", s, end + self.gap, cat, 0      # mute ahead: the secret comes next
            digit = near[i]
            if digit:
                run = run + 1 if prev_end is not None and s - prev_end <= self.gap else 1
                prev_end = e
            if chain is not None and (s > chain[1] or s > chain[3] + CHAIN_MAX):
                chain = None
            hold = max(self.tail, self.hold) if digit else self.tail
            # in a chain: a password takes every word; a code or card number its digits (and [unk])
            if chain is not None and i >= chain[4] and (digit or w == "[unk]" or chain[0] == "password"):
                chain[2] += digit or chain[0] == "password"     # length: digits (every word for a password)
                chain[1] = max(chain[1], e + self.gap)
                yield "token", s - PRE, e + hold, chain[0], chain[2]
            elif digit and run >= 2:
                yield "token", s - PRE, e + hold, "digits", run

    def _digit_like(self, hyp: list[tuple[str, int, int]]) -> list[bool]:
        """Digit words, plus homophones chained (within gap_s) to one: "for to nine" is three digits. Two homophones
        in a row count too ("to to" is how the LM first hears "two two")."""
        d = [w in DIGITS for w, _, _ in hyp]
        for i in range(len(hyp) - 1):
            if hyp[i][0] in HOMOPHONES and hyp[i + 1][0] in HOMOPHONES and hyp[i + 1][1] - hyp[i][2] <= self.gap:
                d[i] = d[i + 1] = True
        grew = True
        while grew:
            grew = False
            for i, (w, s, e) in enumerate(hyp):
                if not d[i] and w in HOMOPHONES and any(
                        0 <= j < len(hyp) and d[j] and max(hyp[j][1] - e, s - hyp[j][2]) <= self.gap for j in (i - 1, i + 1)):
                    d[i] = grew = True
        return d
