"""Vosk spoken-secret spotter: SecretSpotterDriver over an open-vocabulary KaldiRecognizer.

Open vocab, not a restricted digits+triggers grammar: the restricted grammar forces ordinary speech onto digits and
fires far too often. Outbound listens to your mic for digit runs and own-side triggers ("the code is ..."); inbound
listens to the caller for request triggers ("read me the code"). Word timings come from partial results, so a span
is placed while the word is still in the redactor's delay line.

get_model.py appends LOW_LATENCY to conf/model.conf so a digit's span lands ~0.4 s after the word starts, inside the
500 ms delay line; without it Vosk only refreshes partial word timings every ~2 s, too late.

Privacy: recognized words live only inside feed(); spans carry category + length, never text.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.source.types import SR, SecretSpan

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "runs" / "models" / "vosk-model-small-en-us-0.15"
# decoder options for early partial word timings (appended to conf/model.conf by get_model.py)
LOW_LATENCY = ("--determinize-max-delay=3", "--determinize-min-chunk-size=1", "--frames-per-chunk=9")

DIGITS = ("zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
# a homophone counts as a digit only next to a real digit word (within gap_s), so "went to the shop" stays innocent
HOMOPHONES = ("o", "to", "too", "for", "won", "ate")
# Short on purpose: the open-vocabulary LM often hears "my pin is" as "my pin he has".
OWN_TRIGGERS = {("code", "is"): "digits", ("my", "code"): "digits", ("my", "pin"): "digits",
                ("pin", "number"): "digits", ("password", "is"): "password", ("my", "password"): "password",
                ("card", "number"): "card"}
REQUEST_TRIGGERS = (("verification",), ("password",), ("passcode",), ("read", "me"), ("the", "code"),
                    ("your", "code"), ("that", "code"), ("pin", "number"), ("your", "pin"), ("security", "number"),
                    ("card", "number"), ("one", "time", "code"))
SAME = int(0.12 * SR)  # two sightings of a word whose starts differ by less than this are the same word
PRE = int(0.03 * SR)  # span starts this early: Vosk times are on a 30 ms frame grid
MIN_EXT = int(0.1 * SR)  # batch a growing word's extensions
CHAIN_MAX = 10 * SR  # a trigger chain stops after 10 s even without a gap_s pause; a slow card number still fits


class VoskSpotter:
    """SecretSpotterDriver. Outbound: a digit within gap_s of the previous one is redacted (the first of a run
    passes); after an own-side trigger, every token chained within gap_s is redacted too. Each span runs to word
    end + max(tail_s, hold_s) so hold_s mutes ahead of the next token, which the recognizer lags by ~0.4 s.
    Inbound: one "request" span per trigger phrase. Partials get revised, so spans are recomputed each block and
    each is emitted once. No letters A-Z, no min_digits (the pipeline applies that to run length)."""

    def __init__(self, mode: str = "outbound", model_dir: str | Path = MODEL_DIR, gap_s: float = 1.2,
                 tail_s: float = 0.3, hold_s: float = 0.6):
        if mode not in ("outbound", "inbound"):
            raise ValueError(f"mode must be outbound | inbound, got {mode!r}")
        model_dir = Path(model_dir)
        if not (model_dir / "am" / "final.mdl").exists():
            raise FileNotFoundError(f"Vosk model not found at {model_dir} (run app/secret_shield/get_model.py)")
        conf = (model_dir / "conf" / "model.conf").read_text().split()
        self.low_latency = all(o in conf for o in LOW_LATENCY)
        if not self.low_latency:
            log.warning("Vosk model conf lacks the low-latency options: spans will come ~2 s late "
                        "(run app/secret_shield/get_model.py)")
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-2)  # no Kaldi warnings on stderr (they spam)
        self._Rec = KaldiRecognizer
        self.model = Model(str(model_dir))
        self.mode, self.name = mode, f"vosk-spotter-{mode}"
        self.gap, self.tail, self.hold = round(gap_s * SR), round(tail_s * SR), round(hold_s * SR)
        self.reset()

    def reset(self) -> None:
        self.rec = self._Rec(self.model, SR)
        self.rec.SetWords(True)
        self.rec.SetPartialWords(True)
        self._origin: int | None = None  # absolute index of the first sample fed since reset
        self._past: list[tuple[str, int, int]] = []  # recent words of finished utterances
        self._sent: list[list] = []  # [kind, start, emitted end] of every span placed so far

    def feed(self, block: np.ndarray, start: int) -> list[SecretSpan]:
        if self._origin is None:
            self._origin = int(start)
        pcm = (np.clip(np.asarray(block, np.float32).ravel(), -1.0, 1.0) * 32767).astype(np.int16)
        final = self.rec.AcceptWaveform(pcm.tobytes())
        words = json.loads(self.rec.Result() if final else self.rec.PartialResult())
        cur = [(w["word"], self._origin + round(w["start"] * SR), self._origin + round(w["end"] * SR))
               for w in words.get("result" if final else "partial_result", [])]
        hyp = self._past + cur
        if final:  # keep what a later run or trigger chain could still reach
            keep = int(start) + len(pcm) - CHAIN_MAX - 2 * self.gap
            self._past = [t for t in hyp if t[2] > keep]
        out: list[SecretSpan] = []
        for kind, s, e, cat, n in (self._decide(hyp) if self.mode == "outbound" else self._requests(hyp)):
            sent = next((x for x in self._sent if x[0] == kind and abs(x[1] - s) < SAME), None)
            if sent is None:
                out.append(SecretSpan(s, e, cat, n))
                self._sent.append([kind, s, e])
            elif e > sent[2] + MIN_EXT:  # the word grew in a later partial: send only the extension
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
        """Spans the current hypothesis calls for, recomputed each block. Yields (kind, start, end, category, length)."""
        words = [t[0] for t in hyp]
        chain = None  # [category, until, length, trigger end, index after the trigger]
        prev_end, run = None, 0
        near = self._digit_like(hyp)
        for i, (w, s, e) in enumerate(hyp):
            for p, cat in OWN_TRIGGERS.items():
                if tuple(words[i:i + len(p)]) == p:
                    end = hyp[i + len(p) - 1][2]
                    chain = [cat, end + self.gap, 0, end, i + len(p)]
                    yield "trigger", s, end + self.gap, cat, 0  # mute ahead: the secret comes next
            digit = near[i]
            if digit:
                run = run + 1 if prev_end is not None and s - prev_end <= self.gap else 1
                prev_end = e
            if chain is not None and (s > chain[1] or s > chain[3] + CHAIN_MAX):
                chain = None
            hold = max(self.tail, self.hold) if digit else self.tail
            # in a chain: a password takes every word; a code or card number its digits (and [unk])
            if chain is not None and i >= chain[4] and (digit or w == "[unk]" or chain[0] == "password"):
                chain[2] += digit or chain[0] == "password"  # length: digits (every word for a password)
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
