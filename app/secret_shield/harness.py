"""Secret Shield harness: does a spoken-secret spotter fit CallGuard? (contract, latency, quick quality)

    python -m app.secret_shield.harness [--spotter mock|real|module.path:ClassName] [--mode outbound|inbound]

`mock` = the placeholder MockSpotter with one scripted span (so the span checks have something to check), `real` =
the Vosk spotter via app.source.registry (model: python -m app.secret_shield.get_model), anything else = a class
path instantiated with mode=..., or with no arguments. Tested raw (no Quarantine). The quality rows stream a TTS
digit utterance (facebook/mms-tts-eng from the local HF cache) through the spotter and a 500 ms delay line and are
informational. Exit code 1 when a contract or latency check fails.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from typing import Any

import numpy as np

from app.hearsay.harness import Report, Skip, load_driver, probe_audio
from app.source.types import BLOCK, SR, SecretSpan, SecretSpotterDriver

BUDGET_MS = 10.0            # mean per 20 ms block: the spotter shares a worker with the rest of the pipeline
DELAY = SR // 2             # the redactor's delay line (config secret.delay_ms = 500)
CATEGORIES = {"digits", "password", "card", "request"}
FIELDS = {"start", "end", "category", "length"}
# Fake codes only (CLAUDE.md). Outbound: a trigger phrase first, then none (the first digit passes by design).
UTTERANCES = {"outbound": ["the code is four seven two nine", "five eight one six three"],
              "inbound": ["just read me the verification code please", "can you tell me your pin number"]}


def _load(spec: Any, mode: str) -> Any:
    if spec == "mock":   # the placeholder, scripted with one span ~0.6 s in so every span check runs
        from app.secret_shield.mock import MockSpotter
        return MockSpotter(spans=[(0.6, 1.0, "request" if mode == "inbound" else "digits", 0 if mode == "inbound" else 2)],
                           mode=mode)
    return load_driver(spec, "make_spotter", "secret", mode=mode)


def stream(sp: Any, x: np.ndarray) -> tuple[list[tuple[SecretSpan, int]], np.ndarray]:
    """reset(), then feed 20 ms blocks at absolute index i; returns ([(span, index after its block)], ms per block)."""
    sp.reset()
    spans, ms = [], []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        t0 = time.perf_counter()
        got = sp.feed(x[i:i + BLOCK], i)
        ms.append((time.perf_counter() - t0) * 1000)
        assert isinstance(got, list), f"feed() must return a list, got {type(got).__name__}"
        spans += [(s, i + BLOCK) for s in got]
    return spans, np.array(ms)


def check(spotter: Any = "mock", mode: str = "outbound", quality: bool = True) -> Report:
    rep = Report(f"Secret Shield harness: {spotter if isinstance(spotter, str) else type(spotter).__name__} ({mode})")
    box: dict[str, Any] = {}

    def load() -> str:
        t0 = time.perf_counter()
        box["d"] = _load(spotter, mode)
        return f"{type(box['d']).__name__} in {time.perf_counter() - t0:.1f} s"
    if not rep.run("load", "build spotter", load):
        return rep
    d = box["d"]
    tts = _tts() if quality else None           # None when the TTS model isn't local (loading it takes seconds)
    probe = tts(UTTERANCES[mode][0]) if tts else np.concatenate(probe_audio(2 * SR))

    def protocol() -> str:
        assert isinstance(d, SecretSpotterDriver), "not a SecretSpotterDriver (needs name, feed, reset)"
        assert isinstance(d.name, str) and d.name, f"name must be a non-empty str, got {d.name!r}"
        return f"name={d.name!r}"
    rep.run("contract", "Protocol + attributes", protocol)

    def spans() -> str:
        got, ms = stream(d, probe)
        box["ms"] = ms
        seen = set()
        for s, _ in got:
            assert isinstance(s, SecretSpan), f"feed() must return SecretSpan, got {type(s).__name__}"
            assert {f.name for f in dataclasses.fields(s)} == FIELDS and not hasattr(s, "text"), \
                "a span carries start, end, category, length only: never the recognized text"
            assert isinstance(s.start, (int, np.integer)) and isinstance(s.end, (int, np.integer)), "start/end must be ints"
            assert 0 <= s.start < s.end, f"bad span [{s.start}, {s.end})"
            assert s.category in CATEGORIES, f"category {s.category!r} not in {sorted(CATEGORIES)}"
            assert isinstance(s.length, (int, np.integer)) and s.length >= 0, f"length must be an int >= 0: {s.length!r}"
            key = (s.start, s.end, s.category)
            assert key not in seen, f"span {key} emitted twice (each span at most once)"
            seen.add(key)
        box["spans"] = [(s.start, s.end, s.category, s.length) for s, _ in got]
        src = "TTS utterance" if tts else "synthetic probe (no TTS model: may give no spans)"
        return f"{len(got)} span(s) on the {src}; fields ok, no text, categories valid, none repeated"
    rep.run("contract", "feed() -> [SecretSpan]", spans)

    def determinism() -> str:
        if "spans" not in box:
            raise Skip("feed() failed above")
        again = [(s.start, s.end, s.category, s.length) for s, _ in stream(d, probe)[0]]
        first = box["spans"]
        # Vosk's word times jitter by a sample between runs (float seconds -> samples), so allow 1 ms
        same = len(again) == len(first) and all(
            a[2:] == b[2:] and abs(a[0] - b[0]) <= SR // 1000 and abs(a[1] - b[1]) <= SR // 1000
            for a, b in zip(again, first))
        assert same, f"after reset() the same audio gave different spans: {len(again)} vs {len(first)}"
        return "reset() + same audio -> same spans (within 1 ms)"
    rep.run("contract", "reset() + deterministic", determinism)

    def latency() -> str:
        if "ms" not in box:
            raise Skip("feed() failed above")
        ms = box["ms"]
        assert ms.mean() < BUDGET_MS, f"mean {ms.mean():.2f} ms per 20 ms block, budget {BUDGET_MS:.0f} ms"
        return f"mean {ms.mean():.2f} ms, p99 {np.percentile(ms, 99):.1f} ms, max {ms.max():.1f} ms per 20 ms block"
    rep.run("latency", "feed 20 ms block", latency)

    if quality:
        rep.run("quality", "spans on 3 s of silence", lambda: _silence(d), info=True)
        for text in UTTERANCES[mode]:
            rep.run("quality", f"TTS: {text!r}", lambda text=text: _utterance(d, tts, text, mode), info=True)
    return rep


def _silence(d: Any) -> str:
    got, _ = stream(d, np.zeros(3 * SR, np.float32))
    return f"{len(got)} span(s) (want 0; the scripted mock emits its script regardless)"


def _tts():
    """facebook/mms-tts-eng from the local HF cache (never downloads), as a text -> 16 kHz audio function; or None."""
    try:
        import torch
        from transformers import AutoTokenizer, VitsModel
        tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", local_files_only=True)
        model = VitsModel.from_pretrained("facebook/mms-tts-eng", local_files_only=True).eval()
    except Exception:  # noqa: BLE001 - no torch/transformers, or the model isn't cached
        return None

    def render(text: str) -> np.ndarray:
        torch.manual_seed(0)
        with torch.no_grad():
            x = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        return np.concatenate([np.zeros(SR // 4), x, np.zeros(SR)]).astype(np.float32)
    return render


def _words(x: np.ndarray, text: str) -> list[tuple[str, int, int]]:
    """Word times by Vosk forced to the known sentence."""
    from app.secret_shield.spotter import MODEL_DIR
    if not (MODEL_DIR / "am" / "final.mdl").exists():
        raise Skip("no Vosk model for word timing (python -m app.secret_shield.get_model)")
    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-2)
    rec = KaldiRecognizer(Model(str(MODEL_DIR)), SR, json.dumps([text]))
    rec.SetWords(True)
    rec.AcceptWaveform((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    res = json.loads(rec.FinalResult()).get("result", [])
    if " ".join(w["word"] for w in res) != text:
        raise Skip("forced alignment could not place the sentence")
    return [(w["word"], round(w["start"] * SR), round(w["end"] * SR)) for w in res]


def _utterance(d: Any, tts, text: str, mode: str) -> str:
    if tts is None:
        raise Skip("facebook/mms-tts-eng not in the local HF cache")
    x = tts(text)
    got, _ = stream(d, x)
    if mode == "inbound":
        return f"{sum(s.category == 'request' for s, _ in got)} request span(s) (want >= 1)"
    from app.secret_shield.spotter import DIGITS
    # the redactor's rule: a span emitted after the block ending at T can only cut samples >= T - DELAY
    cut = np.zeros(len(x), bool)
    for s, t in got:
        a = max(s.start, t - DELAY, 0)
        cut[a:max(a, min(s.end, len(x)))] = True
    digits = [(s, e) for w, s, e in _words(x, text) if w in DIGITS]
    hidden = sum(cut[s:e].mean() > 0.8 for s, e in digits)
    return f"{hidden}/{len(digits)} digit words redacted (> 80 % cut through a 500 ms delay line); " \
           f"{len(got)} span(s)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.secret_shield.harness", description=__doc__.split("\n")[0])
    ap.add_argument("--spotter", default="mock", help="mock | real | module.path:ClassName (default mock)")
    ap.add_argument("--mode", default="outbound", choices=("outbound", "inbound"))
    ap.add_argument("--no-quality", action="store_true", help="contract + latency only")
    a = ap.parse_args(argv)
    rep = check(a.spotter, a.mode, quality=not a.no_quality)
    rep.print()
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
