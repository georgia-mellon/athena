"""VoskSpotter on real (TTS) speech, streamed in 20 ms blocks through a simulated 500 ms delay line.

Skips unless the Vosk model (scripts/get_vosk_model.py) and the facebook/mms-tts-eng weights are both local.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from callguard.drivers.secret_vosk import DIGITS, MODEL_DIR
from callguard.types import BLOCK, SR, SecretSpotterDriver

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "am" / "final.mdl").exists(),
                                reason="Vosk model missing: run scripts/get_vosk_model.py")
DELAY = SR // 2


@pytest.fixture(scope="module")
def tts():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.AutoTokenizer.from_pretrained("facebook/mms-tts-eng", local_files_only=True)
        model = tr.VitsModel.from_pretrained("facebook/mms-tts-eng", local_files_only=True).eval()
    except OSError:
        pytest.skip("facebook/mms-tts-eng not in the local HF cache")
    assert model.config.sampling_rate == SR

    def render(text: str) -> np.ndarray:
        torch.manual_seed(0)
        with torch.no_grad():
            x = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        return np.concatenate([np.zeros(SR // 4), x, np.zeros(SR)]).astype(np.float32)
    return render


def align(x: np.ndarray, text: str) -> list[tuple[str, int, int]]:
    """Word times by Vosk forced to the known sentence (grammar = that sentence only)."""
    from vosk import KaldiRecognizer, Model
    rec = KaldiRecognizer(Model(str(MODEL_DIR)), SR, json.dumps([text]))
    rec.SetWords(True)
    rec.AcceptWaveform((x * 32767).astype(np.int16).tobytes())
    words = json.loads(rec.FinalResult())["result"]
    assert " ".join(w["word"] for w in words) == text
    return [(w["word"], round(w["start"] * SR), round(w["end"] * SR)) for w in words]


def stream(spotter, x: np.ndarray, origin: int = 5 * SR):
    """Feed 20 ms blocks at absolute index `origin`; returns (spans, redacted mask) with delay-line leak semantics:
    a span returned after the block ending at T can only redact samples >= T - DELAY."""
    mask = np.zeros(len(x), bool)
    spans = []
    for i in range(0, len(x), BLOCK):
        for sp in spotter.feed(x[i:i + BLOCK], origin + i):
            spans.append(sp)
            a, b = max(sp.start - origin, i + BLOCK - DELAY, 0), min(sp.end - origin, len(x))
            mask[a:max(a, b)] = True
    return spans, mask


def test_code_is_redacted(tts):
    from callguard.drivers.secret_vosk import VoskSpotter
    text = "the code is four eight two one nine three"
    x = tts(text)
    sp = VoskSpotter()
    assert isinstance(sp, SecretSpotterDriver) and sp.name == "vosk-spotter-outbound"
    spans, mask = stream(sp, x)
    digits = [(s, e) for w, s, e in align(x, text) if w in DIGITS]
    leaked = [np.mean(~mask[s:e]) > 0.1 for s, e in digits]
    assert sum(leaked) <= 1, leaked
    assert all(s.category == "digits" for s in spans) and max(s.length for s in spans) == 6
    sp.reset()                                          # fresh recognizer, same result on a new stream position
    assert stream(sp, x, origin=0)[1].sum() > 0


def test_run_without_trigger_leaks_at_most_the_first_digit(tts):
    from callguard.drivers.secret_vosk import VoskSpotter
    text = "okay five three nine one seven"
    x = tts(text)
    _, mask = stream(VoskSpotter(), x)
    digits = [(s, e) for w, s, e in align(x, text) if w in DIGITS]
    leaked = [np.mean(~mask[s:e]) > 0.1 for s, e in digits]
    assert leaked[0] and sum(leaked) <= 2, leaked      # the first digit passes by design; the second may clip


@pytest.mark.parametrize("text", ["see you at two", "about five minutes"])
def test_innocent_number_passes(tts, text):
    from callguard.drivers.secret_vosk import VoskSpotter
    spans, _ = stream(VoskSpotter(), tts(text))
    assert spans == []


def test_inbound_request_trigger(tts):
    from callguard.drivers.secret_vosk import VoskSpotter
    sp = VoskSpotter(mode="inbound")
    spans, _ = stream(sp, tts("please read me the verification code"))
    assert sp.name == "vosk-spotter-inbound"
    assert spans and all(s.category == "request" and s.length == 0 for s in spans)
    assert len(spans) == len({s.start for s in spans})  # once per occurrence
