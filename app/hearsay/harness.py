"""Hearsay harness: does a voice-authenticity driver fit CallGuard? (contract, latency, quick quality)

    python -m app.hearsay.harness [--driver mock|real|module.path:ClassName] [--clips N]

`mock` = the placeholder (app.hearsay.mock.MockVoice), `real` = the frozen Hearsay model via app.source.registry,
anything else = a class path instantiated with no arguments. The driver is tested raw (no Quarantine), so a crash
shows up here instead of being swallowed. Exit code 1 when a contract or latency check fails; quality rows are
informational.

"""
from __future__ import annotations

import argparse
import math
import time
from typing import Any

import numpy as np

from app.source.types import SR, VoiceAuthenticityDriver, VoiceScore

WINDOW = 4 * SR          # the pipeline scores 4 s windows of far-end audio
BUDGET_MS = 2000.0       # ... every 2 s, so a window must score in < 2 s on CPU


from app.source.harness import Report, Skip, load_driver, probe_audio, timed  # noqa: E402


def check(driver: Any = "mock", clips: int = 8, quality: bool = True) -> Report:
    """Run every check on `driver` (see load_driver) and return the Report (print it with .print())."""
    rep = Report(f"Hearsay harness: {driver if isinstance(driver, str) else type(driver).__name__}")
    box: dict[str, Any] = {}

    def load() -> str:
        t0 = time.perf_counter()
        box["d"] = load_driver(driver, "make_voice", "voice")
        return f"{type(box['d']).__name__} in {time.perf_counter() - t0:.1f} s"
    if not rep.run("load", "build driver", load):
        return rep
    d = box["d"]
    audio = probe_audio(WINDOW)

    def protocol() -> str:
        assert isinstance(d, VoiceAuthenticityDriver), "not a VoiceAuthenticityDriver (needs name, sample_rate, score)"
        assert isinstance(d.name, str) and d.name, f"name must be a non-empty str, got {d.name!r}"
        assert d.sample_rate == SR, f"sample_rate must be {SR}, got {d.sample_rate}"
        return f"name={d.name!r}, sample_rate={d.sample_rate}"
    rep.run("contract", "Protocol + attributes", protocol)

    def scores() -> str:
        out = []
        for x in audio:
            s = d.score(x)
            assert isinstance(s, VoiceScore), f"score() must return VoiceScore, got {type(s).__name__}"
            assert 0.0 <= s.p_synthetic <= 1.0, f"p_synthetic {s.p_synthetic} outside [0, 1]"
            for f in ("p_synthetic", "margin", "threshold", "latency_ms"):
                assert math.isfinite(getattr(s, f)), f"{f} is not finite: {getattr(s, f)}"
            assert s.latency_ms >= 0, f"latency_ms {s.latency_ms} < 0"
            assert isinstance(s.detail, dict), "detail must be a dict"
            # p = 0.5 is the deployment threshold: p above 0.5 exactly when the margin is above the threshold
            assert (s.p_synthetic - 0.5) * (s.margin - s.threshold) >= 0 or abs(s.p_synthetic - 0.5) < 1e-6, \
                f"p_synthetic {s.p_synthetic:.3f} and margin {s.margin:.3f} vs threshold {s.threshold:.3f} disagree"
            out.append(s)
        box["scores"] = out
        return "VoiceScore, p in [0,1], finite, p>0.5 <=> margin>threshold; p = " + \
            ", ".join(f"{s.p_synthetic:.3f}" for s in out)
    rep.run("contract", "score(4 s) -> VoiceScore", scores)

    def at_threshold() -> str:
        # the real driver maps margin -> p with a logistic centred on the threshold; check that exact point
        from app.hearsay.driver import p_from_margin
        if not hasattr(d, "thr") or not hasattr(d, "s"):
            raise Skip("driver has no thr/s calibration attributes (only checked for HearsayDriver-style drivers)")
        p = p_from_margin(d.thr, d.thr, d.s)
        assert abs(p - 0.5) < 1e-9, f"p at the threshold is {p}, not 0.5"
        return f"p(threshold={d.thr:.3f}) = 0.5"
    rep.run("contract", "p = 0.5 at the threshold", at_threshold)

    def determinism() -> str:
        a, b = d.score(audio[1]).p_synthetic, d.score(audio[1]).p_synthetic
        assert abs(a - b) < 1e-5, f"same audio scored {a:.6f} then {b:.6f}"
        return f"same window twice: {a:.6f} = {b:.6f}"
    rep.run("contract", "deterministic", determinism)

    def latency() -> str:
        ms = timed(lambda: d.score(audio[0]), 3)
        med = float(np.median(ms))
        assert med < BUDGET_MS, f"median {med:.0f} ms per 4 s window, budget {BUDGET_MS:.0f} ms"
        return f"median {med:.0f} ms, max {ms.max():.0f} ms per 4 s window (budget {BUDGET_MS:.0f} ms)"
    rep.run("latency", "score 4 s window", latency)

    if quality:
        rep.run("quality", f"held-out clips (n={clips})", lambda: _quality(d, clips), info=True)
    return rep


def _quality(d: Any, n: int) -> str:
    """Accuracy at p = 0.5 on n clips of Hearsay's frozen headline set (test_internal_testlike: held out, and never
    used to choose anything), half real, half fake, seeded. Whole clips capped at 12 s, as the model was evaluated."""
    import soundfile as sf

    from app.hearsay.driver import HEARSAY_ROOT
    path = HEARSAY_ROOT / "data" / "processed" / "manifest.parquet"
    if not path.exists():
        raise Skip(f"no Hearsay manifest at {path} (set HEARSAY_ROOT)")
    import pandas as pd
    m = pd.read_parquet(path)
    m = m[m.test_internal_testlike]
    pick = pd.concat([m[m.label == lab].sample(n=max(n // 2, 1), random_state=0) for lab in ("bonafide", "spoof")])
    right, flagged_real, missed_fake, ps = 0, 0, 0, []
    for r in pick.itertuples():
        f = HEARSAY_ROOT / r.path
        if not f.exists():
            raise Skip(f"manifest lists {r.path} but the audio isn't there")
        x, sr = sf.read(f, dtype="float32", frames=12 * SR)
        assert sr == SR, f"{r.path}: {sr} Hz"
        p = d.score(x if x.ndim == 1 else x.mean(1)).p_synthetic
        fake = r.label == "spoof"
        right += (p > 0.5) == fake
        flagged_real += (not fake) and p > 0.5
        missed_fake += fake and p <= 0.5
        ps.append(p)
    k = len(pick)
    return (f"accuracy at p=0.5: {right}/{k} = {100 * right / k:.0f} % (reals flagged {flagged_real}, fakes missed "
            f"{missed_fake}); from test_internal_testlike, seed 0")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.hearsay.harness", description=__doc__.split("\n")[0])
    ap.add_argument("--driver", default="mock", help="mock | real | module.path:ClassName (default mock)")
    ap.add_argument("--clips", type=int, default=8, help="held-out clips for the quality check (default 8)")
    ap.add_argument("--no-quality", action="store_true", help="contract + latency only")
    a = ap.parse_args(argv)
    rep = check(a.driver, a.clips, quality=not a.no_quality)
    rep.print()
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
