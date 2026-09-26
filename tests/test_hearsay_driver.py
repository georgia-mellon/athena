"""Hearsay driver: the p mapping (pure), and real-model checks on Hearsay's held-out test_internal_testlike clips
(skipped when HEARSAY_ROOT or its models are missing)."""
import math
import time

import numpy as np
import pytest

from callguard.drivers import hearsay_real as hr


def test_p_mapping():
    thr, fake_med = -1.0, 7.0
    s = hr.scale_for(fake_med, thr)
    assert hr.p_from_margin(thr, thr, s) == 0.5
    assert math.isclose(hr.p_from_margin(fake_med, thr, s), 0.95, rel_tol=1e-9)
    assert math.isclose(hr.p_from_margin(2 * thr - fake_med, thr, s), 0.05, rel_tol=1e-9)  # symmetric
    assert hr.p_from_margin(1e6, thr, s) == 1.0 and hr.p_from_margin(-1e6, thr, s) == 0.0  # no overflow


real_ok = hr.RUN.joinpath("best.pth").exists() and hr.R1_MODEL.exists() and hr.R5_JSON.exists()
needs_hearsay = pytest.mark.skipif(not real_ok, reason="HEARSAY_ROOT models not available")


def _clips(n=4):
    import pandas as pd
    m = pd.read_parquet(hr.HEARSAY_ROOT / "data" / "processed" / "manifest.parquet")
    t = m[m.test_internal_testlike & m.path.map(lambda p: (hr.HEARSAY_ROOT / p).exists())]
    if t.empty:
        pytest.skip("test_internal_testlike audio not present")
    return [(r.path, r.label) for lab in ("bonafide", "spoof")
            for r in t[t.label == lab].sample(n, random_state=0).itertuples()]


@pytest.fixture(scope="module", params=["r4ft", "r5"])
def driver(request):
    return hr.HearsayDriver(mode=request.param, threads=4, device="cpu")


@needs_hearsay
def test_real_driver_separates(driver):
    import soundfile as sf
    from callguard.types import VoiceAuthenticityDriver
    assert isinstance(driver, VoiceAuthenticityDriver)
    for path, label in _clips():
        y, sr = sf.read(str(hr.HEARSAY_ROOT / path), dtype="float32")
        assert sr == 16000
        v = driver.score(y)
        print(f"{driver.name} {label:8s} p={v.p_synthetic:.3f} margin={v.margin:+.2f} {v.latency_ms:.0f} ms {path}")
        assert (v.p_synthetic > 0.5) == (label == "spoof"), (path, label, v)


@needs_hearsay
def test_latency_4s(driver):
    x = np.random.default_rng(0).standard_normal(4 * 16000).astype(np.float32) * 0.1
    driver.score(x)  # warm-up
    ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        driver.score(x)
        ms.append((time.perf_counter() - t0) * 1000)
    print(f"\n{driver.name} CPU latency on 4 s: median {np.median(ms):.0f} ms; thr={driver.thr:.4f} s={driver.s:.4f}")
