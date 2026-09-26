"""Hearsay voice-authenticity driver: the frozen R4ft (XLS-R) model, optionally fused with R1 (R5), read-only.

Same inference path as Hearsay's bench_score.py so the numbers we show are the numbers Hearsay reported:
prep() the whole clip -> up to max_windows 4 s windows (1 centre window for a <= 4 s clip) -> mean logit.
r5 adds the R1 LightGBM on spectral+bio features of the centre 4 s and fuses with the frozen R5 weights.

p_synthetic = sigmoid((margin - thr) / s): thr is the deployment threshold fixed on Hearsay's val_testlike with
bench_score.threshold() (a real flagged as fake costs 4x), and s maps the median val_testlike fake to p = 0.95.
Both are computed once and cached in runs/hearsay_calibration.json, keyed by the checkpoint sha256.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from callguard.types import SR, VoiceScore

REPO = Path(__file__).resolve().parents[2]
HEARSAY_ROOT = Path(os.environ.get("HEARSAY_ROOT") or REPO.parent / "Hearsay")  # same default as config.py
RUN = HEARSAY_ROOT / "data" / "models" / "r4ft_xlsr" / "R4ft_xlsr_light"
R1_MODEL = HEARSAY_ROOT / "data" / "models" / "r1_lgbm_all_full.txt"
R5_JSON = HEARSAY_ROOT / "data" / "scores" / "R5_r4ft_r1.json"
CACHE = REPO / "runs" / "hearsay_calibration.json"
SCORE_FILES = {"r4ft": "R4ft_xlsr_light", "r5": "R5_r4ft_r1"}


def p_from_margin(margin: float, thr: float, s: float) -> float:
    """Logistic centred on the deployment threshold: p = 0.5 exactly at thr."""
    z = (margin - thr) / s
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def scale_for(fake_median: float, thr: float) -> float:
    """s such that a margin at the median val_testlike fake maps to p = 0.95 (logit(0.95) = ln 19)."""
    return (fake_median - thr) / math.log(19)


def _import_hearsay() -> None:
    os.environ["HEARSAY_ROOT"] = str(HEARSAY_ROOT)  # hearsay.audio.ROOT reads it at import
    sys.dont_write_bytecode = True  # read-only upstream: no __pycache__ inside it
    for p in (HEARSAY_ROOT / "src", HEARSAY_ROOT / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=1))


class HearsayDriver:
    """VoiceAuthenticityDriver. Not thread-safe: call score() from one worker thread, never the audio thread."""

    sample_rate = SR

    def __init__(self, mode: str = "r4ft", threads: int = 4, device: str = "auto", max_windows: int | None = None):
        if mode not in SCORE_FILES:
            raise ValueError(f"mode must be one of {list(SCORE_FILES)}, got {mode!r}")
        if not RUN.exists():
            raise FileNotFoundError(f"Hearsay model not found at {RUN} (set HEARSAY_ROOT)")
        _import_hearsay()
        import torch
        import finetune_ssl as fs
        from hearsay import device as hdev

        self.mode, self.name = mode, f"hearsay_{mode}"
        torch.set_num_threads(threads)  # leave cores for the audio thread
        cfg = json.loads((RUN / "config.json").read_text())
        self.sha = cfg["sha256"]
        self._verify_sha(fs.sha256)
        a = fs.parse(["score", "--device", device, "--name", RUN.name])
        for k in fs.ARCH:
            setattr(a, k, cfg["arch"][k])  # config.json froze the architecture with the checkpoint
        self.dev = hdev.resolve(device)
        model = fs.make_model(a, grad_ckpt=False)
        model.load_state_dict(torch.load(RUN / "best.pth", map_location="cpu"))
        self.model, self.args, self.fs = model.to(self.dev).eval(), a, fs
        self.max_windows = max_windows or cfg["max_windows"]
        if mode == "r5":
            import lightgbm as lgb
            from hearsay.features.bio import FEATURES as BIO
            from hearsay.features.spectral import FEATURES as SPEC
            self.booster = lgb.Booster(model_file=str(R1_MODEL))
            self.r1_feats = SPEC + BIO
            self.fusion = json.loads(R5_JSON.read_text())
        self.thr, self.s = self._calibration()

    def _verify_sha(self, sha256) -> None:
        """Hash best.pth against config.json once per (size, mtime); a 1.2 GB hash on every start is too slow."""
        best, c = RUN / "best.pth", _load_cache()
        st = best.stat()
        key = {"size": st.st_size, "mtime": st.st_mtime, "sha256": self.sha}
        if c.get("verified") == key:
            return
        if sha256(best) != self.sha:
            raise RuntimeError(f"{best} does not match the sha256 frozen in config.json")
        c["verified"] = key
        _save_cache(c)

    def _calibration(self) -> tuple[float, float]:
        c = _load_cache()
        hit = c.get(self.sha, {}).get(self.mode)
        if hit:
            return hit["thr"], hit["s"]
        import pandas as pd
        from bench_score import threshold
        from hearsay.evaluate import SCORES, manifest
        m = manifest()
        vt = m[m.val_testlike].set_index("path")
        v = pd.read_parquet(SCORES / f"{SCORE_FILES[self.mode]}.parquet").set_index("path").score.reindex(vt.index)
        ok = v.notna().values
        s_, y = v.values[ok], vt.label.values[ok]
        thr = float(threshold(s_, y))
        s = scale_for(float(np.median(s_[y == "spoof"])), thr)
        c.setdefault(self.sha, {})[self.mode] = {"thr": thr, "s": s, "n": int(ok.sum()),
                                                "source": f"{SCORE_FILES[self.mode]}.parquet val_testlike"}
        _save_cache(c)
        return thr, s

    def _r4ft(self, y: np.ndarray) -> float:
        from hearsay.models.ssl_e2e_data import eval_windows
        import torch
        with torch.inference_mode():
            return float(self.fs.forward_windows(self.model, eval_windows(y, self.max_windows), self.dev, self.args,
                                                 log=lambda *_: None).mean())

    def _r1(self, y: np.ndarray) -> float:
        """R1 P(fake) on the centre 4 s of the prep()'d clip, as extract_classic.features() does at eval."""
        import pandas as pd
        from hearsay.features.bio import extract_bio
        from hearsay.features.spectral import extract_spectral
        n = 4 * SR
        if len(y) > n:
            y = y[(len(y) - n) // 2:][:n]
        row = pd.DataFrame([{**extract_spectral(y), **extract_bio(y)}])[self.r1_feats]
        return float(self.booster.predict(row)[0])

    def _fuse(self, r4: float, r1: float) -> float:
        """R5 margin with the frozen weights; same arithmetic as bench_score.fused() / fuse.apply_transform()."""
        f = self.fusion
        r1 = min(max(r1, 1e-6), 1 - 1e-6)
        raw = {"R4ft_xlsr_light": r4, "R1_lgbm_all_full": math.log(r1 / (1 - r1))}
        return f["intercept"] + sum(f["weights"][m] * (raw[m] - f["mu"][m]) / f["sd"][m] for m in f["models"])

    def score(self, audio: np.ndarray) -> VoiceScore:
        from hearsay.preprocess import prep
        t0 = time.perf_counter()
        audio = np.asarray(audio, np.float32).reshape(-1)
        if len(audio) < SR:
            raise ValueError(f"need >= 1 s of audio (3 s recommended), got {len(audio) / SR:.2f} s")
        y = prep(audio)
        r4 = self._r4ft(y)
        detail = {"mode": self.mode, "r4ft_margin": r4, "scale": self.s, "device": self.dev.type,
                  "seconds": len(audio) / SR}
        margin = r4
        if self.mode == "r5":
            detail["r1_p"] = r1 = self._r1(y)
            margin = self._fuse(r4, r1)
        return VoiceScore(p_synthetic=p_from_margin(margin, self.thr, self.s), margin=margin, threshold=self.thr,
                          latency_ms=(time.perf_counter() - t0) * 1000, detail=detail)
