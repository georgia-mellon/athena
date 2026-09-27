"""Hearsay voice-authenticity driver, read-only: E5 (R4ft + R6 XLS-R + R1 LightGBM fusion; default), R5 (R4ft + R1)
or R4ft alone.

Same inference path as Hearsay's bench_score.py, so our numbers match Hearsay's: prep the clip -> up to max_windows
4 s windows -> mean logit; r5/e5 add the R1 LightGBM and (for e5) a second XLS-R pass, fused with frozen weights.

p_synthetic = sigmoid((margin - thr) / s): thr is the deployment threshold from val_testlike, s maps the median fake
to p = 0.95; both cached in runs/hearsay_calibration.json keyed by checkpoint sha. `ai_p` raises the threshold so
p_synthetic = 0.5 means "calibrated p = ai_p".
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from app.source.types import SR, VoiceScore

REPO = Path(__file__).resolve().parents[2]
HEARSAY_ROOT = Path(os.environ.get("HEARSAY_ROOT") or REPO.parent / "Hearsay")  # same default as config.py
RUN = HEARSAY_ROOT / "data" / "models" / "r4ft_xlsr" / "R4ft_xlsr_light"
RUN6 = HEARSAY_ROOT / "data" / "models" / "r4ft_xlsr" / "R6_xlsr_light"
E5_JSON = HEARSAY_ROOT / "data" / "models" / "e5" / "e5_fusion.json"
R1_MODEL = HEARSAY_ROOT / "data" / "models" / "r1_lgbm_all_full.txt"
R5_JSON = HEARSAY_ROOT / "data" / "scores" / "R5_r4ft_r1.json"
CACHE = REPO / "runs" / "hearsay_calibration.json"
SCORE_FILES = {"r4ft": "R4ft_xlsr_light", "r5": "R5_r4ft_r1", "e5": "E5_r4ft_r6_r1"}  # val_testlike calibration


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

    def __init__(self, mode: str = "e5", threads: int = 4, device: str = "auto", max_windows: int | None = None,
                 ai_p: float = 0.5):
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
        self.fs, self.dev = fs, hdev.resolve(device)
        self.model, self.args, cfg = self._load_xlsr(RUN, device)
        self.sha = cfg["sha256"]
        self.max_windows = max_windows or cfg["max_windows"]
        if mode == "e5":
            if not RUN6.exists():
                raise FileNotFoundError(f"Hearsay R6 model not found at {RUN6} (needed for mode e5)")
            self.model6, self.args6, cfg6 = self._load_xlsr(RUN6, device)
            self.sha = f"{self.sha}+{cfg6['sha256']}"  # the calibration belongs to this checkpoint pair
            self.fusion = json.loads(E5_JSON.read_text())
        if mode in ("r5", "e5"):
            import lightgbm as lgb
            from hearsay.features.bio import FEATURES as BIO
            from hearsay.features.spectral import FEATURES as SPEC
            self.booster = lgb.Booster(model_file=str(R1_MODEL))
            self.r1_feats = SPEC + BIO
            if mode == "r5":
                self.fusion = json.loads(R5_JSON.read_text())
        self.thr, self.s = self._calibration()
        self.thr += self.s * math.log(ai_p / (1 - ai_p))

    def _load_xlsr(self, run: Path, device: str):
        """An XLS-R checkpoint exactly as Hearsay scores it: architecture frozen in config.json, sha256-checked."""
        import torch
        cfg = json.loads((run / "config.json").read_text())
        self._verify_sha(run, cfg["sha256"])
        a = self.fs.parse(["score", "--device", device, "--name", run.name])
        for k in self.fs.ARCH:
            setattr(a, k, cfg["arch"][k])
        model = self.fs.make_model(a, grad_ckpt=False)
        model.load_state_dict(torch.load(run / "best.pth", map_location="cpu"))
        return model.to(self.dev).eval(), a, cfg

    def _verify_sha(self, run: Path, sha: str) -> None:
        """Hash best.pth against config.json once per (size, mtime); a 1.2 GB hash on every start is too slow."""
        best, c = run / "best.pth", _load_cache()
        st = best.stat()
        key = {"size": st.st_size, "mtime": st.st_mtime, "sha256": sha}
        verified = c.setdefault("verified_runs", {})
        if verified.get(run.name) == key:
            return
        if self.fs.sha256(best) != sha:
            raise RuntimeError(f"{best} does not match the sha256 frozen in config.json")
        verified[run.name] = key
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

    def _xlsr(self, y: np.ndarray, model, args) -> float:
        from hearsay.models.ssl_e2e_data import eval_windows
        import torch
        with torch.inference_mode():
            return float(self.fs.forward_windows(model, eval_windows(y, self.max_windows), self.dev, args,
                                                 log=lambda *_: None).mean())

    def _r4ft(self, y: np.ndarray) -> float:
        return self._xlsr(y, self.model, self.args)

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

    def _fuse(self, r4: float, r1: float, r6: float | None = None) -> float:
        """R5 / E5 margin with the frozen weights: sum_k w_k (transform_k(score_k) - mu_k) / sd_k + intercept, the
        arithmetic of bench_score.fused() / fuse.apply_transform() (XLS-R margins as is, R1 as a logit)."""
        f = self.fusion
        r1 = min(max(r1, 1e-6), 1 - 1e-6)
        logit = math.log(r1 / (1 - r1))
        raw = {"R4ft_xlsr_light": r4, "R6_xlsr_light": r6, "R1_lgbm_all_full": logit, "R1": logit}
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
        if self.mode == "e5":
            detail["r6_margin"] = r6 = self._xlsr(y, self.model6, self.args6)
            detail["r1_p"] = r1 = self._r1(y)
            margin = self._fuse(r4, r1, r6)
        elif self.mode == "r5":
            detail["r1_p"] = r1 = self._r1(y)
            margin = self._fuse(r4, r1)
        return VoiceScore(p_synthetic=p_from_margin(margin, self.thr, self.s), margin=margin, threshold=self.thr,
                          latency_ms=(time.perf_counter() - t0) * 1000, detail=detail)
