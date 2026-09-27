"""Keyguard dashboard API. Serves the single-page UI and read-only views over
arena runs, plus a live attack/shield demo. ponytail: one file, stdlib json +
existing keyguard modules; no DB, the filesystem *is* the store."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import CLASSES, CLS_IDX, N_CLASSES, RUNS, SR, KEY_WIN, PRE_S
from . import audio, segment, features, synth
from .eval import metrics
from .shield.shield import Shield, ShieldConfig
from .attackers.supervised import SupervisedAttacker
from .attack_lm import candidate_lattice, gemini_reconstruct
from .memory import recall_runs
from . import demo_cache

WEB = Path(__file__).parent / "web"
# Prefer the call-adapted attacker (keystrokes buried in speech); fall back to
# the clean baseline if the adapted checkpoint hasn't been trained yet.
DEMO_MODEL_PATH = RUNS / "demo_attacker.pt"
BASE_MODEL_PATH = RUNS / "supervised_mbp.pt"
MAX_DEMO_KEYS = 24          # cap so the demo stays snappy
WAVE_POINTS = 400           # downsampled waveform sent to the browser
DEMO_SPEECH_GAIN = 0.05     # call speech level under natural-amplitude keystrokes
KEY_GAP_S = 1.2             # gap between typed keys (realistic typing cadence)
SHIELD_KEY_FRAMES = 6       # STFT frames per stroke the shield inpaints

app = FastAPI(title="Keyguard")

_attacker: SupervisedAttacker | None = None


def _model() -> SupervisedAttacker:
    """Load the trained attacker once. Raises if no checkpoint is present."""
    global _attacker
    if _attacker is None:
        path = DEMO_MODEL_PATH if DEMO_MODEL_PATH.exists() else BASE_MODEL_PATH
        _attacker = SupervisedAttacker().load(path)
    return _attacker


# ---------------------------------------------------------------- arena reads
def _arena_dir() -> Path:
    return RUNS / "arena"


def _run_dirs() -> list[Path]:
    d = _arena_dir()
    if not d.exists():
        return []
    dirs = [p for p in d.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)  # timestamped -> newest first


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@app.get("/api/runs")
def api_runs():
    out = []
    for p in _run_dirs():
        out.append({"run": p.name, "summary": _read_json(p / "summary.json")})
    return out


@app.get("/api/lineage")
def api_lineage(run: str = "latest"):
    dirs = _run_dirs()
    if not dirs:
        return JSONResponse({"error": "no arena runs yet", "nodes": []})
    if run == "latest":
        target = dirs[0]
    else:
        target = next((p for p in dirs if p.name == run), None)
        if target is None:
            return JSONResponse({"error": f"run {run!r} not found", "nodes": []}, 404)
    data = _read_json(target / "lineage.json")
    if data is None:
        return JSONResponse({"error": "lineage.json unreadable", "nodes": []}, 404)
    data["run"] = target.name
    return data


@app.get("/api/grid")
def api_grid():
    return _read_json(RUNS / "grid.json") or {"cells": []}


@app.get("/api/arena")
def api_arena():
    """The real adversarial min-max arena: latest co-train run (streamed live while
    status=='running') + the cross-session history the shield remembers."""
    adv = sorted((RUNS / "arena").glob("adv-*"), key=lambda p: p.name, reverse=True) \
        if (RUNS / "arena").exists() else []
    latest = _read_json(adv[0] / "arena.json") if adv else None
    return {"latest": latest, "history": recall_runs(limit=12)}


# --------------------------------------------------------------- pipelines
def _n(id, label, done):  # node helper
    return {"id": id, "label": label, "status": "implemented" if done else "planned"}


PIPELINES = {
    "supervised": {
        "title": "Attacker A — Supervised (CoAtNet/CNN)",
        "nodes": [
            _n("a_audio", "Call audio", True), _n("a_seg", "Onset segmentation", True),
            _n("a_mel", "Mel-spectrogram", True), _n("a_cnn", "CoAtNet/CNN classifier", True),
            _n("a_llm", "LLM correction", False),
        ],
        "edges": [["a_audio", "a_seg"], ["a_seg", "a_mel"], ["a_mel", "a_cnn"], ["a_cnn", "a_llm"]],
    },
    "selfsup": {
        "title": "Attacker B — Self-supervised",
        "nodes": [
            _n("s_feat", "Features", True), _n("s_umap", "UMAP", False),
            _n("s_clust", "Cluster", False), _n("s_anchor", "Space anchor", False),
            _n("s_hmm", "HMM/EM", False), _n("s_bert", "char-BERT", False),
            _n("s_llm", "LLM", False), _n("s_spread", "Label spread", False),
            _n("s_fb", "Feedback", False),
        ],
        "edges": [["s_feat", "s_umap"], ["s_umap", "s_clust"], ["s_clust", "s_anchor"],
                  ["s_anchor", "s_hmm"], ["s_hmm", "s_bert"], ["s_bert", "s_llm"],
                  ["s_llm", "s_spread"], ["s_spread", "s_fb"], ["s_fb", "s_clust"]],
    },
    "timing": {
        "title": "Attacker C — Timing side-channel",
        "nodes": [
            _n("t_on", "Onsets", True), _n("t_int", "Inter-key intervals", True),
            _n("t_struct", "Word/structure inference", True),
        ],
        "edges": [["t_on", "t_int"], ["t_int", "t_struct"]],
    },
    "shield": {
        "title": "Shield — Keyguard defence",
        "nodes": [
            _n("d_os", "OS key timestamps", False), _n("d_stft", "STFT", True),
            _n("d_inpaint", "Conditioned inpainting", True),
            _n("d_res", "Residue randomization", True),
            _n("d_decoy", "Decoy injection", True), _n("d_mic", "Virtual mic", False),
        ],
        "edges": [["d_os", "d_stft"], ["d_stft", "d_inpaint"], ["d_inpaint", "d_res"],
                  ["d_res", "d_decoy"], ["d_decoy", "d_mic"]],
    },
}


@app.get("/api/pipelines")
def api_pipelines():
    return PIPELINES


# --------------------------------------------------------------- demo
class DemoReq(BaseModel):
    text: str
    protect: str = "none"  # "none" | "keyguard"
    demo_mode: bool = False  # stage insurance: cache-backed, always-lands fallback


def _speech_of_length(n: int) -> np.ndarray:
    """Concatenate/tile the cached libri clips to at least n samples."""
    clips = sorted((synth.SPEECH_DIR).glob("*.wav"))
    if not clips:
        clips = synth.fetch_speech()
    y = np.concatenate([audio.load(c) for c in clips[:4]])
    if len(y) < n:
        y = np.tile(y, int(np.ceil(n / len(y))))
    return y[:n].astype(np.float32)


def _build_mixture(keys: list[str]):
    """Plant one natural-amplitude recorded press per key onto a call at known,
    evenly spaced onsets. Speech sits at DEMO_SPEECH_GAIN (keystrokes audible but
    the call stays intelligible) -- matches how demo_attacker.pt was trained.
    Returns (clean_call, mixed, onsets)."""
    rng = np.random.default_rng(0)
    step = int(KEY_GAP_S * SR)
    pre = int(PRE_S * SR)
    start0 = int(0.5 * SR)
    total = start0 + step * len(keys) + KEY_WIN + SR
    clean = (_speech_of_length(total) * DEMO_SPEECH_GAIN).astype(np.float32)
    mixed = clean.copy()
    onsets = []
    for i, k in enumerate(keys):
        w = synth._one_press(k, synth.KEY_ROOT, rng)
        start = start0 + i * step
        mixed[start:start + len(w)] += w.astype(np.float32)
        onsets.append(start + pre)
    return clean, mixed, np.array(onsets, dtype=int)


def _downsample(y: np.ndarray, n: int = WAVE_POINTS) -> list[float]:
    if len(y) <= n:
        return [round(float(v), 4) for v in y]
    edges = np.linspace(0, len(y), n + 1, dtype=int)
    return [round(float(np.max(np.abs(y[a:b])) if b > a else 0.0), 4)
            for a, b in zip(edges[:-1], edges[1:])]


def _run_demo(keys: list[str], protect: bool) -> dict:
    """Run the real, deterministic attack/shield pipeline for these keys and return
    the API response dict. Raises on model-load / mixture-build failure so callers
    can decide whether to fall back (demo mode) or surface the error."""
    model = _model()
    clean, mixed, onsets = _build_mixture(keys)
    heard = mixed
    if protect:
        cfg = ShieldConfig(strength=1.0, randomize=1.0, decoys=6, key_frames=SHIELD_KEY_FRAMES)
        heard = Shield(cfg, seed=0).apply(mixed, onsets)
        q = metrics.speech_quality(clean, heard, SR)
        stoi = q.get("stoi")
    else:
        q = metrics.speech_quality(clean, mixed, SR)
        stoi = q.get("stoi")

    wins = segment.windows(heard, onsets)
    proba = model.predict_proba(features.mel(wins))
    preds = [CLASSES[i] for i in proba.argmax(1)] if len(proba) else []
    # Pillar 2: hand the per-key top-3 lattice to Gemini, which multiplies weak
    # acoustic guesses against language redundancy into readable text. With the
    # shield on, the lattice is garbage -> Gemini can't reconstruct either.
    lattice = candidate_lattice(proba, CLASSES, k=3)
    reconstructed_lm = gemini_reconstruct(lattice)
    per_key = [
        {"true": t, "pred": p, "correct": t == p, "top3": cand}
        for t, p, cand in zip(keys, preds, lattice)
    ]
    acc = float(np.mean([x["correct"] for x in per_key])) if per_key else 0.0
    lm_acc = (
        float(np.mean([a == b for a, b in zip(keys, reconstructed_lm)]))
        if reconstructed_lm and len(reconstructed_lm) == len(keys)
        else None
    )
    return {
        "typed": "".join(keys),
        "reconstructed": "".join(preds),
        "reconstructed_lm": reconstructed_lm,
        "per_key": per_key,
        "attack_acc": acc,
        "attack_acc_lm": lm_acc,
        "stoi": stoi,
        "protect": "keyguard" if protect else "none",
        "waveform": _downsample(heard),
    }


@app.post("/api/demo")
def api_demo(req: DemoReq):
    keys = [c for c in req.text.upper() if c in CLS_IDX][:MAX_DEMO_KEYS]
    if not keys:
        return JSONResponse({"error": "no attackable keys (A-Z, 0-9) in text"}, 400)

    live: dict | None = None
    try:
        live = _run_demo(keys, req.protect == "keyguard")
    except Exception as e:
        if not req.demo_mode:
            return JSONResponse({"error": f"demo failed: {e}"}, 500)
        # demo mode swallows the error and tries the cache below

    if not req.demo_mode:
        return live

    # Stage insurance: prefer a good live run (and cache it), else a cached good
    # run for this exact input, else best-effort live. See keyguard/demo_cache.py.
    try:
        return demo_cache.pick(live, "".join(keys), req.protect)
    except Exception as e:
        return JSONResponse({"error": f"demo mode: {e}"}, 500)


# --------------------------------------------------------------- static / index
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
