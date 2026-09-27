"""R4ft vs R5 inside CallGuard's pipeline: how fast does each arm the Secret Shield on a fake caller, how often does it
arm on a real one, and does it keep up with the 2 s hop?

Calls (fixed seed, Hearsay's held-out test_internal_testlike rows, read-only from HEARSAY_ROOT):
- 40 real callers: 20-30 s of one bona fide speaker's clips (0.4 s pauses between clips), round-robin over sources,
  LibriSpeech speakers 100 and 2803 excluded (demo voices / secret-shield tuning set).
- 40 fake callers: the same from one (generator, speaker) each, one per generator, modern generators first
  (DiffSSD commercial/zero-shot, MLAAD, SONAR, DFADD, ASVspoof5, ...).

Each call is the far end of `Pipeline.replay(realtime=False)` with the REAL HearsayDriver and mock attacker, shield and
spotters (the Secret Shield is enabled, so its voice arming runs). Fast replay would score every window instantly, so
the voice worker runs on the audio clock instead (`AudioClockWorker`): a verdict reaches the bus `latency_ms` (as
measured on this CPU) after its window was due, and the worker can't start the next window before then, so
pipeline.voice_step's catch-up rule skips windows exactly as it would live.

Then the ai_caller demo scenario in realtime with every pillar real (Keyguard attacker + DSP shield, Vosk spotters),
per mode: the level timeline, when the Secret Shield armed, and voice latency with everything else running. Plus a
back-to-back latency bench per mode, and the arming delay / false arms under other half-lives and arm thresholds,
recomputed from the recorded verdicts with the real ThreatEngine.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python app/hearsay/eval/r5_in_callguard.py
     (writes docs/reports/hearsay_r5_in_callguard.{md,csv}; ~30-40 min on the dev laptop CPU)
"""
from __future__ import annotations

import argparse
import ctypes
import os
import pickle
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.dont_write_bytecode = True  # read-only upstreams: no __pycache__ inside them
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("KEYGUARD_ROOT", str(REPO.parent / "keyboard-acoustic-shield"))
from app.hearsay.driver import HEARSAY_ROOT, HearsayDriver  # noqa: E402
from app.keystroke_guard.mock import MockAttacker, MockShield  # noqa: E402
from app.secret_shield.mock import MockSpotter  # noqa: E402
from app.source.bus import EventBus  # noqa: E402
from app.source.config import Config, ThreatConfig  # noqa: E402
from app.source.pipeline import VOICE_HOP, Pipeline, Scenario, load_scenario  # noqa: E402
from app.source.threat import LEVELS, ThreatEngine  # noqa: E402
from app.source.types import SR, Event  # noqa: E402

OUT = REPO / "docs" / "reports"
CACHE = REPO / "runs" / "cache" / "hearsay_r5_in_callguard.pkl"   # results, for --report-only
SEED = 0
N_CALLS = 40
GAP = np.zeros(int(0.4 * SR), np.float32)
EXCLUDE = {("librispeech", "100"), ("librispeech", "2803")}
FAKE_ORDER = ["diffssd", "mlaad_tiny", "sonar", "dfadd", "asvspoof5", "in_the_wild", "librisevoc", "cvoicefake_en",
              "asvspoof2019_la", "wavefake"]
MODES = ("r4ft", "r5")
THREADS = 4


def cpu_busy(seconds: float = 3.0) -> float:
    """System-wide CPU % over `seconds` (Windows GetSystemTimes; psutil isn't installed)."""
    def times():
        idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        v = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime  # noqa: E731
        return v(idle), v(kern) + v(user)   # kernel time includes idle
    i0, a0 = times()
    time.sleep(seconds)
    i1, a1 = times()
    return 100.0 * (1.0 - (i1 - i0) / max(1, a1 - a0))


def wait_quiet(limit: float = 25.0, tries: int = 10) -> float:
    """Wait (up to `tries` minutes) for other processes to leave the CPU; return the load we start at."""
    load = cpu_busy()
    for _ in range(tries):
        if load <= limit:
            break
        print(f"  CPU busy ({load:.0f} %), waiting a minute", flush=True)
        time.sleep(60)
        load = cpu_busy()
    return load


# --- calls ------------------------------------------------------------------------------------------------------------
def level(y: np.ndarray, target_db: float = -26.0) -> np.ndarray:
    """Meet's AGC stand-in: active-speech RMS (20 ms frames within 40 dB of the loudest) to `target_db` dBFS, peak
    <= 0.99. Dataset levels vary a lot, and the pipeline's VAD has a fixed -45 dBFS floor; prep() normalizes anyway,
    so the gain only decides which windows get scored, not their score."""
    f = y[:len(y) // 320 * 320].reshape(-1, 320).astype(np.float64)
    e = (f ** 2).mean(axis=1) + 1e-12
    rms = np.sqrt(e[e >= e.max() * 1e-4].mean())
    g = min(10 ** (target_db / 20) / rms, 0.99 / (np.abs(y).max() + 1e-9))
    return (y * g).astype(np.float32)


def build_calls() -> list[dict]:
    m = pd.read_parquet(HEARSAY_ROOT / "data" / "processed" / "manifest.parquet")
    t = m[m.test_internal_testlike].copy()
    t = t[t.path.map(lambda p: (HEARSAY_ROOT / p).exists())]
    t = t[[(s, str(k)) not in EXCLUDE for s, k in zip(t.source, t.speaker)]]
    rng = np.random.default_rng(SEED)

    def groups(df, keys):
        tot = df.groupby(keys).duration.sum()
        return tot[tot >= 20].reset_index()

    real = groups(t[t.label == "bonafide"], ["source", "speaker"])
    per_src = {s: list(g.sample(frac=1, random_state=SEED).itertuples()) for s, g in real.groupby("source")}
    picks: list = []
    while len(picks) < N_CALLS and any(per_src.values()):
        for s in sorted(per_src):
            if per_src[s] and len(picks) < N_CALLS:
                picks.append(("real", per_src[s].pop(0)))
    fake = groups(t[t.label == "spoof"], ["source", "generator", "speaker"])
    fake = fake.groupby("generator").sample(1, random_state=SEED)
    fake = fake.assign(o=fake.source.map(FAKE_ORDER.index)).sort_values(["o", "generator"]).head(N_CALLS)
    picks += [("fake", r) for r in fake.itertuples()]

    calls = []
    for i, (kind, g) in enumerate(picks):
        rows = t[(t.source == g.source) & (t.speaker == g.speaker)]
        if kind == "fake":
            rows = rows[rows.generator == g.generator]
        target = int(min(rng.uniform(20, 30), g.duration) * SR)
        parts, n = [], 0
        for p in rows.path.sample(frac=1, random_state=SEED + i):
            y, sr = sf.read(str(HEARSAY_ROOT / p), dtype="float32")
            assert sr == SR, (p, sr)
            y = level(y.mean(axis=1) if y.ndim > 1 else y)
            parts += [y, GAP]
            n += len(y) + len(GAP)
            if n >= target:
                break
        x = np.concatenate(parts)[:target]
        calls.append(dict(call=f"{kind}{sum(c['kind'] == kind for c in calls):02d}", kind=kind, source=g.source,
                          generator=getattr(g, "generator", "bonafide"), speaker=str(g.speaker),
                          dur=round(len(x) / SR, 1), x=x))
    return calls


# --- pipeline runs ----------------------------------------------------------------------------------------------------
class _Capture:
    def __init__(self):
        self.events: list[Event] = []

    def publish(self, ev: Event):
        self.events.append(ev)

    def emit(self, topic, **data):
        self.events.append(Event(topic, data))


class AudioClockWorker:
    """Replaces pipe.voice_step in fast replay: the verdict of a window is published `latency_ms` of *audio* after
    the worker started it, and the next window can't start before then (one worker thread, as deployed)."""

    def __init__(self, pipe: Pipeline):
        self.pipe, self.step0 = pipe, pipe.voice_step
        self.pending: list[Event] = []
        self.ready = 0

    def __call__(self) -> bool:
        p = self.pipe
        now = p.far.total
        if self.pending and now < self.ready:
            return False
        for ev in self.pending:
            p.bus.publish(ev)
        self.pending = []
        cap, bus = _Capture(), p.bus
        p.bus = cap
        try:
            self.step0()
        finally:
            p.bus = bus
        lat = max([e.data.get("latency_ms", 0.0) for e in cap.events] + [0.0])
        self.pending, self.ready = cap.events, now + round(lat / 1000 * SR)
        return False


def instrument(pipe: Pipeline, bus: EventBus) -> dict:
    """Record V / level ticks, voice windows (with the audio time their verdict landed) and the first voice arming."""
    rec = {"ticks": [], "windows": [], "arm": None, "arm_any": None, "levels": []}
    rec["unsubs"] = [
        bus.subscribe("threat.update", lambda e: rec["ticks"].append((pipe.clock(), e.data["V"], e.data["level"],
                                                                      e.data["score"]))),
        bus.subscribe("voice.window", lambda e: rec["windows"].append(dict(e.data, t_ready=pipe.clock()))),
        bus.subscribe("threat.level_change", lambda e: rec["levels"].append((round(pipe.clock(), 2), e.data["to"],
                                                                             e.data["score"], e.data["reasons"])))]
    step0 = pipe.secret_step

    def secret_step():
        r = step0()
        if pipe.armed and rec["arm_any"] is None:
            rec["arm_any"] = (pipe.clock(), pipe.armed_by)
        if pipe.armed and pipe.armed_by == "voice" and rec["arm"] is None:
            rec["arm"] = pipe.clock()
        return r
    pipe.secret_step = secret_step
    return rec


def skipped(windows: list[dict]) -> int:
    t = [w["t_audio"] for w in windows]
    return int(sum(max(0, round((b - a) / (VOICE_HOP / SR)) - 1) for a, b in zip(t, t[1:])))


def run_calls(voice, calls: list[dict], mode: str) -> tuple[list[dict], dict]:
    bus = EventBus(maxlen=100_000)
    pipe = Pipeline(Config(), bus, voice=voice, attacker=MockAttacker(), shield=MockShield(),
                    spotter=MockSpotter(), spotter_in=MockSpotter(mode="inbound"))
    rows, verdicts = [], {}
    for k, c in enumerate(calls):
        rec = instrument(pipe, bus)
        pipe.voice_step = AudioClockWorker(pipe)
        pipe.replay(Scenario(c["call"], c["x"], np.zeros_like(c["x"]), []), realtime=False)
        bus.flush()
        for u in rec["unsubs"]:
            u()
        del pipe.secret_step, pipe.voice_step           # back to the class methods
        sc = [w for w in rec["windows"] if w["scored"]]
        V = np.array([v for _, v, _, _ in rec["ticks"]])
        lat = np.array([w["latency_ms"] for w in sc]) if sc else np.array([np.nan])
        p = np.array([w["p_synthetic"] for w in sc]) if sc else np.array([np.nan])
        flag = [w["t_ready"] for w in sc if w["p_synthetic"] > 0.5]
        verdicts[c["call"]] = [(w["t_ready"], w["p_synthetic"]) for w in sc]
        rows.append(dict({k2: c[k2] for k2 in ("call", "kind", "source", "generator", "speaker", "dur")}, mode=mode,
                         windows=len(rec["windows"]), scored=len(sc), skipped=skipped(rec["windows"]),
                         p_median=round(float(np.median(p)), 3), frac_flagged=round(float(np.mean(p > 0.5)), 3),
                         t_first_flag=round(flag[0], 2) if flag else np.nan,
                         armed=rec["arm"] is not None, t_arm=round(rec["arm"], 2) if rec["arm"] else np.nan,
                         V_max=round(float(V.max()), 3), V_median=round(float(np.median(V)), 3),
                         level_max=max((lv for _, _, lv, _ in rec["ticks"]), key=LEVELS.index),
                         lat_median=round(float(np.median(lat)), 1), lat_p95=round(float(np.percentile(lat, 95)), 1)))
        r = rows[-1]
        print(f"  {mode} {k + 1:2d}/{len(calls)} {c['call']} {c['generator'][:28]:28s} p_med={r['p_median']:.2f} "
              f"Vmax={r['V_max']:.2f} arm={r['t_arm']} lat={r['lat_median']:.0f} ms skip={r['skipped']}", flush=True)
    bus.close()
    return rows, verdicts


def run_demo(voice, mode: str) -> dict:
    """ai_caller in realtime, every pillar real (Keyguard attacker + DSP shield, Vosk spotters) next to Hearsay."""
    cfg = Config()
    cfg.drivers.attacker = cfg.drivers.shield = cfg.drivers.secret = "real"
    attacker, attacker_error = None, None
    try:
        from app.source.registry import make_attacker
        attacker = make_attacker(cfg)
    except Exception as e:  # noqa: BLE001 - e.g. weights vs Keyguard's class list: report it, use the mock
        attacker, attacker_error = MockAttacker(), f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    bus = EventBus(maxlen=100_000)
    pipe = Pipeline(cfg, bus, voice=voice, attacker=attacker)
    rec = instrument(pipe, bus)
    load = cpu_busy()
    t0 = time.perf_counter()
    pipe.replay(load_scenario("ai_caller"), realtime=True)
    bus.flush()
    bus.close()
    sc = [w for w in rec["windows"] if w["scored"]]
    lat = np.array([w["latency_ms"] for w in sc])
    return dict(mode=mode, load_before=load, wall_s=time.perf_counter() - t0, levels=rec["levels"], arm=rec["arm"],
                arm_any=rec["arm_any"], attacker_error=attacker_error,
                drivers=(pipe.attacker.name, pipe.shield.name, getattr(pipe.spotter, "name", None), pipe.secret_error),
                windows=len(rec["windows"]), scored=len(sc), skipped=skipped(rec["windows"]),
                lat_median=float(np.median(lat)), lat_p95=float(np.percentile(lat, 95)), lat_max=float(lat.max()),
                verdicts=[(round(w["t_audio"], 1), round(w["p_synthetic"], 2)) for w in sc])


def bench(voice, x: np.ndarray, wait_min: int = 0, n: int = 20) -> dict:
    voice.score(x[:4 * SR])
    load = wait_quiet(tries=wait_min)
    ms = []
    for i in range(n):
        w = x[(i * SR) % (len(x) - 4 * SR):][:4 * SR]
        t0 = time.perf_counter()
        voice.score(w)
        ms.append((time.perf_counter() - t0) * 1000)
    return dict(load_before=load, load_after=cpu_busy(2), median=float(np.median(ms)),
                p95=float(np.percentile(ms, 95)), max=float(np.max(ms)))


# --- threshold sensitivity (the real ThreatEngine on the recorded verdicts) -------------------------------------------
class _NullBus:
    def subscribe(self, *a):
        return lambda: None

    def emit(self, *a, **k):
        pass


def first_arm(verdicts, dur: float, half_life: float, arm: float) -> float | None:
    now = [0.0]
    eng = ThreatEngine(_NullBus(), replace(ThreatConfig(), voice_half_life_windows=half_life / 2), now=lambda: now[0])
    vs = sorted(verdicts)
    j = 0
    for k in range(int(dur / 0.25) + 1):
        now[0] = k * 0.25
        while j < len(vs) and vs[j][0] <= now[0]:
            eng.on_event(Event("voice.verdict", {"p_synthetic": vs[j][1]}))
            j += 1
        eng.tick()
        if eng.V >= arm:
            return now[0]
    return None


def sensitivity(df: pd.DataFrame, verdicts: dict) -> pd.DataFrame:
    out = []
    for mode in MODES:
        d = df[df["mode"] == mode]
        for hl in (3.0, 4.0, 6.0):
            for arm in (0.4, 0.5, 0.6):
                t = {r.call: first_arm(verdicts[mode][r.call], r.dur, hl, arm) for r in d.itertuples()}
                fk = [t[c] for c in d[d.kind == "fake"].call]
                rl = [t[c] for c in d[d.kind == "real"].call]
                armed = [x for x in fk if x is not None]
                out.append(dict(mode=mode, half_life_s=hl, arm_voice=arm, fakes_armed=len(armed), n_fake=len(fk),
                                t_arm_median=np.median(armed) if armed else np.nan,
                                t_arm_p90=np.percentile(armed, 90) if armed else np.nan,
                                real_false_arms=sum(x is not None for x in rl), n_real=len(rl)))
    return pd.DataFrame(out)


# --- report -----------------------------------------------------------------------------------------------------------
def summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for mode in MODES:
        d = df[df["mode"] == mode]
        f, r = d[d.kind == "fake"], d[d.kind == "real"]
        out.append({"mode": mode,
                    "fakes armed": f"{f.armed.sum()}/{len(f)}",
                    "arm time median (s)": f"{f.t_arm.median():.1f}",
                    "arm time p90 (s)": f"{f.t_arm.quantile(0.9):.1f}",
                    "first flagged window median (s)": f"{f.t_first_flag.median():.1f}",
                    "real false arms": f"{r.armed.sum()}/{len(r)}",
                    "real V max (median / max)": f"{r.V_max.median():.2f} / {r.V_max.max():.2f}",
                    "fake V median (median)": f"{f.V_median.median():.2f}",
                    "windows flagged, fake / real": f"{f.frac_flagged.mean():.0%} / {r.frac_flagged.mean():.0%}",
                    "WARN reached, fake / real": f"{(f.level_max.isin(['WARN', 'CRITICAL'])).sum()} / "
                                                 f"{(r.level_max.isin(['WARN', 'CRITICAL'])).sum()}",
                    "latency median / p95 (ms)": f"{d.lat_median.median():.0f} / {d.lat_p95.quantile(0.95):.0f}",
                    "windows skipped": f"{d.skipped.sum()} of {d.windows.sum() + d.skipped.sum()}"})
    return pd.DataFrame(out).set_index("mode").T


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table (tabulate isn't installed, so no DataFrame.to_markdown)."""
    d = df.reset_index() if index else df
    cell = lambda v: "" if isinstance(v, float) and np.isnan(v) else str(v)  # noqa: E731
    rows = [list(map(str, d.columns)), ["---"] * len(d.columns)] + [[cell(v) for v in r] for r in d.values]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def write_report(df, sens, demos, benches, elapsed, load0):
    s = summary(df)
    r4, r5 = (df[df["mode"] == m] for m in MODES)
    fails = df[((df.kind == "fake") & ~df.armed) | ((df.kind == "real") & df.armed)]
    miss = fails[["mode", "call", "kind", "source", "generator", "speaker", "p_median", "frac_flagged", "V_max"]]
    per_src = (df.groupby(["mode", "kind", "source"])
               .agg(calls=("call", "size"), armed=("armed", "sum"), t_arm=("t_arm", "median"),
                    p_median=("p_median", "median")).round(2).reset_index())
    lines = [
        "# Hearsay R5 vs R4ft inside CallGuard",
        "",
        f"`app/hearsay/eval/r5_in_callguard.py`, {time.strftime('%Y-%m-%d')}, dev laptop CPU "
        f"({os.cpu_count()} logical cores), Hearsay driver on {THREADS} torch threads, {elapsed / 60:.0f} min. "
        f"CPU load from other processes when the run started: {load0:.0f} %.",
        "",
        "## Question",
        "",
        "CallGuard switches its voice pillar from R4ft (the XLS-R fine-tune alone) to R5 (Hearsay's submitted fusion: "
        "R4ft + the R1 LightGBM on classic features, frozen weights). The Secret Shield arms when the smoothed voice "
        "risk V >= `secret.arm_voice` (0.5; stays armed while V >= `keep_voice` 0.3; V = EMA of p_synthetic, "
        "half-life 6 s). So the model decides when your spoken codes get bleeped. Does R5 arm sooner on fake callers, "
        "arm less on real ones, and keep up with the 2 s hop?",
        "",
        "## Method",
        "",
        f"- {len(r5[r5.kind == 'real'])} real and {len(r5[r5.kind == 'fake'])} fake simulated callers from Hearsay's "
        "held-out `test_internal_testlike` rows (never used for training or for the threshold, which comes from "
        "`val_testlike`). A call is 20-30 s of one speaker's clips with 0.4 s pauses: a real call is one bona fide "
        "speaker, round-robin over the 11 bona fide sources; each clip is levelled to -26 dBFS active-speech RMS (a "
        "stand-in for Meet's AGC: dataset levels vary and the pipeline's VAD has a fixed -45 dBFS floor; Hearsay's "
        "prep() normalizes, so this only decides which windows get scored); a fake call is one (generator, speaker), one per "
        "generator, modern ones first (DiffSSD's ElevenLabs, PlayHT, OpenVoice v2, XTTS v2, YourTTS...; MLAAD's "
        "FishTTS, Llasa, MegaTTS3, Dia, OuteTTS; SONAR's OpenAI, VoiceBox, xTTS; DFADD's NaturalSpeech 2, StyleTTS 2; "
        "ASVspoof5). LibriSpeech speakers 100 and 2803 excluded. Seed 0.",
        "- Each call is the far end of `Pipeline.replay(realtime=False)` with the real `HearsayDriver` and mock "
        "attacker/shield/spotters (Secret Shield enabled, default config). The voice worker runs on the audio clock: "
        "a verdict lands `latency_ms` (measured live on this CPU) after its window was due, and the worker can't "
        "start the next window before then, so `voice_step`'s catch-up (jump to the newest window when > 2 s behind) "
        "skips windows exactly as it would live. Arm time = audio seconds from the caller's first sample until "
        "`armed_by == \"voice\"`. The first window is due at 4 s.",
        "- Without typing, the score is 100 * 0.7 * V, so a voice alone tops out at WARN (70); CRITICAL needs "
        "typing or a blocked secret. Levels below are for voice alone.",
        "",
        "## Results: simulated calls",
        "",
        md(s),
        "",
        "Arm time and V are per call; latency is the median of the per-call medians and the 95th percentile of the "
        "per-call p95s. The windows count includes silence-skipped windows (VAD < 0.5).",
        "",
        "### By source",
        "",
        md(per_src, index=False),
        "",
        "### Calls the arming got wrong (fake never armed, or real armed)",
        "",
        md(miss, index=False) if len(miss) else "None.",
        "",
        "## Latency per 4 s window (back to back, no other CallGuard pillar)",
        "",
        md(pd.DataFrame(benches).round(0), index=False),
        "",
        "`load_before` = system CPU % from other processes just before the bench (a Keyguard `adversarial_eval` job "
        "with 8 worker processes was running during parts of this experiment); `load_after` includes the bench.",
        "",
        "## ai_caller demo scenario, realtime, every pillar real",
        "",
        "`demo/scenarios/ai_caller`: colleague (real, 0-12 s), cloned-voice AI agent (12-40 s) asking for the reset "
        "code, typed at 20.5 s (shield off) and 29.5 s (shield on from 29 s), colleague again (40-60 s). Realtime "
        "replay with the real Keyguard attacker + DSP shield and the Vosk spotters on their worker threads, so the "
        "voice latency here is with the other pillars running.",
        "",
    ]
    for d in demos:
        lines += [f"### {d['mode']}", "",
                  f"- drivers: attacker `{d['drivers'][0]}`, shield `{d['drivers'][1]}`, spotter `{d['drivers'][2]}`"
                  + (f" (secret shield error: {d['drivers'][3]})" if d["drivers"][3] else "")
                  + (f"; the real attacker failed to load, so the mock stood in: `{d['attacker_error']}`"
                     if d.get("attacker_error") else ""),
                  f"- Secret Shield armed by voice at **{d['arm']:.1f} s**" if d["arm"] is not None else
                  "- Secret Shield never armed by voice",
                  f"- first arming of any kind: {d['arm_any'][1]} at {d['arm_any'][0]:.1f} s" if d["arm_any"] else "",
                  f"- voice windows: {d['scored']} scored of {d['windows']}, {d['skipped']} skipped by catch-up; "
                  f"latency median {d['lat_median']:.0f} ms, p95 {d['lat_p95']:.0f} ms, max {d['lat_max']:.0f} ms "
                  f"(CPU load before: {d['load_before']:.0f} %)",
                  f"- verdicts (t_audio s, p): {', '.join(f'{t:g}:{p:.2f}' for t, p in d['verdicts'])}",
                  "", "| t (s) | level | score | reasons |", "|---|---|---|---|"]
        lines += [f"| {t:.1f} | {lv} | {sc:.0f} | {'; '.join(r)} |" for t, lv, sc, r in d["levels"]]
        lines.append("")
    lines += [
        "## Threshold sensitivity",
        "",
        "First arming recomputed with the real `ThreatEngine` on each call's recorded verdicts (the time each "
        "verdict landed), for other half-lives and `arm_voice`. `keep_voice` only matters once armed, so it doesn't "
        "change these. Row half_life 6 / arm 0.5 is the shipped config and should match the pipeline run above "
        "(to the 0.25 s tick).",
        "",
        md(sens.round(2), index=False),
        "",
        "## Recommendation",
        "",
        RECOMMENDATION.strip(),
        "",
        "## Caveats",
        "",
        "- Clean 16 kHz dataset audio, not a Meet call: no Opus codec, echo cancellation or noise suppression, which "
        "Hearsay never saw. The meeting-path numbers still need a real call.",
        "- One speaker per call and no silence at the start; a real call has hellos and gaps (V decays toward 0 in "
        "silence after `voice_stale_s` 4 s).",
        "- 3 of the 7 ASVspoof2019 LA bona fide calls had no window pass the pipeline's VAD (energy + zero-crossing "
        "gate, even after levelling), so they were never scored: their 'no false arm' says nothing about the model.",
        "- 40 + 40 calls, one seed; the 10/40 false arms carry a wide interval (95 % CI roughly 13-41 %).",
        "- Latency depends on what else runs: a Keyguard `adversarial_eval` job (8 processes) ran throughout, and a "
        "test suite overlapped the r5 call runs and the r5 demo. No quiet re-measurement was possible in the time "
        "box; the quiet numbers are from the smoke run (see the recommendation).",
        "- The real Keyguard attacker failed to load in the demo (provisional KeyNet weights have 36 classes, "
        "Keyguard's CLASSES now has 37 with space), so the mock attacker stood in there.",
        "",
    ]
    (OUT / "hearsay_r5_in_callguard.md").write_text("\n".join(lines), encoding="utf-8")


RECOMMENDATION = """
**Ship R5 as the default (done: `drivers.hearsay_mode = "r5"`, `r4ft` still selectable); leave the arming
thresholds alone.** Inside the pipeline the two modes behave almost the same:

- Arming: both arm on 37/40 fake callers (misses: MLAAD MegaTTS3 and Nari Dia, CVoiceFake Griffin-Lim, the same 3
  for both). R5 arms about 1 s later (median 13.2 s vs 12.0 s from the caller's first word, p90 17.4 vs 15.3 s),
  because the fusion is a little less extreme on fakes (fake V median 0.42 vs 0.47).
- False arms: both arm on the same 10/40 real callers (LibriSeVoc x4, ASVspoof5 x2, In-the-Wild x2, DFADD, MLAAD
  bona fide): R5 doesn't fix them. 4 of them are confident errors (p near 1 on most windows); the rest are single
  high windows that push V just past 0.5. LibriSpeech, LJSpeech, SONAR and ASVspoof2019 bona fide never arm. A false
  arm costs little here: the shield then listens and bleeps codes you read out, it doesn't cut the call.
- Latency: R5 costs about +90 ms per 4 s window on a quiet CPU (smoke run at 6-8 % load: r4ft 392-399 ms, r5 483 ms
  median; the benches above were taken with the Keyguard adversarial_eval job at 19-24 % load). It is fast enough for
  the 2 s hop: no window was ever skipped by the catch-up in any run, including the realtime ai_caller replay next to
  the Keyguard DSP shield and both Vosk spotters with the CPU at 100 % (another test suite was running on top of the
  Keyguard job): r5 median 1.66 s, max 1.81 s per window there. The r5 call-run latencies (median ~1.7 s) were
  measured under the same load, so they are an upper bound; the arming times include that latency.
- Thresholds: the data doesn't clearly support a change. A 3-4 s half-life arms 3-4 s sooner but adds 3-5 false
  arms out of 40; `arm_voice` 0.6 halves false arms (10 -> 5) but loses a fake and adds ~3 s. The honest fix for
  false arms is the model (R6) or calibration on meeting audio, not the EMA. Kept 0.5 / 0.3 / 6 s.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=N_CALLS)
    ap.add_argument("--only", default="calls,bench,demo", help="parts to (re)run, the rest comes from the cache; "
                    "'' = just rewrite the report. bench waits up to --wait min for a quiet CPU")
    ap.add_argument("--wait", type=int, default=0)
    args = ap.parse_args()
    parts = {p for p in args.only.split(",") if p}
    res = pickle.loads(CACHE.read_bytes()) if CACHE.exists() and parts != {"calls", "bench", "demo"} else {
        "benches": [], "demos": {}}
    t_start = time.perf_counter()
    if parts:
        load0 = cpu_busy()
        print(f"CPU load before start: {load0:.0f} %", flush=True)
        calls = build_calls()
        calls = ([c for c in calls if c["kind"] == "real"][:args.calls]
                 + [c for c in calls if c["kind"] == "fake"][:args.calls])
        print(f"{len(calls)} calls, {sum(c['dur'] for c in calls) / 60:.0f} min of audio", flush=True)
        rows, verdicts = [], {}
        for mode in MODES:
            voice = HearsayDriver(mode=mode, threads=THREADS, device="cpu")
            if "bench" in parts:
                real = np.concatenate([c["x"] for c in calls if c["kind"] == "real"][:3])
                res["benches"].append(dict(mode=mode, when=time.strftime("%H:%M"), **bench(voice, real, args.wait)))
                print(f"{mode} bench: {res['benches'][-1]}", flush=True)
            if "calls" in parts:
                r, verdicts[mode] = run_calls(voice, calls, mode)
                rows += r
            if "demo" in parts:
                d = res["demos"][mode] = run_demo(voice, mode)
                print(f"{mode} demo: arm={d['arm']} levels={[(t, lv) for t, lv, _, _ in d['levels']]}", flush=True)
            del voice
        if "calls" in parts:
            res.update(df=pd.DataFrame(rows), verdicts=verdicts, elapsed=time.perf_counter() - t_start, load0=load0)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(res))
    df, verdicts = res["df"], res["verdicts"]
    df.to_csv(OUT / "hearsay_r5_in_callguard.csv", index=False)
    sens = sensitivity(df, verdicts)
    print(summary(df).to_string())
    print(sens.round(2).to_string(index=False))
    write_report(df, sens, [res["demos"][m] for m in MODES if m in res["demos"]], res["benches"], res["elapsed"],
                 res["load0"])


if __name__ == "__main__":
    main()
