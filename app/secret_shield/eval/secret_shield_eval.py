"""Plan 06 §5: does the spoken-secret shield stop a code being read out, without muting normal speech?

Streams audio exactly as deployed: 20 ms blocks into VoskSpotter, spans applied to a 500 ms delay line. A span
returned after the block ending at sample T can only redact samples >= T + worker_lag - 500 ms; anything earlier has
already left (leaked). This is the Redactor's `leaked_samples` rule (app/secret_shield/redactor.py).

Data (fixed seeds, CPU):
- Sensitive: 40 TTS renders (facebook/mms-tts-eng) of fake codes: "the code is" + 4-8 digits, "my pin is" + 4,
  card numbers (4-4-4-4, read with pauses), "my password is" + words + digits, and runs with no trigger phrase
  ("okay it's ..."), each clean and with Keyguard-shielded keystrokes mixed in (harrison test presses, +5 dB
  speech over key-window power, ~2.5 keys/s, run through the real KeyguardShield driver with oracle key events).
  NOT our voices: plan 06 wants consenting teammates' recordings, which don't exist yet (owner to-do).
- Innocent: 20 TTS sentences with casual numbers ("see you at two", a few deliberately hard: "my kids are seven
  and nine") + 40 LibriSpeech bona fide clips from Hearsay's test_internal split, speakers 100 and 2803 excluded
  (they're the demo voices, and the only clips used for tuning, see --tune).
- Inbound: the request-trigger spotter on the same innocent audio (false triggers/min) + 8 TTS requests.

Ground truth word times: Vosk forced to the known sentence (grammar = that one sentence) on the clean render. Same
model family as the spotter, so alignment errors correlate with it; 30 ms frame grid. A secret word counts as
leaked if > 20 % of its samples left the delay line unredacted (~60 ms of a digit: enough to hear its onset);
"audible" = > 50 %.

Run: .venv/Scripts/python app/secret_shield/eval/secret_shield_eval.py  [--tune]   (writes docs/reports/secret_shield.{md,csv})
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.dont_write_bytecode = True  # read-only upstreams: no __pycache__ inside them
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from app.secret_shield.spotter import MODEL_DIR, VoskSpotter  # noqa: E402
from app.source.types import BLOCK, SR  # noqa: E402

HEARSAY_ROOT = Path(os.environ.get("HEARSAY_ROOT", REPO.parent / "Hearsay"))
CACHE = REPO / "runs" / "cache" / "secret_shield"
OUT = REPO / "docs" / "reports"
SEED = 0
DELAY = SR // 2
LAGS = (0.0, 0.1)            # worker lag: the pipeline's secret_step runs on a worker, not per block
LEAK_FRAC, AUDIBLE_FRAC = 0.2, 0.5
WORDS = "zero one two three four five six seven eight nine".split()
PW_WORDS = "blue falcon maple river silver tiger purple ocean happy garden".split()
INNOCENT = [
    "see you at two", "it takes about five minutes", "we are in room three", "we have two options",
    "call me back at four", "there were three of us and one dog", "one or two things to check first",
    "the meeting moved to five thirty", "i will be there in ten minutes", "give me one second",
    "she has two cats and a dog", "take the second left after the bridge", "it was built in the old town",
    "we should leave by nine tomorrow", "the train is at seven fifteen", "i need four eggs and some milk",
    # deliberately hard: two digit words close together
    "my kids are seven and nine", "it costs four ninety nine", "gate six or seven, not sure", "chapter one two",
]
REQUESTS = [
    "just read me the verification code please", "what is the code we sent you", "can you tell me your pin number",
    "please confirm your card number", "read me the one time code", "what is your password",
    "i need your social security number", "tell me the code on your screen",
]


# --- audio --------------------------------------------------------------------------------------------------------
class TTS:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", local_files_only=True)
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng", local_files_only=True).eval()
        assert self.model.config.sampling_rate == SR

    def __call__(self, text: str, seed: int) -> np.ndarray:
        f = CACHE / f"tts_{hashlib.sha1(f'{text}|{seed}'.encode()).hexdigest()[:16]}.npy"
        if f.exists():
            return np.load(f)
        self.torch.manual_seed(seed)
        with self.torch.no_grad():
            x = self.model(**self.tok(text, return_tensors="pt")).waveform[0].numpy()
        x = np.concatenate([np.zeros(SR // 4), x, np.zeros(SR)]).astype(np.float32)
        np.save(f, x)
        return x


def align(x: np.ndarray, text: str) -> list[tuple[str, int, int]] | None:
    """Word times: Vosk constrained to the known sentence. None if it can't place the sentence."""
    from vosk import KaldiRecognizer, Model, SetLogLevel
    global _ALIGN_MODEL
    SetLogLevel(-2)
    if "_ALIGN_MODEL" not in globals():
        _ALIGN_MODEL = Model(str(MODEL_DIR))
    words = " ".join(text.replace(",", " ").split())
    rec = KaldiRecognizer(_ALIGN_MODEL, SR, json.dumps([words]))
    rec.SetWords(True)
    rec.AcceptWaveform((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    res = json.loads(rec.FinalResult()).get("result", [])
    if " ".join(w["word"] for w in res) != words:
        return None
    return [(w["word"], round(w["start"] * SR), round(w["end"] * SR)) for w in res]


def power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2)) + 1e-12


def add_keys(x: np.ndarray, presses: np.ndarray, rng) -> np.ndarray:
    """Mix harrison test presses (Poisson, ~2.5/s) at +5 dB speech over key-window power, then run the real
    Keyguard DSP shield (the mic chain: mic -> Keyguard shield -> secret shield) with oracle key events."""
    from app.keystroke_guard.driver import KeyguardShield
    from keyguard.config import PRE_S
    global _SHIELD
    if "_SHIELD" not in globals():
        _SHIELD = KeyguardShield()
    sp = power(x[np.abs(x) > 0.02 * np.abs(x).max()])
    y, onsets, t = x.copy(), [], int(rng.exponential(SR / 2.5))
    while t + presses.shape[1] < len(x):
        k = presses[rng.integers(len(presses))]
        y[t:t + len(k)] += k * np.sqrt(sp / 10 ** 0.5 / power(k))
        onsets.append(t + int(PRE_S * SR))
        t += presses.shape[1] + int(rng.exponential(SR / 2.5))
    _SHIELD.reset()
    lat = _SHIELD.latency
    y = np.concatenate([y, np.zeros(lat + BLOCK, np.float32)])
    out = []
    for i in range(0, len(y) - BLOCK + 1, BLOCK):
        ev = [o for o in onsets if i - BLOCK <= o < i]      # OS key events arrive about a block late
        out.append(_SHIELD.process(y[i:i + BLOCK], ev))
    return np.concatenate(out)[lat:lat + len(x)].astype(np.float32)


# --- streaming ----------------------------------------------------------------------------------------------------
def stream(spotter: VoskSpotter, x: np.ndarray):
    """Feed 20 ms blocks; returns (spans [(start, end, category, length, T)], per-block ms)."""
    spotter.reset()
    spans, ms = [], []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        t0 = time.perf_counter()
        got = spotter.feed(x[i:i + BLOCK], i)
        ms.append((time.perf_counter() - t0) * 1000)
        spans += [(s.start, s.end, s.category, s.length, i + BLOCK) for s in got]
    return spans, np.array(ms)


def redacted(spans, n: int, lag: float) -> np.ndarray:
    mask = np.zeros(n, bool)
    for a, b, _, _, T in spans:
        a = max(a, T + round(lag * SR) - DELAY, 0)
        if b > a:
            mask[a:min(b, n)] = True
    return mask


def secret_words(ali, n_secret: int):
    return ali[len(ali) - n_secret:]


# --- data ---------------------------------------------------------------------------------------------------------
def sensitive_texts(rng, n: int = 40):
    """(kind, text, n_secret_words, first_is_secret_after_trigger)."""
    def digits(k):
        return [WORDS[i] for i in rng.integers(0, 10, k)]
    out = []
    for i in range(n):
        kind = ("code", "pin", "card", "password", "no-trigger")[i % 5]
        if kind == "code":
            d = digits(rng.integers(4, 9)); out.append((kind, "the code is " + " ".join(d), len(d)))
        elif kind == "pin":
            d = digits(4); out.append((kind, "my pin is " + " ".join(d), 4))
        elif kind == "card":
            g = [" ".join(digits(4)) for _ in range(4)]; out.append((kind, "my card number is " + ", ".join(g), 16))
        elif kind == "password":
            w = list(rng.choice(PW_WORDS, 2, replace=False)) + digits(2)
            out.append((kind, "my password is " + " ".join(w), 4))
        else:
            d = digits(rng.integers(4, 9)); out.append((kind, rng.choice(["okay it's ", "sure, ", "yes "]) + " ".join(d), len(d)))
    return out


def librispeech(n: int, tune: bool):
    m = pd.read_parquet(HEARSAY_ROOT / "data" / "processed" / "manifest.parquet")
    m = m[(m.label == "bonafide") & (m.source == "librispeech") & (m.split == "test_internal") & (m.duration >= 4)]
    demo = m.speaker.astype(str).isin(["100", "2803"])
    m = m[demo] if tune else m[~demo].sample(n=n, random_state=SEED)
    out = []
    for r in m.itertuples():
        x, sr = sf.read(HEARSAY_ROOT / r.path, dtype="float32")
        assert sr == SR, (r.path, sr)
        out.append((f"ls:{r.speaker}", x if x.ndim == 1 else x.mean(1)))
    return out


# --- evaluation ---------------------------------------------------------------------------------------------------
def eval_sensitive(sp, items, rows, tag):
    for sid, kind, cond, x, ali, n_secret in items:
        spans, ms = stream(sp, x)
        sec = secret_words(ali, n_secret)
        first = {}
        for w, a, b in sec:          # detection lag: first span overlapping the word, relative to its start
            ts = [T for s0, s1, _, _, T in spans if s0 < b and s1 > a]
            first[(a, b)] = (min(ts) - a) / SR if ts else np.nan
        for lag in LAGS:
            mask = redacted(spans, len(x), lag)
            esc = [1 - mask[a:b].mean() for _, a, b in sec]
            rows.append(dict(variant=tag, set="sensitive", id=sid, kind=kind, cond=cond, lag_ms=int(lag * 1000),
                             n_secret=n_secret, leaked=sum(e > LEAK_FRAC for e in esc),
                             audible=sum(e > AUDIBLE_FRAC for e in esc), first_leaked=float(esc[0] > LEAK_FRAC),
                             det_lag_med=float(np.nanmedian(list(first.values()))),
                             det_lag_max=float(np.nanmax(list(first.values()))), dur=len(x) / SR,
                             redacted_s=mask.sum() / SR, blocks=len(ms), ms_mean=ms.mean(), ms_p99=np.percentile(ms, 99),
                             ms_max=ms.max()))


def eval_innocent(sp, items, rows, tag, direction):
    for sid, cond, x in items:
        spans, ms = stream(sp, x)
        mask = redacted(spans, len(x), 0.0)
        trig = len({s0 // SR for s0, _, _, n, _ in spans if n == 0})   # trigger-phrase spans carry length 0
        rows.append(dict(variant=tag, set=f"innocent-{direction}", id=sid, kind=sid.split(":")[0], cond=cond, lag_ms=0,
                         dur=len(x) / SR, redacted_s=mask.sum() / SR if direction == "out" else 0.0,
                         triggers=trig if direction == "out" else len({s0 for s0, *_ in spans}),
                         runs=len({s0 // SR for s0, _, c, n, _ in spans if c == "digits" and n >= 2}),
                         blocks=len(ms), ms_mean=ms.mean(), ms_p99=np.percentile(ms, 99), ms_max=ms.max()))


def build(tune: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 if tune else SEED)
    tts = TTS()
    sens = []
    presses = None
    if not tune:
        from app.keystroke_guard.driver import harrison_split
        presses = harrison_split()[2]
    dropped = 0
    for i, (kind, text, n_secret) in enumerate(sensitive_texts(rng, 12 if tune else 40)):
        x = tts(text, seed=i)
        ali = align(x, text)
        if ali is None:
            dropped += 1
            continue
        sens.append((f"s{i}", kind, "clean", x, ali, n_secret))
        if presses is not None:
            sens.append((f"s{i}", kind, "keys", add_keys(x, presses, rng), ali, n_secret))
    inn = [(f"tts:{i}", "clean", tts(t, seed=100 + i)) for i, t in enumerate(INNOCENT)]
    if presses is not None:
        inn += [(f"tts:{i}", "keys", add_keys(x, presses, rng)) for i, (_, _, x) in enumerate(list(inn))]
    inn += [(sid, "clean", x) for sid, x in librispeech(40, tune)]
    req = [(f"req:{i}", "clean", tts(t, seed=200 + i)) for i, t in enumerate(REQUESTS)]
    return sens, inn, req, dropped


def cpu_with_hearsay(sens):
    """Per-block ms of the outbound spotter while Hearsay scores 4 s windows back to back on another thread
    (worst case: deployed, it scores every 2 s)."""
    try:
        from app.hearsay.driver import HearsayDriver
        voice = HearsayDriver(mode="r4ft", threads=4)
    except Exception as e:  # noqa: BLE001 - report, don't fail the eval
        return None, f"Hearsay not loadable here ({type(e).__name__}: {e})"
    stop = threading.Event()
    clip = np.concatenate([x for _, _, c, x, _, _ in sens if c == "clean"])[:4 * SR]
    n = [0]

    def loop():
        while not stop.is_set():
            voice.score(clip); n[0] += 1
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    sp, ms = VoskSpotter(), []
    t0 = time.perf_counter()
    for _, _, c, x, _, _ in sens[:20]:
        if c == "clean":
            ms.append(stream(sp, x)[1])
    stop.set(); th.join()
    return np.concatenate(ms), f"{n[0]} Hearsay scores in {time.perf_counter() - t0:.0f} s alongside"


def summarize(df: pd.DataFrame, variant: str, lag: int = 0) -> dict:
    s = df[(df.variant == variant) & (df.set == "sensitive") & (df.lag_ms == lag)]
    io = df[(df.variant == variant) & (df.set == "innocent-out")]
    ii = df[(df.variant.str.startswith(variant.split("/")[0])) & (df.set == "innocent-in")]
    ls, tt = io[io.kind == "ls"], io[io.kind == "tts"]
    per_min = lambda d, col: 60 * d[col].sum() / max(d.dur.sum(), 1e-9)  # noqa: E731
    return dict(variant=variant, lag_ms=lag, sequences=len(s), leaked_per_seq=s.leaked.mean(),
                fully_blocked=(s.leaked == 0).mean(), le1=(s.leaked <= 1).mean(), audible_per_seq=s.audible.mean(),
                fr_s_per_min_ls=per_min(ls, "redacted_s"), fr_s_per_min_tts=per_min(tt, "redacted_s"),
                out_trig_per_min=per_min(io, "triggers"), in_trig_per_min=per_min(ii, "triggers") if len(ii) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="compare variants on the tuning set (seed 1000 TTS, "
                    "LibriSpeech speakers 100/2803 only); writes nothing to docs/reports/")
    args = ap.parse_args()
    t_start = time.perf_counter()
    sens, inn, req, dropped = build(args.tune)
    print(f"sensitive {len(sens)} (dropped {dropped} unalignable), innocent {len(inn)}, requests {len(req)}", flush=True)
    variants = {"default": {}} if args.tune else {"default": {}, "hold0": {"hold_s": 0.0}}
    rows = []
    for tag, kw in variants.items():
        sp = VoskSpotter(**kw)
        eval_sensitive(sp, sens if tag == "default" else [s for s in sens if s[2] == "clean"], rows, tag)
        eval_innocent(sp, inn if tag == "default" else [i for i in inn if i[1] == "clean"], rows, tag, "out")
        print(f"  {tag}: done ({time.perf_counter() - t_start:.0f} s)", flush=True)
    for tag, kw in {"default": {}}.items():
        spi = VoskSpotter(mode="inbound", **kw)
        eval_innocent(spi, [i for i in inn if i[1] == "clean"], rows, tag, "in")
        eval_innocent(spi, req, rows, tag, "req")
    df = pd.DataFrame(rows)
    summ = [summarize(df, v, int(lag * 1000)) for v in variants for lag in (LAGS if v == "default" else (0,))]
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    if args.tune:
        return
    hs_ms, hs_note = cpu_with_hearsay(sens)
    df.to_csv(OUT / "secret_shield.csv", index=False)
    write_report(df, summ, hs_ms, hs_note, dropped, time.perf_counter() - t_start)


def write_report(df, summ, hs_ms, hs_note, dropped, elapsed):
    d = df[(df.variant == "default") & (df.lag_ms == 0)]
    s = d[d.set == "sensitive"]
    req = d[d.set == "innocent-req"]
    blocks = lambda q: (q.ms_mean * q.blocks).sum() / q.blocks.sum()  # noqa: E731
    out_ms, in_ms = blocks(d[d.set == "innocent-out"]), blocks(d[d.set == "innocent-in"])
    S = pd.DataFrame(summ)
    by = (s.groupby(["kind", "cond"]).agg(n=("id", "size"), leaked=("leaked", "mean"), full=("leaked", lambda v: (v == 0).mean()),
                                         le1=("leaked", lambda v: (v <= 1).mean()), first=("first_leaked", "mean"),
                                         lag_med=("det_lag_med", "median"), lag_max=("det_lag_max", "max"))).round(2)
    fmt = lambda v, p=2: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"  # noqa: E731
    lines = [
        "# Spoken-secret shield: evaluation (plan 06 §5)", "",
        "Generated by `app/secret_shield/eval/secret_shield_eval.py` (seed 0, CPU, "
        f"{elapsed / 60:.0f} min). Spotter: `VoskSpotter` (Vosk small en-us 0.15, open vocabulary, low-latency decoder "
        "options), 20 ms blocks, 500 ms delay line, gap_s 1.2, tail_s 0.3, hold_s 0.6.", "",
        "**Data caveat:** every sensitive utterance is a TTS render (facebook/mms-tts-eng, one synthetic voice). Plan 06 "
        "asks for our own recorded voices from consenting teammates; none exist yet. **Owner to-do:** record ~20 "
        "code/card/PIN readings per teammate and rerun. Real people read digits faster, with fillers and accents, "
        "and the numbers below may not hold.", "",
        "## Headline", "",
        "| metric | target (plan 06) | result |", "|---|---|---|",
    ]
    h = S[(S.variant == "default") & (S.lag_ms == 0)].iloc[0]
    h1 = S[(S.variant == "default") & (S.lag_ms == 100)].iloc[0]
    lines += [
        f"| secret words leaked per sequence (> 20 % escaped) | 0 (acceptance ≤ 1) | {h.leaked_per_seq:.2f} "
        f"({h1.leaked_per_seq:.2f} with 100 ms worker lag) |",
        f"| sequences fully blocked | - | {100 * h.fully_blocked:.0f} % |",
        f"| sequences with ≤ 1 word leaked | - | {100 * h.le1:.0f} % ({100 * h1.le1:.0f} % at 100 ms lag) |",
        f"| secret words > 50 % audible per sequence | - | {h.audible_per_seq:.2f} |",
        f"| false redaction, LibriSpeech (s per min of speech) | < 1 | {h.fr_s_per_min_ls:.2f} |",
        f"| false redaction, casual-number TTS sentences (s/min) | < 1 | {h.fr_s_per_min_tts:.2f} |",
        f"| outbound false trigger phrases per min (innocent) | - | {h.out_trig_per_min:.2f} |",
        f"| inbound false request triggers per min (innocent) | - | {h.in_trig_per_min:.2f} |",
        f"| inbound requests caught (TTS, {len(req)}) | - | {int((req.triggers > 0).sum())}/{len(req)} |",
        f"| CPU per 20 ms block, outbound / inbound spotter (mean) | ≪ 20 ms | {out_ms:.2f} / {in_ms:.2f} ms "
        f"(p99 {d.ms_p99.max():.1f}, max {d.ms_max.max():.1f}) |",
        f"| same, outbound, Hearsay scoring concurrently | - | "
        + (f"{hs_ms.mean():.2f} ms (p99 {np.percentile(hs_ms, 99):.1f}, max {hs_ms.max():.1f}); {hs_note}" if hs_ms is not None else hs_note)
        + " |",
        "| end-to-end added latency | - | 500 ms (the delay line; constant while the feature is on) |", "",
        "## By sequence kind (lag 0)", "",
        "`first` = share of sequences whose first secret word leaked. `lag` = seconds from a secret word's start until "
        "the first span covering it was emitted (median of per-sequence medians / worst).", "",
        "| kind | cond | n | leaked/seq | fully blocked | ≤ 1 leaked | first leaked | lag med | lag max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (kind, cond), r in by.iterrows():
        lines.append(f"| {kind} | {cond} | {r.n} | {r.leaked:.2f} | {100 * r.full:.0f} % | {100 * r.le1:.0f} % | "
                     f"{100 * r['first']:.0f} % | {fmt(r.lag_med)} | {fmt(r.lag_max)} |")
    lines += ["", "## Ablation: no mute-ahead (hold_s 0, clean only)", "",
              "| variant | leaked/seq | fully blocked | ≤ 1 leaked | FR LibriSpeech s/min | FR TTS s/min |", "|---|---|---|---|---|---|"]
    for _, r in S[S.lag_ms == 0].iterrows():
        lines.append(f"| {r.variant} | {r.leaked_per_seq:.2f} | {100 * r.fully_blocked:.0f} % | {100 * r.le1:.0f} % | "
                     f"{r.fr_s_per_min_ls:.2f} | {r.fr_s_per_min_tts:.2f} |")
    lines += ["", "(`default` above includes the keys condition; `hold0` is clean only, compare with the clean rows.)", ""]
    lines += verdict(h, h1, s)
    lines += ["", "## Method notes", "",
              f"- {dropped} sensitive renders dropped because forced alignment could not place the sentence.",
              "- Leak rule = the Redactor's: a span emitted after the block ending at T redacts only samples ≥ T + lag − 500 ms.",
              "- Worker lag 100 ms models the pipeline's secret_step running on a worker (it reads the ring in steps, "
              "up to 1 s at a time); spans arriving later than the delay line allows leak.",
              "- Tuning (open vocabulary vs restricted grammar, hold_s, trigger lists, homophones) used `--tune`: a "
              "different TTS seed and only LibriSpeech speakers 100/2803, which this evaluation excludes. There the "
              "restricted grammar of plan 06 §4 (digits + triggers + [unk]) leaked less (0.33 words/sequence) but "
              "redacted 6.6 s/min of LibriSpeech and fired 37.7 inbound request triggers/min: every utterance gets "
              "forced onto grammar words. The open-vocabulary LM got 0.16 s/min and 0 triggers/min, so it ships.",
              "- CPU spikes (p99/max above) come from Vosk finishing an utterance (endpoint + final lattice). Spans "
              "wait behind them, which is part of why some words leak; the spotter must run on its own worker.",
              "- Keys condition: harrison test presses (Keyguard `harrison_split`, seed 0), shielded by the real "
              "`KeyguardShield`; the shield's lookahead is trimmed so word times line up.",
              "- The recognized text never leaves the spotter; this script reads only spans (category + length).", ""]
    (OUT / "secret_shield.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'secret_shield.md'}")


def verdict(h, h1, s) -> list[str]:
    trig = s[s.kind != "no-trigger"]
    nt = s[s.kind == "no-trigger"]
    ok_leak = h.leaked_per_seq <= 1 and h.le1 >= 0.9
    ok_fr = h.fr_s_per_min_ls < 1 and h.fr_s_per_min_tts < 1
    return ["## Verdict", "",
            f"- **Leak (target 0, acceptance ≤ 1 per sequence): {'meets acceptance' if ok_leak else 'misses acceptance'}** "
            f"on TTS: {h.leaked_per_seq:.2f} words leaked per sequence, {100 * h.fully_blocked:.0f} % fully blocked. "
            f"With a trigger phrase first (\"the code is\") {100 * (trig.leaked == 0).mean():.0f} % are fully blocked; "
            f"without one the first digit passes by design ({100 * nt.first_leaked.mean():.0f} % of those leak it) "
            f"and {100 * (nt.leaked <= 1).mean():.0f} % stay within ≤ 1.",
            f"- **False redaction (target < 1 s/min): {'meets' if ok_fr else 'misses'}**: {h.fr_s_per_min_ls:.2f} s/min on "
            f"LibriSpeech, {h.fr_s_per_min_tts:.2f} s/min on the casual-number sentences (which include hard cases on purpose).",
            f"- Worker lag matters: at 100 ms, {h1.leaked_per_seq:.2f} leaked per sequence. The pipeline should feed the "
            "spotter at least every 20-40 ms while armed.",
            "- Not yet evidence for real voices (see the data caveat)."]


if __name__ == "__main__":
    main()
