"""Test KeyGuard on YOUR computer: type live, see what the attacker reads, then
watch the defender protect it. Records the mic while logging your real keystrokes
(ground truth) via pynput.

IMPORTANT: per-key acoustics are keyboard-specific, so the SKAID-trained model
reads a NEW keyboard poorly zero-shot. Run `calibrate` once (~2 min) to adapt it
to your keyboard, then `record`.

    uv run python3 -m keyguard.agents.live calibrate     # adapt to your keyboard
    uv run python3 -m keyguard.agents.live record --secs 20
    uv run python3 -m keyguard.agents.live record --secs 20 --defend

Needs a working mic + permission for global key capture (pynput).
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
import torch

from .. import config, audio
from ..ctc import train_overlap as T
from ..ctc.model import logmel, DEVICE
from ..ctc.data import VOCAB, SYM_OF_KEY, BLANK
from ..ctc.overlap_eval import cer_str
from ..config import SR, RUNS
from . import defense_audio as DA

CAL_PROMPTS = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS 1234567890",
    "HOW QUICKLY DAFT JUMPING ZEBRAS VEX THE 0 9 8 7",
]
LIVE_CKPT = str(RUNS / "ctc_live.pt")


_DEVICE = None       # input device index; set via --device (None = system default)
_EXCLUSIVE = False   # WASAPI exclusive mode (bypasses Windows mic processing); set via --exclusive


def list_devices():
    import sounddevice as sd
    print("input devices (use --device N):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:2d}  {d['name'][:55]:55}  in={d['max_input_channels']} "
                  f"sr={int(d['default_samplerate'])} api={sd.query_hostapis(d['hostapi'])['name']}")


def _resolve_mic(spec):
    """KEYGUARD_MIC -> device index. None/'' = system default; all-digits = index;
    anything else = case-insensitive name substring (stable across index reshuffles)."""
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower():
            return i
    raise SystemExit(f"KEYGUARD_MIC='{spec}' matched no input device — run `devices` to list them.")


def _record(secs: float, device=None):
    """Record mic (device default sr -> 16k) + log real keystrokes. Returns
    (y16, events[list of {t,key}])."""
    import sounddevice as sd
    from pynput import keyboard
    import librosa
    dev = device if device is not None else _DEVICE
    if dev is None:
        dev = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    info = sd.query_devices(dev) if dev is not None else sd.query_devices(sd.default.device[0])
    sr_dev = int(info.get("default_samplerate", 48000))
    nch = max(1, int(info.get("max_input_channels", 1)))   # WDM-KS demands native channel count
    extra = None
    if _EXCLUSIVE:
        try:
            extra = sd.WasapiSettings(exclusive=True)       # bypasses Windows APO (noise suppression)
        except Exception:
            extra = None
    if dev is not None:
        print(f"   [mic] device #{dev}: {info['name']} @ {sr_dev}Hz, {nch}ch"
              + (" (WASAPI exclusive)" if extra else ""))
    chunks, events = [], []
    state = {"t0": None}

    def cb(indata, frames, tinfo, status):
        chunks.append(indata[:, 0].copy())                  # first mic channel

    def on_press(key):
        if state["t0"] is None:
            return
        ch = " " if key == keyboard.Key.space else getattr(key, "char", None)
        if ch and (ch.isalnum() or ch == " "):
            events.append({"t": time.monotonic() - state["t0"], "key": ch.upper()})

    try:
        stream = sd.InputStream(samplerate=sr_dev, channels=nch, dtype="float32", device=dev,
                                callback=cb, extra_settings=extra)
    except Exception as ex:
        raise SystemExit(
            f"\ncould not open device #{dev} ({ex}).\n"
            "WDM-KS devices often won't open via PortAudio. Try instead:\n"
            "  --device 9 --exclusive   (WASAPI exclusive on the Intel mic; bypasses suppression)\n"
            "  or the system default (omit --device) after raising the Windows input level and\n"
            "  disabling mic enhancements/noise-suppression. See `devices` for the list.")
    kl = keyboard.Listener(on_press=on_press)
    print(f"\n>>> RECORDING for {secs:.0f}s — start typing now (talk too, to test noise)! <<<")
    state["t0"] = time.monotonic(); stream.start(); kl.start()
    time.sleep(secs)
    stream.stop(); stream.close(); kl.stop()
    y = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    y16 = librosa.resample(y, orig_sr=sr_dev, target_sr=SR).astype(np.float32)
    print(f"    captured {len(y16)/SR:.1f}s audio, {len(events)} keys")
    return y16, events


def _load_net(prefer_live=True):
    ck = LIVE_CKPT if (prefer_live and __import__("os").path.exists(LIVE_CKPT)) else "runs/ctc_skaid_crnn.pt"
    net = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    net.load_state_dict(torch.load(ck, map_location=DEVICE)); net.eval()
    return net, ck


def calibrate(rounds=1, secs=45):
    """Collect REAL continuous typing on YOUR keyboard and fine-tune the attacker.
    Starts from the bank-adapted model (runs/ctc_live.pt) if present, so it combines
    your per-key acoustic bank with real continuous typing — closing the synth->real
    gap. Otherwise starts from the SKAID model."""
    import os
    from scipy.signal import find_peaks
    net, base = _load_net(prefer_live=True)     # build on the bank-adapted model if we have it
    print(f"Calibrating from {base} (real continuous typing -> fine-tune). Type prompts clearly.")
    samples = []  # (logmel, frame_key_target)
    for r in range(rounds):
        for pr in CAL_PROMPTS:
            print(f"\nTYPE THIS (then keep still ~1s):\n   {pr}")
            y, ev = _record(secs=max(8, len(pr) * 0.4))
            keys = [e for e in ev if e["key"] in SYM_OF_KEY]
            if len(keys) < 8:
                print("   (too few keys captured, skipping)"); continue
            m = logmel(y)
            onset_samp = np.array([int(e["t"] * SR) for e in keys])
            ids = np.array([SYM_OF_KEY[e["key"]] for e in keys], np.int64)
            samples.append((m, T.frame_key_target(m.shape[0], onset_samp, ids)))
    if not samples:
        print("No calibration data captured."); return
    import torch.nn as nn
    w = torch.ones(len(VOCAB), device=DEVICE); w[BLANK] = 0.05
    fce = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    net.train()
    rng = np.random.default_rng(0)
    for step in range(400):
        m, fk = samples[int(rng.integers(len(samples)))]
        logits, _ = net(torch.from_numpy(m)[None].to(DEVICE))
        loss = fce(logits.reshape(-1, logits.shape[-1]), torch.from_numpy(fk).to(DEVICE).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); os.makedirs(RUNS, exist_ok=True); torch.save(net.state_dict(), LIVE_CKPT)
    print(f"\nCalibrated attacker saved -> {LIVE_CKPT}. Now run: record")


def _extract_clips(y, events, clip_len):
    """Cut a CLIP-length clip per keystroke, robust to quiet mics + keylog latency:
    peak-normalize, and for each keylog time search a window forward for the actual
    click (loudest smoothed-energy point), keep it only if that peak stands out
    above the recording's own noise floor (relative, not a fixed absolute gate)."""
    import numpy as np
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) + 1e-9
    y = y / peak                                   # peak-normalize (handles quiet mic gain)
    env = np.abs(y)
    w = max(1, int(0.003 * SR))
    env = np.convolve(env, np.ones(w) / w, mode="same")
    floor = float(np.median(env)) + 1e-6           # noise floor of THIS recording
    pre, post = int(0.05 * SR), int(0.30 * SR)     # search from just before keylog to +300ms (mic latency)
    cs = []
    for e in events:
        t = int(e["t"] * SR)
        a0, b0 = max(0, t - pre), min(len(y), t + post)
        if b0 - a0 < w:
            continue
        pk = a0 + int(np.argmax(env[a0:b0]))
        # keylog is ground truth (a key WAS pressed), so keep the clip unless the
        # window is essentially flat noise (no transient the mic could capture).
        if env[pk] < 1.5 * floor:
            continue
        a = max(0, pk - int(0.02 * SR))
        c = np.zeros(clip_len, np.float32)
        s = y[a:a + clip_len]; c[:len(s)] = s
        cs.append(c)
    return cs


def diag(secs=6.0):
    """Are your keystrokes actually reaching the mic? Tap ~8-10 keys during the
    window; this reports the recorded level, the noise floor, how many transients
    stand out, and per-keystroke how loud the click is vs the floor. Saves the wav
    so we can inspect. If clicks are <~3x the floor, it's mic gain/placement, not
    the code."""
    import os
    import numpy as np
    import soundfile as sf
    from ..config import RUNS
    print("Tap ~8-10 keys during the window (any keys), at your normal volume.")
    y, ev = _record(secs)
    if len(y) < 10:
        print("no audio captured — mic/device problem."); return
    rms = float(np.sqrt(np.mean(y ** 2))); peak = float(np.max(np.abs(y)))
    dbfs = 20 * np.log10(peak + 1e-9)
    env = np.abs(y.astype(np.float32))
    w = max(1, int(0.003 * SR)); env = np.convolve(env, np.ones(w) / w, "same")
    floor = float(np.median(env)) + 1e-9
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(env, height=3 * floor, distance=int(0.05 * SR))
    print(f"\naudio: rms={rms:.4f}  peak={peak:.4f}  ({dbfs:.0f} dBFS)  "
          f"noise_floor={floor:.4f}")
    print(f"transients >3x floor: {len(peaks)}   |   keylog events: {len(ev)}")
    ratios = []
    for e in ev:
        t = int(e["t"] * SR); a0, b0 = max(0, t - int(0.05 * SR)), min(len(y), t + int(0.3 * SR))
        pk = float(env[a0:b0].max()) if b0 > a0 else 0.0
        ratios.append(pk / floor)
        print(f"  '{e['key']}' @{e['t']:.2f}s: click {pk:.4f} = {pk/floor:.1f}x floor"
              + ("  <-- too quiet" if pk < 3 * floor else ""))
    os.makedirs(RUNS / "demo", exist_ok=True)
    sf.write(RUNS / "demo" / "diag.wav", y, SR)
    med = float(np.median(ratios)) if ratios else 0.0
    print(f"\nmedian click/floor = {med:.1f}x  (saved {RUNS/'demo'/'diag.wav'})")
    if med < 3:
        print("VERDICT: mic is barely hearing your keys. Move the laptop mic closer to the\n"
              "  keyboard, raise the input gain (Windows Sound settings -> Input -> device\n"
              "  properties -> boost/level), disable mic AGC/noise-suppression, or use an\n"
              "  external mic. Keystroke acoustic attacks need audible key sound.")
    else:
        print("VERDICT: keys are audible -> bank capture should work; if a key still gets 0,\n"
              "  tap it a bit harder/steadier.")


def bank(per_key=20, secs_per_key=9.0, out="data/live_bank.npz", missing_only=False, min_clips=8):
    """Build a clean, BALANCED per-key acoustic bank for YOUR keyboard: for each
    key you tap it ~per_key times (QUIETLY); we cut a 0.12s clip per keystroke
    (labeled by the keylog) into an .npz the trainer/synth engine uses
    (KEYGUARD_BANK=data/live_bank.npz). MERGES with an existing bank, so you can
    re-run with --missing to only fill in keys you don't have enough of yet."""
    import os
    import numpy as np
    from ..config import CLASSES
    from ..ctc.data import CLIP
    clips = {}
    if os.path.exists(out):                         # merge with what we already have
        d = np.load(out)
        clips = {k: d[k] for k in d.files}
        print(f"merging into existing {out}: {len(clips)}/36 keys present")
    todo = [k for k in CLASSES if (not missing_only) or len(clips.get(k, [])) < min_clips]
    print(f"capturing {len(todo)} keys: {todo}\n"
          f"tap each ~{per_key} times and STAY QUIET (no talking) — this is the CLEAN "
          f"alphabet; noise robustness is tested separately. Run in a real terminal.\n")
    for i, k in enumerate(todo):
        try:
            input(f"  [{i+1}/{len(todo)}] press ENTER, then tap '{k}' ~{per_key} times (quietly)...")
        except EOFError:
            print("  (no stdin — run in a terminal, not via '!')"); break
        y, ev = _record(secs_per_key)
        cs = _extract_clips(y, [e for e in ev if e["key"] == k], CLIP)
        if cs:
            clips[k] = np.stack(cs)                 # replace with the fresh (quiet) capture
        print(f"      got {len(cs)} clean '{k}' clips")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, **clips)
    miss = [k for k in CLASSES if k not in clips]
    print(f"\nsaved {out}: {len(clips)}/36 keys, {sum(len(v) for v in clips.values())} clips.")
    if miss:
        print(f"MISSING {miss} -- re-run to add just these:  "
              f"python -m keyguard.agents.live bank --missing")
    else:
        print("all 36 keys captured. Next, adapt the attacker to your keyboard:\n"
              "  KEYGUARD_BANK=data/live_bank.npz KEYGUARD_CKPT=runs/ctc_live.pt \\\n"
              "    python -m keyguard.ctc.train_overlap 3000\n"
              "then: python -m keyguard.agents.live record --secs 20")


def _save_sample(y, ev, out_dir="data/continuous/live"):
    """Append this recording as a SKAID-format sample the GPU trainer ingests via
    KEYGUARD_SKAID=<out_dir>/labels.jsonl. Row schema matches continuous_convert.py."""
    import os, json, time
    import soundfile as sf
    keys = [e for e in ev if e["key"] in SYM_OF_KEY]
    if len(keys) < 4:
        print("  (too few keys — sample not saved)"); return
    # Dead-mic guard: a silent recording (mic muted / wrong device / no permission)
    # is worthless and, if trained on, forces the model to memorize text from
    # silence. Reject it loudly instead of saving garbage.
    rms = float(np.sqrt(np.mean(np.square(np.asarray(y, dtype=np.float64)))))
    if rms < 1e-3:
        print(f"  !! MIC SILENT (rms={rms:.2e}) — NOT saved. The mic captured no "
              f"audio.\n     Fix: check mic permission/mute, pick the right input "
              f"with `devices`/`--device`, then retry.")
        return
    os.makedirs(out_dir, exist_ok=True)
    name = f"live_{int(time.time())}.wav"
    sf.write(os.path.join(out_dir, name), y, SR)
    row = {"wav": name, "keys": "".join(e["key"] for e in keys),
           "onset_samples": [int(e["t"] * SR) for e in keys]}
    with open(os.path.join(out_dir, "labels.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  saved overlap sample -> {out_dir}/{name} ({len(keys)} keys, labels.jsonl appended)")


# DIVERSE, key-balanced prompt bank. Generalization needs the model to see MANY
# different words/digraphs, not the same phrase repeated — repeated phrases make
# the net memorize text instead of learning per-key acoustics. Multiple distinct
# pangrams (full A-Z coverage) + wide everyday vocabulary + digit runs.
_COLLECT_TEXTS = [
    # pangrams (each covers all 26 letters)
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "sphinx of black quartz judge my vow",
    "jackdaws love my big sphinx of quartz",
    "we promptly judged antique ivory buckles for the next prize",
    "grumpy wizards make toxic brew for the evil queen and jack",
    # varied natural sentences (different vocabulary each)
    "she sells seashells beside the sunny shore every summer morning",
    "our flight departs at seven so we should leave home by four thirty",
    "the museum exhibit featured ancient pottery glassware and jewelry",
    "he whispered that the treasure was buried beneath the old oak tree",
    "modern engines convert chemical energy into motion with great efficiency",
    "curious children explored the dusty attic looking for hidden letters",
    "the chef garnished the plate with basil thyme and a squeeze of lemon",
    "volunteers gathered downtown to plant maple trees along the avenue",
    "quantum computers exploit superposition to explore many states at once",
    "a gentle breeze carried the scent of pine across the quiet valley",
    "the journalist verified every quote before publishing the exclusive story",
    "grandma baked cinnamon rolls while jazz played softly in the kitchen",
    "the hikers followed the winding trail up the rugged snowy mountain",
    "please water the ferns twice a week and keep them out of direct light",
    "the committee will vote on the zoning proposal next thursday evening",
    "brilliant fireworks exploded above the harbor as the crowd cheered loudly",
    # numbers / mixed
    "order 3 boxes of size 12 bolts and 7 packs of 40 washers by friday",
    "the invoice total is 1284 dollars due within 30 days of receipt",
    "flight 726 boards at gate 19 and lands around 8 45 local time",
    "call 8005551212 for support or dial extension 4073 during work hours",
]


def collect(n=24, secs=25):
    """Bulk-collect REAL typing for training: shows a prompt, you type it (vary
    speed, talk/cough sometimes for noise coverage), saves each as a SKAID-format
    sample. Run on the machine whose mic hears the keys (e.g. your Mac), then push
    data/continuous/live/ and we retrain. Aim for ~15-30 min total (~1-3k keys).

    Prompts are DISTINCT (shuffled, no repeats until the bank is exhausted) so the
    attacker learns per-key acoustics instead of memorizing a few phrases. Each
    saved clip is level-checked; a silent (dead-mic) recording is rejected, not
    saved. Do ONE quick warm-up clip and confirm it saved before doing the rest."""
    import random
    order = list(range(len(_COLLECT_TEXTS)))
    random.shuffle(order)
    print(f"Collecting up to {n} sessions x ~{secs}s from {len(_COLLECT_TEXTS)} "
          f"distinct prompts. Type each naturally; vary speed; occasionally "
          f"talk/cough for noise coverage.\n"
          f"TIP: for best signal put a phone mic ~15cm from the keyboard (a far "
          f"built-in mic is much weaker). Watch for the 'MIC SILENT' warning.\n")
    saved = 0
    for i in range(n):
        pr = _COLLECT_TEXTS[order[i % len(order)]]
        try:
            input(f"  [{i+1}/{n}] ENTER, then type (~{secs}s):\n     {pr}\n  > ")
        except EOFError:
            print("  (run in a terminal)"); return
        y, ev = _record(secs)
        import numpy as _np
        rms = float(_np.sqrt(_np.mean(_np.square(_np.asarray(y, dtype=_np.float64)))))
        print(f"     level rms={rms:.4f}  keys={sum(1 for e in ev if e['key'] in SYM_OF_KEY)}")
        _save_sample(y, ev)
        if rms >= 1e-3:
            saved += 1
    print(f"\ndone. saved {saved}/{n} audible samples. push data/continuous/live/ "
          f"(labels.jsonl + wavs) and we retrain.")


def record(secs, defend=False):
    net, ck = _load_net()
    y, ev = _record(secs)
    typed = "".join(e["key"] for e in ev if e["key"] in SYM_OF_KEY)
    read = DA.decode(net, y)
    _save_sample(y, ev)   # persist raw overlap wav + labels for GPU-side retraining
    print(f"\nattacker model: {ck}")
    print(f"YOU TYPED : {typed}")
    print(f"ATTACKER  : {read}   (CER {cer_str(typed, read):.0%})")
    if "skaid" in ck:
        print("  NOTE: using the SKAID model on your keyboard (zero-shot) -> expect high CER."
              "\n  Run `calibrate` first to adapt to your keyboard.")
    if defend and len(ev) >= 8:
        keys = [e for e in ev if e["key"] in SYM_OF_KEY]
        s = len(keys) // 3; e = min(s + 7, len(keys))
        lo = max(0, int(keys[s]["t"] * SR) - int(0.03 * SR))
        hi = min(len(y), int(keys[e - 1]["t"] * SR) + int(0.12 * SR))
        yp, info = DA.craft(net, y, lo, hi, mode="protect", snr_db=18.0)
        read2 = DA.decode(net, yp)
        print(f"\nDEFENDER protects keystrokes [{s}:{e}] "
              f"(t {info['span_s'][0]:.2f}-{info['span_s'][1]:.2f}s), "
              f"perturbation SNR {info['snr_db']:.1f}dB, STOI {info['stoi']:.3f} (inaudible)")
        print(f"ATTACKER (defended): {read2}   (CER {cer_str(typed, read2):.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None, help="input device index (see `devices`)")
    ap.add_argument("--exclusive", action="store_true", help="WASAPI exclusive mode (bypass mic noise-suppression)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    sub.add_parser("calibrate")
    dp = sub.add_parser("diag"); dp.add_argument("--secs", type=float, default=6.0)
    bp = sub.add_parser("bank"); bp.add_argument("--per-key", type=int, default=20); bp.add_argument("--missing", action="store_true")
    cp = sub.add_parser("collect"); cp.add_argument("--n", type=int, default=12); cp.add_argument("--secs", type=float, default=25)
    rp = sub.add_parser("record"); rp.add_argument("--secs", type=float, default=20); rp.add_argument("--defend", action="store_true")
    a = ap.parse_args()
    import os
    # --device wins; else KEYGUARD_MIC env (index OR name substring — names survive the
    # device-index reshuffle that happens when mics connect/disconnect).
    _DEVICE = a.device if a.device is not None else _resolve_mic(os.environ.get("KEYGUARD_MIC"))
    _EXCLUSIVE = a.exclusive
    if a.cmd == "devices":
        list_devices()
    elif a.cmd == "calibrate":
        calibrate()
    elif a.cmd == "diag":
        diag(a.secs)
    elif a.cmd == "bank":
        bank(per_key=a.per_key, missing_only=a.missing)
    elif a.cmd == "collect":
        collect(a.n, a.secs)
    else:
        record(a.secs, a.defend)
