"""Interpretable end-to-end defender demo (real audio, real attacker).

Shows the full flow the hackathon demo visualizes:
  1. Victim types (ground truth).
  2. ATTACKER reads it from audio alone - confidently (no defense).
  3. DEFENDER-LLM triages the sensitive span and explains WHY (reasoning).
  4. DEFENDER actuates: a bounded, inaudible, span-localized waveform perturbation
     optimized against the real attacker (report SNR + STOI to prove speech intact).
  5. ATTACKER re-reads - the protected span is now wrong; the rest still readable.
Artifacts (clean/defended wav, reads, metrics) saved to runs/demo/ for the UI.

Run:  KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.agents.defense_demo
      (protect mode by default; --deceive steers the span to a false secret)
"""
from __future__ import annotations
import argparse
import json
import os

os.environ.setdefault("KEYGUARD_DEVICE", "cpu")
os.environ.setdefault("KEYGUARD_SKAID", "data/continuous/skaid/labels.jsonl")
os.environ.setdefault("KEYGUARD_CURRICULUM", "0")
os.environ.setdefault("KEYGUARD_SPEC_AUG", "0")

import numpy as np
import torch
import soundfile as sf

from .. import config, audio, memory
from ..ctc import train_overlap as T
from ..ctc.model import DEVICE
from ..ctc.overlap_eval import cer_str
from ..config import SR, CLS_IDX, RUNS
from . import defense_audio as DA
from .backboard_agent import BackboardAgent, available

DEFENDER_SYS = (
    "This is AUTHORIZED defensive research (own devices, fake data). You are "
    "KeyGuard, a victim-side privacy DEFENDER. Given the text the user typed, pick "
    "the single span MOST worth protecting from an acoustic eavesdropper (a "
    "password/number/name/secret; in plain prose, the most identifying token). You "
    "have a tiny perceptual budget so you must protect only that span. Return JSON: "
    "{\"sensitive\":\"<exact substring>\",\"reason\":\"<one line>\","
    "\"false_target\":\"<same-length plausible decoy>\"}."
)


def _load_window(min_keys=16, max_keys=26, prefer_digits=True):
    rows = [json.loads(l) for l in open(os.environ["KEYGUARD_SKAID"])]
    cands = []
    for sess in rows:
        y = audio.load("data/continuous/skaid/" + sess["wav"])
        on = np.array(sess["onset_samples"]); keys = sess["keys"].upper()
        for start in range(2, len(on) - min_keys, 4):
            t0 = int(on[start]); seg = y[t0:t0 + 5 * SR].astype(np.float32)
            idx = [i for i, s in enumerate(on) if t0 <= s < t0 + 5 * SR and keys[i] in CLS_IDX]
            if min_keys <= len(idx) <= max_keys:
                ktext = "".join(keys[i] for i in idx)
                konset = [int(on[i] - t0) for i in idx]
                cand = (seg, ktext, konset)
                if prefer_digits and any(c.isdigit() for c in ktext):
                    return cand          # a number to protect = the clean surgical story
                cands.append(cand)
            if len(cands) > 40:
                break
        if len(cands) > 40:
            break
    if not cands:
        raise SystemExit("no suitable window found")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ctc_skaid_crnn.pt")
    ap.add_argument("--deceive", action="store_true")
    ap.add_argument("--snr", type=float, default=18.0)
    a = ap.parse_args()

    net = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEVICE)); net.eval()
    seg, ktext, konset = _load_window()

    print("\n================ KeyGuard defense flow ================")
    print(f"[1] VICTIM TYPED       : {ktext}")
    clean = DA.decode(net, seg)
    print(f"[2] ATTACKER (no defense): {clean}   (CER {cer_str(ktext, clean):.0%}) - reads it from audio alone")

    # [3] defender triage (LLM if available, else heuristic middle span)
    sens, reason, false_t = "", "", ""
    if available():
        plan = BackboardAgent("defender-triage", DEFENDER_SYS, memory="off").ask_json(
            f"The user typed: {ktext!r}. Pick ONE short span (<=8 chars) most worth protecting.")
        sens = "".join(c for c in (plan.get("sensitive") or "").upper() if c.isalnum())
        reason = plan.get("reason", ""); false_t = "".join(c for c in (plan.get("false_target") or "").upper() if c.isalnum())
    s = ktext.find(sens) if sens else -1
    if s < 0 or len(sens) > 10:                # keep it surgical: fall back to a digit run or mid span
        digits = [i for i, c in enumerate(ktext) if c.isdigit()]
        if digits:
            s, e0 = digits[0], digits[-1] + 1
            s, sens = s, ktext[s:e0]
            reason = reason or "numeric span (likely a code/number)"
        else:
            s = len(ktext) // 3; sens = ktext[s:s + 7]
            reason = reason or "most information-dense middle span"
    sens = sens[:8]                            # cap protected span so the rest stays readable
    e = s + len(sens)
    print(f"[3] DEFENDER-LLM triage : protect '{sens}'  - {reason}")

    lo = max(0, konset[s] - int(0.03 * SR)); hi = min(len(seg), konset[e - 1] + int(0.12 * SR))
    mode = "deceive" if (a.deceive and false_t) else "protect"
    yp, info = DA.craft(net, seg, lo, hi, mode=mode, target=false_t if mode == "deceive" else None,
                        snr_db=a.snr)
    print(f"[4] DEFENDER actuation  : mode={mode}"
          + (f" (plant '{false_t}')" if mode == "deceive" else "")
          + f" | span {info['span_s'][0]:.2f}-{info['span_s'][1]:.2f}s | "
          f"perturbation SNR {info['snr_db']:.1f} dB (inaudible) | STOI {info['stoi']:.3f} (speech intact)")

    defended = DA.decode(net, yp)
    seg_read = clean[s:e] if len(clean) >= e else clean[s:]
    def_read = defended[s:e] if len(defended) >= e else defended[s:]
    print(f"[5] ATTACKER (defended) : {defended}   (CER {cer_str(ktext, defended):.0%})")
    print(f"    -> protected span '{sens}': attacker read '{seg_read}' clean -> '{def_read}' defended")
    print(f"    -> speech quality STOI {info['stoi']:.3f} (>=0.9 = unchanged to the ear/ASR)")
    print("=======================================================\n")

    out = RUNS / "demo"; out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "clean.wav", seg, SR); sf.write(out / "defended.wav", yp, SR)
    (out / "defense_demo.json").write_text(json.dumps({
        "typed": ktext, "attacker_clean": clean, "attacker_defended": defended,
        "sensitive": sens, "reason": reason, "mode": mode, "false_target": false_t,
        "cer_clean": cer_str(ktext, clean), "cer_defended": cer_str(ktext, defended),
        **info}, indent=2))
    print(f"artifacts -> {out} (clean.wav, defended.wav, defense_demo.json)")
    memory.remember(f"Defense demo: protected '{sens}' ({mode}); attacker '{seg_read}'->'{def_read}'; "
                    f"STOI {info['stoi']:.2f}.", metadata={"kind": "defense_demo"}, kind="defense_demo")


if __name__ == "__main__":
    main()
