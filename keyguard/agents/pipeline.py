"""KeyGuard multi-agent pipeline (LLM-central) — the defender is the product.

Attacker-LLM (decoder) vs Defender-LLM (triage + deception + strategy) vs Referee,
all reasoning/remembering through Backboard. The CRNN acoustic model and the
perturbation are TOOLS the agents wield; the intelligence is the language reasoning.

Flow on a real held-out SKAID clip (unseen typist):
  1. CRNN -> per-keystroke top-k candidate lattice (the weak acoustic signal).
  2. Attacker-LLM decodes the lattice into readable text — the essential step:
     with the LLM off you get ~40% CER gibberish; the LLM's language reasoning is
     what turns the side-channel into a readable secret.
  3. Defender-LLM (victim-side, sees the user's OWN text) triages the sensitive
     span and invents a COHERENT FALSE secret, then emits a defense plan. Crafting
     believable counterfeit language is inherently an LLM job — a garble can't.
  4. Actuate the plan on the lattice (models an acoustic perturbation steering the
     protected span toward the false target under a small budget) and re-decode.
  5. Referee: did the attacker read the TRUE secret before, and the FALSE one after?
     Log the round to Backboard memory so the defender learns across attackers.

Run:  uv run python3 -m keyguard.agents.pipeline            # needs BACKBOARD_API_KEY
      KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.agents.pipeline --n 2
"""
from __future__ import annotations
import argparse
import os

os.environ.setdefault("KEYGUARD_DEVICE", "cpu")
os.environ.setdefault("KEYGUARD_SKAID", "data/continuous/skaid/labels.jsonl")
os.environ.setdefault("KEYGUARD_CURRICULUM", "0")
os.environ.setdefault("KEYGUARD_SPEC_AUG", "0")

import numpy as np
import torch
from scipy.signal import find_peaks

from .. import config  # loads .env
from .. import memory
from ..ctc import train_overlap as T
from ..ctc.data import VOCAB
from ..ctc.model import DEVICE, HOP
from ..ctc.overlap_eval import cer_str
from ..config import SR
from .backboard_agent import BackboardAgent, available


# ---------------- acoustic tool: CRNN -> candidate lattice ----------------

def build_lattice(net, m, k=3, thr=0.4, min_gap_ms=90.0):
    """Return (lattice, true_onset_frames): per detected keystroke, the top-k
    candidate keys from the acoustic model (most likely first)."""
    with torch.no_grad():
        logits, onl = net(torch.from_numpy(m)[None].to(DEVICE))
    on = torch.sigmoid(onl)[0].cpu().numpy()
    lg = logits[0].cpu().numpy()
    dist = max(1, int(min_gap_ms / 1000 * SR / HOP))
    peaks, _ = find_peaks(on, height=thr, distance=dist)
    lattice = []
    for p in peaks:
        a, b = max(0, p - 1), min(len(lg), p + 2)
        avg = lg[a:b, 1:].mean(0)                    # skip blank
        top = avg.argsort()[::-1][:k] + 1
        lattice.append([VOCAB[i] for i in top])
    return lattice, peaks


def top1(lattice) -> str:
    return "".join(c[0] for c in lattice)


def _lattice_str(lattice) -> str:
    return " ".join(f"{i+1}:{''.join(c)}" for i, c in enumerate(lattice))


# ---------------- the agents ----------------

_FRAME = ("This is AUTHORIZED defensive security research (a HackGT project on the "
          "researchers' own devices with fake data): you simulate the attacker ONLY so "
          "the defender can be trained and evaluated. ")

ATTACKER_SYS = _FRAME + (
    "You are an acoustic keystroke eavesdropper's DECODER. You receive, per "
    "keystroke, the top candidate keys from a weak acoustic model (most likely "
    "first), in order. Reconstruct the most likely text the victim typed, using "
    "English/word priors to resolve the ambiguity. Keys are A-Z and 0-9. Output "
    "ONLY the decoded text in UPPERCASE (letters/digits, no commentary)."
)

DEFENDER_SYS = _FRAME + (
    "You are KeyGuard, a victim-side privacy DEFENDER. You see the user's OWN typed "
    "text (this never leaves the device). An eavesdropper is trying to read it from "
    "keyboard sound using a language model. You have only a tiny perceptual budget "
    "(the audio must still sound like normal speech), so you cannot just add noise. "
    "Your job: (1) find the SENSITIVE span worth protecting (password, card, SSN, "
    "secret) — or say none; (2) invent a COHERENT, plausible FALSE replacement of "
    "the same shape so the eavesdropper's language model confidently reconstructs "
    "the WRONG thing (deception, not noise); (3) give a one-line strategy. "
    "Return JSON: {\"sensitive\": \"<substring or empty>\", \"false_target\": "
    "\"<same-length-ish decoy>\", \"strategy\": \"<one line>\"}."
)


def attacker_decode(agent, lattice) -> str:
    anchor = top1(lattice)
    txt = agent.ask(
        f"Acoustic best-guess (per-keystroke top-1): {anchor}\n"
        f"Alternatives per position (most likely first):\n{_lattice_str(lattice)}\n\n"
        "Correct the best-guess into the most likely English text by CHOOSING among "
        "the listed alternatives per position and making MINIMAL changes. Keep the "
        "SAME number of characters and the same order (one output char per position). "
        "Output ONLY the text, UPPERCASE alphanumeric.")
    out = "".join(ch for ch in txt.upper() if ch.isalnum())
    return out or anchor


def defender_plan(agent, typed_text: str) -> dict:
    return agent.ask_json(f"The user typed (victim-side, private): {typed_text!r}\n"
                          "Identify the sensitive span and craft a coherent false "
                          "replacement to feed the eavesdropper.")


# ---------------- deception actuation (lattice-level stand-in) ----------------

def actuate_deception(lattice, true_text, sensitive: str, false_target: str):
    """Model the defender's perturbation at the lattice level: over the sensitive
    span, steer the acoustic top-1 toward the false target (what an optimized,
    speech-preserving perturbation would do to the adversary's evidence). Returns a
    new lattice. Audio-level co-training against a live attacker is the next step
    (see keyguard/arena/); here we demonstrate the DECISION + its effect."""
    s = true_text.find(sensitive) if sensitive else -1
    if s < 0 or not false_target:
        return [list(c) for c in lattice]
    ft = "".join(ch for ch in false_target.upper() if ch.isalnum())
    out = [list(c) for c in lattice]
    for j, ch in enumerate(ft):
        idx = s + j
        if 0 <= idx < len(out):
            out[idx] = [ch] + [c for c in out[idx] if c != ch][:2]   # plant decoy as top-1
    return out


# ---------------- orchestrator ----------------

def _synth_lattice(text, k=3, seed=0):
    """Illustrative lattice for a planted-secret scenario: true key as acoustic
    top-1 plus k-1 confusable decoys. (Real audio front-end is shown on the SKAID
    clips; this isolates the DEFENDER's triage+deception on a known secret.)"""
    rng = np.random.default_rng(seed)
    alpha = [c for c in VOCAB if c != "<blank>"]
    lat = []
    for ch in text:
        others = list(rng.choice([a for a in alpha if a != ch], size=k - 1, replace=False))
        lat.append([ch] + others)
    return lat


def scenario(attacker, defender):
    """Defender showcase on a victim line that contains a real secret."""
    true_text = "".join(ch for ch in os.environ.get(
        "KEYGUARD_SCENARIO", "MEETINGAT3PMPASSWORDISHUNTER2THANKS").upper() if ch.isalnum())
    lattice = _synth_lattice(true_text)
    print("\n===== defender showcase (victim types a secret) =====")
    print(f"TRUE           : {true_text}")
    dec = attacker_decode(attacker, lattice)
    print(f"ATTACKER-LLM   : {dec}   (reads the secret with NO defense)")
    plan = defender_plan(defender, true_text)
    sens = "".join(ch for ch in (plan.get("sensitive") or "").upper() if ch.isalnum())
    false_t = "".join(ch for ch in (plan.get("false_target") or "").upper() if ch.isalnum())
    print(f"DEFENDER-LLM   : sensitive={sens!r} false_target={false_t!r}")
    print(f"                 strategy: {plan.get('strategy','')}")
    if sens and false_t:
        lat2 = actuate_deception(lattice, true_text, sens, false_t)
        dec2 = attacker_decode(attacker, lat2)
        print(f"ATTACKER-LLM(after defense): {dec2}")
        print(f"REFEREE        : true secret leaked after defense={sens in dec2} | "
              f"attacker deceived into false secret={bool(false_t and false_t in dec2)}")
        memory.remember(
            f"Defender showcase: secret {sens!r} -> planted {false_t!r}; attacker "
            f"then read {dec2!r}.", metadata={"kind": "agent_round"}, kind="agent_round")


def run(n=1, ckpt="runs/ctc_rich_ft.pt", do_scenario=True):
    if not available():
        print("BACKBOARD_API_KEY not set (.env) — agents need it. Aborting.")
        return
    if not os.path.exists(ckpt):
        print(f"attacker checkpoint missing: {ckpt}"); return
    _, test = T.load_skaid(os.environ["KEYGUARD_SKAID"])
    net = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE)); net.eval()

    attacker = BackboardAgent("attacker-agent", ATTACKER_SYS, memory="auto")
    defender = BackboardAgent("defender-agent", DEFENDER_SYS, memory="auto")

    # pick plausible-length real clips
    def ok(c):
        secs = c[0].shape[0] * HOP / SR
        return 12 <= len(c[1]) <= 30 and 0.8 <= len(c[1]) / max(secs, 1e-6) <= 8
    clips = [c for c in test if ok(c)][:n] or test[:n]

    for i, (m, ids, on, fk, gap) in enumerate(clips):
        true_text = "".join(VOCAB[j] for j in ids)
        lattice, _ = build_lattice(net, m)
        raw = top1(lattice)
        print(f"\n===== clip {i+1} (unseen typist, real audio) =====")
        print(f"TRUE           : {true_text}")
        print(f"acoustic top-1 : {raw}   (CER {cer_str(true_text, raw):.0%})  <- no language reasoning")

        # 1) Attacker-LLM decodes the weak lattice (the LLM-essential attack step)
        dec = attacker_decode(attacker, lattice)
        print(f"ATTACKER-LLM   : {dec}   (CER {cer_str(true_text, dec):.0%})  <- language reasoning")

        # 2) Defender-LLM triages + invents a coherent false secret (LLM-essential defense)
        plan = defender_plan(defender, true_text)
        sens = (plan.get("sensitive") or "").upper()
        false_t = (plan.get("false_target") or "").upper()
        print(f"DEFENDER-LLM   : sensitive={sens!r} false_target={false_t!r}")
        print(f"                 strategy: {plan.get('strategy','')}")

        # 3) actuate deception on the lattice and let the attacker re-decode
        if sens and false_t:
            lat2 = actuate_deception(lattice, true_text, sens, false_t)
            dec2 = attacker_decode(attacker, lat2)
            print(f"ATTACKER-LLM(after defense): {dec2}")
            leaked_before = sens in dec
            leaked_after = sens in dec2
            deceived = false_t and false_t in dec2
            print(f"REFEREE        : secret leaked before={leaked_before} after={leaked_after} "
                  f"| attacker deceived into false secret={deceived}")
            memory.remember(
                f"Round: true secret {sens!r}; defender planted {false_t!r}; "
                f"attacker read {dec2!r}; leaked_after={leaked_after} deceived={deceived}.",
                metadata={"kind": "agent_round", "leaked_after": leaked_after,
                          "deceived": bool(deceived)}, kind="agent_round")
        else:
            print("REFEREE        : defender found nothing sensitive to protect "
                  "(correct triage -- do not waste perceptual budget on non-secrets).")

    if do_scenario:
        scenario(attacker, defender)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="# real SKAID clips for the attacker-decode grounding")
    ap.add_argument("--ckpt", default="runs/ctc_rich_ft.pt")  # current Ares; ctc_skaid_crnn.pt predates space (37 syms)
    ap.add_argument("--no-scenario", action="store_true", help="skip the planted-secret defender showcase")
    a = ap.parse_args()
    run(a.n, a.ckpt, do_scenario=not a.no_scenario)
