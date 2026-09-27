"""Unified LLM interface — the reasoning core of KeyGuard's agents.

Priority: direct Gemini (GEMINI_API_KEY, fast + reliable) -> Backboard proxy
(BACKBOARD_API_KEY) -> a transparent rule-based stand-in (so a demo still runs
offline, clearly labelled as the fallback). Every call is guarded: a network or
parse hiccup degrades to the next tier instead of crashing the demo.

The LLM is central on BOTH sides of the arms race:
  * Attacker-LLM  — turns the weak per-key acoustic guess into readable English.
  * Defender-LLM  — triages the sensitive span and invents a COHERENT false
                    secret (deception), reasoning a fixed transform cannot do.
"""
from __future__ import annotations
import json
import os
import re

from .. import config  # loads .env -> os.environ

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


def backend() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("BACKBOARD_API_KEY"):
        return "backboard"
    return "rule-based"


def _gemini(system: str, prompt: str, temperature: float = 0.4) -> str:
    import httpx
    key = os.environ["GEMINI_API_KEY"]
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
    }
    r = httpx.post(_GEMINI_URL.format(m=GEMINI_MODEL), params={"key": key},
                   json=body, timeout=45)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


_AGENTS: dict = {}


def _agent_ask(agent: str, system: str, prompt: str) -> str:
    """A named persistent Backboard agent (kg-<agent>) with memory='auto': it recalls what it learned in earlier
    rounds and runs (Backboard is the LLM and the memory store). '' when no key or on failure."""
    if not os.environ.get("BACKBOARD_API_KEY"):
        return ""
    from .backboard_agent import BackboardAgent
    a = _AGENTS.get(agent) or _AGENTS.setdefault(agent, BackboardAgent(agent, system, memory="auto"))
    try:
        return a.ask(prompt) or ""
    except Exception:
        return ""


last_route = ""   # CallGuard: which path answered the last ask(), e.g. "backboard:google/gemini-2.5-flash"


def _bb_route() -> str:
    from .backboard_agent import MODEL, PROVIDER
    return f"backboard:{PROVIDER or 'default'}/{MODEL or 'default'}"


def ask(system: str, prompt: str, temperature: float = 0.4, agent: str | None = None) -> str:
    """Plain-text LLM reply; '' on failure (caller falls back). `agent` (e.g. "ares", "athena") routes through that
    agent's Backboard memory first, then the usual Gemini -> Backboard chain. Sets `last_route`."""
    global last_route
    last_route = "none"
    if agent:
        txt = _agent_ask(agent, system, prompt)
        if txt:
            last_route = _bb_route()
            return txt
    b = backend()
    try:
        if b == "gemini":
            txt = _gemini(system, prompt, temperature) or ""
            last_route = f"gemini-direct:{GEMINI_MODEL}" if txt else "none"
            return txt
        if b == "backboard":
            from .backboard_agent import BackboardAgent
            txt = BackboardAgent("agent", system, memory="off").ask(prompt) or ""
            last_route = _bb_route() if txt else "none"
            return txt
    except Exception:
        return ""
    return ""


def ask_json(system: str, prompt: str, temperature: float = 0.2, agent: str | None = None) -> dict:
    """Ask for JSON and parse the first object; {} on failure."""
    txt = ask(system, prompt + "\n\nRespond with ONLY a JSON object.", temperature, agent=agent)
    if not txt:
        return {}
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Rule-based stand-ins (used only when no LLM key is present) — kept transparent
# so the demo still shows the SHAPE of the reasoning offline.
# ---------------------------------------------------------------------------
_SECRET_RE = re.compile(
    r"(password|passcode|code|pin|account|ssn|card|cvv|otp)\W*(is|:)?\W*([A-Za-z0-9]{3,})",
    re.I)


def rule_triage(text: str) -> dict:
    """Fallback sensitivity triage: find a credential-like token."""
    m = _SECRET_RE.search(text)
    if m:
        return {"sensitive": m.group(3).upper(), "kind": m.group(1).lower(),
                "reason": "matched a credential keyword + token (rule-based)"}
    toks = [t for t in re.findall(r"[A-Za-z0-9]{5,}", text)
            if any(c.isdigit() for c in t)]
    if toks:
        return {"sensitive": toks[0].upper(), "kind": "alphanumeric-secret",
                "reason": "longest alphanumeric token with a digit (rule-based)"}
    return {"sensitive": "", "kind": "none", "reason": "no obvious secret (rule-based)"}


def rule_decoy(secret: str) -> str:
    """Fallback coherent false target: same length, letters->plausible letters,
    digits->different digits (so it reads as a real-looking wrong secret)."""
    import random
    rng = random.Random(hash(secret) & 0xffff)
    pools = {"v": "AEIOU", "c": "BCDFGHJKLMNPRSTVWYZ"}
    out = []
    for ch in secret.upper():
        if ch.isdigit():
            out.append(str((int(ch) + rng.randint(1, 8)) % 10))
        elif ch in "AEIOU":
            out.append(rng.choice(pools["v"]))
        elif ch.isalpha():
            out.append(rng.choice(pools["c"]))
        else:
            out.append(ch)
    return "".join(out)


def demo():
    print(f"LLM backend: {backend()}")
    r = rule_triage("meet me at noon my password is hunter2 thanks")
    print("rule_triage:", r)
    print("rule_decoy(HUNTER2):", rule_decoy("HUNTER2"))
    if backend() == "gemini":
        print("gemini says:", ask("You are terse.", "Reply with one word: PONG")[:40])


if __name__ == "__main__":
    demo()
