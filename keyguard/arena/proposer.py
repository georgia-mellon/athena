"""Proposers that mutate a ShieldConfig each arena generation.

Two backends behind one interface `propose(current, history, n) -> list[dict]`:
- LLMProposer : asks Grok (xAI, for the SpaceX prize) or Claude to read the
  min-max game so far and return candidate configs as JSON.
- MutationProposer : guided random search. Always available, no API key, and
  the honest fallback so the arena runs offline.

The LLM only proposes *config deltas*; the harness is the sole judge, so a bad
LLM idea just loses a round -- it can never fake a win.
"""
from __future__ import annotations
import os
import json
import dataclasses
import random
from ..shield.shield import ShieldConfig

# search bounds the proposer must stay inside
BOUNDS = {
    "strength": (0.0, 1.0),
    "randomize": (0.0, 2.0),
    "decoys": (0, 60),
    "ctx_frames": (4, 24),
    "guard_frames": (1, 5),
    "key_frames": (8, 28),       # ~64-224ms; a keystroke spans ~110ms so search wide
}


def _clip(cfg: dict) -> dict:
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        v = cfg.get(k, getattr(ShieldConfig(), k))
        v = max(lo, min(hi, v))
        out[k] = int(round(v)) if isinstance(getattr(ShieldConfig(), k), int) else float(v)
    return out


class MutationProposer:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def propose(self, current: dict, history: list, n: int) -> list[dict]:
        base = current or dataclasses.asdict(ShieldConfig())
        cands = []
        for _ in range(n):
            c = dict(base)
            for k, (lo, hi) in self.rng.sample(list(BOUNDS.items()),
                                               k=self.rng.randint(1, 3)):
                span = (hi - lo)
                c[k] = base[k] + self.rng.uniform(-0.3, 0.3) * span
            cands.append(_clip(c))
        return cands


class LLMProposer:
    """Grok/Claude backend. Falls back to mutation on any error."""

    def __init__(self, provider="xai", seed=0):
        self.provider = provider
        self.fallback = MutationProposer(seed)
        self.key = os.environ.get("XAI_API_KEY" if provider == "xai"
                                  else "ANTHROPIC_API_KEY", "")

    def available(self) -> bool:
        return bool(self.key)

    def propose(self, current: dict, history: list, n: int) -> list[dict]:
        if not self.key:
            return self.fallback.propose(current, history, n)
        try:
            return self._ask(current, history, n)
        except Exception as e:  # noqa
            print(f"[proposer] LLM failed ({e}); using mutation fallback")
            return self.fallback.propose(current, history, n)

    def _ask(self, current, history, n):
        import httpx
        sys = (
            "You are the BLUE agent in an acoustic-keystroke privacy arena. A "
            "shield removes keystroke sounds from a call; a RED attacker retrains "
            "on the shielded audio and tries to read the keys. You tune the shield "
            "config to MINIMIZE the adaptive attacker's accuracy while keeping "
            "speech quality (STOI/PESQ) high. Propose diverse configs as JSON.\n"
            f"Config fields and bounds: {json.dumps(BOUNDS)}.\n"
            "Return ONLY a JSON list of exactly "
            f"{n} objects with those fields."
        )
        user = json.dumps({"current": current, "history_tail": history[-6:]})
        if self.provider == "xai":
            url = "https://api.x.ai/v1/chat/completions"
            model = "grok-4"
        else:
            url = "https://api.anthropic.com/v1/messages"
            model = "claude-opus-5-5"
        with httpx.Client(timeout=60) as cl:
            if self.provider == "xai":
                r = cl.post(url, headers={"Authorization": f"Bearer {self.key}"},
                            json={"model": model, "messages": [
                                {"role": "system", "content": sys},
                                {"role": "user", "content": user}]})
                txt = r.json()["choices"][0]["message"]["content"]
            else:
                r = cl.post(url, headers={"x-api-key": self.key,
                                          "anthropic-version": "2023-06-01"},
                            json={"model": model, "max_tokens": 1024,
                                  "system": sys,
                                  "messages": [{"role": "user", "content": user}]})
                txt = r.json()["content"][0]["text"]
        txt = txt[txt.index("["):txt.rindex("]") + 1]
        return [_clip(c) for c in json.loads(txt)][:n]


def get_proposer(name="auto", seed=0):
    if name == "mutation":
        return MutationProposer(seed)
    p = LLMProposer("xai" if name in ("auto", "xai") else "anthropic", seed)
    if name == "auto" and not p.available():
        return MutationProposer(seed)
    return p
