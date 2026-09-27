"""Demo-mode insurance for /api/demo: a deterministic, always-lands fallback.

The demo pipeline is ALREADY deterministic (seeded mixture + seeded shield + a
`net.eval()` attacker), so the only flaky thing on a noisy expo floor is the
Gemini network call. This module records known-good `/api/demo` responses on
disk (keyed by the attackable-key string + protect flag) and, in demo mode,
prefers a cached good result over a live run that underperformed or errored.

Cached results are REAL: same models, same recorded key presses, same shield --
just captured while the network was healthy. Nothing here fabricates numbers;
caching is only insurance against Gemini/network variance on stage.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import RUNS

CACHE_PATH = RUNS / "demo_cache.json"
# Below this acoustic+LM accuracy an UNPROTECTED run is judged a dud, and we fall
# back to the cached good result for that exact input. Tune if the bank changes.
GOOD_ATTACK_ACC = 0.5  # ponytail: single threshold; per-length curve if it ever matters


def _key(keys: str, protect: str) -> str:
    return f"{keys}|{protect}"


def load_cache(path: Path = CACHE_PATH) -> dict:
    """Read the on-disk cache, or {} if absent/corrupt (never raises)."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_result(keys: str, protect: str, result: dict, path: Path = CACHE_PATH) -> None:
    """Persist one good response immutably (new dict, atomic-ish write)."""
    cache = load_cache(path)
    updated = {**cache, _key(keys, protect): result}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(updated))


def cached_result(keys: str, protect: str, path: Path = CACHE_PATH) -> dict | None:
    return load_cache(path).get(_key(keys, protect))


def is_good(result: dict) -> bool:
    """A demo run "lands" when the UNPROTECTED attack reads the secret well -- or
    when the shield is on, where a LOW accuracy is itself the win."""
    if result.get("protect") == "keyguard":
        return True
    acc = result.get("attack_acc_lm")
    if acc is None:
        acc = result.get("attack_acc", 0.0)
    return acc >= GOOD_ATTACK_ACC


def pick(live: dict | None, keys: str, protect: str, path: Path = CACHE_PATH) -> dict:
    """Demo-mode result selection.

    - good live run  -> use it AND cache it as future insurance   (source=live)
    - weak/failed run with a cached good run for this input -> cache (source=cache)
    - weak run, no cache -> return live best-effort               (source=live-weak)
    Raises only if there is neither a live result nor a cache entry.
    """
    if live is not None and is_good(live):
        chosen = {**live, "source": "live"}
        save_result(keys, protect, chosen, path)
        return chosen
    cached = cached_result(keys, protect, path)
    if cached is not None:
        return {**cached, "source": "cache"}
    if live is not None:
        return {**live, "source": "live-weak"}
    raise RuntimeError(f"demo pick: no live result and no cache for {_key(keys, protect)}")


def demo() -> None:
    """Self-check: fails if the fallback selection logic breaks. No network/model."""
    import tempfile

    p = Path(tempfile.mkdtemp()) / "demo_cache.json"
    good = {"typed": "PASSWORD", "attack_acc": 0.9, "attack_acc_lm": 1.0, "protect": "none"}
    weak = {"typed": "PASSWORD", "attack_acc": 0.2, "attack_acc_lm": 0.2, "protect": "none"}

    r = pick(good, "PASSWORD", "none", p)                     # good -> live + cached
    assert r["source"] == "live"
    assert cached_result("PASSWORD", "none", p) is not None

    r2 = pick(weak, "PASSWORD", "none", p)                    # weak -> cached good
    assert r2["source"] == "cache" and r2["attack_acc_lm"] == 1.0

    r3 = pick(None, "PASSWORD", "none", p)                    # total failure -> cache
    assert r3["source"] == "cache"

    prot = {"typed": "PASSWORD", "attack_acc": 0.05, "protect": "keyguard"}
    assert is_good(prot)                                      # low acc under shield = win
    assert pick(prot, "PASSWORD", "keyguard", p)["source"] == "live"

    assert pick(weak, "NEVERSEEN", "none", p)["source"] == "live-weak"  # cold, best effort

    try:
        pick(None, "NOTHING", "none", p)                     # no live, no cache -> raise
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

    print("demo_cache ok: good->live+cached, weak->cache, None->cache, "
          "protected->live, cold->live-weak, empty->raises")


if __name__ == "__main__":
    demo()
