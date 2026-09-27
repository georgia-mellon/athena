"""Backboard-backed agent memory for KeyGuard (attacker + defense arena).

Gives our agents persistent, cross-session/cross-machine memory via Backboard
(official `backboard-sdk`): every training experiment and every adversarial-arena
run is stored as a semantically-searchable fact, so the self-evolving loop can
recall "what have I already tried and how good was it?" and the shield's defense
track record survives across sessions and never silently regresses.

Design:
  * A local JSONL (`runs/arena_memory.jsonl`) is ALWAYS written — offline source of
    truth so the dashboard/demo work with no network and no key.
  * Backboard is mirrored best-effort when BACKBOARD_API_KEY is set (in .env).
  * The SDK is async; we wrap it with asyncio.run for the sync codebase. Every
    network call is guarded — memory is a convenience, never a hard dependency.

Assistant: one shared Backboard assistant (name from BACKBOARD_ASSISTANT, default
"keyguard-attacker"); its id is cached in runs/backboard_assistant.txt.
"""
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone

from .config import RUNS

LOCAL = RUNS / "arena_memory.jsonl"
ASSISTANT_FILE = RUNS / "backboard_assistant.txt"
ASSISTANT_NAME = os.environ.get("BACKBOARD_ASSISTANT", "keyguard-attacker")


def _run(coro):
    """Run an async SDK coroutine from sync code (best-effort)."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # already inside a loop (rare here) -> new loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _client():
    key = os.environ.get("BACKBOARD_API_KEY")
    if not key:
        return None
    try:
        from backboard import BackboardClient
        return BackboardClient(api_key=key)
    except Exception:
        return None


async def _assistant_id_async(client) -> str | None:
    if ASSISTANT_FILE.exists():
        cached = ASSISTANT_FILE.read_text().strip()
        if cached:
            return cached
    lst = await client.list_assistants(name=ASSISTANT_NAME)
    a = lst[0] if lst else await client.create_assistant(
        name=ASSISTANT_NAME,
        description="KeyGuard acoustic-keystroke attacker + adversarial-shield agent memory")
    aid = str(a.assistant_id)
    ASSISTANT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ASSISTANT_FILE.write_text(aid)
    return aid


def remember(content: str, metadata: dict | None = None, kind: str = "note") -> None:
    """Store one fact: append locally (always) + mirror to Backboard (best-effort)."""
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    rec = {"created": now_iso(), "kind": kind, "content": content,
           "metadata": metadata or {}}
    with LOCAL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    client = _client()
    if client is None:
        return

    async def _go():
        try:
            aid = await _assistant_id_async(client)
            if aid:
                md = {"source": "keyguard", "kind": kind, **(metadata or {})}
                await client.add_memory(aid, content, metadata=md)
        finally:
            try:
                await client.aclose()
            except Exception:
                pass
    try:
        _run(_go())
    except Exception:
        pass  # local JSONL is the source of truth


def search(query: str, limit: int = 5) -> list[dict]:
    """Semantic recall from Backboard; falls back to local substring match."""
    client = _client()
    if client is not None:
        async def _go():
            try:
                aid = await _assistant_id_async(client)
                if not aid:
                    return None
                return await client.search_memories(aid, query, limit=limit)
            finally:
                try:
                    await client.aclose()
                except Exception:
                    pass
        try:
            res = _run(_go())
            if isinstance(res, dict) and res.get("memories"):
                return res["memories"]
        except Exception:
            pass
    # offline fallback: naive local match
    q = query.lower()
    hits = [r for r in recall_runs(200) if q in json.dumps(r).lower()]
    return hits[:limit]


# ---- arena-specific helpers (kept for keyguard/shield/adversarial.py) ----

def _summary_text(run: dict) -> str:
    return (f"Arena run {run['created']}: clean attack {run['clean_acc']:.1%}, "
            f"shielded to {run['final_shielded']:.1%} after {run['rounds']} "
            f"retraining rounds at {run['snr_db']:.1f} dB (chance {run['chance']:.1%}).")


def remember_run(run: dict) -> None:
    """Record one adversarial-arena run (local + Backboard)."""
    remember(_summary_text(run), metadata={k: run[k] for k in run if k != "content"},
             kind="arena_run")


def remember_experiment(tag: str, cer: float, config: dict, note: str = "") -> None:
    """Record one attacker training experiment so the self-evolving loop remembers
    what was tried and how well it did."""
    content = (f"Attacker experiment '{tag}': MEAN CER {cer:.1%} on real held-out "
               f"typing. config={json.dumps(config, sort_keys=True)}. {note}".strip())
    remember(content, metadata={"tag": tag, "cer": cer, **config}, kind="experiment")


def recall_runs(limit: int = 20) -> list[dict]:
    """Past records, newest first, from the local mirror (offline-safe)."""
    if not LOCAL.exists():
        return []
    rows = []
    for line in LOCAL.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[::-1][:limit]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def demo():
    import tempfile, pathlib
    global LOCAL
    orig = LOCAL
    LOCAL = pathlib.Path(tempfile.mkdtemp()) / "arena_memory.jsonl"
    try:
        assert recall_runs() == []
        run = {"created": now_iso(), "clean_acc": 0.88, "final_shielded": 0.05,
               "snr_db": -18.0, "rounds": 4, "chance": 0.028}
        remember_run(run)                                # no key -> local only, no raise
        remember_experiment("crnn+framece", 0.375, {"model": "crnn", "blank_w": 0.05})
        got = recall_runs()
        assert len(got) == 2, got
        assert "shielded to 5.0%" in _summary_text(run)
        print(f"memory ok: local JSONL write/recall works ({LOCAL.name}); "
              f"Backboard mirror active when BACKBOARD_API_KEY set "
              f"(assistant '{ASSISTANT_NAME}').")
    finally:
        LOCAL = orig


if __name__ == "__main__":
    demo()
