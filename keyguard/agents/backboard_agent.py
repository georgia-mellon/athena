"""A thin LLM agent that reasons (and optionally remembers) through Backboard.

Each agent is a Backboard assistant with a role system-prompt; `.ask()` sends a
message and returns the reply text. `memory='auto'` lets an agent accumulate and
recall memory across rounds (the Defender uses this to learn which strategies beat
which attackers); `memory='off'` for stateless calls. Backboard proxies to a real
LLM provider (anthropic/google/openai/...), so Backboard is both the reasoning
backbone AND the memory store — one dependency for the whole multi-agent system.

Sync wrappers over the async SDK; every call is guarded so a network/LLM hiccup
degrades gracefully rather than crashing the pipeline.
"""
from __future__ import annotations
import asyncio
import json
import os
import re
from typing import Optional

from .. import config  # noqa: F401  Athena: loads .env before the env reads below

# Athena: Ares and Athena reason on Gemini through Backboard (Backboard = memory layer), overridable by env
PROVIDER = os.environ.get("BACKBOARD_PROVIDER") or "google"
MODEL = os.environ.get("BACKBOARD_MODEL") or "gemini-2.5-flash"


def _run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
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


class BackboardAgent:
    def __init__(self, role: str, system_prompt: str, memory: str = "off"):
        self.role = role
        self.system_prompt = system_prompt
        self.memory = memory
        self.name = f"kg-{role}"
        self._aid: Optional[str] = None

    async def _ensure(self, client):
        if self._aid:
            return self._aid
        lst = await client.list_assistants(name=self.name)
        a = lst[0] if lst else await client.create_assistant(
            name=self.name, description=f"KeyGuard {self.role} agent",
            system_prompt=self.system_prompt)
        self._aid = str(a.assistant_id)
        return self._aid

    def ask(self, content: str) -> str:
        client = _client()
        if client is None:
            return ""

        async def _go():
            try:
                aid = await self._ensure(client)
                kw = dict(content=content, assistant_id=aid, system_prompt=self.system_prompt,
                          memory=self.memory)
                if PROVIDER:
                    kw["llm_provider"] = PROVIDER
                if MODEL:
                    kw["model_name"] = MODEL
                r = await client.send_message(**kw)
                return getattr(r, "content", "") or ""
            finally:
                try:
                    await client.aclose()
                except Exception:
                    pass
        try:
            return _run(_go()) or ""
        except Exception:
            return ""

    def ask_json(self, content: str) -> dict:
        """Ask and parse the first JSON object in the reply (robust to prose)."""
        txt = self.ask(content + "\n\nRespond with ONLY a JSON object.")
        if not txt:
            return {}
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}


def available() -> bool:
    return bool(os.environ.get("BACKBOARD_API_KEY"))
