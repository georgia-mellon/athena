"""Hook sinks: console, JSONL file, HTTP webhook, each subscribed to its topic globs.

Sinks run on the bus thread except the webhook, which has its own thread and bounded queue so a slow or dead
endpoint never backs up the bus. Key identities (`truth`, `key`, `typed`) are scrubbed from every payload
before it leaves the process.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

from .bus import EventBus
from .config import REPO, HookConfig
from .types import Event

log = logging.getLogger(__name__)
PRIVATE = frozenset({"truth", "key", "typed", "top1"})  # top1 = what the attacker read


def _scrub(x):
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if k not in PRIVATE}
    if isinstance(x, (list, tuple)):
        return [_scrub(v) for v in x]
    return x


def to_json(ev: Event) -> str:
    """The event as one JSON line; numpy scalars/arrays and other oddities are converted, never fatal."""
    def default(o):
        return o.tolist() if hasattr(o, "tolist") else str(o)
    return json.dumps({"topic": ev.topic, "t": ev.t, "data": _scrub(ev.data)}, default=default)


def console_sink(ev: Event) -> None:
    print(f"[athena] {to_json(ev)}", flush=True)


def jsonl_sink(path: str | Path) -> Callable[[Event], None]:
    p = Path(path)
    p = p if p.is_absolute() else REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def sink(ev: Event) -> None:
        line = to_json(ev) + "\n"
        with lock, p.open("a", encoding="utf-8") as f:
            f.write(line)
    return sink


class WebhookSink:
    """POSTs the event JSON to `url` from a daemon thread, with a timeout and retries (backoff 0.5 s, 1 s, ...)."""

    def __init__(self, url: str, timeout_s: float = 3.0, retries: int = 2, maxsize: int = 100, post=None):
        import httpx
        self.url, self.timeout_s, self.retries = url, timeout_s, retries
        self._post = post or httpx.post
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = self.failed = self.sent = 0
        threading.Thread(target=self._run, name="athena-webhook", daemon=True).start()

    def __call__(self, ev: Event) -> None:
        try:
            self._q.put_nowait(to_json(ev))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            body = self._q.get()
            for attempt in range(self.retries + 1):
                try:
                    r = self._post(self.url, content=body, headers={"content-type": "application/json"},
                                   timeout=self.timeout_s)
                    if r.status_code < 500:
                        self.sent += 1
                        break
                except Exception as e:  # network errors: retry, then give up on this event
                    log.debug("webhook attempt %d failed: %s", attempt, e)
                if attempt == self.retries:
                    self.failed += 1
                else:
                    time.sleep(0.5 * 2 ** attempt)
            self._q.task_done()


def make_sink(h: HookConfig) -> Callable[[Event], None]:
    if h.kind == "console":
        return console_sink
    if h.kind == "jsonl":
        return jsonl_sink(h.path)
    if h.kind == "webhook":
        return WebhookSink(h.url, h.timeout_s, h.retries)
    raise ValueError(f"unknown hook kind {h.kind!r}")


def install(bus: EventBus, hooks: list[HookConfig]) -> list[Callable[[], None]]:
    """Subscribe one sink per hook to each of its topic globs; returns the unsubscribe functions."""
    unsubs = []
    for h in hooks:
        sink = make_sink(h)
        unsubs += [bus.subscribe(t, sink) for t in h.topics]
    return unsubs
