"""EventBus: thread-safe pub/sub between the audio threads, drivers, ThreatEngine, hooks and the server.

Publishers include the audio threads, so `publish` only appends to a bounded deque and returns; one worker thread
does the dispatch. When the queue is full the oldest event is dropped and counted: stale telemetry is worth less
than a stalled mic. Subscribers match topics with shell globs (`threat.*`, `*`).
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import threading
from collections import deque
from typing import Callable

from .types import Event

log = logging.getLogger(__name__)
Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, maxlen: int = 4096):
        self._q: deque[Event] = deque(maxlen=maxlen)
        self._cv = threading.Condition()
        self._subs: list[tuple[str, Handler]] = []
        self._busy = False
        self._closed = False
        self.dropped = 0
        self._worker = threading.Thread(target=self._run, name="athena-bus", daemon=True)
        self._worker.start()

    def publish(self, event: Event) -> None:
        """Never blocks on subscribers; drops the oldest queued event when full."""
        with self._cv:
            if len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(event)
            self._cv.notify()

    def emit(self, topic: str, **data) -> None:
        self.publish(Event(topic, data))

    def subscribe(self, pattern: str, fn: Handler) -> Callable[[], None]:
        """Call `fn(event)` on the bus thread for topics matching `pattern`. Returns an unsubscribe function."""
        entry = (pattern, fn)
        with self._cv:
            self._subs = [*self._subs, entry]  # copy-on-write: dispatch iterates a snapshot

        def unsubscribe() -> None:
            with self._cv:
                self._subs = [s for s in self._subs if s is not entry]
        return unsubscribe

    def async_queue(self, pattern: str, loop: asyncio.AbstractEventLoop | None = None,
                    maxsize: int = 256) -> tuple[asyncio.Queue, Callable[[], None]]:
        """Bridge for asyncio consumers (the WebSocket server): events land in a bounded asyncio.Queue on `loop`.

        A slow client loses its oldest events (counted in `self.dropped`) instead of backing up the bus.
        """
        loop = loop or asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

        def put(ev: Event) -> None:  # runs on the loop thread
            if q.full():
                q.get_nowait()
                with self._cv:
                    self.dropped += 1
            q.put_nowait(ev)

        def forward(ev: Event) -> None:
            try:
                loop.call_soon_threadsafe(put, ev)
            except RuntimeError:  # loop closed
                pass
        return q, self.subscribe(pattern, forward)

    def flush(self, timeout: float = 2.0) -> bool:
        """Wait until every queued event is dispatched (tests, shutdown). True if drained in time."""
        with self._cv:
            return self._cv.wait_for(lambda: not self._q and not self._busy, timeout)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._cv:
                self._busy = False
                self._cv.notify_all()
                self._cv.wait_for(lambda: self._q or self._closed)
                if self._closed and not self._q:
                    return
                ev = self._q.popleft()
                subs = self._subs
                self._busy = True
            for pattern, fn in subs:
                if fnmatch.fnmatchcase(ev.topic, pattern):
                    try:
                        fn(ev)
                    except Exception:
                        log.exception("subscriber %r failed on %s", fn, ev.topic)
