import asyncio
import threading

from app.source.bus import EventBus
from app.source.types import Event


def test_order_and_glob():
    bus, got = EventBus(), []
    bus.subscribe("threat.*", lambda e: got.append(e.data["i"]))
    for i in range(100):
        bus.emit("threat.update" if i % 2 == 0 else "voice.verdict", i=i)
    assert bus.flush()
    assert got == list(range(0, 100, 2))


def test_unsubscribe_and_bad_subscriber():
    bus, got = EventBus(), []
    bus.subscribe("*", lambda e: 1 / 0)  # must not kill the worker
    unsub = bus.subscribe("*", lambda e: got.append(e.topic))
    bus.emit("a")
    assert bus.flush()
    unsub()
    bus.emit("b")
    assert bus.flush()
    assert got == ["a"]


def test_drop_oldest_when_full_publisher_never_blocks():
    bus, gate, got = EventBus(maxlen=5), threading.Event(), []
    bus.subscribe("*", lambda e: (gate.wait(2), got.append(e.data["i"])))
    bus.emit("x", i=-1)             # the worker takes this one and blocks on the gate
    while not bus._busy:
        pass
    for i in range(20):             # returns at once although the subscriber is stuck
        bus.emit("x", i=i)
    gate.set()
    assert bus.flush()
    assert bus.dropped == 15
    assert got == [-1, 15, 16, 17, 18, 19]


def test_async_bridge_bounded():
    async def main():
        bus = EventBus()
        q, unsub = bus.async_queue("threat.*", maxsize=2)
        for i in range(5):
            bus.publish(Event("threat.update", {"i": i}))
        bus.flush()
        await asyncio.sleep(0.05)
        items = [q.get_nowait().data["i"] for _ in range(q.qsize())]
        unsub()
        return items, bus.dropped
    items, dropped = asyncio.run(main())
    assert items == [3, 4] and dropped == 3
