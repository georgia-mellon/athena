from app.source.bus import EventBus
from app.source.config import ThreatConfig
from app.source.threat import ThreatEngine
from app.source.types import Event


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def engine():
    clock, bus = Clock(), EventBus()
    eng = ThreatEngine(bus, ThreatConfig(), now=clock)
    changes = []
    bus.subscribe("threat.level_change", lambda e: changes.append(e.data["to"]))
    return eng, clock, bus, changes


def feed(eng, topic, **data):
    eng.on_event(Event(topic, data))


def run(eng, clock, seconds, *, p=None, typing=False, raw=None, shielded=None):
    """Advance 0.5 s per step: a verdict every 2 s, a stroke (with its readouts) every 0.5 s."""
    out = None
    for i in range(int(seconds * 2)):
        clock.t += 0.5
        if p is not None and i % 4 == 0:
            feed(eng, "voice.verdict", p_synthetic=p)
        if typing:
            feed(eng, "keys.stroke")
            hit = {k: v for k, v in (("raw", raw), ("shielded", shielded)) if v is not None}
            if hit:
                feed(eng, "keys.readout", hit=hit)
        out = eng.tick()
    return out


def test_silence_is_safe():
    eng, clock, *_ = engine()
    assert run(eng, clock, 10)["level"] == "SAFE"


def test_human_voice_is_safe():
    eng, clock, *_ = engine()
    assert run(eng, clock, 30, p=0.05)["level"] == "SAFE"


def test_synthetic_voice_alone_is_warn():
    eng, clock, *_ = engine()
    u = run(eng, clock, 30, p=0.95)
    assert u["level"] == "WARN" and u["V"] > 0.9
    assert any("synthetic voice" in r for r in u["reasons"])


def test_synthetic_voice_plus_leaky_typing_shield_off_is_critical():
    eng, clock, bus, changes = engine()
    feed(eng, "shield.state", mode="off")
    u = run(eng, clock, 30, p=0.95, typing=True, raw=True, shielded=True)
    assert u["level"] == "CRITICAL" and u["E"] > 0.85
    assert "typing while an unverified voice is speaking" in u["reasons"]
    bus.flush()
    assert changes[-1] == "CRITICAL"


def test_shield_on_with_low_leak_is_lower():
    off, c1, *_ = engine()
    feed(off, "shield.state", mode="off")
    u_off = run(off, c1, 30, p=0.95, typing=True, raw=True, shielded=True)
    on, c2, *_ = engine()
    feed(on, "shield.state", mode="dsp")
    u_on = run(on, c2, 30, p=0.95, typing=True, raw=True, shielded=False)
    assert u_on["L"] == 0.0 and u_on["E"] > 0.85
    assert u_on["score"] < u_off["score"] and u_on["level"] != "CRITICAL"
    assert any("blocking" in r for r in u_on["reasons"])


def test_shield_failure_counts_as_off():
    eng, clock, *_ = engine()
    feed(eng, "shield.state", mode="dsp", failed=True)
    u = run(eng, clock, 30, p=0.95, typing=True, raw=True, shielded=False)
    assert u["level"] == "CRITICAL" and any("shield failed" in r for r in u["reasons"])


def test_chance_level_attacker_is_no_exposure():
    eng, *_ = engine()
    for i in range(20):
        feed(eng, "keys.readout", hit={"raw": i == 0}, chance=1 / 20)
    assert eng.tick()["E"] == 0.0


def test_voice_holds_in_silence_until_evidence_or_flush():
    eng, clock, *_ = engine()
    run(eng, clock, 30, p=0.95)
    V = eng.V
    assert eng.level == "WARN"
    assert run(eng, clock, 60)["level"] == "WARN" and eng.V == V      # no information: the verdict plateaus
    assert run(eng, clock, 30, p=0.05)["level"] == "SAFE"             # human-sounding speech brings it down
    run(eng, clock, 30, p=0.95)
    eng.flush_voice()                                                 # a new speaker
    assert eng.V == 0 and eng.tick()["level"] in ("SAFE", "WATCH")


def test_hysteresis_rule():
    eng, *_ = engine()
    eng.level = "WARN"
    assert eng._hysteresis(47) == "WARN"
    assert eng._hysteresis(44.9) == "WATCH"
    assert eng._hysteresis(75) == "CRITICAL"
    eng.level = "SAFE"
    assert eng._hysteresis(24.9) == "SAFE"
    assert eng._hysteresis(25) == "WATCH"


def test_tick_publishes_update():
    eng, clock, bus, _ = engine()
    got = []
    bus.subscribe("threat.update", lambda e: got.append(e.data))
    eng.tick()
    bus.flush()
    assert got and set(got[0]) >= {"score", "level", "V", "E", "L", "T", "reasons"}


def test_shield_change_restarts_residual_leak():
    eng, clock, *_ = engine()
    run(eng, clock, 5, typing=True, raw=True, shielded=True)       # shield off: shielded == raw
    assert eng.tick()["L"] > 0.5
    feed(eng, "shield.state", mode="dsp")
    assert eng.tick()["L"] == 0.0
