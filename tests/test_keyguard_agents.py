"""Live Ares-vs-Athena arms race (app/keystroke_guard/agents.py) and its pipeline wiring.

Offline: no LLM keys (llm falls back to rule-based), keyguard memory redirected to tmp_path. The real-model tests
need runs/keyguard/ctc_rich_ft.pt and data/keyguard/live_bank_rich.npz and use small craft/adapt steps.
"""
import sys
import threading
import time

import numpy as np
import pytest

from app.keystroke_guard.agents import AgentWorker, ArmsRace, Burst, demo_burst, load_net
from app.keystroke_guard.mock import MockAttacker, MockShield
from app.hearsay.mock import MockVoice
from app.source import config
from app.source.bus import EventBus
from app.source.pipeline import Pipeline, Scenario
from app.source.types import SR

REAL = (config.REPO / "runs/keyguard/ctc_rich_ft.pt").exists() and \
    (config.REPO / "data/keyguard/live_bank_rich.npz").exists()
LOG_KEYS = {"line", "secret", "decoy", "kind", "attacker", "defender", "backend", "clean_span_read",
            "smart_dict_clean", "smart_dict_defended", "rounds", "moves", "final_read", "source", "t_audio",
            "shield", "route", "agents", "protected", "seconds"}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    from keyguard import memory                     # keyguard.config load_dotenv()s on import: do it BEFORE delenv
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BACKBOARD_API_KEY", raising=False)
    monkeypatch.setattr(memory, "LOCAL", tmp_path / "arena_memory.jsonl")
    monkeypatch.setattr(memory, "ASSISTANT_FILE", tmp_path / "backboard_assistant.txt")
    return tmp_path


class FakeReader:
    """A population reader that always guesses A, B, C."""
    name, net = "fake", None

    def read(self, audio, onsets):
        return [["A", "B", "C"] for _ in onsets]


def _match(race_kw, offline):
    events = []
    net = load_net()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    race = ArmsRace(net, lambda topic, **d: events.append((topic, d)), device="cpu", steps=30, adapt_steps=10,
                    **race_kw)
    out = race.run(demo_burst("hey meet me at noon my password is hunter2 thanks"))
    return out, events, net, before


@pytest.mark.skipif(not REAL, reason="Keyguard weights / key bank not present")
def test_arms_race_on_demo_burst(offline):
    out, events, net, before = _match({"rounds": 2, "population": [FakeReader()]}, offline)
    assert LOG_KEYS <= set(out)
    assert out["secret"] == "HUNTER2"
    d = out["decoy"]
    assert len(d) == 7 and [c.isdigit() for c in d] == [c.isdigit() for c in "HUNTER2"] and d != "HUNTER2"
    assert len(out["rounds"]) == 2 and "after_retrain_read" in out["rounds"][0]
    assert set(out["agents"]) == {"ctc", "fake"} and out["agents"]["fake"] == {"before": "AAAAAAA", "after": "AAAAAAA"}
    assert out["protected"] in (True, False) and out["route"] == "none" and out["backend"] == "rule-based"

    titles = [(m["agent"], m["title"]) for m in out["moves"]]
    head = [("⚔️ ARES·ctc", "Acoustic read (no defense)"), ("⚔️ ARES·fake", "Acoustic read (no defense)"),
            ("⚔️ ARES", "Opening read (no defense)"), ("🦉 ATHENA", "Triage + deception plan (LLM)"),
            ("⚔️ ARES", "Smart-dictionary attack"), ("🦉 ATHENA", "Deploy shield (round 0)"),
            ("⚔️ ARES", "Evolve — retrain on the shielded audio (round 0)")]
    assert titles[:len(head)] == head
    assert titles[-4:] == [("🦉 ATHENA", "Deploy shield (round 1)"),
                           ("⚔️ ARES", "Smart-dictionary attack (under shield)"),
                           ("⚔️ ARES·fake", "Re-read under Athena's shield"), ("🏁 OUTCOME", "Final state")]
    moves = [d for t, d in events if t == "keyguard.move"]
    assert [(m["agent"], m["title"]) for m in moves] == titles       # streamed live, same order
    assert [t for t, _ in events][-1] == "keyguard.arms_race"

    assert all((net.state_dict()[k] == v).all() for k, v in before.items())   # live attacker untouched
    rec = (offline / "arena_memory.jsonl").read_text().strip().splitlines()
    assert len(rec) == 1 and '"kind": "arms_race"' in rec[0]


@pytest.mark.skipif(not REAL, reason="Keyguard weights / key bank not present")
def test_population_missing_runs_with_ctc_only(offline, monkeypatch):
    monkeypatch.setitem(sys.modules, "app.keystroke_guard.population", None)   # import fails
    out, _, _, _ = _match({"rounds": 1}, offline)
    assert set(out["agents"]) == {"ctc"} and out["secret"] == "HUNTER2"
    assert "population unavailable" in out["population"]


@pytest.mark.skipif(not REAL, reason="Keyguard weights / key bank not present")
def test_nothing_sensitive(offline):
    events = []
    race = ArmsRace(load_net(), lambda t, **d: events.append((t, d)), device="cpu", population=[])
    out = race.run(demo_burst("see you at noon"))
    assert out["protected"] is None and out["rounds"] == []
    assert out["moves"][-1]["agent"] == "🦉 ATHENA" and "Nothing sensitive" in out["moves"][-1]["result"]


# --- pipeline: burst detection ---------------------------------------------------------------------------------
class FakeWorker:
    def __init__(self):
        self.bursts, self.stopped = [], 0

    def submit(self, b):
        self.bursts.append(b)

    def stop(self):
        self.stopped += 1


def test_pipeline_cuts_two_bursts():
    cfg = config.load(env={})
    cfg.drivers.secret = "mock"
    bus = EventBus()
    pipe = Pipeline(cfg, bus, voice=MockVoice(), attacker=MockAttacker(), shield=MockShield())
    assert pipe.agents is None                                       # mock attacker: no matches
    pipe.agents = fake = FakeWorker()
    mic = (0.003 * np.random.default_rng(0).standard_normal(12 * SR)).astype(np.float32)
    typed = [(1.0, "a"), (1.3, "b"), (1.6, "space"), (1.9, "c"), (6.0, "x"), (6.25, "y"), (6.5, "7")]
    keys = [(int(t * SR), k) for t, k in typed]
    for s, _ in keys:
        mic[s:s + 200] += 0.3
    pipe.replay(Scenario("bursts", np.zeros_like(mic), mic, keys), realtime=False)
    assert [b.keys for b in fake.bursts] == ["AB C", "XY7"]
    for b, group in zip(fake.bursts, (keys[:4], keys[4:])):
        start = group[0][0] - SR // 2
        assert list(b.onsets) == [s - start for s, _ in group]
        assert len(b.audio) == group[-1][0] + SR // 2 - start
        assert np.allclose(b.audio, mic[start:group[-1][0] + SR // 2])   # raw mic, before the shield
        assert b.t_audio == round(group[0][0] / SR, 3)
    pipe.stop()
    assert fake.stopped == 1


def test_agents_off_in_config():
    cfg = config.load(env={"CALLGUARD_KEYGUARD_AGENTS": "0"})
    cfg.drivers.secret = "mock"

    class CTC(MockAttacker):
        pass
    CTC.__name__ = "KeyguardCTCAttacker"
    att = CTC()
    att.net = object()
    assert Pipeline(cfg, EventBus(), voice=MockVoice(), attacker=att, shield=MockShield()).agents is None
    cfg.keyguard.agents = True
    assert isinstance(Pipeline(cfg, EventBus(), voice=MockVoice(), attacker=att, shield=MockShield()).agents,
                      AgentWorker)


# --- config -----------------------------------------------------------------------------------------------------
def test_keyguard_config(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[keyguard]\nagents = false\nrounds = 3\nburst_gap_s = 1.5\nsnr_db = 12\ndevice = "cpu"\n')
    k = config.load(p, env={}).keyguard
    assert (k.agents, k.rounds, k.burst_gap_s, k.snr_db, k.device) == (False, 3, 1.5, 12.0, "cpu")
    assert config.load(env={}).keyguard.rounds == 2
    for bad in ("rounds = 0", "burst_gap_s = 0"):
        p.write_text(f"[keyguard]\n{bad}\n")
        with pytest.raises(ValueError):
            config.load(p, env={})


# --- worker -----------------------------------------------------------------------------------------------------
def test_worker_reports_errors_and_keeps_going():
    bus = EventBus()
    seen = []
    bus.subscribe("*", lambda e: seen.append((e.topic, e.data)))
    done = threading.Event()

    class Race:
        calls = 0

        def run(self, burst):
            Race.calls += 1
            if Race.calls == 1:
                raise RuntimeError("boom")
            done.set()
            return {"secret": burst.keys}

    w = AgentWorker(Race(), bus)
    b = Burst(np.zeros(SR, np.float32), np.array([100]), "A", 1.0, "dsp")
    w.submit(b)
    t0 = time.monotonic()
    while not any(t == "driver.error" for t, _ in seen) and time.monotonic() - t0 < 5:
        bus.flush()
        time.sleep(0.02)
    w.submit(b)
    assert done.wait(5)
    time.sleep(0.05)
    bus.flush()
    err = next(d for t, d in seen if t == "driver.error")
    assert err["driver"] == "keyguard agents" and err["kind"] == "agents" and "boom" in err["error"]
    assert err["quarantined"] is False
    assert [d for t, d in seen if t == "keyguard.burst"][0] == {"t_audio": 1.0, "n_keys": 1, "shield": "dsp"}
    assert w.latest == {"secret": "A"}
    w.stop()
