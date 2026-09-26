import json

import numpy as np

from app.source import config, hooks
from app.source.bus import EventBus
from app.source.config import HookConfig
from app.source.types import Event


def test_jsonl_sink_filters_topics_and_scrubs_keys(tmp_path):
    bus, path = EventBus(), tmp_path / "ev.jsonl"
    hooks.install(bus, [HookConfig("jsonl", topics=["threat.*", "keys.readout"], path=str(path))])
    bus.emit("threat.update", score=np.float32(80.5), level="CRITICAL")
    bus.emit("voice.verdict", p_synthetic=0.9)
    bus.emit("keys.readout", stream="raw", correct=True, truth="k", guess={"key": "k", "p": 0.8})
    assert bus.flush()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [x["topic"] for x in lines] == ["threat.update", "keys.readout"]
    assert lines[0]["data"]["score"] == 80.5
    assert "truth" not in lines[1]["data"] and lines[1]["data"]["guess"] == {"p": 0.8}


def test_webhook_retries_then_succeeds():
    calls = []

    class R:
        def __init__(self, code):
            self.status_code = code

    def post(url, content, headers, timeout):
        calls.append(json.loads(content))
        if len(calls) == 1:
            raise OSError("down")
        return R(200)

    sink = hooks.WebhookSink("http://x", retries=2, post=post)
    sink(Event("threat.level_change", {"to": "CRITICAL"}))
    sink._q.join()
    assert sink.sent == 1 and sink.failed == 0 and len(calls) == 2
    assert calls[0]["data"]["to"] == "CRITICAL"


def test_config_toml_and_env(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[drivers]\nvoice = "mock"\n[server]\nport = 9000\n[[hooks]]\nkind = "jsonl"\ntopics = ["*"]\n')
    cfg = config.load(p, env={"CALLGUARD_SERVER_PORT": "9100", "HEARSAY_ROOT": "C:/h",
                              "CALLGUARD_WEBHOOK_URL": "http://hook"})
    assert cfg.drivers.voice == "mock" and cfg.server.port == 9100
    assert str(cfg.hearsay_root).replace("\\", "/") == "C:/h"
    assert [h.kind for h in cfg.hooks] == ["jsonl", "webhook"]


def test_config_rejects_bad_driver(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[drivers]\nvoice = "fake"\n')
    try:
        config.load(p, env={})
    except ValueError as e:
        assert "drivers.voice" in str(e)
    else:
        raise AssertionError("bad driver accepted")


def test_example_toml_loads():
    cfg = config.load(config.REPO / "callguard.example.toml", env={})
    assert cfg.server.port == 8765 and cfg.threat.tick_hz >= 2
    assert cfg.hearsay_root.name == "Hearsay"
