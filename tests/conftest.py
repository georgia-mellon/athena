import pytest


@pytest.fixture(autouse=True)
def _caller_audio_from_the_tab(monkeypatch):
    """Tests feed the caller's audio through /meet/far (the extension's tab path) and must never record this machine's
    speakers: every loaded config uses meet_far_source = "tab" (the app's default is "system")."""
    from app.source import config
    real = config.load

    def load(*a, **k):
        cfg = real(*a, **k)
        cfg.devices.meet_far_source = "tab"
        return cfg
    monkeypatch.setattr(config, "load", load)
