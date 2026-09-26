"""Driver registry and the quarantine wrapper (plan 02 §3, spec F8).

Drivers are picked by config, never by code edits. Config is a plain dict or an object with attributes; driver keys
are read from a ``drivers`` section first, then from the top level:

    voice     = "real" | "mock"   -> real: callguard.drivers.hearsay_real.HearsayDriver(mode=voice_mode, threads=threads)
    attacker  = "real" | "mock"   -> real: callguard.drivers.keyguard_real.KeyguardAttacker(weights=attacker_weights)
    shield    = "real" | "mock"   -> real: callguard.drivers.keyguard_real.KeyguardShield(mode=shield_mode)
    hearsay_mode (alias voice_mode) = "r4ft" | "r5" (default "r4ft"), threads = 4, device = "auto",
    attacker_weights = None, shield_mode = "dsp" ("off" is a runtime switch, so it still builds the dsp shield),
    mock_latency_ms = 0.0 (MockVoice sleep, to mimic the real timing profile)

Real drivers are imported lazily so a missing upstream repo only fails when "real" is actually asked for.
"""
from __future__ import annotations

import importlib
import time
import traceback
from typing import Any, Callable

import numpy as np

from callguard.types import Event, KeyGuess, VoiceScore

_MISSING = object()


def _opt(cfg: Any, key: str, default: Any = None) -> Any:
    """Look up `key` in cfg.drivers, then in cfg; dicts and attribute objects both work."""
    def get(obj: Any, k: str) -> Any:
        if obj is None:
            return _MISSING
        if isinstance(obj, dict):
            return obj.get(k, _MISSING)
        return getattr(obj, k, _MISSING)

    for scope in (get(cfg, "drivers"), cfg):
        v = get(scope, key) if scope is not _MISSING else _MISSING
        if v is not _MISSING and v is not None:
            return v
    return default


def _real(module: str, cls: str, **kwargs: Any) -> Any:
    return getattr(importlib.import_module(f"callguard.drivers.{module}"), cls)(**kwargs)


def _kind(cfg: Any, key: str) -> str:
    kind = str(_opt(cfg, key, "mock")).lower()
    if kind not in ("real", "mock"):
        raise ValueError(f"drivers.{key} must be 'real' or 'mock', got {kind!r}")
    return kind


def make_voice(cfg: Any = None):
    if _kind(cfg, "voice") == "real":
        mode = _opt(cfg, "hearsay_mode", _opt(cfg, "voice_mode", "r4ft"))
        return _real("hearsay_real", "HearsayDriver", mode=mode, threads=int(_opt(cfg, "threads", 4)),
                     device=_opt(cfg, "device", "auto"))
    from callguard.drivers.mock import MockVoice
    return MockVoice(latency_ms=float(_opt(cfg, "mock_latency_ms", 0.0)))


def make_attacker(cfg: Any = None):
    if _kind(cfg, "attacker") == "real":
        return _real("keyguard_real", "KeyguardAttacker", weights=_opt(cfg, "attacker_weights") or None)
    from callguard.drivers.mock import MockAttacker
    return MockAttacker()


def make_shield(cfg: Any = None):
    if _kind(cfg, "shield") == "real":
        mode = _opt(cfg, "shield_mode", "dsp")
        return _real("keyguard_real", "KeyguardShield", mode="dsp" if mode == "off" else mode)
    from callguard.drivers.mock import MockShield
    return MockShield()


def make_spotter(cfg: Any = None, mode: str = "outbound"):
    """Spoken-secret spotter (plan 06). Real = Vosk; raises FileNotFoundError when its model isn't downloaded."""
    if _kind(cfg, "secret") == "real":
        return _real("secret_vosk", "VoskSpotter", mode=mode)
    from callguard.drivers.mock import MockSpotter
    return MockSpotter(mode=mode)


class Quarantine:
    """Runs a driver's calls with timing; a raising driver can't take the pipeline down.

    Each failure publishes ``driver.error`` through `on_error(Event)`. After `max_failures` consecutive failures the
    driver is quarantined: it's no longer called and every call returns the safe fallback at once (voice: None,
    attacker: [], shield: the input block unchanged). A success resets the streak. Attributes other than the wrapped
    methods (name, classes, sample_rate, ...) pass through, so the wrapper still satisfies the driver Protocol.
    It can't pre-empt a call that hangs: slow models belong on worker threads, not the audio thread.
    """
    kind = "driver"

    def __init__(self, driver: Any, on_error: Callable[[Event], None] | None = None, max_failures: int = 3):
        self.driver, self.on_error, self.max_failures = driver, on_error, max_failures
        self.failures = 0
        self.quarantined = False
        self.last_latency_ms = 0.0
        # isinstance() on a runtime_checkable Protocol looks attributes up statically (3.12), so copy them here.
        for attr in ("name", "sample_rate", "classes"):
            if hasattr(driver, attr):
                setattr(self, attr, getattr(driver, attr))

    def __getattr__(self, item: str) -> Any:  # only reached for attributes not set on the wrapper
        return getattr(self.__dict__["driver"], item)

    def _call(self, method: str, fallback: Any, *args: Any, **kw: Any) -> Any:
        if self.quarantined:
            return fallback
        t0 = time.perf_counter()
        try:
            out = getattr(self.driver, method)(*args, **kw)
        except Exception as e:  # noqa: BLE001 - any driver failure must be contained
            self.failures += 1
            self.quarantined = self.failures >= self.max_failures
            self._publish(method, e)
            return fallback
        finally:
            self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.failures = 0
        return out

    def _publish(self, method: str, e: Exception) -> None:
        if self.on_error is None:
            return
        data = {"driver": getattr(self.driver, "name", type(self.driver).__name__), "kind": self.kind,
                "method": method, "error": f"{type(e).__name__}: {e}", "failures": self.failures,
                "quarantined": self.quarantined, "trace": traceback.format_exc(limit=3)}
        try:
            self.on_error(Event("driver.error", data))
        except Exception:  # noqa: BLE001 - a broken sink must not break the audio path
            pass


class QuarantinedVoice(Quarantine):
    kind = "voice"

    def score(self, audio: np.ndarray) -> VoiceScore | None:
        return self._call("score", None, audio)


class QuarantinedAttacker(Quarantine):
    kind = "attacker"

    def read(self, audio: np.ndarray, onsets: np.ndarray, **kw: Any) -> list[KeyGuess]:
        return self._call("read", [], audio, onsets, **kw)


class QuarantinedShield(Quarantine):
    kind = "shield"

    def process(self, block: np.ndarray, key_events: list[int]) -> np.ndarray:
        return self._call("process", block, block, key_events)

    def reset(self) -> None:
        self._call("reset", None)


class QuarantinedSpotter(Quarantine):
    kind = "secret"

    def feed(self, block: np.ndarray, start: int) -> list:
        return self._call("feed", [], block, start)

    def reset(self) -> None:
        self._call("reset", None)


def guard(driver: Any, on_error: Callable[[Event], None] | None = None, max_failures: int = 3) -> Quarantine:
    """Wrap a driver in the Quarantine subclass matching the Protocol it implements."""
    from callguard.types import KeystrokeAttackerDriver, SecretSpotterDriver, ShieldDriver, VoiceAuthenticityDriver
    for proto, cls in ((VoiceAuthenticityDriver, QuarantinedVoice), (KeystrokeAttackerDriver, QuarantinedAttacker),
                       (SecretSpotterDriver, QuarantinedSpotter), (ShieldDriver, QuarantinedShield)):
        if isinstance(driver, proto):
            return cls(driver, on_error, max_failures)
    raise TypeError(f"{type(driver).__name__} implements no driver Protocol")
