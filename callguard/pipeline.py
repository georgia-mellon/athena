"""Pipeline: audio in -> shield -> rings -> voice / attacker workers -> bus -> ThreatEngine (plan 02 §1).

One code path for both modes. Live: MicShieldStream (mic -> shield -> VB-CABLE) + LoopbackStream (far end) +
the pynput KeyClock. Replay: a scenario's WAVs are pushed block by block through the same MicShieldStream.process
and far-end ring, with a ScriptedKeyClock. Downstream nothing knows which mode it is in.

The workers are step functions keyed on *audio* sample counters, not wall time:
- `voice_step`: every 2 s of far-end audio, if the last 4 s is mostly speech, score it -> voice.window/voice.verdict.
- `attack_step`: each key event -> keys.stroke; once 0.5 s of audio after it exists, run the attacker at the same
  onset on the raw ring and on the shielded ring (shifted by the shield's latency) -> keys.readout.
Live and realtime replay run them on worker threads; fast replay (tests) calls them inline after every block, which
makes a run deterministic. The audio thread only ever runs the shield; a raising driver is quarantined and the
block passes through (MicShieldStream + drivers.base.Quarantine).

Contracts reconciled here (HANDOFF.md): key events reach the shield as absolute sample indices; the attacker never
sees the truth (KeyGuess.truth is filled in after the call; only a mock with `wants_truth` gets it); driver errors
go to `bus.publish`; this object is the server's `controls` (set_shield, scenario).
"""
from __future__ import annotations

import csv
import logging
import threading
import time
import tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from callguard.audio.keys import KeyClock, ScriptedKeyClock
from callguard.audio.replay import FileSource
from callguard.audio.ring import Ring
from callguard.audio.streams import MicShieldStream
from callguard.audio.vad import speech_fraction
from callguard.bus import EventBus
from callguard.config import REPO, SHIELD_MODES, Config
from callguard.drivers.base import guard, make_attacker, make_shield, make_voice
from callguard.threat import ThreatEngine
from callguard.types import BLOCK, SR, Event

log = logging.getLogger(__name__)

VOICE_WIN = 4 * SR          # scored window (plan 01 F1)
VOICE_HOP = 2 * SR          # one verdict per 2 s of far-end audio
SPEECH_MIN = 0.5            # VAD speech fraction needed to score a window
ATTACK_CTX = SR // 2        # samples either side of an onset handed to the attacker
LATE = 4 * BLOCK            # OS key events up to 80 ms late still reach the shield
TICK = SR // 4              # fast replay: a threat tick every 0.25 s of audio
SCENARIOS = REPO / "demo" / "scenarios"
DEMO_AUDIO = REPO / "demo" / "audio"


@dataclass
class Scenario:
    name: str
    far: np.ndarray                             # far-end (meeting) audio, 16 kHz
    mic: np.ndarray                             # local mic audio, same length
    keys: list[tuple[int, str]]                 # (absolute sample, key)
    actions: list[tuple[float, dict]] = field(default_factory=list)  # (t_s, {"shield": mode}) scripted controls
    segments: list[dict] = field(default_factory=list)              # {"name", "start", "end", "voice"} for reports

    @property
    def seconds(self) -> float:
        return len(self.far) / SR


def load_scenario(name: str, root: Path = SCENARIOS, audio: Path = DEMO_AUDIO) -> Scenario:
    """demo/scenarios/<name>.toml; its far_end / mic / keys paths are relative to demo/audio."""
    spec = tomllib.loads((root / f"{name}.toml").read_text(encoding="utf-8"))
    missing = [spec[k] for k in ("far_end", "mic", "keys") if not (audio / spec[k]).exists()]
    if missing:
        raise FileNotFoundError(f"scenario {name!r} audio missing ({', '.join(missing)}); build it with "
                                f"`uv run python demo/build_scenario_audio.py`")
    far, mic = FileSource(audio / spec["far_end"]).audio, FileSource(audio / spec["mic"]).audio
    n = max(len(far), len(mic))
    far, mic = np.pad(far, (0, n - len(far))), np.pad(mic, (0, n - len(mic)))
    with open(audio / spec["keys"], newline="") as f:
        keys = [(round(float(r[0]) * SR), r[1].strip()) for r in csv.reader(f) if r and r[0][:1].isdigit()]
    actions = [(float(a.pop("t")), a) for a in spec.get("actions", [])]
    return Scenario(name, far, mic, keys, actions, spec.get("segments", []))


class Pipeline:
    """Owns the drivers (built once; models are slow to load) and one run's state (rebuilt by `reset`)."""

    def __init__(self, cfg: Config, bus: EventBus, voice: Any = None, attacker: Any = None, shield: Any = None):
        self.cfg, self.bus = cfg, bus
        self.voice = guard(voice or make_voice(cfg), self._on_error)
        self.attacker = guard(attacker or make_attacker(cfg), self._on_error)
        self.shield = guard(shield or make_shield(cfg), self._on_error)
        self.shield_mode = cfg.drivers.shield_mode
        self.mode = "idle"                      # idle | live | replay
        self._run_stop: threading.Event | None = None
        self._run_thread: threading.Thread | None = None
        self._live: list = []
        self.reset(ScriptedKeyClock([]), clock=lambda: 0.0)

    # --- state ------------------------------------------------------------------------------------------------
    def reset(self, keyclock: KeyClock, clock: Callable[[], float]) -> None:
        """Fresh rings, counters and ThreatEngine for a new run. Workers must be stopped."""
        d = self.cfg.devices
        self.keyclock, self.clock = keyclock, clock
        self.mic = MicShieldStream(self._hook, mic=d.mic or None, out=d.virtual_out or None, keyclock=keyclock)
        self.far = Ring(30.0)
        self.shield.reset()
        self._sent: set[int] = set()            # key samples already given to the shield
        self._seen: set = set()                 # key events already published as keys.stroke
        self._pending: deque = deque()          # key events waiting for their post-onset audio
        self._next_voice = VOICE_WIN
        self._readout = {"raw": deque(maxlen=self.cfg.threat.readout_window),
                         "shielded": deque(maxlen=self.cfg.threat.readout_window)}
        if getattr(self, "engine", None):
            self.engine.stop()
        self.engine = ThreatEngine(self.bus, self.cfg.threat, now=clock)
        self._publish_shield()

    def _on_error(self, ev: Event) -> None:
        self.bus.publish(ev)
        if ev.data.get("kind") == "shield" and ev.data.get("quarantined"):
            self._publish_shield(failed=True)

    def _publish_shield(self, failed: bool = False) -> None:
        self.bus.emit("shield.state", mode=self.shield_mode, failed=failed or self.shield.quarantined,
                      driver=self.shield.name, latency_ms=round(self._shield_lag() / SR * 1000, 1))

    def _shield_lag(self) -> int:
        return 0 if self.shield.quarantined else int(getattr(self.shield, "latency", 0) or 0)

    # --- audio thread -----------------------------------------------------------------------------------------
    def _hook(self, block: np.ndarray, start: int) -> np.ndarray:
        """Runs on the audio thread: new key events (absolute samples, each once) -> shield. Mode 'off' still runs
        the shield with no events, so its delay stays constant and switching modes doesn't jump the stream."""
        evs: list[int] = []
        if self.shield_mode != "off":
            evs = [s for s in self.keyclock.in_range(start - LATE, start + len(block)) if s not in self._sent]
            self._sent.update(evs)
        return self.shield.process(block, evs)

    # --- workers ----------------------------------------------------------------------------------------------
    def voice_step(self) -> bool:
        """Score the next due far-end window. True if a window was due (scored or skipped as silence)."""
        total = self.far.total
        if total < self._next_voice:
            return False
        if total - self._next_voice > VOICE_HOP:        # fell behind (slow model): jump to the newest window
            self._next_voice = total
        stop = self._next_voice
        self._next_voice += VOICE_HOP
        x = self.far.read_range(stop - VOICE_WIN, stop)
        if x is None:
            return True
        speech = speech_fraction(x)
        info = {"t_audio": round(stop / SR, 2), "speech": round(speech, 2)}
        if speech < SPEECH_MIN:
            self.bus.emit("voice.window", scored=False, **info)
            return True
        v = self.voice.score(x)
        if v is None:                                   # driver failed; driver.error already published
            return True
        out = dict(info, scored=True, p_synthetic=round(float(v.p_synthetic), 4), margin=round(float(v.margin), 3),
                   threshold=v.threshold, latency_ms=round(v.latency_ms, 1), driver=self.voice.name)
        self.bus.emit("voice.window", **out)
        self.bus.emit("voice.verdict", **out)
        return True

    def attack_step(self) -> bool:
        """Publish new key events, attack the ones whose audio is complete. True if anything happened."""
        pos, did = self.mic.position, False
        for e in self.keyclock.between(max(0, pos - 2 * SR), pos):
            if e not in self._seen:
                self._seen.add(e)
                self._pending.append(e)
                self.bus.emit("keys.stroke", t_audio=round(e.sample / SR, 3))
                did = True
        lag = self._shield_lag()
        while self._pending and self._pending[0].sample + ATTACK_CTX + lag <= self.mic.shielded.total:
            self._attack(self._pending.popleft(), lag)
            did = True
        return did

    def _attack(self, e, lag: int) -> None:
        classes = list(self.attacker.classes)
        truth = str(e.key).upper()
        if truth not in classes:                        # space, shift, ...: nothing to score
            return
        raw = self.mic.raw.read_range(e.sample - ATTACK_CTX, e.sample + ATTACK_CTX)
        shd = self.mic.shielded.read_range(e.sample + lag - ATTACK_CTX, e.sample + lag + ATTACK_CTX)
        if raw is None or shd is None:                  # overwritten (we fell far behind): skip
            return
        kw = {"truths": [truth]} if getattr(self.attacker, "wants_truth", False) else {}
        hit = {}
        for stream, audio in (("raw", raw), ("shielded", shd)):
            guesses = self.attacker.read(audio, np.array([ATTACK_CTX]), **kw)
            if not guesses:                             # driver failed
                return
            g = guesses[0]
            g.truth = truth
            top1, p = g.top[0]
            hit[stream] = truth in [k for k, _ in g.top[:3]]   # exposure = true key in the top 3 (see threat.py)
            self._readout[stream].append({"top1": top1, "p": round(float(p), 3), "truth": truth, "hit": hit[stream]})
        acc = {s: float(np.mean([r["hit"] for r in q])) for s, q in self._readout.items()}
        self.bus.emit("keys.readout", hit=hit, k=len(classes), chance=min(3, len(classes)) / len(classes),
                      raw=list(self._readout["raw"]), shielded=list(self._readout["shielded"]),
                      acc_raw=acc["raw"], acc_shielded=acc["shielded"], driver=self.attacker.name)

    def _workers(self, stop: threading.Event) -> list[threading.Thread]:
        def loop(step):
            while not stop.is_set():
                try:
                    busy = step()
                except Exception:                       # noqa: BLE001 - a worker bug must not kill the run
                    log.exception("worker %s failed", step.__name__)
                    busy = False
                if not busy:
                    stop.wait(0.02)
        ts = [threading.Thread(target=loop, args=(f,), name=f"callguard-{f.__name__}", daemon=True)
              for f in (self.voice_step, self.attack_step)]
        for t in ts:
            t.start()
        return ts

    # --- replay -----------------------------------------------------------------------------------------------
    def replay(self, sc: Scenario, realtime: bool = True, stop: threading.Event | None = None,
               play: bool = False) -> None:
        """Run a scenario through the pipeline. realtime=False: as fast as possible, workers inline (tests).
        play=True also plays far end + shielded mic on the default speaker (paces the run)."""
        stop = stop or threading.Event()
        kc = ScriptedKeyClock(sc.keys)
        self.reset(kc, clock=lambda: self.mic.position / SR)
        self.mode = "replay"
        actions = sorted(sc.actions, key=lambda a: a[0])
        out = _open_speaker() if (play and realtime) else None
        threads = []
        if realtime:
            threads = self._workers(stop)
            self.engine.start()
        t0 = time.monotonic()
        try:
            for i, s in enumerate(range(0, len(sc.far), BLOCK)):
                if stop.is_set():
                    break
                while actions and actions[0][0] * SR <= s:
                    self._apply(actions.pop(0)[1])
                far = _block(sc.far, s)
                self.far.write(far)
                kc.anchor(self.mic.position + BLOCK)
                y = self.mic.process(_block(sc.mic, s))
                if out is not None:
                    out.write(np.clip(far + y, -1, 1).astype(np.float32)[:, None])
                elif realtime:
                    wait = t0 + (i + 1) * BLOCK / SR - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                else:
                    while self.voice_step():
                        pass
                    self.attack_step()
                    if (s // BLOCK) % (TICK // BLOCK) == 0:
                        self.bus.flush()                # the engine must see this tick's events first
                        self.engine.tick()
            if not realtime:                            # drain the tail
                self.attack_step()
                self.bus.flush()
                self.engine.tick()
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=10)
            if out is not None:
                out.stop()
                out.close()
            self.engine.stop()
            self.mode = "idle"

    def _apply(self, action: dict) -> None:
        if "shield" in action:
            self.set_shield(action["shield"])

    # --- live -------------------------------------------------------------------------------------------------
    def start_live(self) -> None:
        from callguard.audio.streams import LoopbackStream
        d = self.cfg.devices
        kc = KeyClock(offset_s=d.key_offset_s)
        self.reset(kc, clock=time.monotonic)
        self.mode = "live"
        self._run_stop = threading.Event()
        self.mic.start()
        loop = LoopbackStream(d.loopback or None).start()
        self.far = loop.ring
        kc.start()
        self._live = [self.mic, loop, kc]
        self._workers(self._run_stop)
        self.engine.start()

    def stop(self) -> None:
        if self._run_stop is not None:
            self._run_stop.set()
        if self._run_thread is not None:
            self._run_thread.join(timeout=10)
            self._run_thread = None
        for x in self._live:
            x.stop()
        self._live = []
        self.engine.stop()

    # --- controls (the server's `controls` object) ------------------------------------------------------------
    def set_shield(self, mode: str) -> str:
        if mode not in SHIELD_MODES:
            raise ValueError(f"shield mode must be one of {SHIELD_MODES}")
        if mode == "adversarial":
            raise ValueError("adversarial shield not available yet (waits for Keyguard's streaming D); use dsp")
        if mode != self.shield_mode:
            self._readout["shielded"].clear()         # the shielded readout restarts with the new mode
        self.shield_mode = mode
        self.bus.publish(Event("control.shield", {"mode": mode}))
        self._publish_shield()
        return mode

    def scenario(self, action: str, name: str = "ai_caller", on_end: Callable[[], None] | None = None,
                 play: bool = True) -> str:
        """Start/stop a realtime replay on a background thread (dashboard button, CLI)."""
        if self.mode == "live":
            raise ValueError("scenarios run in replay mode (callguard run --mode replay)")
        if action == "stop":
            if self._run_stop is not None:
                self._run_stop.set()
            return "stopped"
        if action != "start":
            raise ValueError(f"unknown scenario action {action!r}")
        sc = load_scenario(name)
        self.stop()                                     # a running replay restarts from the top
        self._run_stop = stop = threading.Event()
        self.bus.publish(Event("control.scenario", {"action": "start", "name": name, "seconds": sc.seconds}))

        def run():
            try:
                self.replay(sc, realtime=True, stop=stop, play=play)
            except Exception as e:                      # noqa: BLE001
                log.exception("scenario %s failed", name)
                self.bus.emit("driver.error", driver="scenario", kind="replay", error=repr(e))
            self.bus.publish(Event("control.scenario", {"action": "end", "name": name}))
            if on_end is not None:
                on_end()
        self._run_thread = threading.Thread(target=run, name="callguard-replay", daemon=True)
        self._run_thread.start()
        return f"started {name} ({sc.seconds:.0f} s)"


def _block(x: np.ndarray, s: int) -> np.ndarray:
    b = x[s:s + BLOCK]
    return b if len(b) == BLOCK else np.pad(b, (0, BLOCK - len(b)))


def _open_speaker():
    """Default output device for replay playback, or None (then the run is paced by sleeping)."""
    try:
        import sounddevice as sd
        out = sd.OutputStream(samplerate=SR, channels=1, blocksize=BLOCK, dtype="float32")
        out.start()
        return out
    except Exception as e:  # noqa: BLE001 - no speaker: run silently
        log.warning("replay playback unavailable (%s); running silently", e)
        return None
