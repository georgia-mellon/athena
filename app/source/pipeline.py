"""Pipeline: audio in -> shield -> rings -> voice / attacker workers -> bus -> ThreatEngine (plan 02 §1).

One code path for all modes. Live: MicShieldStream (mic -> shield -> VB-CABLE) + LoopbackStream (far end) +
the pynput KeyClock. Meet: the browser bridge (app/source/connectors/meet) pushes the page's mic blocks through
`meet_mic` (the same MicShieldStream.process) and the remote participants' audio into `meet_far`. Replay: a scenario's WAVs are pushed block by block through the same MicShieldStream.process
and far-end ring, with a ScriptedKeyClock. Downstream nothing knows which mode it is in.

The workers are step functions keyed on *audio* sample counters, not wall time:
- `voice_step`: every 2 s of far-end audio, if the last 4 s is mostly speech, score it -> voice.window/voice.verdict.
- `attack_step`: each key event -> keys.stroke; once 0.5 s of audio after it exists, run the attacker at the same
  onset on the raw ring and on the shielded ring (shifted by the output latency) -> keys.readout.
- `secret_step` (plan 06): while armed (unverified voice, an inbound "read me the code" trigger, or manual), feed the
  outbound audio (after the Keyguard shield, before the delay line) to the spotter and mark its spans on the
  Redactor; the audio thread applies them as the samples leave the constant delay line -> secret.blocked.
- keyguard bursts: attack_step also collects the typed keys into a burst; after `keyguard.burst_gap_s` of silence
  the raw mic clip around it goes to the AgentWorker (app/keystroke_guard/agents.py), which plays one
  Ares-vs-Athena match on its own thread (keyguard.* events). Only with the real CTC attacker.
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

from app.source.audio.keys import KeyClock, ScriptedKeyClock
from app.secret_shield.redactor import Redactor
from app.source.audio.replay import FileSource
from app.source.audio.ring import Ring
from app.source.audio.streams import MicShieldStream
from app.source.audio.vad import speech_fraction
from app.source.bus import EventBus
from app.source.config import REPO, SHIELD_MODES, Config
from app.source.registry import guard, make_attacker, make_shield, make_spotter, make_voice
from app.source.threat import ThreatEngine
from app.source.types import BLOCK, SR, Event

log = logging.getLogger(__name__)

VOICE_WIN = 4 * SR          # scored window (plan 01 F1)
VOICE_HOP = 2 * SR          # one verdict per 2 s of far-end audio
SPEECH_MIN = 0.5            # VAD speech fraction needed to score a window
SPEECH_DB = -45.0           # default VAD gate: a 20 ms frame counts as speech above this level (dashboard slider)
PAGE_STALE_S = 3.0          # a /meet/status page that hasn't reported for this long is gone
LEVEL_HZ = 5                # audio.level (the dashboard's meters)


def dbfs(x: np.ndarray) -> float:
    if len(x) == 0:
        return -90.0
    return round(max(-90.0, 10 * float(np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12))), 1)
ATTACK_CTX = SR // 2        # samples before an onset handed to the attacker
ATTACK_POST = int(0.3 * SR) # ... and after it: covers Keyguard's KEY_WIN (0.3 s from onset - 20 ms)
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


class ArrivalAnchor:
    """Meet mode's KeyClock anchor: when was stream sample `s` captured, given when its block arrived? arrival -
    s / SR = a constant (capture + one-way latency) + positive jitter (socket bursts, page scheduling). Its lower
    envelope over the last `window_s` rejects the jitter and still follows slow clock drift. The constant part left
    over is calibrated with devices.meet_key_offset_s."""

    def __init__(self, window_s: float = 3.0):
        self.window_s = window_s
        self._q: deque[tuple[float, float]] = deque()   # (arrival, offset), offsets increasing: [0] is the min

    def reset(self) -> None:
        self._q.clear()

    def __call__(self, sample: int, arrival: float) -> float:
        """Monotonic time at which `sample` was (least-delayed estimate) captured."""
        off, q = arrival - sample / SR, self._q
        while q and q[-1][1] >= off:
            q.pop()
        q.append((arrival, off))
        while q[0][0] < arrival - self.window_s:
            q.popleft()
        return sample / SR + q[0][1]


class Pipeline:
    """Owns the drivers (built once; models are slow to load) and one run's state (rebuilt by `reset`)."""

    def __init__(self, cfg: Config, bus: EventBus, voice: Any = None, attacker: Any = None, shield: Any = None,
                 spotter: Any = None, spotter_in: Any = None):
        self.cfg, self.bus = cfg, bus
        self.voice = guard(voice or make_voice(cfg), self._on_error)
        self.attacker = guard(attacker or make_attacker(cfg), self._on_error)
        self.shield = guard(shield or make_shield(cfg), self._on_error)
        sc, self.secret_error = cfg.secret, None
        self.spotter = self.spotter_in = self.redactor = None
        if sc.enabled:
            try:
                self.spotter = guard(spotter or make_spotter(cfg, "outbound"), self._on_error)
                if sc.arm_on_request:
                    self.spotter_in = guard(spotter_in or make_spotter(cfg, "inbound"), self._on_error)
                self.redactor = Redactor(round(sc.delay_ms / 1000 * SR), sc.style)
            except Exception as e:                      # noqa: BLE001 - no model: the feature is off, audio is not
                self.spotter = self.spotter_in = None
                self.secret_error = f"{type(e).__name__}: {e}"
                log.warning("secret shield disabled: %s", self.secret_error)
        self.shield_mode = cfg.drivers.shield_mode
        try:
            self._driver_mode(self.shield_mode)
        except ValueError as e:                         # e.g. deltas not trained: start on dsp, say why
            log.warning("%s; starting with shield mode dsp", e)
            self.bus.emit("driver.error", driver=self.shield.name, kind="shield", error=str(e))
            self.shield_mode = "dsp"
        self.mode = "idle"                     # idle | live | replay | meet
        self.meet_session = None                # launcher.MeetSession while a Meet window is open
        self.meet_port: int | None = None       # the server the bridge talks to (None: cfg.server.port)
        self.meet_owners: dict[str, str | None] = {"mic": None, "far": None}   # origin of each /meet stream's page
        self.meet_pages: dict[int, tuple[float, str | None, dict]] = {}   # /meet/status: id -> (time, origin, status)
        self.meet_command: Callable[[dict], bool] | None = None        # set by the router: a command to the page
        self.speech_db = SPEECH_DB
        self.voice_source = "far"               # far = the caller (default) | mic = your own mic (solo tests)
        self.far_source = cfg.devices.meet_far_source   # meet mode: system (speaker loopback) | tab (the extension)
        self.far_error: str | None = None       # the system-audio capture failed: why (dashboard)
        self._call_phase: str | None = None     # meet.call: none | open | call | local, from the extension's reports
        self.ready = False                      # drivers loaded and the voice model warmed up (system.state)
        self._run_stop: threading.Event | None = None
        self._run_thread: threading.Thread | None = None
        self._live: list = []
        self.agents = self._make_agents()
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
        self._pending: deque = deque()          # (key event, shield mode when typed) waiting for its audio
        self._clear_shielded = False
        self._mic_errors = 0
        self._next_voice = VOICE_WIN
        self._last_speech: int | None = None    # far-end sample where the last scored speech window ended
        self._burst: list[tuple[int, str]] = []  # (absolute sample, key char) of the current typing burst
        self._readout = {"raw": deque(maxlen=self.cfg.threat.readout_window),
                         "shielded": deque(maxlen=self.cfg.threat.readout_window)}
        if getattr(self, "engine", None):
            self.engine.stop()
        self.engine = ThreatEngine(self.bus, self.cfg.threat, now=clock)
        self.outbound = Ring(30.0)              # shield output before the delay line: what the spotter hears
        for x in (self.redactor, self.spotter, self.spotter_in):
            if x is not None:
                x.reset()
        self._out_cursor = self._far_cursor = 0
        self._last_request = -10 * SR
        self._keep_until = self._bypass_until = -1e9
        self._manual: bool | None = None
        self.armed, self.armed_by, self._auto_by, self._run = False, "", "", None
        self._publish_shield()
        self._publish_secret()

    def _make_agents(self):
        """AgentWorker for the live arms race, or None (off in config, or not the real CTC attacker)."""
        k = self.cfg.keyguard
        drv = getattr(self.attacker, "driver", self.attacker)   # unwrap the Quarantine
        net = getattr(drv, "net", None)
        if not k.agents or net is None or type(drv).__name__ != "KeyguardCTCAttacker":
            return None
        try:
            from app.keystroke_guard.agents import AgentWorker, ArmsRace
            return AgentWorker(ArmsRace(net, self.bus.emit, rounds=k.rounds, snr_db=k.snr_db, steps=k.steps,
                                        device=k.device), self.bus)
        except Exception as e:                          # noqa: BLE001 - the matches are extra; the call is not
            log.warning("keyguard agents unavailable: %s", e)
            return None

    def _on_error(self, ev: Event) -> None:
        self.bus.publish(ev)
        if ev.data.get("kind") == "shield" and ev.data.get("quarantined"):
            self._publish_shield(failed=True)
        if ev.data.get("kind") == "secret" and ev.data.get("quarantined"):
            self.secret_error = f"spotter failed: {ev.data.get('error')}"   # fails open: the delay line just passes
            self._publish_secret()

    def _publish_shield(self, failed: bool = False) -> None:
        self.bus.emit("shield.state", mode=self.shield_mode, failed=failed or self.shield.quarantined,
                      driver=self.shield.name, latency_ms=round(self._shield_lag() / SR * 1000, 1))

    def _shield_lag(self) -> int:
        """Output delay of the shielded stream vs the raw mic: Keyguard shield lookahead + the secret delay line."""
        lag = 0 if self.shield.quarantined else int(getattr(self.shield, "latency", 0) or 0)
        return lag + (self.redactor.latency if self._secret_on() else 0)

    def _secret_on(self) -> bool:
        return self.redactor is not None and self.spotter is not None

    # --- audio thread -----------------------------------------------------------------------------------------
    def _hook(self, block: np.ndarray, start: int) -> np.ndarray:
        """Runs on the audio thread: new key events (absolute samples, each once) -> shield. Mode 'off' still runs
        the shield with no events, so its delay stays constant and switching modes doesn't jump the stream."""
        evs: list[int] = []
        if self.shield_mode != "off":
            evs = [s for s in self.keyclock.in_range(start - LATE, start + len(block)) if s not in self._sent]
            self._sent.update(evs)
        y = self.shield.process(block, evs)
        if self._secret_on():                           # constant delay line whenever the feature is on
            self.outbound.write(y)
            y = self.redactor.process(y, start)
        return y

    # --- workers ----------------------------------------------------------------------------------------------
    def _voice_ring(self) -> Ring:
        return self.mic.raw if self.voice_source == "mic" else self.far

    def voice_step(self) -> bool:
        """Score the next due window of the judged stream (the caller, or your mic). True if one was due."""
        ring = self._voice_ring()
        total = ring.total
        if total < self._next_voice:
            return False
        if total - self._next_voice > VOICE_HOP:        # fell behind (slow model): jump to the newest window
            self._next_voice = total
        stop = self._next_voice
        self._next_voice += VOICE_HOP
        x = ring.read_range(stop - VOICE_WIN, stop)
        if x is None:
            return True
        speech = speech_fraction(x, energy_db=self.speech_db)
        info = {"t_audio": round(stop / SR, 2), "speech": round(speech, 2)}
        if speech < SPEECH_MIN:
            self.bus.emit("voice.window", scored=False, **info)
            return True
        gap = self.cfg.threat.new_speaker_gap_s
        if gap and self._last_speech is not None and stop - VOICE_WIN - self._last_speech >= gap * SR:
            self._flush("gap")                          # a long silence, then speech: treat it as a new speaker
        self._last_speech = stop
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
                self._pending.append((e, self.shield_mode))
                self.bus.emit("keys.stroke", t_audio=round(e.sample / SR, 3))
                did = True
        lag = self._shield_lag()
        while self._pending and self._pending[0][0].sample + ATTACK_POST + lag <= self.mic.shielded.total:
            e, mode = self._pending.popleft()
            if self._clear_shielded:                    # the shield mode changed: its readout restarts
                self._readout["shielded"].clear()
                self._clear_shielded = False
            if mode == self.shield_mode:                # typed under another mode: not evidence for this one
                self._attack(e, lag)
            self._track_burst(e)
            did = True
        self._flush_burst()
        if self.mic.errors != self._mic_errors:        # the hook failed and passed raw audio: say so
            self._mic_errors = self.mic.errors
            self.bus.emit("driver.error", driver="audio hook", kind="stream", error=self.mic.last_error,
                          failures=self.mic.errors, quarantined=False)
        return did

    def _track_burst(self, e) -> None:
        if self.agents is None:
            return
        k = str(e.key)
        ch = " " if k.lower() == "space" else k.upper()
        if len(ch) == 1 and (ch.isalnum() and ch.isascii() or ch == " "):
            self._burst.append((e.sample, ch))

    def _flush_burst(self) -> None:
        """Hand the finished burst (burst_gap_s of silence, its tail in the raw ring) to the agents. Never waits."""
        b = self._burst
        if not b or self.agents is None:
            return
        last = b[-1][0]
        if self.mic.position - last < self.cfg.keyguard.burst_gap_s * SR or self.mic.raw.total < last + SR // 2:
            return
        self._burst = []
        start = max(0, b[0][0] - SR // 2)
        audio = self.mic.raw.read_range(start, last + SR // 2)
        if audio is None or not "".join(c for _, c in b).strip():   # overwritten (burst too long) / only spaces
            return
        from app.keystroke_guard.agents import Burst
        self.agents.submit(Burst(audio, np.array([s - start for s, _ in b]), "".join(c for _, c in b),
                                 round(b[0][0] / SR, 3), self.shield_mode))

    def _attack(self, e, lag: int) -> None:
        classes = list(self.attacker.classes)
        truth = str(e.key).upper()
        if truth not in classes:                        # space, shift, ...: nothing to score
            return
        raw = self.mic.raw.read_range(e.sample - ATTACK_CTX, e.sample + ATTACK_POST)
        shd = self.mic.shielded.read_range(e.sample + lag - ATTACK_CTX, e.sample + lag + ATTACK_POST)
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
            self._readout[stream].append({"top1": top1, "p": round(float(p), 3), "exact": top1 == truth,
                                          "hit": hit[stream]})   # no truth: typed keys never leave the pipeline
        acc = {s: float(np.mean([r["hit"] for r in q])) for s, q in self._readout.items()}
        self.bus.emit("keys.readout", hit=hit, k=len(classes), chance=min(3, len(classes)) / len(classes),
                      raw=list(self._readout["raw"]), shielded=list(self._readout["shielded"]),
                      acc_raw=acc["raw"], acc_shielded=acc["shielded"], driver=self.attacker.name)

    # --- spoken-secret shield (plan 06) -----------------------------------------------------------------------
    def secret_step(self) -> bool:
        """Arm/disarm, listen for inbound triggers, spot outbound secrets and mark them. True if work was done."""
        if not self._secret_on():
            return False
        c, now, did = self.cfg.secret, self.clock(), False
        V = self.engine.V
        if c.arm_on_voice and (V >= c.arm_voice or (self.armed and V >= c.keep_voice)):
            self._auto_by = self._auto_by if now < self._keep_until else "voice"
            self._keep_until = max(self._keep_until, now + c.disarm_after_s)
        armed = self._manual if self._manual is not None else now < self._keep_until
        by = "manual" if self._manual else (self._auto_by if armed else "")
        if armed != self.armed or by != self.armed_by:
            if armed and not self.armed:                # fresh listen: nothing from before arming
                self.spotter.reset()
                self._out_cursor = self.outbound.total
                self._run = None
            self.armed, self.armed_by = armed, by
            self._publish_secret()
        if armed:
            did |= self._feed(self.spotter, self.outbound, "_out_cursor", self._on_span)
        else:
            self._out_cursor = self.outbound.total      # not listening while disarmed (plan 06 section 8)
        self._close_run(self._out_cursor)
        return did

    def request_step(self) -> bool:
        """Listen to the far end for "read me the code" (own thread: Vosk's 0.15-0.4 s end-of-utterance spikes must
        not hold up the outbound spotter, whose spans have to beat the delay line)."""
        if not self._secret_on() or self.spotter_in is None:
            return False
        return self._feed(self.spotter_in, self.far, "_far_cursor", self._on_request)

    def _feed(self, drv, ring: Ring, cursor: str, on_span) -> bool:
        pos, end = getattr(self, cursor), ring.total
        pos = max(pos, end - ring.cap + BLOCK)          # fell a whole ring behind: skip ahead
        end = min(end, pos + SR)                        # at most 1 s per step
        n = (end - pos) // BLOCK * BLOCK
        if n <= 0:
            return False
        x = ring.read_range(pos, pos + n)
        if x is None:
            setattr(self, cursor, ring.total)
            return False
        for s in range(0, n, BLOCK):
            for sp in drv.feed(x[s:s + BLOCK], pos + s):
                on_span(sp)
        setattr(self, cursor, pos + n)
        return True

    def _on_request(self, sp) -> None:
        if sp.start - self._last_request < 3 * SR:       # one request sentence can match several phrases
            return
        self._last_request = sp.start
        self.bus.emit("secret.request", t_audio=round(sp.start / SR, 2))
        if self.cfg.secret.arm_on_request:
            now = self.clock()
            if now >= self._keep_until or not self._auto_by:
                self._auto_by = "request"
            self._keep_until = max(self._keep_until, now + self.cfg.secret.disarm_after_s)

    def _on_span(self, sp) -> None:
        allowed = self.clock() < self._bypass_until
        leaked = 0 if allowed else self.redactor.mark(sp.start, sp.end)
        r = self._run
        if r is not None and sp.start - r["end"] <= self.cfg.secret.gap_s * SR and sp.category == r["category"]:
            r["end"], r["length"] = max(r["end"], sp.end), max(r["length"], sp.length)
            r["leaked"] += leaked
        else:
            self._close_run(None)
            self._run = {"start": sp.start, "end": sp.end, "category": sp.category, "length": sp.length,
                         "leaked": leaked, "allowed": allowed}

    def _close_run(self, pos: int | None) -> None:
        """Publish the current run once the stream is gap_s past it (pos None = now)."""
        r = self._run
        if r is None or (pos is not None and pos < r["end"] + self.cfg.secret.gap_s * SR):
            return
        self._run = None
        if r["category"] == "digits" and r["length"] < self.cfg.secret.min_digits:
            return
        self.bus.emit("secret.blocked", category=r["category"], length=r["length"], armed_by=self.armed_by,
                      allowed=r["allowed"], leaked_ms=round(r["leaked"] / SR * 1000),
                      t_audio=round(r["start"] / SR, 2))

    def _publish_secret(self) -> None:
        c, on = self.cfg.secret, self._secret_on()
        self.bus.emit("secret.state", enabled=on, armed=self.armed, armed_by=self.armed_by, manual=self._manual,
                      allowed=self.clock() < self._bypass_until, allow_s=c.allow_s, delay_ms=c.delay_ms if on else 0,
                      error=self.secret_error, driver=self.spotter.name if self.spotter is not None else None)

    def warm_up(self) -> None:
        """Run the voice model once so the first real window isn't slow (first inference loads kernels), then
        mark the system ready (system.state)."""
        t0 = time.perf_counter()
        self.voice.score((np.random.default_rng(0).standard_normal(VOICE_WIN) * 0.01).astype(np.float32))
        log.info("voice model warmed up in %.1f s", time.perf_counter() - t0)
        self.ready = True
        self._publish_system()

    def _publish_system(self) -> None:
        loop = getattr(self, "_loopback", None)
        self.bus.emit("system.state", ready=self.ready, voice=self.voice.name, speech_db=self.speech_db,
                      speech_min=SPEECH_MIN, voice_source=self.voice_source, far_source=self.far_source,
                      far_error=loop.last_error if loop is not None else None)

    def set_voice_source(self, source: str) -> str:
        """Dashboard control: which stream Hearsay judges. far = the caller (the product); mic = your own mic, to test
        alone (e.g. a phone playing an AI voice into your laptop's mic during a Meet). Starts a fresh voice history."""
        if source not in ("far", "mic"):
            raise ValueError("voice source must be far | mic")
        self.voice_source = source
        self._next_voice = self._voice_ring().total + VOICE_WIN
        self._last_speech = None
        self._flush("source")
        self._publish_system()
        return source

    def set_speech_db(self, db: float) -> float:
        """Dashboard control: the VAD gate. A far-end window is scored when >= SPEECH_MIN of its frames are louder."""
        db = float(db)
        if not -80.0 <= db <= -10.0:
            raise ValueError("speech threshold must be between -80 and -10 dBFS")
        self.speech_db = round(db, 1)
        self._publish_system()
        return self.speech_db

    def flush_voice(self) -> str:
        """Dashboard control: a new speaker starts now. V back to 0, and no window mixes audio from before."""
        self._next_voice = self.far.total + VOICE_WIN
        self._last_speech = None
        self._flush("manual")
        return "flushed"

    def _flush(self, reason: str) -> None:
        self.engine.flush_voice()
        self.bus.emit("voice.flush", reason=reason)

    def secret(self, action: str) -> str:
        """Dashboard control: allow (bypass for allow_s) | arm | disarm | auto."""
        if not self._secret_on():
            raise ValueError(f"secret shield unavailable ({self.secret_error or 'disabled in config'})")
        if action == "allow":
            self._bypass_until = self.clock() + self.cfg.secret.allow_s
        elif action in ("arm", "disarm", "auto"):
            self._manual = {"arm": True, "disarm": False, "auto": None}[action]
        else:
            raise ValueError(f"unknown secret action {action!r}")
        self.bus.publish(Event("control.secret", {"action": action}))
        self._publish_secret()
        return action

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
        def levels():                                   # audio.level for the dashboard's meters (None = no audio)
            seen = {"mic": 0, "far": 0}                 # a stream counts once samples arrive
            while not stop.wait(1 / LEVEL_HZ):
                out = {}
                for k, total, ring in (("mic", self.mic.position, self.mic.raw), ("far", self.far.total, self.far)):
                    out[f"{k}_db"] = dbfs(ring.read_last(SR // LEVEL_HZ)) if total != seen[k] else None
                    seen[k] = total
                self.bus.emit("audio.level", **out)
        ts = [threading.Thread(target=loop, args=(f,), name=f"athena-{f.__name__}", daemon=True)
              for f in (self.voice_step, self.attack_step, self.secret_step, self.request_step)]
        ts.append(threading.Thread(target=levels, name="athena-levels", daemon=True))
        for t in ts:
            t.start()
        return ts

    # --- replay -----------------------------------------------------------------------------------------------
    def replay(self, sc: Scenario, realtime: bool = True, stop: threading.Event | None = None,
               play: bool = False) -> bool:
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
        completed = False
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
                    self.request_step()
                    self.secret_step()
                    if (s // BLOCK) % (TICK // BLOCK) == 0:
                        self.bus.flush()                # the engine must see this tick's events first
                        self.engine.tick()
            completed = not stop.is_set()
            if not realtime:                            # drain the tail
                self.attack_step()
                self.request_step()
                self.secret_step()
                self._close_run(None)
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
        return completed

    def _apply(self, action: dict) -> None:
        if "shield" in action:
            self.set_shield(action["shield"])

    # --- live -------------------------------------------------------------------------------------------------
    def start_live(self) -> None:
        from app.source.audio.streams import LoopbackStream
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

    # --- Google Meet (app/source/connectors/meet) ----------------------------------------------------------------
    def start_meet(self, keyclock: KeyClock | None = None) -> None:
        """Meet mode: no local audio devices; the bridge in the Meet page feeds meet_mic / meet_far over the
        server's /meet WebSockets. `keyclock` is for tests (default: the pynput KeyClock). Needs no server: it can
        run before the server starts; sockets that are already open keep their owner (meet_owners). Idempotent."""
        if self.mode == "meet":
            return
        kc = keyclock or KeyClock(offset_s=self.cfg.devices.meet_key_offset_s)
        self.reset(kc, clock=time.monotonic)
        self.mode = "meet"
        self._run_stop = stop = threading.Event()
        kc.start()
        self._live = [kc]
        self._meet = {"mic": 0.0, "far": 0.0, "rtt_ms": 0.0, "last": None}
        self._loopback = None
        if self.far_source == "system":
            self._start_system_audio()
        self._meet_anchor = ArrivalAnchor()
        self._workers(stop)
        self.engine.start()
        threading.Thread(target=self._meet_watch, args=(stop,), name="athena-meet-state", daemon=True).start()
        self._publish_meet()

    def meet_mic(self, block: np.ndarray) -> np.ndarray:
        """One mic block from the page -> Keyguard shield -> secret delay line -> back to the page (same size).
        Outside meet mode it passes straight through (never mute the user)."""
        block = np.nan_to_num(np.asarray(block, np.float32).reshape(-1))
        if self.mode != "meet":
            return block
        self._meet["mic"] = now = time.monotonic()
        end = self.mic.position + len(block)            # the block's last sample: its arrival, jitter removed
        self.keyclock.anchor(end, self._meet_anchor(end, now))
        return self.mic.process(block)

    def meet_far(self, block: np.ndarray) -> None:
        """Remote participants' audio as the Meet tab's extension hears it (used when far_source == "tab")."""
        self._far_in(block, "tab")

    def _far_in(self, block: np.ndarray, source: str) -> None:
        """The caller's audio -> far-end ring (Hearsay, the request listener). Two sources, one used at a time:
        "system" = what the speakers play (WASAPI loopback: Meet in any browser, Zoom, the test room), "tab" = the
        extension's tap inside the Meet page."""
        if self.mode == "meet" and source == self.far_source:
            self._meet["far"] = time.monotonic()
            self.far.write(np.nan_to_num(np.asarray(block, np.float32).reshape(-1)))

    def _start_system_audio(self) -> None:
        """Capture the default speaker's output (loopback) for the whole meet-mode run; it only feeds the far ring
        while far_source == "system". No loopback device (macOS, no speaker): say why, the tab source still works."""
        from app.source.audio.streams import LoopbackStream
        loop = LoopbackStream(self.cfg.devices.loopback or None, on_block=lambda x: self._far_in(x, "system"))
        self._live.append(loop.start())
        self._loopback = loop

    def set_far_source(self, source: str) -> str:
        """Dashboard control: where the caller's audio comes from in meet mode (system | tab)."""
        if source not in ("system", "tab"):
            raise ValueError("caller audio source must be system | tab")
        self.far_source = source
        if source == "system" and self.mode == "meet" and getattr(self, "_loopback", None) is None:
            self._start_system_audio()
        if self.voice_source == "far":                  # a different stream: start the voice history over
            self._next_voice = self.far.total + VOICE_WIN
            self._last_speech = None
            self._flush("source")
        self._publish_system()
        return source

    def meet_link(self, kind: str, delta: int = 0, rtt_ms: float | None = None, owner: str | None = None) -> None:
        """Router bookkeeping, on the server loop: `owner` = the origin of the page that owns /meet/<kind> now (None:
        nobody); delta +1 = a new owner, -1 = it left; the bridge's measured round trip."""
        self.meet_owners[kind] = owner                  # kept outside meet mode too: start_meet may come later
        if self.mode != "meet":
            return
        if kind == "mic" and delta > 0:
            self._meet_flush()
        if rtt_ms is not None:
            self._meet["rtt_ms"] = float(rtt_ms)
        self._publish_meet()

    def _meet_flush(self) -> None:
        """A new mic socket: drop what the previous one left in the delay line (it would go out ~0.5 s into the new
        stream) and the shield's lookahead, and re-fit the key anchor. Called on the server loop, the thread that
        runs meet_mic, so it lands between blocks."""
        r = self.redactor
        if r is not None:
            kept = r.leaked_samples, r.redacted_samples
            r.reset()
            r.leaked_samples, r.redacted_samples = kept
        self.shield.reset()
        self._meet_anchor.reset()

    def _meet_watch(self, stop: threading.Event) -> None:
        while not stop.wait(0.25):                      # mic/far flip to False a second after frames stop
            self._publish_meet()

    def announce(self) -> None:
        """Re-publish the current shield / secret / meet state, for a dashboard that started listening after them."""
        self._publish_shield()
        self._publish_secret()
        self._publish_system()
        if getattr(self, "_meet", None) is not None:
            self._meet["last"] = None                   # force meet.state out even if nothing changed
            self._publish_meet()

    def meet_page(self, pid: int, origin: str | None, status: dict | None) -> None:
        """Router: a page with the bridge reported its state ({"site", "in_call", ...}; None = it closed)."""
        if status is None:
            self.meet_pages.pop(pid, None)
        else:
            self.meet_pages[pid] = (time.monotonic(), origin, status)
        self._publish_meet()

    def _call_event(self, page: dict | None) -> None:
        """meet.call on every change of what the extension reports: a Meet tab opened / closed, a call joined / left
        (the dashboard's event log, and any hook subscribed to meet.*)."""
        phase = "none" if not page else "local" if page.get("site") != "meet" else "call" if page.get("in_call") else "open"
        prev, self._call_phase = self._call_phase, phase
        if prev is None or prev == phase:
            return
        event = {("open", "call"): "joined", ("local", "call"): "joined", ("none", "call"): "joined",
                 ("call", "open"): "left", ("call", "none"): "left", ("call", "local"): "left"}.get((prev, phase))
        event = event or {"open": "meet_open", "none": "meet_closed", "local": "test_room"}.get(phase, phase)
        self.bus.emit("meet.call", event=event, site=page.get("site") if page else None)

    def _page(self) -> dict | None:
        """The page the dashboard shows: a live Meet page if there is one, else the newest live page."""
        now = time.monotonic()
        live = [(o == "https://meet.google.com", pid, st) for pid, (t, o, st) in self.meet_pages.items()
                if now - t < PAGE_STALE_S]
        return max(live, key=lambda x: x[:2])[2] if live else None

    def _publish_meet(self) -> None:
        m, now, sess = getattr(self, "_meet", None), time.monotonic(), self.meet_session
        if m is None or self.mode != "meet":
            return
        owners, page = self.meet_owners, self._page()
        self._call_event(page)
        state = dict(page=page and page.get("site"),     # "meet" | "local" (test room) | None: what the extension sees
                     in_call=bool(page and page.get("in_call")),
                     connected=any(owners.values()) or bool(sess and sess.alive),
                     owner=owners["mic"] or owners["far"],  # the page being protected
                     url=sess.url if sess and sess.alive else None,
                     mic=now - m["mic"] < 1.0, far=now - m["far"] < 1.0,
                     browser=sess.browser if sess and sess.alive else None,
                     latency_ms=round(self._shield_lag() / SR * 1000 + m["rtt_ms"], 1))
        key = {k: v for k, v in state.items() if k != "latency_ms"} | {"lat": round(state["latency_ms"], -1)}
        if key != m["last"]:                            # on changes only (latency in 10 ms steps)
            m["last"] = key
            self.bus.emit("meet.state", **state)

    def meet(self, action: str, url: str | None = None) -> str:
        """Dashboard control: join (open the Meet link as a normal tab in the user's Chrome; the Athena extension
        connects it) | extension (show the extension folder, to install it) | leave."""
        from app.source.connectors.meet import launcher
        if self.mode != "meet":
            raise ValueError("Google Meet needs meet mode (athena app, or athena run --mode meet)")
        page = self._page()
        if action == "leave":                           # leave the call itself: the page clicks Meet's Leave button
            if self.meet_session is not None:
                self.meet_session.close()
                self.meet_session = None
                result = "left"
            elif page and page.get("site") == "meet" and self.meet_command and self.meet_command({"cmd": "leave"}):
                result = "leaving the call"
            else:
                raise ValueError("no Google Meet tab is connected (is the Athena extension installed?)")
        elif action == "join":
            url = launcher.meet_url(url)                # ValueError on anything but a Meet / local URL
            if page and page.get("site") == "meet" and self.meet_command and self.meet_command({"cmd": "open", "url": url}):
                result = f"opened {url} in your Meet tab"
            else:
                result = f"opened {url} in {launcher.open_tab(url)}"
        elif action == "extension":
            result = launcher.show_extension()
        else:
            raise ValueError(f"unknown meet action {action!r}")
        self.bus.publish(Event("control.meet", {"action": action, "url": url}))
        self._publish_meet()
        return result

    def stop(self) -> None:
        if self.agents is not None:
            self.agents.stop()
        if self.meet_session is not None:
            self.meet_session.close()
            self.meet_session = None
        if self._run_stop is not None:
            self._run_stop.set()
        if self._run_thread is not None:
            self._run_thread.join(timeout=10)
            self._run_thread = None
        for x in self._live:
            x.stop()
        self._live = []
        self.engine.stop()
        if self.mode == "meet":                         # late bridge frames now pass straight through
            self.mode = "idle"

    # --- controls (the server's `controls` object) ------------------------------------------------------------
    def set_shield(self, mode: str) -> str:
        if mode not in SHIELD_MODES:
            raise ValueError(f"shield mode must be one of {SHIELD_MODES}")
        self._driver_mode(mode)                         # raises (mode unchanged) when adversarial can't load
        if mode != self.shield_mode:
            self._clear_shielded = True               # the worker restarts the shielded readout (no cross-thread edit)
        self.shield_mode = mode
        self.bus.publish(Event("control.shield", {"mode": mode}))
        self._publish_shield()
        return mode

    def _driver_mode(self, mode: str) -> None:
        """Pipeline mode -> the shield driver's mode. "off" leaves the driver as is (it just gets no key events), so
        every switch keeps the same driver and the same output delay. ValueError when "adversarial" can't run."""
        if mode == "off":
            return
        set_mode = getattr(self.shield, "set_mode", None)   # Quarantine forwards it; mocks don't have one
        if set_mode is None:
            if mode == "adversarial":
                raise ValueError(f"shield driver {self.shield.name!r} has no adversarial mode")
            return
        from app.keystroke_guard.driver import DASHBOARD_ADVERSARIAL
        try:
            set_mode(DASHBOARD_ADVERSARIAL if mode == "adversarial" else mode)
        except (OSError, KeyError, RuntimeError) as e:  # missing / unreadable runs/adversarial_deltas.pt
            raise ValueError(f"adversarial shield unavailable: {e}") from e

    def scenario(self, action: str, name: str = "ai_caller", on_end: Callable[[], None] | None = None,
                 play: bool = True) -> str:
        """Start/stop a realtime replay on a background thread (dashboard button, CLI)."""
        if self.mode in ("live", "meet"):
            raise ValueError("scenarios run in replay mode (athena run --mode replay)")
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
            completed = False
            try:
                completed = self.replay(sc, realtime=True, stop=stop, play=play)
            except Exception as e:                      # noqa: BLE001
                log.exception("scenario %s failed", name)
                self.bus.emit("driver.error", driver="scenario", kind="replay", error=repr(e))
            self.bus.publish(Event("control.scenario", {"action": "end", "name": name, "completed": completed}))
            if on_end is not None and completed:
                on_end()
        self._run_thread = threading.Thread(target=run, name="athena-replay", daemon=True)
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
