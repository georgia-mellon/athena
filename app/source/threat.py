"""ThreatEngine: fuses the voice verdicts and the keystroke readouts into one 0-100 score (plan 02 §4).

Consumed payloads (producers must send these keys; extra keys are ignored):
- `voice.verdict`  {"p_synthetic": float}              one per scored far-end speech window
- `keys.stroke`    {}                                  one per OS key event (timing only)
- `keys.readout`   {"hit": {"raw": bool, "shielded": bool}, "chance"?: float}   one per attacked keystroke; "hit" =
                   the true key is in the attacker's top-3 (a 9-key code then falls to ~3^9 guesses), "chance" = the
                   hit rate of a guessing attacker (default 3 / num_classes) (the pipeline
                   also sends the dashboard's lists: raw/shielded guesses, acc_raw, acc_shielded, chance)
- `shield.state`   {"mode": "off"|"dsp"|"adversarial", "failed"?: bool}
- `secret.blocked` {"category", "length"}            a spoken secret redacted from the outbound mic (plan 06)
- `secret.request` {}                                the far end asked for a code ("read me the code")
Published:
- `threat.update`        {"score", "level", "V", "E", "L", "T", "typing", "shield", "reasons"}  on every tick()
- `threat.level_change`  {"from", "to", "score", "reasons"}

Time comes from the injected `now()` (monotonic seconds), never from Event.t, so tests are deterministic.
What leaks is E with the shield off and L with it on: a working shield stops E from raising the score.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

from .bus import EventBus
from .config import ThreatConfig
from .types import Event

LEVELS = ("SAFE", "WATCH", "WARN", "CRITICAL")


PRIOR = 3  # pseudo-keystrokes at chance: one lucky read isn't "100 % readable"


def _above_chance(hits: deque, chance: float) -> float:
    if not hits:
        return 0.0
    acc = (sum(hits) + PRIOR * chance) / (len(hits) + PRIOR)
    return max(0.0, (acc - chance) / (1.0 - chance))


class ThreatEngine:
    def __init__(self, bus: EventBus, cfg: ThreatConfig | None = None, now: Callable[[], float] = time.monotonic):
        self.bus, self.cfg, self.now = bus, cfg or ThreatConfig(), now
        self._lock = threading.Lock()
        self.V = 0.0
        self._p_last: float | None = None
        self._t_verdict = -1e9
        self._t_ema = now()
        self._voice_since: float | None = None      # start of the current run of p >= se_voice, for the reasons
        self._hits = {"raw": deque(maxlen=self.cfg.readout_window), "shielded": deque(maxlen=self.cfg.readout_window)}
        self._chance = min(3, self.cfg.num_classes) / self.cfg.num_classes
        self._strokes: deque[float] = deque()
        self._secrets: deque[tuple[float, str, int]] = deque()   # (t, category, length) of blocked secrets
        self._t_request = -1e9
        self.shield = "off"
        self.shield_failed = False
        self.level = "SAFE"
        self.score = 0.0
        self._unsubs = [bus.subscribe(t, self.on_event)
                        for t in ("voice.verdict", "keys.stroke", "keys.readout", "shield.state", "secret.blocked",
                                  "secret.request")]
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- inputs -------------------------------------------------------------------------------------------------
    def on_event(self, ev: Event) -> None:
        d, now = ev.data, self.now()
        with self._lock:
            if ev.topic == "voice.verdict":
                self._advance_v(now)
                p = min(1.0, max(0.0, float(d["p_synthetic"])))
                if p >= self.cfg.se_voice and self._voice_since is None:
                    self._voice_since = now
                elif p < self.cfg.se_voice:
                    self._voice_since = None
                self._p_last, self._t_verdict = p, now
            elif ev.topic == "keys.stroke":
                self._strokes.append(now)
            elif ev.topic == "keys.readout":
                if d.get("chance"):
                    self._chance = float(d["chance"])
                for stream, ok in d["hit"].items():
                    self._hits[stream].append(bool(ok))
            elif ev.topic == "secret.blocked" and not d.get("allowed"):
                self._secrets.append((now, str(d.get("category", "secret")), int(d.get("length", 0))))
            elif ev.topic == "secret.request":
                self._t_request = now
            elif ev.topic == "shield.state":
                if d.get("mode", self.shield) != self.shield:
                    self._hits["shielded"].clear()     # L = what leaks with the *current* shield
                self.shield = d.get("mode", self.shield)
                self.shield_failed = bool(d.get("failed", False))

    def _advance_v(self, now: float) -> None:
        """Continuous-time EMA toward the last p_synthetic while speech is fresh, toward 0 in silence."""
        dt = max(0.0, now - self._t_ema)
        self._t_ema = now
        fresh = self._p_last is not None and now - self._t_verdict <= self.cfg.voice_stale_s
        target = self._p_last if fresh else 0.0
        if not fresh:
            self._voice_since = None
        self.V += (target - self.V) * (1.0 - 0.5 ** (dt / self.cfg.voice_half_life_s))

    # --- scoring ------------------------------------------------------------------------------------------------
    def tick(self) -> dict:
        """Recompute, publish threat.update (and threat.level_change on a level change), return the update."""
        c, now = self.cfg, self.now()
        with self._lock:
            self._advance_v(now)
            while self._strokes and now - self._strokes[0] > c.typing_window_s:
                self._strokes.popleft()
            while self._secrets and now - self._secrets[0][0] > c.secret_window_s:
                self._secrets.popleft()
            S = len(self._secrets)
            asked = now - self._t_request <= c.secret_window_s
            V = self.V
            E = _above_chance(self._hits["raw"], self._chance)
            L = _above_chance(self._hits["shielded"], self._chance)
            T = min(len(self._strokes), c.typing_saturation) / c.typing_saturation
            typing = T >= c.typing_on
            shield_on = self.shield != "off" and not self.shield_failed
            leak = L if shield_on else E

            score = 100.0 * (1.0 - (1.0 - c.w_v * V) * (1.0 - c.w_l * leak * typing))
            se = V >= c.se_voice and typing
            if se:
                score = max(score, c.se_floor + c.se_gain * V * leak)
            if S and V >= c.se_voice:           # you started reading a secret to an unverified voice (plan 06 §6)
                score = max(score, c.critical + 10 if asked else c.warn)
            score = min(100.0, max(0.0, score))
            level = self._hysteresis(score)
            reasons = self._reasons(now, V, E, L, T, typing, shield_on, se)
            if S:
                reasons = [r for r in reasons if r != "no threat signals"]
                cat, n = self._secrets[-1][1:]
                what = f"{n}-digit code" if cat == "digits" else cat
                who = "caller asked for a code and you started reading it: " if asked and V >= c.se_voice else ""
                reasons.insert(0, f"{who}{what} blocked from your voice" + (f" ({S} in {c.secret_window_s:.0f} s)"
                                                                            if S > 1 else ""))
            old, self.level, self.score = self.level, level, score
            shield = self.shield + (" (failed)" if self.shield_failed else "")

        update = {"score": round(score, 1), "level": level, "V": round(V, 3), "E": round(E, 3), "S": S,
                  "L": round(L, 3), "T": round(T, 2), "typing": typing, "shield": shield, "reasons": reasons}
        self.bus.emit("threat.update", **update)
        if level != old:
            self.bus.emit("threat.level_change", **{"from": old, "to": level, "score": update["score"],
                                                     "reasons": reasons})
        return update

    def _hysteresis(self, score: float) -> str:
        c = self.cfg

        def lvl(s: float) -> int:
            return sum(s >= t for t in (c.watch, c.warn, c.critical))
        cur, up, down = LEVELS.index(self.level), lvl(score), lvl(score + c.hysteresis)
        if up > cur:
            cur = up
        elif down < cur:
            cur = down
        return LEVELS[cur]

    def _reasons(self, now, V, E, L, T, typing, shield_on, se) -> list[str]:
        c, r = self.cfg, []
        if V >= c.se_voice:
            since = f" for {now - self._voice_since:.0f} s" if self._voice_since is not None else ""
            r.append(f"synthetic voice {V:.2f}{since}")
        elif V >= 0.25:
            r.append(f"unverified voice {V:.2f}")
        if se:
            r.append("typing while an unverified voice is speaking")
        if self.shield_failed:
            r.append("shield failed: mic is passing through unprotected")
        if typing and E > 0:
            if shield_on:
                r.append(f"shield ({self.shield}) blocking a {E:.0%} readable keyboard; residual leak {L:.0%}")
            else:
                r.append(f"keystrokes {E:.0%} readable by an eavesdropper (shield off)")
        elif typing:
            r.append(f"typing ({T * c.typing_saturation:.0f} keys in {c.typing_window_s:.0f} s)")
        return r or ["no threat signals"]

    # --- driving ------------------------------------------------------------------------------------------------
    def start(self) -> None:
        """Tick at cfg.tick_hz on a daemon thread."""
        def loop():
            while not self._stop.wait(1.0 / self.cfg.tick_hz):
                self.tick()
        self._thread = threading.Thread(target=loop, name="callguard-threat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for u in self._unsubs:
            u()
