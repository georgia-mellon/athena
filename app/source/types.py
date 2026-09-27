"""Shared contracts (plan 02 §3). Every package codes against these; change only through the integrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

SR = 16_000
BLOCK = 320  # 20 ms at 16 kHz: the audio thread's block size


@dataclass
class VoiceScore:
    p_synthetic: float          # [0, 1]; 0.5 = the deployment threshold
    margin: float               # raw model margin/logit (higher = more synthetic)
    threshold: float            # the margin at which p_synthetic = 0.5
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyGuess:
    onset: int                  # sample index in the analysed buffer
    top: list[tuple[str, float]]  # [(key, prob), ...] best first
    truth: str | None = None    # demo ground truth (local only, never logged)


@dataclass
class Event:
    topic: str                  # e.g. "voice.verdict", "threat.update"
    data: dict[str, Any]
    t: float = field(default_factory=time.time)


@runtime_checkable
class VoiceAuthenticityDriver(Protocol):
    name: str
    sample_rate: int

    def score(self, audio: np.ndarray) -> VoiceScore: ...


@runtime_checkable
class KeystrokeAttackerDriver(Protocol):
    """`onsets` are sample indices within `audio`. The driver never gets the true keys: the pipeline fills
    KeyGuess.truth after the call (only a mock may opt in with `wants_truth = True`)."""
    name: str
    classes: list[str]

    def read(self, audio: np.ndarray, onsets: np.ndarray) -> list[KeyGuess]: ...


@runtime_checkable
class ShieldDriver(Protocol):
    """Streaming, same length out. `key_events` are ABSOLUTE sample indices on the block clock (sum of block lengths
    since reset()); each event is passed once, possibly a block or two late (OS key events lag the sound).
    Optional `latency` attribute: output delay in samples (0 if absent)."""
    name: str

    def process(self, block: np.ndarray, key_events: list[int]) -> np.ndarray: ...

    def reset(self) -> None: ...


@dataclass
class SecretSpan:
    """A stretch of outbound (or inbound, for category "request") audio to act on. No text field, by design: the
    recognized words never leave the spotter (plan 06 §8)."""
    start: int                  # absolute sample index on the stream the spotter was fed
    end: int
    category: str               # "digits" | "password" | "card" (outbound) | "request" (inbound trigger phrase)
    length: int                 # tokens in the run so far (digits spoken); 0 for trigger phrases


@runtime_checkable
class SecretSpotterDriver(Protocol):
    """Streaming spotter (plan 06). `feed` gets consecutive blocks with the absolute index of their first sample and
    returns spans as soon as it can place them (partial results), each at most once. Outbound mode redacts every
    digit/letter token that follows another within gap_s (the first token of a run passes: the delay line can't
    wait for a whole sequence), plus the words after an own-side trigger ("the code is ...")."""
    name: str

    def feed(self, block: np.ndarray, start: int) -> list[SecretSpan]: ...

    def reset(self) -> None: ...


# Event topics (plan 01 F6). Payload keys are documented next to each producer.
TOPICS = (
    "voice.window", "voice.verdict",
    "keys.stroke", "keys.readout",
    "shield.state", "driver.error",
    "threat.update", "threat.level_change",
    "control.scenario", "control.shield",
    "secret.state", "secret.blocked", "secret.request", "control.secret",
    "meet.state", "control.meet",
)
