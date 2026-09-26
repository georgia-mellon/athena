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
    name: str
    classes: list[str]

    def read(self, audio: np.ndarray, onsets: np.ndarray) -> list[KeyGuess]: ...


@runtime_checkable
class ShieldDriver(Protocol):
    name: str

    def process(self, block: np.ndarray, key_events: list[int]) -> np.ndarray: ...

    def reset(self) -> None: ...


# Event topics (plan 01 F6). Payload keys are documented next to each producer.
TOPICS = (
    "voice.window", "voice.verdict",
    "keys.stroke", "keys.readout",
    "shield.state", "driver.error",
    "threat.update", "threat.level_change",
    "control.scenario", "control.shield",
)
