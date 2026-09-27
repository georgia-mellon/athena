"""Config: dataclasses, loaded from TOML (`callguard.toml`, gitignored; see `callguard.example.toml`) plus env.

Precedence: defaults < TOML < env. Env overrides:
- `HEARSAY_ROOT`, `KEYGUARD_ROOT`, `CALLGUARD_ATTACKER_WEIGHTS` (top-level paths);
- `CALLGUARD_<SECTION>_<FIELD>` for any scalar field, e.g. `CALLGUARD_SERVER_PORT=9000`, `CALLGUARD_DRIVERS_VOICE=mock`;
- `CALLGUARD_WEBHOOK_URL` appends a webhook hook for `threat.level_change`, so the URL never lands in a file.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER_KINDS = ("real", "mock")
SHIELD_MODES = ("off", "dsp", "adversarial")
HEARSAY_MODES = ("r4ft", "r5", "e5")


@dataclass
class ThreatConfig:
    """Plan 02 §4. Scores are 0-100; V, E, L, T are in [0, 1]."""
    voice_half_life_s: float = 6.0     # EMA half-life of V
    voice_stale_s: float = 4.0         # no verdict for this long = silence, V decays toward 0
    readout_window: int = 20           # keystrokes for E and L
    num_classes: int = 36              # K (A-Z0-9): chance = 3/K for top-3 hits, unless the readout says otherwise
    typing_window_s: float = 10.0
    typing_saturation: int = 10        # strokes in the window at which T = 1
    typing_on: float = 0.2             # T >= this counts as "typing" (T_on); 0.2 = 2 strokes in 10 s
    w_v: float = 0.7
    w_l: float = 0.9
    se_voice: float = 0.5              # social-engineering rule: V at/above this while typing
    se_floor: float = 60.0
    se_gain: float = 40.0
    watch: float = 25.0
    warn: float = 50.0
    critical: float = 75.0
    hysteresis: float = 5.0
    tick_hz: float = 4.0               # threat.update rate (>= 2 Hz)
    secret_window_s: float = 60.0      # S = spoken secrets blocked in this window (plan 06 section 6)


@dataclass
class DriversConfig:
    voice: str = "real"                # real | mock, per slot
    attacker: str = "real"
    shield: str = "real"
    secret: str = "real"               # spoken-secret spotter (plan 06): real = Vosk (runs/models), mock
    hearsay_mode: str = "e5"           # e5 (final Hearsay model, default) | r5 (R4ft + R1) | r4ft (XLS-R alone)
    hearsay_ai_p: float = 0.7          # decision threshold: a voice is AI when its calibrated p >= this (0.5 = Hearsay's)
    shield_mode: str = "dsp"           # off | dsp | adversarial
    device: str = "auto"               # torch device: auto | cpu | cuda


@dataclass
class SecretConfig:
    """Spoken-secret shield (plan 06)."""
    enabled: bool = True
    delay_ms: float = 500.0            # constant outbound delay while enabled (the redactor's lookahead)
    style: str = "tone"                # mute | tone | noise
    arm_on_voice: bool = True          # arm while Hearsay V >= arm_voice
    arm_voice: float = 0.5
    keep_voice: float = 0.3            # stays armed while V >= this ...
    arm_on_request: bool = True        # ... or after an inbound "read me the code" trigger
    disarm_after_s: float = 60.0       # ... for this long after the last reason to be armed
    allow_s: float = 30.0              # the dashboard's Allow button
    min_digits: int = 3                # a run this long is reported as a blocked secret
    gap_s: float = 1.2                 # tokens further apart than this start a new run


@dataclass
class DevicesConfig:
    """Device names (substring match; empty = system default). See docs/plans/03."""
    mic: str = ""
    virtual_out: str = "CABLE Input"
    loopback: str = ""
    key_offset_s: float = 0.0          # KeyClock calibration: + if key sounds land later than their OS timestamps
    meet_key_offset_s: float = 0.0     # the same for meet mode: + the page's capture + socket latency (~0.02-0.08)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class HookConfig:
    kind: str                          # console | jsonl | webhook
    topics: list[str] = field(default_factory=lambda: ["threat.level_change"])
    url: str = ""                      # webhook
    path: str = "logs/events.jsonl"    # jsonl, relative to the repo unless absolute
    timeout_s: float = 3.0
    retries: int = 2


@dataclass
class Config:
    hearsay_root: Path = REPO.parent / "Hearsay"
    keyguard_root: Path = REPO.parent / "keyboard-acoustic-shield"
    attacker_weights: str = ""         # empty = Keyguard's CTC attacker, KEYGUARD_ROOT/runs/ctc_rich_ft.pt
    drivers: DriversConfig = field(default_factory=DriversConfig)
    devices: DevicesConfig = field(default_factory=DevicesConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    secret: SecretConfig = field(default_factory=SecretConfig)
    hooks: list[HookConfig] = field(default_factory=lambda: [HookConfig("console")])


SECTIONS = {"drivers": DriversConfig, "devices": DevicesConfig, "server": ServerConfig, "threat": ThreatConfig,
            "secret": SecretConfig}


def _coerce(value, like):
    if isinstance(like, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(like, (int, float, Path)):
        return type(like)(value)
    return value


def _fill(obj, values: dict, where: str):
    names = {f.name for f in fields(obj)}
    for k, v in values.items():
        if k not in names:
            raise ValueError(f"unknown config key {where}.{k}")
        setattr(obj, k, _coerce(v, getattr(obj, k)))


def load(path: str | Path | None = None, env: dict[str, str] | None = None) -> Config:
    """Load config. `path=None` reads `<repo>/callguard.toml` if present; `env` defaults to os.environ."""
    env = os.environ if env is None else env
    cfg = Config()
    p = Path(path) if path else REPO / "callguard.toml"
    raw = tomllib.loads(p.read_text(encoding="utf-8")) if (path or p.exists()) else {}

    hooks = raw.pop("hooks", None)
    if hooks is not None:
        cfg.hooks = [HookConfig(**h) for h in hooks]
    for name in SECTIONS:
        _fill(getattr(cfg, name), raw.pop(name, {}), name)
    _fill(cfg, raw, "root")

    for key, attr in (("HEARSAY_ROOT", "hearsay_root"), ("KEYGUARD_ROOT", "keyguard_root"),
                      ("CALLGUARD_ATTACKER_WEIGHTS", "attacker_weights")):
        if env.get(key):
            setattr(cfg, attr, _coerce(env[key], getattr(cfg, attr)))
    for name, cls in SECTIONS.items():
        section = getattr(cfg, name)
        for f in fields(cls):
            key = f"CALLGUARD_{name}_{f.name}".upper()
            if key in env:
                setattr(section, f.name, _coerce(env[key], getattr(section, f.name)))
    if env.get("CALLGUARD_WEBHOOK_URL"):
        cfg.hooks.append(HookConfig("webhook", url=env["CALLGUARD_WEBHOOK_URL"]))

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    d = cfg.drivers
    for slot in ("voice", "attacker", "shield", "secret"):
        if getattr(d, slot) not in DRIVER_KINDS:
            raise ValueError(f"drivers.{slot} must be one of {DRIVER_KINDS}, got {getattr(d, slot)!r}")
    if d.shield_mode not in SHIELD_MODES:
        raise ValueError(f"drivers.shield_mode must be one of {SHIELD_MODES}")
    if d.hearsay_mode not in HEARSAY_MODES:
        raise ValueError(f"drivers.hearsay_mode must be one of {HEARSAY_MODES}")
    if not 0.0 < d.hearsay_ai_p < 1.0:
        raise ValueError("drivers.hearsay_ai_p must be in (0, 1)")
    if cfg.secret.style not in ("mute", "tone", "noise"):
        raise ValueError("secret.style must be mute | tone | noise")
    for h in cfg.hooks:
        if h.kind not in ("console", "jsonl", "webhook"):
            raise ValueError(f"unknown hook kind {h.kind!r}")
        if h.kind == "webhook" and not h.url:
            raise ValueError("webhook hook needs a url")
    t = cfg.threat
    if not (0 < t.watch < t.warn < t.critical <= 100) or t.tick_hz < 2:
        raise ValueError("threat thresholds must satisfy 0 < watch < warn < critical <= 100, tick_hz >= 2")
