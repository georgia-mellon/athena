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

REPO = Path(__file__).resolve().parents[1]
DRIVER_KINDS = ("real", "mock")
SHIELD_MODES = ("off", "dsp", "adversarial")
HEARSAY_MODES = ("r4ft", "r5")


@dataclass
class ThreatConfig:
    """Plan 02 §4. Scores are 0-100; V, E, L, T are in [0, 1]."""
    voice_half_life_s: float = 6.0     # EMA half-life of V
    voice_stale_s: float = 4.0         # no verdict for this long = silence, V decays toward 0
    readout_window: int = 20           # keystrokes for E and L
    num_classes: int = 36              # K for the chance rescale (A-Z0-9), unless the readout says otherwise
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


@dataclass
class DriversConfig:
    voice: str = "real"                # real | mock, per slot
    attacker: str = "real"
    shield: str = "real"
    hearsay_mode: str = "r4ft"         # r4ft (fast) | r5 (submitted fusion)
    shield_mode: str = "dsp"           # off | dsp | adversarial
    device: str = "auto"               # torch device: auto | cpu | cuda


@dataclass
class DevicesConfig:
    """Device names (substring match; empty = system default). See plans/03."""
    mic: str = ""
    virtual_out: str = "CABLE Input"
    loopback: str = ""
    key_offset_s: float = 0.0          # KeyClock calibration: + if key sounds land later than their OS timestamps


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
    attacker_weights: str = ""         # empty = provisional KeyNet (plan 04)
    drivers: DriversConfig = field(default_factory=DriversConfig)
    devices: DevicesConfig = field(default_factory=DevicesConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    hooks: list[HookConfig] = field(default_factory=lambda: [HookConfig("console")])


SECTIONS = {"drivers": DriversConfig, "devices": DevicesConfig, "server": ServerConfig, "threat": ThreatConfig}


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
    for slot in ("voice", "attacker", "shield"):
        if getattr(d, slot) not in DRIVER_KINDS:
            raise ValueError(f"drivers.{slot} must be one of {DRIVER_KINDS}, got {getattr(d, slot)!r}")
    if d.shield_mode not in SHIELD_MODES:
        raise ValueError(f"drivers.shield_mode must be one of {SHIELD_MODES}")
    if d.hearsay_mode not in HEARSAY_MODES:
        raise ValueError(f"drivers.hearsay_mode must be one of {HEARSAY_MODES}")
    for h in cfg.hooks:
        if h.kind not in ("console", "jsonl", "webhook"):
            raise ValueError(f"unknown hook kind {h.kind!r}")
        if h.kind == "webhook" and not h.url:
            raise ValueError("webhook hook needs a url")
    t = cfg.threat
    if not (0 < t.watch < t.warn < t.critical <= 100) or t.tick_hz < 2:
        raise ValueError("threat thresholds must satisfy 0 < watch < warn < critical <= 100, tick_hz >= 2")
