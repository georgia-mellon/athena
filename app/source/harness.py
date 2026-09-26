"""Shared pieces of the pillar harnesses (app.hearsay / app.keystroke_guard / app.secret_shield .harness): the
PASS/FAIL report, the driver loader, timing and probe audio."""
from __future__ import annotations

import importlib
import inspect
import time
from typing import Any, Callable

import numpy as np

from app.source.types import SR


class Skip(Exception):
    """Raised by a check when its data or model is missing."""


class Report:
    """Rows of (section, check, status, detail); status is PASS | FAIL | SKIP | INFO."""

    def __init__(self, title: str):
        self.title, self.rows = title, []

    def run(self, section: str, check: str, fn: Callable[[], str], info: bool = False) -> bool:
        """Run one check: a returned string is the detail (PASS, or INFO for quality rows), AssertionError = FAIL,
        Skip = SKIP, any other exception = FAIL with its type (a driver that crashes doesn't fit)."""
        try:
            status, detail = ("INFO" if info else "PASS"), fn()
        except Skip as e:
            status, detail = "SKIP", str(e)
        except AssertionError as e:
            status, detail = "FAIL", str(e) or "assertion failed"
        except Exception as e:  # noqa: BLE001 - report it, keep checking
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        self.rows.append((section, check, status, detail))
        return status != "FAIL"

    @property
    def ok(self) -> bool:
        return not any(r[2] == "FAIL" for r in self.rows)

    def print(self) -> None:
        w = max(len(r[1]) for r in self.rows) if self.rows else 10
        print(f"\n{self.title}")
        for section, check, status, detail in self.rows:
            print(f"  {section:<9} {check:<{w}}  {status:<4}  {detail}")
        n = {s: sum(r[2] == s for r in self.rows) for s in ("PASS", "FAIL", "SKIP", "INFO")}
        print(f"  => {'FITS' if self.ok else 'DOES NOT FIT'}  ({', '.join(f'{v} {k}' for k, v in n.items() if v)})")


def load_driver(spec: Any, factory: str, key: str, **kw: Any) -> Any:
    """spec: a driver instance (returned as is), "mock" | "real" (app.source.registry.<factory>, raw, no
    Quarantine), or "module.path:ClassName" (instantiated with **kw, or with no arguments if it doesn't take them)."""
    if not isinstance(spec, str):
        return spec
    if spec in ("mock", "real"):
        from app.source import registry
        return getattr(registry, factory)({key: spec}, **kw)
    mod, _, cls = spec.partition(":")
    if not cls:
        raise ValueError(f"driver must be mock | real | module.path:ClassName, got {spec!r}")
    klass = getattr(importlib.import_module(mod), cls)
    params = inspect.signature(klass).parameters          # pass only what it takes; a TypeError inside __init__
    if not any(q.kind is q.VAR_KEYWORD for q in params.values()):  # is the driver's bug and must surface
        kw = {k: v for k, v in kw.items() if k in params}
    return klass(**kw)


def timed(fn: Callable[[], Any], n: int) -> np.ndarray:
    """Per-call milliseconds of n calls, after one warm-up call."""
    fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return np.array(out)


def probe_audio(n: int, seed: int = 0) -> list[np.ndarray]:
    """Two deterministic test signals: noise, and a harmonic 'voiced' tone with a slow pitch wobble."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    tone = sum(np.sin(k * phase) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2)
    return [(0.05 * rng.standard_normal(n)).astype(np.float32), (0.1 * tone).astype(np.float32)]


# --- Hearsay checks -------------------------------------------------------------------------------------------------
