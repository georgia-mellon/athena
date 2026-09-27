"""Keystroke Guard harness: do an attacker and a shield fit Athena? (contract, latency, quick quality)

    python -m app.keystroke_guard.harness [--attacker mock|real|module.path:ClassName]
                                          [--shield mock|real|module.path:ClassName] [--presses N] [--data bank|harrison]
                                          [--shield-mode dsp|adversarial|dsp+adversarial]

`mock` = the placeholders, `real` = Keyguard's CTC attacker / streaming DSP shield, anything else = a class path
instantiated with no arguments. Quality is informational (bank presses are optimistic, the CTC attacker's domain).
Exit code 1 when a contract or latency check fails.
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from app.source.harness import Report, Skip, load_driver, probe_audio, timed
from app.source.types import BLOCK, SR, KeyGuess, KeystrokeAttackerDriver, ShieldDriver

ATTACK_BUDGET_MS = 50.0     # per keystroke: the readout keeps up with fast typing
SHIELD_BUDGET_MS = 20.0     # per 20 ms block with a key active: the audio thread must keep real time
NOISE = 0.002               # bank quality rows: noise floor to match Keyguard's synth


def _click(n: int, at: int) -> np.ndarray:
    """A crude keystroke: a 5 ms decaying click on a quiet noise floor."""
    x = (0.002 * np.random.default_rng(1).standard_normal(n)).astype(np.float32)
    k = np.arange(80)
    x[at:at + 80] += (0.5 * np.exp(-k / 15) * np.sin(2 * np.pi * 3000 * k / SR)).astype(np.float32)
    return x


def check(attacker: Any = "mock", shield: Any = "mock", presses: int = 0, quality: bool = True,
          shield_mode: str | None = None, data: str = "bank") -> Report:
    """Run every check on the attacker and the shield (either may be None to skip it); returns the Report.
    presses: 0 = 10 per key from the bank / all 360 harrison test presses. shield_mode: set_mode() on the shield."""
    name = lambda s: s if isinstance(s, str) else type(s).__name__  # noqa: E731
    mode = f" ({shield_mode})" if shield_mode else ""
    rep = Report(f"Keystroke Guard harness: attacker={name(attacker)}, shield={name(shield)}{mode}")
    box: dict[str, Any] = {}
    if attacker is not None:
        _attacker(rep, attacker, box)
    if shield is not None:
        _shield(rep, shield, box, shield_mode)
    if quality:
        load = lambda: _presses(presses, data)  # noqa: E731
        rep.run("quality", f"attacker on {data} presses", lambda: _q_attack(box, load), info=True)
        rep.run("quality", "shield vs that attacker", lambda: _q_shield(box, load), info=True)
    return rep


def _attacker(rep: Report, spec: Any, box: dict) -> None:
    def load() -> str:
        t0 = time.perf_counter()
        box["a"] = load_driver(spec, "make_attacker", "attacker")
        return f"attacker {type(box['a']).__name__} in {time.perf_counter() - t0:.1f} s"
    if not rep.run("load", "build attacker", load):
        return
    a = box["a"]
    audio = _click(SR, SR // 2)
    onsets = np.array([0, SR // 2, SR - 1])     # the edges must work too (windows are zero-padded)

    def protocol() -> str:
        assert isinstance(a, KeystrokeAttackerDriver), "not a KeystrokeAttackerDriver (needs name, classes, read)"
        assert isinstance(a.name, str) and a.name, f"name must be a non-empty str, got {a.name!r}"
        c = a.classes
        assert isinstance(c, list) and c and all(isinstance(k, str) for k in c), "classes must be a list of str"
        assert len(set(c)) == len(c), "classes has duplicates"
        return f"name={a.name!r}, {len(c)} classes"
    rep.run("contract", "attacker Protocol + attributes", protocol)

    def read() -> str:
        out = a.read(audio, onsets)
        assert isinstance(out, list) and len(out) == len(onsets), "read() must return one KeyGuess per onset"
        for g, o in zip(out, onsets):
            assert isinstance(g, KeyGuess), f"got {type(g).__name__}, not KeyGuess"
            assert g.onset == int(o), f"KeyGuess.onset {g.onset} != onset {o}"
            assert g.truth is None, "truth must be None: the attacker never gets the true key"
            assert g.top, "top is empty"
            keys, probs = [k for k, _ in g.top], [p for _, p in g.top]
            assert all(k in a.classes for k in keys), f"top has keys outside classes: {keys}"
            assert len(set(keys)) == len(keys), f"top repeats a key: {keys}"
            assert all(0.0 <= p <= 1.0 for p in probs), f"probabilities outside [0, 1]: {probs}"
            assert probs == sorted(probs, reverse=True), f"top is not sorted best first: {probs}"
            assert sum(probs) <= 1.0 + 1e-4, f"top probabilities sum to {sum(probs):.3f} > 1"
        assert a.read(audio, np.array([], dtype=int)) == [], "read() with no onsets must return []"
        box["k"] = len(out[0].top)
        return f"one KeyGuess per onset (edges too), top-{box['k']} sorted, keys in classes, truth None, [] for none"
    rep.run("contract", "read() -> [KeyGuess]", read)

    def determinism() -> str:
        x, y = a.read(audio, onsets), a.read(audio, onsets)
        assert [g.top for g in x] == [g.top for g in y], "same audio and onsets gave different guesses"
        return "same input twice, same guesses"
    rep.run("contract", "attacker deterministic", determinism)

    def latency() -> str:
        one = timed(lambda: a.read(audio, onsets[1:2]), 20)
        med = float(np.median(one))
        assert med < ATTACK_BUDGET_MS, f"median {med:.1f} ms per keystroke, budget {ATTACK_BUDGET_MS:.0f} ms"
        return f"median {med:.1f} ms, max {one.max():.1f} ms per keystroke (budget {ATTACK_BUDGET_MS:.0f} ms)"
    rep.run("latency", "attacker read 1 keystroke", latency)


def _shield(rep: Report, spec: Any, box: dict, mode: str | None = None) -> None:
    def load() -> str:
        t0 = time.perf_counter()
        box["s"] = load_driver(spec, "make_shield", "shield")
        if mode:
            box["s"].set_mode(mode)
        return f"shield {type(box['s']).__name__} in {time.perf_counter() - t0:.1f} s"
    if not rep.run("load", "build shield", load):
        return
    s = box["s"]
    n = 50 * BLOCK
    speech = probe_audio(n)[1]

    def protocol() -> str:
        assert isinstance(s, ShieldDriver), "not a ShieldDriver (needs name, process, reset)"
        assert isinstance(s.name, str) and s.name, f"name must be a non-empty str, got {s.name!r}"
        lat = getattr(s, "latency", 0)
        assert isinstance(lat, (int, np.integer)) and lat >= 0, f"latency must be an int >= 0 samples, got {lat!r}"
        box["lat"] = int(lat)
        return f"name={s.name!r}, latency {int(lat)} samples = {1000 * lat / SR:.0f} ms"
    rep.run("contract", "shield Protocol + latency", protocol)
    lat = box.get("lat", 0)

    def stream(x: np.ndarray, events: dict[int, list[int]]) -> np.ndarray:
        s.reset()
        out = [s.process(x[i:i + BLOCK], events.get(i, [])) for i in range(0, len(x), BLOCK)]
        for o in out:
            assert isinstance(o, np.ndarray) and len(o) == BLOCK, f"process() returned {len(o)} samples for {BLOCK}"
            assert np.all(np.isfinite(o)), "process() returned NaN/inf"
        return np.concatenate(out)

    def passthrough() -> str:
        y = stream(speech, {})
        assert np.array_equal(y[lat:], speech[:len(speech) - lat]), \
            f"with no key events the output must equal the input delayed by latency ({lat}); " \
            f"max diff {np.abs(y[lat:] - speech[:len(speech) - lat]).max():.2e}"
        return f"exact pass-through (delayed {lat} samples) with no key events"
    rep.run("contract", "no keys -> exact pass-through", passthrough)

    def keyed() -> str:
        x = speech + _click(n, 20 * BLOCK)
        y = stream(x, {21 * BLOCK: [20 * BLOCK + 10]})    # OS key events arrive about a block late
        d = np.abs(y[lat:] - x[:len(x) - lat])
        assert d.max() > 1e-4, "a key event changed nothing: the shield doesn't act on keystrokes"
        hit = np.flatnonzero(d > 1e-4)
        return f"same length, finite; changed samples {hit[0] - 20 * BLOCK - 10:+d}..{hit[-1] - 20 * BLOCK - 10:+d} " \
               f"around the key (late event)"
    rep.run("contract", "key event -> audio changed", keyed)

    def latency() -> str:
        s.reset()
        blk, i = speech[:BLOCK], [0]

        def step():  # a key event every block: every block is key-touched
            i[0] += 1
            return s.process(blk, [i[0] * BLOCK])
        ms = timed(step, 100)
        p95 = float(np.percentile(ms, 95))
        assert p95 < SHIELD_BUDGET_MS, f"p95 {p95:.1f} ms per block with a key active, budget {SHIELD_BUDGET_MS:.0f} ms"
        return f"median {np.median(ms):.1f} ms, p95 {p95:.1f} ms, max {ms.max():.1f} ms per 20 ms block, key active; " \
               f"+{1000 * lat / SR:.0f} ms constant delay"
    rep.run("latency", "shield 20 ms block, key active", latency)


def _harrison(presses: int):
    from app.keystroke_guard.driver import HARRISON, harrison_split
    if not HARRISON.exists():
        raise Skip(f"no {HARRISON} (run python -m app.keystroke_guard.get_assets)")
    _, _, X, y = harrison_split()
    from keyguard.config import CLASSES, PRE_S
    if presses:
        idx = np.random.default_rng(0).permutation(len(X))[:presses]
        X, y = X[idx], y[idx]
    return X, np.array([CLASSES[i] for i in y]), int(PRE_S * SR)


def _bank(presses: int):
    from app.keystroke_guard.driver import BANK, keyguard_bank
    if not BANK.exists():
        raise Skip(f"no {BANK} (run python -m app.keystroke_guard.get_assets)")
    from keyguard.config import KEY_WIN, PRE_S
    bank, rng = keyguard_bank(), np.random.default_rng(0)
    per_key = presses // len(bank) if presses else 10
    X, keys = [], []
    for k in sorted(bank):
        for c in bank[k][rng.permutation(len(bank[k]))[:per_key]]:
            X.append(np.pad(c, (0, max(0, KEY_WIN - len(c))))[:KEY_WIN])    # same layout as harrison windows
            keys.append(k)
    X = np.stack(X) + np.float32(NOISE) * rng.standard_normal((len(X), KEY_WIN), dtype=np.float32)
    return X, np.array(keys), int(PRE_S * SR)


def _presses(presses: int, data: str):
    if data not in ("bank", "harrison"):
        raise ValueError(f"data must be bank | harrison, got {data!r}")
    return _bank(presses) if data == "bank" else _harrison(presses)


def _topk(a: Any, audio: np.ndarray, onsets: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    guesses = a.read(audio, onsets)
    ranks = [[k for k, _ in g.top].index(t) if t in [k for k, _ in g.top] else 99 for g, t in zip(guesses, truth)]
    return 100 * np.mean(np.array(ranks) < 1), 100 * np.mean(np.array(ranks) < 3)


def _q_attack(box: dict, load) -> str:
    if "a" not in box:
        raise Skip("no attacker")
    X, keys, pre = load()
    # presses laid end to end: each is a KEY_WIN window with its onset PRE_S in, so read() cuts it back exactly
    audio, onsets = X.reshape(-1), np.arange(len(X)) * X.shape[1] + pre
    top1, top3 = _topk(box["a"], audio, onsets, keys)
    box["clean"] = (top1, top3)
    c = len(set(keys))
    return f"keys-only, oracle onsets, n={len(X)}: top-1 {top1:.1f} %, top-3 {top3:.1f} % " \
           f"(chance {100 / c:.1f} / {300 / c:.1f} %)"


def _q_shield(box: dict, load) -> str:
    if "a" not in box or "s" not in box:
        raise Skip("needs both an attacker and a shield")
    a, s = box["a"], box["s"]
    X, keys, pre = load()
    lat, gap = int(getattr(s, "latency", 0)), SR // 4
    # presses 250 ms apart, streamed through the shield in 20 ms blocks; each key event arrives one block late
    hop = X.shape[1] + gap
    x = np.zeros(len(X) * hop + lat + BLOCK, np.float32)
    for i, w in enumerate(X):
        x[i * hop: i * hop + len(w)] = w
    onsets = np.arange(len(X)) * hop + pre
    s.reset()
    out = []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        out.append(s.process(x[i:i + BLOCK], [int(o) for o in onsets if i - BLOCK <= o < i]))
    y = np.concatenate(out)[lat:]
    top1, top3 = _topk(a, y, onsets, keys)
    t1, t3 = box.get("clean", (float("nan"), float("nan")))
    return f"shielded, oracle onsets, n={len(X)}: top-1 {top1:.1f} %, top-3 {top3:.1f} % (unshielded {t1:.1f} / {t3:.1f} %)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.keystroke_guard.harness", description=__doc__.split("\n")[0])
    ap.add_argument("--attacker", default="mock", help="mock | real | module.path:ClassName | none (default mock)")
    ap.add_argument("--shield", default="mock", help="mock | real | module.path:ClassName | none (default mock)")
    ap.add_argument("--presses", type=int, default=0, help="quality presses (default 0 = 10 per key / all 360 harrison)")
    ap.add_argument("--data", default="bank", choices=("bank", "harrison"), help="quality presses: Keyguard's bank "
                    "(default, the CTC attacker's domain) or harrison test presses")
    ap.add_argument("--shield-mode", help="set_mode() on the shield: dsp | adversarial | dsp+adversarial (real)")
    ap.add_argument("--no-quality", action="store_true", help="contract + latency only")
    a = ap.parse_args(argv)
    none = lambda v: None if v == "none" else v  # noqa: E731
    rep = check(none(a.attacker), none(a.shield), a.presses, quality=not a.no_quality, shield_mode=a.shield_mode,
                data=a.data)
    rep.print()
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
