"""Convert newly-downloaded external keystroke-audio datasets into
data/pool/<name>.npz {wins (n, KEY_WIN) float32, labels (n,) str}.

ponytail: two source shapes only (one-wav-per-key-with-many-presses, and
one-dir-per-key-with-one-press-per-file), so two small functions beat a
plugin framework. Add a third shape only when a third dataset needs it.

Run: uv run python3 -m keyguard.pool_convert_external
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .audio import load
from .config import CLS_IDX, DATA, SR
from .segment import onsets_n, windows

POOL = DATA / "pool"
CAP = 40                # max presses/key/keyboard
MIN_KEYS = 20           # of 36
MIN_WINDOWS = 100


def _key(stem: str) -> str | None:
    k = stem.upper()
    return k if k in CLS_IDX else None


def _pack(wins_by_key: dict[str, np.ndarray], name: str) -> Path | None:
    n_keys, n_wins = len(wins_by_key), sum(len(v) for v in wins_by_key.values())
    if n_keys < MIN_KEYS or n_wins < MIN_WINDOWS:
        print(f"SKIP {name}: {n_keys} keys, {n_wins} windows (below threshold)")
        return None
    wins = np.concatenate(list(wins_by_key.values())).astype(np.float32)
    labels = np.array([k for k, v in wins_by_key.items() for _ in range(len(v))])
    POOL.mkdir(parents=True, exist_ok=True)
    out = POOL / f"{name}.npz"
    np.savez_compressed(out, wins=wins, labels=labels)
    print(f"WROTE {out}: {n_wins} windows, {n_keys} keys, "
          f"presses/key min={min(len(v) for v in wins_by_key.values())} "
          f"max={max(len(v) for v in wins_by_key.values())}")
    return out


def convert_multi_press_dir(src_dir: Path, name: str, presses_per_file: int) -> Path | None:
    """One wav file per key, each containing many presses back-to-back."""
    wins_by_key = {}
    for f in sorted(Path(src_dir).glob("*.wav")):
        key = _key(f.stem)
        if key is None:
            continue
        y = load(f)
        on = onsets_n(y, presses_per_file)
        wins_by_key[key] = windows(y, on)[:CAP]
    return _pack(wins_by_key, name)


def convert_single_press_dirs(src_root: Path, name: str) -> Path | None:
    """One subdirectory per key, each holding individual single-press wav files."""
    wins_by_key = {}
    for keydir in sorted(Path(src_root).iterdir()):
        key = _key(keydir.name) if keydir.is_dir() else None
        if key is None:
            continue
        wins = []
        for f in sorted(keydir.glob("*.wav"))[:CAP]:
            y = load(f)
            on = onsets_n(y, 1)
            if len(on) == 0:               # ponytail: too short/quiet for onset detection -> clip start
                on = np.array([int(0.02 * SR)])
            wins.append(windows(y, on)[0])
        if wins:
            wins_by_key[key] = np.stack(wins)
    return _pack(wins_by_key, name)


def demo() -> None:
    """ponytail self-check: pack a synthetic 20-key domain, verify round-trip."""
    rng = np.random.default_rng(0)
    fake = {chr(ord("A") + i): rng.standard_normal((5, 4800)).astype(np.float32) for i in range(20)}
    out = _pack(fake, "_pool_convert_demo")
    assert out is not None and out.exists()
    d = np.load(out)
    assert d["wins"].shape == (100, 4800) and len(d["labels"]) == 100
    out.unlink()
    print("demo OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
        raise SystemExit(0)

    convert_multi_press_dir(
        DATA / "external/kad_spatam/mechanical_keyboard_dataset",
        "kad_spatam_mechanical", presses_per_file=75,
    )
    convert_single_press_dirs(
        DATA / "external/hf_sakuzas/training_data",
        "hf_sakuzas_keyboard",
    )
