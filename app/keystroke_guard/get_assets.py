"""Copy the vendored Keyguard's weights and data (never committed) from a Keyguard checkout into Athena.

Weights/data go to runs/keyguard/ and data/keyguard/ (where keyguard.config.RUNS / DATA point). Source checkout:
KEYGUARD_ROOT, else the first sibling that has the attacker weights. Idempotent: existing files are kept.
Run: uv run python -m app.keystroke_guard.get_assets [--root PATH]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS, DATA = REPO / "runs" / "keyguard", REPO / "data" / "keyguard"   # = keyguard.config.RUNS / DATA
# (relative path in the Keyguard checkout, destination root, required?)
ASSETS = [
    ("runs/ctc_rich_ft.pt", RUNS, True),        # Ares: the CTC attacker
    ("data/live_bank_rich.npz", DATA, True),    # per-key press bank (demo utterance, demo audio, harness)
    ("data/pool/harrison.npz", DATA, True),     # harrison presses (KeyNet, adversarial deltas)
    ("data/speech", DATA, False),               # LibriSpeech clips (adversarial deltas, demo)
    ("data/harrison/MBPWavs", DATA, False),     # per-key wavs (demo mixture)
    ("runs/demo_attacker.pt", RUNS, False),     # demo attacker
    ("runs/supervised_mbp.pt", RUNS, False),    # ... its fallback
    ("runs/demo_cache.json", RUNS, False),      # ... its demo-mode cache
    ("runs/pareto.json", RUNS, False),
    ("runs/arena", RUNS, False),                # past co-training runs
    ("runs/arena_memory.jsonl", RUNS, False),   # Ares/Athena local memory
]


def source_root() -> Path | None:
    if os.environ.get("KEYGUARD_ROOT"):
        return Path(os.environ["KEYGUARD_ROOT"])
    for p in (REPO / "upstream" / "keyguard", REPO.parents[1] / "keyboard", REPO.parent / "keyboard-acoustic-shield"):
        if (p / "runs" / "ctc_rich_ft.pt").exists():
            return p
    return None


def dest(rel: str, root: Path) -> Path:
    return root / Path(rel).relative_to(Path(rel).parts[0])  # strip the leading runs/ or data/


def missing() -> list[str]:
    """Required assets not yet in Athena."""
    return [rel for rel, root, req in ASSETS if req and not dest(rel, root).exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, help="Keyguard checkout to copy from (default: KEYGUARD_ROOT or a sibling)")
    a = ap.parse_args(argv)
    src = a.root or source_root()
    if src is None or not src.is_dir():
        print("no Keyguard checkout found: pass --root or set KEYGUARD_ROOT", file=sys.stderr)
        return 1
    if src.resolve() == REPO.resolve():
        print("--root must be the Keyguard checkout, not Athena", file=sys.stderr)
        return 1
    failed = []
    for rel, root, required in ASSETS:
        s, d = src / rel, dest(rel, root)
        if d.exists():
            print(f"ok    {d.relative_to(REPO)} (present)")
            continue
        if not s.exists():
            print(f"{'MISS' if required else 'skip'}  {rel} (not in {src})")
            if required:
                failed.append(rel)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, d)
        print(f"copy  {rel} -> {d.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
