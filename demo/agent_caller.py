"""Play the AI agent's WAV lines into a Zoom call (run on the second device that joins as "IT Support").

Point Zoom's microphone on that device at a virtual cable (e.g. "CABLE Output") and play into its input side:
    python demo/agent_caller.py --list
    python demo/agent_caller.py demo/audio/agent_lines --device "CABLE Input" --pause 2.5
Arguments are WAV files or folders (a folder's *.wav play in name order).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", nargs="*", type=Path, help="WAV files or folders, played in order")
    ap.add_argument("--device", help="output device name substring (default: system default)")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds of silence between lines")
    ap.add_argument("--list", action="store_true", help="list output devices and exit")
    a = ap.parse_args()
    if a.list:
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                print(f"{i:3d}  {d['name']}")
        return
    files = [f for p in a.lines for f in (sorted(p.glob("*.wav")) if p.is_dir() else [p])]
    if not files:
        ap.error("no WAV lines given")
    dev = None
    if a.device:
        hits = [i for i, d in enumerate(sd.query_devices())
                if d["max_output_channels"] > 0 and a.device.lower() in d["name"].lower()]
        if not hits:
            ap.error(f"no output device matches {a.device!r} (see --list)")
        dev = hits[0]
    for i, f in enumerate(files):
        x, sr = sf.read(f, dtype="float32")
        print(f"[{i + 1}/{len(files)}] {f.name} ({len(x) / sr:.1f} s)")
        sd.play(x, sr, device=dev)
        sd.wait()
        if i < len(files) - 1:
            time.sleep(a.pause)


if __name__ == "__main__":
    main()
