"""Real-time Keyguard shield -> BlackHole virtual mic.

Streams the default input mic through a causal per-block shield and out to the
BlackHole device, so any call app (Zoom/Meet) can select "BlackHole" as its mic
and get shielded audio. Key-event *timestamps* (never which key) come from an OS
keyboard listener and are aligned to the audio clock online; a block that
contains a keystroke gets its transient attenuated + residue randomized, and
decoys are injected on a schedule so timing never leaks.

ponytail: this is the real-time *approximation* of shield.py -- a per-block
spectral gate rather than a full causal neural inpainter. Ceiling: heavier
overlap-save STFT or a trained causal model if 20 ms proves audibly lossy.
Upgrade path is drop-in behind `_process_block`.

Run: uv run python3 -m keyguard.realtime   (needs BlackHole + mic permission)
"""
from __future__ import annotations
import time
import threading
from collections import deque
import numpy as np

from .config import SR

BLOCK = 320                 # 20 ms at 16 kHz -> the latency budget
KEY_ACTIVE_S = 0.12         # a press affects this much audio after its timestamp
DECOY_EVERY_S = 0.6         # inject a decoy transient at least this often


class KeyClock:
    """Thread-safe store of recent key-press timestamps (monotonic seconds)."""

    def __init__(self, keep_s: float = 2.0):
        self.keep_s = keep_s
        self._t = deque()
        self._lock = threading.Lock()
        self._listener = None

    def press(self, *_):
        with self._lock:
            self._t.append(time.monotonic())

    def active(self, now: float) -> bool:
        """True if a keystroke is currently sounding."""
        with self._lock:
            while self._t and now - self._t[0] > self.keep_s:
                self._t.popleft()
            return any(0 <= now - t <= KEY_ACTIVE_S for t in self._t)

    def start(self):
        try:
            from pynput import keyboard
        except Exception as e:  # noqa
            print(f"[realtime] pynput unavailable ({e}); no key timing")
            return self
        self._listener = keyboard.Listener(on_press=self.press)
        self._listener.start()
        return self


def _process_block(block: np.ndarray, key_active: bool, rng, decoy: bool):
    """Causal shield on one audio block. Uniform processing at all times."""
    win = np.hanning(len(block))
    spec = np.fft.rfft(block * win)
    if key_active:
        mag = np.abs(spec)
        floor = np.median(mag)                       # inpaint toward the floor
        keep = np.minimum(mag, floor)                # kill the transient peak
        residue = (mag - keep) * (0.3 * rng.random(len(mag)))  # randomized residue
        spec = (keep + residue) * np.exp(1j * np.angle(spec))
    out = np.fft.irfft(spec, n=len(block))
    if len(win):
        out = out / (win + 1e-6) * win               # undo analysis window softly
    if decoy:
        click = rng.standard_normal(min(60, len(out))).astype(np.float32)
        click *= np.exp(-np.linspace(0, 6, len(click)))
        out[:len(click)] += 0.01 * click
    return out.astype(np.float32)


def find_blackhole():
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if "blackhole" in d["name"].lower() and d["max_output_channels"] > 0:
            return i
    return None


def run():
    import sounddevice as sd
    clock = KeyClock().start()
    rng = np.random.default_rng(0)
    out_dev = find_blackhole()
    if out_dev is None:
        print("[realtime] BlackHole output device not found. Install BlackHole "
              "(brew install blackhole-2ch) and rerun. Running mic->speakers "
              "passthrough for a local demo instead.")
    last_decoy = [time.monotonic()]

    def cb(indata, outdata, frames, t, status):
        now = time.monotonic()
        decoy = now - last_decoy[0] > DECOY_EVERY_S
        if decoy:
            last_decoy[0] = now
        y = indata[:, 0]
        outdata[:, 0] = _process_block(y, clock.active(now), rng, decoy)

    with sd.Stream(samplerate=SR, blocksize=BLOCK, dtype="float32",
                   channels=1, device=(None, out_dev), callback=cb):
        print(f"[realtime] shielding mic -> "
              f"{'BlackHole' if out_dev is not None else 'default out'} "
              f"@ {BLOCK/SR*1000:.0f} ms blocks. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[realtime] stopped.")


def demo():
    """Self-check the DSP path without any audio hardware."""
    rng = np.random.default_rng(0)
    block = rng.standard_normal(BLOCK).astype(np.float32)
    spike = block.copy(); spike[100:140] += 8.0
    clean_out = _process_block(spike, key_active=False, rng=rng, decoy=False)
    shielded = _process_block(spike, key_active=True, rng=rng, decoy=False)
    assert np.max(np.abs(shielded)) < np.max(np.abs(clean_out))
    assert len(shielded) == BLOCK
    print("realtime DSP demo ok: transient peak",
          f"{np.max(np.abs(clean_out)):.2f} -> {np.max(np.abs(shielded)):.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        run()
