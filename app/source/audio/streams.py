"""Live audio streams.

MicShieldStream: physical mic -> hook(block, start_sample) -> VB-CABLE "CABLE Input" (Zoom's mic = "CABLE Output").
  The hook is the shield. The audio thread runs it and copies raw + shielded blocks into rings for the workers.
  Any hook exception or bad output -> pass through the raw block and `errors += 1`, so the call never goes silent.
LoopbackStream: the speaker the meeting plays to, captured via WASAPI loopback (soundcard) on a thread -> ring.

Streams open at 16 kHz and let the OS convert (WASAPI auto_convert, MME and soundcard do it natively), so no
resampler runs in the callback; add a stateful polyphase resampler only if a device refuses 16 kHz.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from app.source.audio import devices as dv
from app.source.audio.keys import KeyClock
from app.source.audio.ring import Ring
from app.source.types import BLOCK, SR

Hook = Callable[[np.ndarray, int], np.ndarray]  # (block, absolute start sample) -> same-length block


class MicShieldStream:
    def __init__(self, hook: Hook | None = None, mic: str | None = None, out: str | None = dv.CABLE_IN,
                 keyclock: KeyClock | None = None, ring_seconds: float = 30.0):
        self.hook = hook
        self.mic, self.out = mic, out
        self.keyclock = keyclock
        self.raw = Ring(ring_seconds)       # what the mic heard (attacker's view without the shield)
        self.shielded = Ring(ring_seconds)  # what the meeting receives
        self.position = 0                   # absolute index of the next sample
        self.errors = 0
        self.last_error: str | None = None
        self.xruns = 0
        self.output_name: str | None = None
        self._in_latency = 0.0
        self._stream = None

    def process(self, block: np.ndarray) -> np.ndarray:
        """One block through the hook with pass-through on failure. Also the entry point for replay mode."""
        block = np.asarray(block, np.float32).reshape(-1)
        start = self.position
        y = block
        if self.hook is not None:
            try:
                out = np.asarray(self.hook(block, start), np.float32).reshape(-1)
                if out.shape != block.shape or not np.all(np.isfinite(out)):
                    raise ValueError(f"hook returned shape {out.shape} / non-finite samples")
                y = out
            except Exception as e:  # noqa: BLE001 - never let a model break the audio path
                self.errors += 1
                self.last_error = repr(e)
        self.raw.write(block)  # Ring.write copies into its own buffer
        self.shielded.write(y)
        self.position = start + len(block)
        return y

    def _callback(self, indata, outdata, frames, t, status) -> None:
        if status:
            self.xruns += 1
        if self.keyclock is not None:
            # the block's last sample was captured ~input latency before now
            self.keyclock.anchor(self.position + frames, time.monotonic() - self._in_latency)
        y = self.process(indata[:, 0])
        if outdata is not None:
            outdata[:] = y[:, None]

    def start(self) -> MicShieldStream:
        import sounddevice as sd
        devs = dv.list_devices()
        err = None
        for api in dv.PREFERRED_HOSTAPIS:  # both ends must share a host API
            mic = dv.pick_mic(devs, self.mic)
            mic = mic and dv.find_device(mic["name"][:31], "input", hostapi=api, devices=devs)  # MME cuts names at 31
            out = self.out and dv.find_device(self.out, "output", hostapi=api, devices=devs)
            if not mic:
                continue
            extra = sd.WasapiSettings(auto_convert=True) if api == "Windows WASAPI" else None
            try:
                if out:
                    ch = min(2, out["outputs"])
                    s = sd.Stream(device=(mic["index"], out["index"]), samplerate=SR, blocksize=BLOCK,
                                  channels=(1, ch), dtype="float32", callback=self._callback,
                                  extra_settings=(extra, extra))
                    self._in_latency = s.latency[0]
                else:
                    s = sd.InputStream(device=mic["index"], samplerate=SR, blocksize=BLOCK, channels=1,
                                       dtype="float32", extra_settings=extra,
                                       callback=lambda i, f, t, st: self._callback(i, None, f, t, st))
                    self._in_latency = s.latency
            except Exception as e:  # noqa: BLE001 - try the next host API
                err = e
                continue
            self._stream = s
            self.output_name = out["name"] if out else None
            if not out:
                print(f"[streams] no output device matching {self.out!r}: mic is analysed but NOT routed to the "
                      f"meeting.\n{dv.INSTALL_STEPS if self.out == dv.CABLE_IN else ''}")
            s.start()
            return self
        raise RuntimeError(f"could not open mic stream ({err or 'no input device'})")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class LoopbackStream:
    """Far-end audio: everything the chosen speaker plays (the meeting's inbound voices) -> ring at 16 kHz mono."""

    def __init__(self, speaker: str | None = None, ring_seconds: float = 30.0, on_block=None):
        self.speaker = speaker
        self.ring = Ring(ring_seconds)
        self.on_block = on_block            # instead of the ring: meet mode routes system audio through the pipeline
        self.errors = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        try:
            import soundcard as sc
            name = self.speaker or sc.default_speaker().name
            mic = sc.get_microphone(name, include_loopback=True)
            with mic.recorder(samplerate=SR, channels=1, blocksize=BLOCK) as rec:
                while not self._stop.is_set():
                    x = rec.record(numframes=BLOCK)[:, 0].astype("float32")
                    (self.on_block or self.ring.write)(x)
        except Exception as e:  # noqa: BLE001 - inbound scoring stops, the call does not
            self.errors += 1
            self.last_error = repr(e)
            print(f"[streams] loopback capture stopped: {e}")

    def start(self) -> LoopbackStream:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="loopback", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
