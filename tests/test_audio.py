"""WP2 audio I/O: synthetic signals only, no devices."""
import time

import numpy as np
import soundfile as sf

from app.source.audio.devices import find_device, pick_mic, routing_status
from app.source.audio.keys import KeyClock, ScriptedKeyClock
from app.source.audio.replay import FileSource, load_key_track, load_wav, mix
from app.source.audio.ring import Ring
from app.source.audio.streams import MicShieldStream
from app.source.audio.vad import Vad, speech_fraction
from app.source.types import BLOCK, SR

rng = np.random.default_rng(0)
t = np.arange(SR) / SR
TONE = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
NOISE = (0.3 * rng.standard_normal(SR)).astype(np.float32)
SILENCE = np.zeros(SR, np.float32)


def test_ring_absolute_indexing_and_wrap():
    r = Ring(seconds=1.0, sr=100)  # cap 100
    for i in range(0, 250, 30):
        r.write(np.arange(i, min(i + 30, 250), dtype=np.float32))
    assert r.total == 250
    assert np.array_equal(r.read_range(160, 170), np.arange(160, 170))
    assert np.array_equal(r.read_last(5), np.arange(245, 250))
    assert r.read_range(100, 110) is None  # overwritten
    assert r.read_range(245, 251) is None  # not written yet
    assert len(r.read_last(1000)) == 100
    r.write(np.arange(1000, 1300, dtype=np.float32))  # longer than capacity
    assert np.array_equal(r.read_last(3), [1297, 1298, 1299]) and r.total == 550


def test_vad_tone_noise_silence():
    assert speech_fraction(TONE) > 0.9
    assert speech_fraction(NOISE) < 0.1   # loud but broadband: zero-crossing rate too high
    assert speech_fraction(SILENCE) == 0.0
    v = Vad(hangover=3)
    assert v(TONE[:BLOCK])
    assert [v(SILENCE[:BLOCK]) for _ in range(4)] == [True, True, True, False]


def test_keyclock_maps_time_to_samples():
    kc = KeyClock()
    assert kc.press("a", t=5.0) is None  # no stream anchored yet
    kc.anchor(32_000, t=10.0)
    assert kc.press("a", t=10.5).sample == 40_000
    assert kc.press("b", t=9.9).sample == 30_400
    kc.offset_s = 0.01
    assert kc.sample_of(10.0) == 32_160
    assert kc.in_range(30_000, 35_000) == [30_400]
    assert [e.key for e in kc.recent()] == ["a", "b"]


def test_scripted_keyclock_and_track(tmp_path):
    kc = ScriptedKeyClock.from_seconds([(1.0, "x"), (0.5, "y")])
    assert kc.in_range(0, SR * 2) == [8_000, 16_000]
    kc.anchor(10_000)
    assert [e.key for e in kc.recent()] == ["y"]
    p = tmp_path / "keys.csv"
    p.write_text("t,key\n0.25,p\n0.75,w\n")
    assert load_key_track(p).in_range(0, SR) == [4_000, 12_000]


def test_replay_blocks_timing_resample_and_mix(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.ones(4410, np.float32) * 0.1, 44_100)  # 0.1 s at 44.1 kHz
    x = load_wav(p)
    assert abs(len(x) - 1600) <= 1
    src = FileSource(np.ones(1000, np.float32), offset_s=0.01)
    blocks = list(src.blocks())
    assert len(src) == 1160 and len(blocks) == 4 and all(len(b) == BLOCK for b in blocks)
    assert blocks[0][159] == 0 and blocks[0][160] == 1 and blocks[-1][-1] == 0
    t0 = time.monotonic()
    n = sum(1 for _ in FileSource(np.zeros(BLOCK * 10, np.float32)).blocks(realtime=True))
    assert n == 10 and 0.17 <= time.monotonic() - t0 < 0.5  # 10 blocks x 20 ms, first one immediate
    m = mix(FileSource(np.ones(10, np.float32)), FileSource(np.ones(20, np.float32), gain=2))
    assert len(m) == 20 and m.audio[0] == 3 and m.audio[15] == 2


def test_mic_callback_passthrough_on_hook_error():
    def bad(block, start):
        raise RuntimeError("model died")

    kc = KeyClock()
    s = MicShieldStream(hook=bad, keyclock=kc)
    indata = TONE[:BLOCK, None]
    outdata = np.zeros((BLOCK, 2), np.float32)
    s._callback(indata, outdata, BLOCK, None, None)
    assert np.array_equal(outdata[:, 0], TONE[:BLOCK]) and np.array_equal(outdata[:, 1], TONE[:BLOCK])
    assert s.errors == 1 and "model died" in s.last_error and s.position == BLOCK and kc.now_sample == BLOCK

    s.hook = lambda b, start: b[:10]  # wrong length -> pass-through too
    assert np.array_equal(s.process(TONE[:BLOCK]), TONE[:BLOCK]) and s.errors == 2

    starts = []
    s.hook = lambda b, start: starts.append(start) or b * 0.5
    y = s.process(TONE[:BLOCK])
    assert np.allclose(y, TONE[:BLOCK] * 0.5) and starts == [2 * BLOCK]
    assert np.allclose(s.raw.read_last(BLOCK), TONE[:BLOCK])
    assert np.allclose(s.shielded.read_last(BLOCK), TONE[:BLOCK] * 0.5)


def test_routing_status_without_cable():
    devs = [dict(index=0, name="Microphone Array", hostapi="MME", inputs=2, outputs=0, samplerate=48000.0),
            dict(index=1, name="Speakers", hostapi="MME", inputs=0, outputs=2, samplerate=48000.0)]
    ok, msg = routing_status(devs, speakers=["Speakers"])
    assert not ok and "vb-audio.com/Cable" in msg
    devs += [dict(index=2, name="CABLE Input (VB-Audio Virtual Cable)", hostapi="MME", inputs=0, outputs=2,
                  samplerate=48000.0),
             dict(index=3, name="CABLE Output (VB-Audio Virtual Cable)", hostapi="MME", inputs=2, outputs=0,
                  samplerate=48000.0)]
    ok, msg = routing_status(devs, speakers=["Speakers"])
    assert ok and "MISSING" not in msg
    assert pick_mic(devs)["index"] == 0  # never the virtual cable
    assert find_device("cable input", "output", devices=devs)["index"] == 2
