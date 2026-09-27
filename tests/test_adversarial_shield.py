"""Adversarial keystroke shield: the runtime delta stage, the KeyguardShield modes and the pipeline switch.
No trained model needed: a fake deltas file is written to tmp_path. Driver-level tests skip without the Keyguard data (get_assets)."""
import numpy as np
import pytest
import torch

from app.hearsay.mock import MockVoice
from app.keystroke_guard import adversarial as adv
from app.keystroke_guard import driver as kr
from app.keystroke_guard.mock import MockAttacker
from app.source import config
from app.source.bus import EventBus
from app.source.pipeline import Pipeline
from app.source.types import BLOCK, SR, ShieldDriver

K, BUDGET_DB, GAIN = 4, -18.0, 0.5
RADIUS = 10 ** (BUDGET_DB / 20) * np.sqrt(adv.KEY_WIN)
keyguard = pytest.mark.skipif(not kr.HARRISON.exists(), reason="Keyguard data missing (get_assets)")


@pytest.fixture
def deltas(tmp_path):
    u = torch.randn(K, adv.KEY_WIN, generator=torch.Generator().manual_seed(0))
    u *= RADIUS / u.norm(dim=1, keepdim=True)
    path = tmp_path / "deltas.pt"
    torch.save({"deltas": u, "meta": {"level_gain": GAIN, "budget_db": BUDGET_DB, "K": K}}, path)
    return path


def clicks(n, onsets, seed=0):
    x = (0.003 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)
    for o in onsets:
        x[o:o + 400] += (0.4 * np.exp(-np.arange(400) / 60)).astype(np.float32)
    return x


def stream(proc, x, events, late=BLOCK):
    """20 ms blocks; each key event handed over `late` samples after its onset (OS events lag the sound)."""
    out = [proc(x[i:i + BLOCK], [e for e in events if i <= e + late < i + BLOCK]) for i in range(0, len(x), BLOCK)]
    return np.concatenate(out)


def test_missing_deltas_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="adversarial train"):
        adv.DeltaStage.load(tmp_path / "missing.pt")


def test_delta_added_at_absolute_position_under_budget(deltas):
    st = adv.DeltaStage.load(deltas, seed=0, harden=())       # no per-stroke shift: exact placement
    lat, e = 4 * BLOCK, 10 * BLOCK + 37
    x = clicks(40 * BLOCK, [e])
    buf_len = 24 * BLOCK + lat

    def proc(block, events, t=[0]):                 # the driver's delay line, stage only (out = zeros)
        t[0] += len(block)
        st.add(events)
        return st.apply(np.zeros(len(block), np.float32), t[0] - lat - len(block), x[max(0, t[0] - buf_len):t[0]], t[0])
    y = stream(proc, x, [e])[lat:]                  # absolute index s of the stream = y[s]
    (a, k, level, scale), = st.history
    assert a == e - adv.PRE and 0 <= k < K and scale == 1.0   # one stroke: the cap doesn't bite
    assert level == pytest.approx(GAIN * np.sqrt(np.mean(x[a:a + adv.EST].astype(np.float64) ** 2)), rel=1e-5)
    want = np.zeros_like(y)
    want[a:a + adv.KEY_WIN] = level * st.deltas[k][:len(y) - a]
    np.testing.assert_allclose(y, want, atol=1e-7)
    assert np.linalg.norm(y) <= level * RADIUS * (1 + 1e-5)   # ||delta|| <= level * 10^(budget/20) * sqrt(KEY_WIN)


def test_random_choice_among_k(deltas):
    st = adv.DeltaStage.load(deltas, seed=3)
    x = np.full(200 * adv.KEY_WIN, 0.1, np.float32)
    for i in range(40):                                      # each stroke on time: its start hasn't gone out yet
        a = i * adv.KEY_WIN + 2000
        st.add([a + adv.PRE])
        start = a - adv.SHIFT - BLOCK // 2               # the block reaches the earliest shifted start
        t = start + BLOCK + 4 * BLOCK                        # the driver's 80 ms lookahead covers EST + SHIFT
        st.apply(np.zeros(BLOCK, np.float32), start, x[:t], t)
    ks = [k for _, k, _, _ in st.history]
    assert len(ks) == 40 and set(ks) == set(range(K))       # every delta used, none fixed


@keyguard
def test_driver_modes_contract_and_constant_latency(deltas):
    s = kr.KeyguardShield(deltas=deltas)
    assert isinstance(s, ShieldDriver) and s.mode == "dsp"
    lat = s.latency
    x = clicks(60 * BLOCK, [])
    for mode in kr.SHIELD_MODES:                    # no key events -> exact pass-through, same delay, every mode
        s.set_mode(mode)
        s.reset()
        y = stream(s.process, x, [])
        assert s.latency == lat and len(y) == len(x)
        np.testing.assert_array_equal(y[lat:], x[:len(x) - lat])
    s.reset()                                       # switching mid-stream: no jump in the stream
    out = []
    for n, i in enumerate(range(0, len(x), BLOCK)):
        s.set_mode(kr.SHIELD_MODES[n % 3])
        out.append(s.process(x[i:i + BLOCK], []))
    np.testing.assert_array_equal(np.concatenate(out)[lat:], x[:len(x) - lat])


@keyguard
def test_driver_adversarial_adds_the_delta_only(deltas):
    e = 20 * BLOCK + 11
    x = clicks(60 * BLOCK, [e])
    s = kr.KeyguardShield(mode="adversarial", deltas=deltas)
    y = stream(s.process, x, [e])[s.latency:]
    (a, k, level, scale), = s.adv.history
    d = y - x[:len(y)]
    assert np.all(np.isfinite(y)) and abs(a - (e - adv.PRE)) <= adv.SHIFT   # per-stroke random shift (hardening)
    np.testing.assert_allclose(d[a:a + adv.KEY_WIN], scale * level * s.adv.deltas[k][:len(d) - a], atol=1e-6)
    assert np.abs(d[:a]).max() == 0 and np.abs(d[a + adv.KEY_WIN:]).max() == 0
    s.set_mode("dsp+adversarial")                   # DSP inpainting too: changes more than the delta
    s.reset()
    y2 = stream(s.process, x, [e])[s.latency:]
    assert np.abs(y2 - x[:len(y2)] - d).max() > 1e-3


@keyguard
def test_driver_missing_deltas_keeps_mode(tmp_path):
    s = kr.KeyguardShield(deltas=tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        s.set_mode("adversarial")
    assert s.mode == "dsp"
    x = clicks(30 * BLOCK, [])
    np.testing.assert_array_equal(stream(s.process, x, [])[s.latency:], x[:len(x) - s.latency])


class FakeShield:
    """A ShieldDriver with set_mode, like KeyguardShield, without Keyguard."""
    name, latency = "fake", 4 * BLOCK

    def __init__(self, fail=False):
        self.fail, self.modes = fail, []
        self.reset()

    def set_mode(self, mode):
        if self.fail and "adversarial" in mode:
            raise FileNotFoundError("adversarial deltas not found at runs/adversarial_deltas.pt")
        self.modes.append(mode)

    def reset(self):
        self._buf = np.zeros(self.latency, np.float32)

    def process(self, block, key_events):
        y = np.concatenate([self._buf, block])
        self._buf = y[len(block):]
        return y[:len(block)]


def pipe_with(shield, mode="dsp"):
    cfg = config.load(env={})
    cfg.drivers.secret, cfg.drivers.shield_mode = "mock", mode
    bus = EventBus()
    errors, states = [], []
    bus.subscribe("driver.error", lambda e: errors.append(e.data))
    bus.subscribe("shield.state", lambda e: states.append(e.data))
    return Pipeline(cfg, bus, voice=MockVoice(), attacker=MockAttacker(), shield=shield), bus, errors, states


def test_pipeline_set_shield_round_trip():
    sh = FakeShield()
    pipe, bus, _, states = pipe_with(sh)
    lag = pipe._shield_lag()
    for mode in ("adversarial", "off", "dsp", "adversarial", "dsp"):
        assert pipe.set_shield(mode) == mode and pipe.shield_mode == mode
        assert pipe._shield_lag() == lag            # one driver, one delay, whatever the mode
    assert sh.modes == ["dsp", kr.DASHBOARD_ADVERSARIAL, "dsp", kr.DASHBOARD_ADVERSARIAL, "dsp"]
    bus.flush()
    assert states[-1]["mode"] == "dsp" and not states[-1]["failed"]
    bus.close()


def test_pipeline_adversarial_unavailable_is_a_clear_error_and_dsp_still_works():
    pipe, bus, errors, _ = pipe_with(FakeShield(fail=True), mode="adversarial")
    assert pipe.shield_mode == "dsp"                # configured adversarial, no deltas: starts on dsp
    bus.flush()
    assert errors and "deltas not found" in errors[-1]["error"]
    with pytest.raises(ValueError, match="adversarial shield unavailable"):
        pipe.set_shield("adversarial")
    assert pipe.shield_mode == "dsp"
    assert pipe.set_shield("off") == "off" and pipe.set_shield("dsp") == "dsp"
    bus.close()
