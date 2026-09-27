"""Adversarial keystroke shield: K universal bounded perturbations, trained offline against an attacker ensemble,
added live at each OS key event (the streaming form of Keyguard's keyguard/shield/adversarial.py).

Runtime (DeltaStage, run inside KeyguardShield modes "adversarial" / "dsp+adversarial"): at each key event e, pick one
of the K deltas at random and add it on [e - PRE, e - PRE + KEY_WIN), scaled by the stroke's level. The level is the
RMS of the outgoing audio over [e - PRE, e - PRE + EST) (the first 60 ms, always inside the 80 ms lookahead by the
time the delta's first sample is due) times `level_gain`, the median full-window / first-60-ms RMS ratio of the
harrison train presses. So for keys alone the budget is Keyguard's (||delta||_2 <= key_rms * 10^(budget/20) *
sqrt(KEY_WIN), per stroke instead of one global key_rms); under speech it is relative to what the window really
carries (speech + key), which is what hides it. No state is learned at runtime: no warm-up, first stroke included.

Training (python -m app.keystroke_guard.adversarial train): maximize an attacker ensemble's error (clipped margin
loss) on harrison TRAIN presses through Keyguard's differentiable torch_logmel, with expectation over transformation:
onset misalignment uniform +/-40 ms (the delta and the level window move, the attacker's window does not), gain
+/-6 dB, train-speaker speech (Hearsay test_internal pools of eval/attack_under_speech.py, demo voices excluded)
at +0..+20 dB over the key in 70 % of the draws, and the runtime level estimate. Each delta is projected to the L2
budget after every step; a pairwise cosine penalty keeps the K deltas apart. Two phases: vs 3 attackers, then vs 4
after an adversarial retrain on the phase-1 deltas (co_train's move). `eval` measures the TEST split.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from app.keystroke_guard.driver import N_KEYS  # noqa: F401  (harrison A-Z0-9: the one class set)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
DELTAS = RUNS / "adversarial_deltas.pt"
KEY_WIN = 4800      # keyguard.config.KEY_WIN (0.30 s at 16 kHz); asserted when training
PRE = 320           # keyguard.config.PRE_S * SR: Keyguard's window starts 20 ms before the onset
EST = 960           # level window: [onset - PRE, onset - PRE + EST), 60 ms
JITTER = 640        # EOT onset misalignment, +/-40 ms
TAPER_IN, TAPER_OUT = 48, 320   # raised-cosine edges of every delta (3 ms in, 20 ms out), applied inside training
FADE = 80           # 5 ms: a delta cut short (next stroke, mode switch) fades out instead of stopping dead
DEDUP = 480         # events within 30 ms of an accepted one are the same press
SHIFT = 160         # hardening "shift": random +/-10 ms per stroke (OS error +/-30 ms + this stays within JITTER)
GRID = 160          # the cap checks attacker windows cut every 10 ms within +/-JITTER of each stroke's onset
HARDEN = ("shift",)  # per-stroke hardening the runtime ships (of "sign", "shift", "mix"; chosen by `eval`, README)

log = logging.getLogger(__name__)


def taper() -> np.ndarray:
    w = np.ones(KEY_WIN, np.float32)
    w[:TAPER_IN] = 0.5 - 0.5 * np.cos(np.pi * np.arange(TAPER_IN) / TAPER_IN)
    w[KEY_WIN - TAPER_OUT:] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(TAPER_OUT) + 1) / TAPER_OUT)
    return w


def _crop(x: np.ndarray, org: int, lo: int, hi: int) -> np.ndarray:
    """x (absolute index of x[0] = org) over [lo, hi), zero outside."""
    out = np.zeros(hi - lo, np.float64)
    a, b = max(lo, org), min(hi, org + len(x))
    if a < b:
        out[a - lo:b - lo] = x[a - org:b - org]
    return out


# --- runtime ----------------------------------------------------------------------------------------------------
class DeltaStage:
    """Adds one delta per key press on the absolute sample clock. No torch at runtime: plain numpy.

    Per press (event e, nominal start a = e - PRE), when the delta is due to go out: pick one of the K deltas with an
    OS-entropy rng (seed=None; an explicit seed is for tests), harden it (HARDEN), scale it by the level of
    [a', a' + EST) of `buf` (the audio the attacker will hear minus the delta: raw input, or the DSP output in
    dsp+adversarial) and add it on [a', a' + KEY_WIN). The running perturbation is cut (5 ms fade) where the new one
    starts, and the new one is scaled down, if needed, so every attacker window within +/-JITTER of every recent
    stroke keeps ||perturbation||^2 <= that stroke's budget (level * 10^(budget/20))^2 * KEY_WIN: overlapping strokes
    never stack past the per-stroke budget. A late event (its start already went out) shifts the delta later, up to
    JITTER, instead of dropping its head; later than that, the stroke gets no delta.
    """

    def __init__(self, deltas: np.ndarray, level_gain: float, seed: int | None = None, meta: dict | None = None,
                 harden: tuple = HARDEN):
        # (K, KEY_WIN), level-relative units; tapered here so no stroke starts or ends with a step (ponytail: the
        # taper isn't in training yet; retrain with it applied inside the perturbation for the optimum)
        self.deltas = np.ascontiguousarray(np.asarray(deltas, np.float32) * taper(), np.float32)
        assert self.deltas.ndim == 2 and self.deltas.shape[1] == KEY_WIN, self.deltas.shape
        self.level_gain, self.seed, self.meta, self.harden = float(level_gain), seed, meta or {}, tuple(harden)
        self.level_gain_dsp = float(self.meta.get("level_gain_dsp", level_gain))
        self.rel = 10 ** (float(self.meta.get("budget_db", -18.0)) / 20)
        self.reset()

    @classmethod
    def load(cls, path: str | Path | None = None, seed: int | None = None, **kw) -> "DeltaStage":
        path = Path(path or DELTAS)
        if not path.exists():
            raise FileNotFoundError(f"adversarial deltas not found at {path}; train them with "
                                    f"`python -m app.keystroke_guard.adversarial train`")
        import torch
        d = torch.load(path, map_location="cpu")
        return cls(d["deltas"].numpy(), d["meta"]["level_gain"], seed, d["meta"], **kw)

    @property
    def shift(self) -> int:
        return SHIFT if "shift" in self.harden else 0

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._pending: list[int] = []
        self._recent: deque = deque(maxlen=16)                  # accepted events (dedup)
        self._strokes: deque = deque()                          # (nominal start, energy budget) the cap protects
        self._p0, self._p = 0, np.zeros(0, np.float32)          # summed perturbation on [p0, p0 + len(p))
        self.history: deque = deque(maxlen=64)                  # (start, k, level, scale) per stroke

    def add(self, events) -> None:
        for e in sorted(int(e) for e in events):
            if all(abs(e - x) >= DEDUP for x in self._recent):
                self._recent.append(e)
                self._pending.append(e)

    def fade_out(self, at: int) -> None:
        """Mode switched away: drop pending strokes, fade what is still to go out over FADE samples from `at`."""
        self._pending.clear()
        self._cut(at)

    def _cut(self, at: int) -> None:
        i = max(at - self._p0, 0)
        f = self._p[i:i + FADE]
        f *= np.linspace(1, 0, FADE, endpoint=False, dtype=np.float32)[:len(f)]
        self._p[i + FADE:] = 0

    def apply(self, out: np.ndarray, start: int, buf: np.ndarray, t: int, dsp: bool = False) -> np.ndarray:
        """out = the outgoing block [start, start + len(out)); buf = the level source (raw input, or the DSP output
        aligned with it), buf[-1] is sample t - 1. Needs t - start - len(out) (the lookahead) >= EST + SHIFT."""
        n = len(out)
        for e in [e for e in self._pending if e - PRE - self.shift < start + n]:   # due: may start in this block
            self._pending.remove(e)
            self._stroke(e - PRE, start, buf, t - len(buf), self.level_gain_dsp if dsp else self.level_gain)
        keep = start + n - KEY_WIN - 3 * JITTER                  # older samples: no protected window reaches them
        if keep > self._p0:
            self._p, self._p0 = self._p[keep - self._p0:], keep
        i = start - self._p0
        seg = self._p[max(i, 0):max(i + n, 0)]
        if len(seg) and i >= 0:
            out[:len(seg)] += seg
        return out

    def _stroke(self, a: int, start: int, buf: np.ndarray, base: int, gain: float) -> None:
        rng, K = self.rng, len(self.deltas)
        a2 = a + (int(rng.integers(-self.shift, self.shift + 1)) if self.shift else 0)
        if a2 < start:                                          # late event: its head would already be out
            if start - a > JITTER:
                log.debug("key event %d samples late for its delta; stroke skipped", start - a)
                return
            a2 = start                                          # shift later instead (within the trained jitter)
        seg = buf[max(a2 - base, 0):max(a2 + EST - base, 0)]
        level = gain * float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) if len(seg) else 0.0
        k = int(rng.integers(K))
        d = self.deltas[k].astype(np.float64)
        if "mix" in self.harden:                                # convex mix of 2: ||.|| <= the budget still
            w = rng.random()
            d = w * d + (1 - w) * self.deltas[(k + 1 + int(rng.integers(K - 1))) % K]
        if "sign" in self.harden and rng.random() < 0.5:
            d = -d
        d *= level
        need = a2 + KEY_WIN - self._p0                          # grow the buffer to hold the new delta
        if need > len(self._p):
            self._p = np.concatenate([self._p, np.zeros(need - len(self._p), np.float32)])
        self._cut(a2)                                           # the running perturbation fades out where this starts
        while self._strokes and self._strokes[0][0] + JITTER + KEY_WIN <= a2:
            self._strokes.popleft()
        self._strokes.append((a, (level * self.rel) ** 2 * KEY_WIN))
        s = self._cap(d, a2)
        i = a2 - self._p0
        self._p[i:i + KEY_WIN] += (s * d).astype(np.float32)
        self.history.append((a2, k, level, s))

    def _cap(self, d: np.ndarray, a2: int) -> float:
        """Largest s in [0, 1] keeping ||p + s d||^2 <= budget on every protected window (a quadratic in s each)."""
        s = 1.0
        for a, budget in self._strokes:
            for w in range(a - JITTER, a + JITTER + 1, GRID):
                q = _crop(d, a2, w, w + KEY_WIN)
                qq = q @ q
                if qq == 0:
                    continue
                p = _crop(self._p, self._p0, w, w + KEY_WIN)
                pq, pp = p @ q, p @ p
                disc = pq * pq - qq * (pp - budget)
                s = min(s, max(0.0, (-pq + np.sqrt(disc)) / qq) if disc >= 0 else 0.0)
        return s


# --- training ---------------------------------------------------------------------------------------------------
def _torch(threads: int):
    os.environ.setdefault("KEYGUARD_DEVICE", "cpu")
    sys.dont_write_bytecode = True
    import torch
    torch.set_num_threads(threads)
    from app.keystroke_guard.driver import _import_keyguard
    _import_keyguard()
    from keyguard.config import KEY_WIN as KW, PRE_S, SR
    assert (KW, int(PRE_S * SR)) == (KEY_WIN, PRE), "Keyguard's window changed: update KEY_WIN / PRE here"
    return torch


def speech_bank(which: str, n: int, seed: int, length: int = KEY_WIN) -> np.ndarray:
    """(n, length) speech excerpts from attack_under_speech's `which` speaker pool ("train" | "test"). Its rng is
    replayed (load_keys, then load_speech) so the speaker split is the same one the speech-aug attacker used; the
    pools already exclude the demo voices (LibriSpeech 100 and 2803)."""
    from app.keystroke_guard.eval import attack_under_speech as aus
    rng = np.random.default_rng(aus.SEED)
    aus.load_keys(rng)
    pool = aus.load_speech(rng)[0][which]
    r = np.random.default_rng(seed)
    out = np.empty((n, length), np.float32)
    for i in range(n):
        x = pool[r.integers(len(pool))]
        a = r.integers(0, len(x) - length + 1)
        out[i] = x[a:a + length]
    return out


def keyguard_speech(n: int, seed: int, length: int = KEY_WIN) -> np.ndarray:
    """(n, length) excerpts of Keyguard's own speech clips (data/speech/*.wav, a few LibriSpeech speakers): the
    stand-in when Hearsay's pools aren't on the machine. Far fewer speakers than speech_bank."""
    from app.keystroke_guard.driver import keyguard_root
    from app.source.audio.replay import load_wav
    pool = [load_wav(p) for p in sorted((keyguard_root() / "data" / "speech").glob("*.wav"))]
    pool = [x for x in pool if len(x) > length]
    if not pool:
        raise FileNotFoundError(f"no speech clips in {keyguard_root() / 'data' / 'speech'}")
    r = np.random.default_rng(seed)
    out = np.empty((n, length), np.float32)
    for i in range(n):
        x = pool[r.integers(len(pool))]
        a = r.integers(0, len(x) - length + 1)
        out[i] = x[a:a + length]
    return out


def _rms(x):
    return x.pow(2).mean(1, keepdim=True).clamp_min(1e-12).sqrt()


def _mix(torch, X, S, db_lo=0.0, db_hi=20.0, p=0.7):
    """Keys X plus speech S at U(db_lo, db_hi) dB speech-to-key power in a fraction p of the rows."""
    db = torch.empty(len(X), 1).uniform_(db_lo, db_hi)
    on = (torch.rand(len(X), 1) < p).float()
    return X + on * S * (_rms(X) / _rms(S)) * 10 ** (db / 20)


def _perturb(torch, mix, U, gain, k, j):
    """mix + the runtime delta: delta k moved by j samples (the OS onset error), scaled by gain * RMS of the level
    window, which moves with it. Differentiable in U."""
    n = mix.shape[1]
    idx = torch.arange(n)[None, :] - j[:, None]
    placed = U[k[:, None], idx.clamp(0, n - 1)] * ((idx >= 0) & (idx < n))
    w = torch.arange(EST)[None, :] + j[:, None]
    seg = mix.gather(1, w.clamp(0, n - 1)) * ((w >= 0) & (w < n))
    return mix + gain * _rms(seg) * placed


def _augment(torch, X, S, U, gain, copies, jitter=JITTER):
    """Attacker training data: X plus `copies` speech mixes (p=0.7), each with a random delta if U is given."""
    Xs = [X]
    with torch.no_grad():
        for _ in range(copies):
            m = _mix(torch, X, S[torch.randint(len(S), (len(X),))])
            if U is not None:
                m = _perturb(torch, m, U, gain, torch.randint(len(U), (len(X),)),
                             torch.randint(-jitter, jitter + 1, (len(X),)))
            Xs.append(m)
    return torch.cat(Xs)


def _train_net(torch, net, X, y, S, U, gain, epochs, seed, copies=3):
    from keyguard.shield.adversarial import train_attacker
    torch.manual_seed(seed)
    Xa = _augment(torch, X, S, U, gain, copies)
    train_attacker(net, Xa, y.repeat(copies + 1), epochs=epochs)
    return net.eval()


def _keynet(torch, path=None):
    from keyguard.attackers.supervised import KeyNet
    net = KeyNet(N_KEYS)
    if path is not None:
        state = torch.load(path, map_location="cpu")
        net.load_state_dict(state.get("state_dict", state))
    return net.eval()


def _ctc_keys(torch):
    """Keyguard's CTC attacker (driver.KeyguardCTCAttacker, MtlCRNN) as a differentiable window classifier:
    (B, KEY_WIN) windows, onset PRE in -> (B, 37) non-blank logits at the onset frame +/-1, read exactly as the driver
    reads it (1 s context, Keyguard's noise floor) through defense_audio.torch_logmel (Keyguard's differentiable match
    of ctc.model.logmel). Takes the raw waveform (`wants_wave`), not KeyNet's features."""
    from app.keystroke_guard.driver import CTC_CTX, CTC_WIN, KeyguardCTCAttacker
    from keyguard.agents.defense_audio import torch_logmel as ctc_logmel
    from keyguard.ctc.model import HOP

    class CTCKeys(torch.nn.Module):
        wants_wave = True

        def __init__(self):
            super().__init__()
            self.net = KeyguardCTCAttacker().net
            self.floor = 0.002 * torch.randn(2 * CTC_CTX, generator=torch.Generator().manual_seed(0))

        def forward(self, w):
            a = CTC_CTX - PRE
            x = torch.nn.functional.pad(w, (a, 2 * CTC_CTX - a - w.shape[1])) + self.floor
            logits, _ = self.net(torch.stack([ctc_logmel(r) for r in x]))
            f = CTC_CTX // HOP
            return logits[:, f - CTC_WIN:f + CTC_WIN + 1, 1:].mean(1)
    return CTCKeys().eval()


def _margin(torch, logits, y, kappa):
    true = logits.gather(1, y[:, None])[:, 0]
    other = logits.masked_fill(torch.nn.functional.one_hot(y, logits.shape[1]).bool(), -1e9).amax(1)
    return torch.relu(true - other + kappa).mean()


def optimize(torch, U, nets, X, y, S, gain, radius, steps, lr, lam, kappa, bs=64, jitter=JITTER, log_every=100):
    """EOT ascent on the ensemble's error; each delta projected to the L2 `radius` after every step."""
    from keyguard.shield.adversarial import torch_logmel
    for net in nets.values():
        net.eval().requires_grad_(False)
    U = U.clone().requires_grad_(True)
    opt = torch.optim.Adam([U], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    K = len(U)
    for step in range(steps):
        i = torch.randint(len(X), (bs,))
        with torch.no_grad():
            m = _mix(torch, X[i], S[torch.randint(len(S), (bs,))])
        k = torch.arange(bs) % K                                   # every delta gets gradient each step
        j = torch.randint(-jitter, jitter + 1, (bs,))
        g = 10 ** (torch.empty(bs, 1).uniform_(-6, 6) / 20)
        wave = g * _perturb(torch, m, U, gain, k, j)
        feats = torch_logmel(wave) if any(not getattr(n, "wants_wave", False) for n in nets.values()) else None
        attack = sum(_margin(torch, net(wave if getattr(net, "wants_wave", False) else feats), y[i], kappa)
                     for net in nets.values())
        un = U / U.norm(dim=1, keepdim=True)
        c = un @ un.T
        div = (c - torch.eye(K)).pow(2).sum() / (K * (K - 1))    # mean squared pairwise cosine
        loss = attack + lam * div
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            U.mul_((radius / U.norm(dim=1, keepdim=True)).clamp(max=1.0))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:4d}  margin loss {attack.item() / len(nets):.3f}/attacker  mean cos^2 {div.item():.3f}",
                  flush=True)
    return U.detach()


def train_ctc(args) -> dict:
    """Deltas vs Keyguard's CTC attacker (CallGuard's default attacker) on Keyguard's per-key bank (the teammate's
    MacBook: the demo keyboard), 10 presses per key held out. Same budget, EOT, speech and runtime format as `train`;
    no adaptive-retrain phase (that step is KeyNet-specific), so both phases run against the CTC attacker."""
    torch = _torch(args.threads)
    from app.keystroke_guard.driver import keyguard_bank
    from keyguard.config import CLS_IDX
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    rng, Xs, ys = np.random.default_rng(args.seed), [], []
    for key, clips in sorted(keyguard_bank().items()):
        for c in clips[rng.permutation(len(clips))[10:]]:          # the first 10 per key stay out (harness uses 10/key)
            Xs.append(np.pad(c, (0, max(0, KEY_WIN - len(c))))[:KEY_WIN])
            ys.append(CLS_IDX[key])
    Xtr = np.stack(Xs).astype(np.float32)
    X, y = torch.from_numpy(Xtr), torch.tensor(ys).long()
    S = torch.from_numpy(speech_bank("train", 4000, args.seed) if args.speech == "hearsay"
                         else keyguard_speech(4000, args.seed))
    xr = np.sqrt(np.mean(Xtr.astype(np.float64) ** 2, 1))
    gain = float(np.median(xr / np.sqrt(np.mean(Xtr[:, :EST].astype(np.float64) ** 2, 1))))
    radius = 10 ** (args.budget_db / 20) * np.sqrt(KEY_WIN)
    nets = {"keyguard-ctc (ctc_rich_ft)": _ctc_keys(torch)}
    print(f"bank presses {len(X)}, speech excerpts {len(S)}, level_gain {gain:.3f}, budget {args.budget_db} dB, "
          f"K={args.k}, vs {list(nets)}", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    U = torch.randn(args.k, KEY_WIN, generator=g)
    U *= radius / U.norm(dim=1, keepdim=True)
    U = optimize(torch, U, nets, X, y, S, gain, radius, 2 * args.steps, args.lr, args.lam, args.kappa)
    un = U / U.norm(dim=1, keepdim=True)
    cos = (un @ un.T)[~torch.eye(args.k, dtype=bool)]
    meta = {"budget_db": args.budget_db, "K": args.k, "level_gain": gain, "level_window": EST, "pre": PRE,
            "key_win": KEY_WIN, "sr": 16000, "seed": args.seed, "steps": 2 * args.steps, "lr": args.lr,
            "lam": args.lam, "kappa": args.kappa, "jitter": JITTER, "gain_db": [-6, 6],
            "speech": ("attack_under_speech train-speaker pool" if args.speech == "hearsay"
                       else "Keyguard data/speech clips") + ", +0..+20 dB, p=0.7",
            "split": "Keyguard bank (live_bank_rich), all but 10 presses per key", "attackers": list(nets),
            "pairwise_cos_mean_abs": float(cos.abs().mean()), "pairwise_cos_max": float(cos.max()),
            "train_seconds": round(time.perf_counter() - t0), "threads": args.threads}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"deltas": U.float(), "meta": meta}, args.out)
    print(json.dumps(meta, indent=1))
    print(f"saved {args.out} ({meta['train_seconds']} s)")
    return meta


def train(args) -> dict:
    torch = _torch(args.threads)
    from app.keystroke_guard.driver import PROVISIONAL, PROVISIONAL_AUG, harrison_split
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    Xtr, ytr, _, _ = harrison_split()                             # TRAIN split only; the test presses stay unseen
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    S = torch.from_numpy(speech_bank("train", 4000, args.seed))
    xr = np.sqrt(np.mean(Xtr.astype(np.float64) ** 2, 1))
    gain = float(np.median(xr / np.sqrt(np.mean(Xtr[:, :EST].astype(np.float64) ** 2, 1))))
    radius = 10 ** (args.budget_db / 20) * np.sqrt(KEY_WIN)
    print(f"train presses {len(X)}, speech excerpts {len(S)}, level_gain {gain:.3f}, budget {args.budget_db} dB "
          f"(L2 radius {radius:.2f} in level units), K={args.k}", flush=True)

    nets = {"keynet-clean (provisional)": _keynet(torch, PROVISIONAL),
            "keynet-speechaug (provisional)": _keynet(torch, PROVISIONAL_AUG)}
    p1 = RUNS / "adv_attacker_keynet_s1_speechaug.pt"
    if p1.exists():
        nets["keynet-s1-speechaug"] = _keynet(torch, p1)
    else:
        net = _train_net(torch, _keynet(torch), X, y, S, None, gain, args.epochs, seed=1)
        torch.save({"state_dict": net.state_dict(), "split_seed": 0, "seed": 1, "data": "harrison train + speech"}, p1)
        nets["keynet-s1-speechaug"] = net
    print(f"attackers ready ({time.perf_counter() - t0:.0f} s)", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    U = torch.randn(args.k, KEY_WIN, generator=g)
    U *= radius / U.norm(dim=1, keepdim=True)
    print(f"phase 1: {args.steps} steps vs {len(nets)} attackers", flush=True)
    U = optimize(torch, U, nets, X, y, S, gain, radius, args.steps, args.lr, args.lam, args.kappa)
    print(f"phase 1 done ({time.perf_counter() - t0:.0f} s); adversarial retrain (co_train) on the phase-1 deltas",
          flush=True)
    adv = copy.deepcopy(nets["keynet-s1-speechaug"]).requires_grad_(True)
    nets["keynet-s2-advretrain"] = _train_net(torch, adv, X, y, S, U, gain, args.epochs // 2, seed=2)
    torch.save({"state_dict": nets["keynet-s2-advretrain"].state_dict(), "split_seed": 0, "seed": 2,
                "data": "harrison train + speech + phase-1 deltas (warm start keynet-s1)"},
               RUNS / "adv_attacker_keynet_s2_advretrain.pt")
    print(f"phase 2: {args.steps} steps vs {len(nets)} attackers ({time.perf_counter() - t0:.0f} s)", flush=True)
    U = optimize(torch, U, nets, X, y, S, gain, radius, args.steps, args.lr, args.lam, args.kappa)

    un = U / U.norm(dim=1, keepdim=True)
    cos = (un @ un.T)[~torch.eye(args.k, dtype=bool)]
    meta = {"budget_db": args.budget_db, "K": args.k, "level_gain": gain, "level_window": EST, "pre": PRE,
            "key_win": KEY_WIN, "sr": 16000, "seed": args.seed, "steps_per_phase": args.steps, "lr": args.lr,
            "lam": args.lam, "kappa": args.kappa, "jitter": JITTER, "gain_db": [-6, 6],
            "speech": "attack_under_speech train-speaker pool, +0..+20 dB, p=0.7",
            "split": "harrison_split(seed 0, 60/40 per key) TRAIN (540 presses)",
            "attackers": list(nets), "held_out": ["widecnn-s3-speechaug", "keynet-s4-retrained-on-deltas"],
            "pairwise_cos_mean_abs": float(cos.abs().mean()), "pairwise_cos_max": float(cos.max()),
            "train_seconds": round(time.perf_counter() - t0), "threads": args.threads}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"deltas": U.float(), "meta": meta}, args.out)
    print(json.dumps(meta, indent=1))
    print(f"saved {args.out} ({meta['train_seconds']} s)")
    return meta


# --- evaluation (TEST split) -------------------------------------------------------------------------------------
def _acc(torch, net, W, y) -> float:
    from keyguard.shield.adversarial import torch_logmel
    with torch.no_grad():
        p = torch.cat([net(torch_logmel(W[i:i + 256])).argmax(1) for i in range(0, len(W), 256)])
    return float((p == y).float().mean())


def _stream(Xte, speech, mode, deltas, seed=0):
    """Test presses 250 ms apart (optionally over speech) through the live KeyguardShield in 20 ms blocks, key
    events one block late (as the harness); returns the windows an attacker cuts at the true onsets."""
    from app.keystroke_guard.driver import KeyguardShield
    from app.source.types import BLOCK
    s = KeyguardShield(mode=mode, deltas=deltas, seed=seed)
    hop = KEY_WIN + 4000
    x = np.zeros(len(Xte) * hop + s.latency + BLOCK, np.float32)
    for i, w in enumerate(Xte):
        x[i * hop:i * hop + KEY_WIN] += w
        if speech is not None:
            x[i * hop:(i + 1) * hop] += speech[i]
    onsets = np.arange(len(Xte)) * hop + PRE
    out = [s.process(x[i:i + BLOCK], [int(o) for o in onsets if i - BLOCK <= o < i])
           for i in range(0, len(x) - BLOCK + 1, BLOCK)]
    y = np.concatenate(out)[s.latency:]
    return np.stack([y[o - PRE:o - PRE + KEY_WIN] for o in onsets])


def evaluate(args) -> dict:
    torch = _torch(args.threads)
    from keyguard.attackers.population import WideCNN
    from app.keystroke_guard.driver import PROVISIONAL, PROVISIONAL_AUG, harrison_split
    t0 = time.perf_counter()
    torch.manual_seed(args.seed + 100)
    d = torch.load(args.out, map_location="cpu")
    U, meta = d["deltas"], d["meta"]
    Xtr, ytr, Xte, yte = harrison_split()
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    Xt, yt = torch.from_numpy(Xte), torch.from_numpy(yte).long()
    S = torch.from_numpy(speech_bank("train", 4000, args.seed))            # attacker training only
    St = speech_bank("test", len(Xte) * 2, args.seed + 1)                   # evaluation mixtures: test speakers
    nets = {"keynet-clean (provisional)": _keynet(torch, PROVISIONAL),
            "keynet-speechaug (provisional)*": _keynet(torch, PROVISIONAL_AUG),
            "keynet-s1-speechaug": _keynet(torch, RUNS / "adv_attacker_keynet_s1_speechaug.pt"),
            "keynet-s2-advretrain": _keynet(torch, RUNS / "adv_attacker_keynet_s2_advretrain.pt")}
    ph = RUNS / "adv_attacker_widecnn_s3_speechaug.pt"
    wide = WideCNN(N_KEYS)
    if ph.exists():
        wide.load_state_dict(torch.load(ph, map_location="cpu")["state_dict"])
    else:
        _train_net(torch, wide, X, y, S, None, meta["level_gain"], args.epochs, seed=3)
        torch.save({"state_dict": wide.state_dict(), "split_seed": 0, "seed": 3, "held_out": True}, ph)
    nets["HELD-OUT widecnn-s3-speechaug"] = wide.eval()
    print(f"retraining a fresh KeyNet on train presses + the FROZEN deltas ({time.perf_counter() - t0:.0f} s)",
          flush=True)
    nets["ADAPTIVE keynet-s4-retrained-on-deltas"] = _train_net(torch, _keynet(torch).requires_grad_(True), X, y, S,
                                                                U, meta["level_gain"], args.epochs, seed=4)
    gain = meta["level_gain"]
    n = len(Xt)
    ks = torch.randint(len(U), (n,))
    jit = torch.randint(-JITTER, JITTER + 1, (n,))
    sp10 = torch.from_numpy(St[:n]) * (_rms(Xt) / _rms(torch.from_numpy(St[:n]))) * 10 ** 0.5   # +10 dB
    noise = torch.randn(U.shape)                  # control: white noise with the same per-stroke L2 as the deltas
    noise *= U.norm(dim=1, keepdim=True) / noise.norm(dim=1, keepdim=True)
    with torch.no_grad():
        conds = {"clean": Xt,
                 "white noise, same budget, jitter": _perturb(torch, Xt, noise, gain, ks, jit),
                 "delta, aligned": _perturb(torch, Xt, U, gain, ks, torch.zeros(n, dtype=torch.long)),
                 "delta, +/-40 ms jitter": _perturb(torch, Xt, U, gain, ks, jit),
                 "+10 dB speech": Xt + sp10,
                 "+10 dB speech + delta (jitter)": _perturb(torch, Xt + sp10, U, gain, ks, jit)}
    sp = speech_bank("test", len(Xte), args.seed + 2, length=KEY_WIN + 4000)   # one excerpt under each press
    sp *= (np.sqrt(np.mean(Xte ** 2, 1)) / np.sqrt(np.mean(sp ** 2, 1)) * 10 ** 0.5)[:, None]   # +10 dB
    for speech, tag in ((None, ""), (sp, ", +10 dB speech")):
        for mode in ("dsp", "adversarial", "dsp+adversarial"):
            conds[f"stream {mode}{tag}"] = torch.from_numpy(_stream(Xte, speech, mode, args.out))
    res = {c: {name: _acc(torch, net, W.float(), yt) for name, net in nets.items()} for c, W in conds.items()}
    dlt = conds["stream adversarial"].numpy() - Xte                  # what the live stage really added, keys only
    snr = 20 * np.log10(np.sqrt(np.mean(Xte ** 2, 1)) / np.sqrt(np.mean(dlt ** 2, 1) + 1e-20))
    out = {"n_test": n, "chance": 1 / N_KEYS, "results": res, "train_meta": meta,
           "stream_adversarial_snr_db": {"median": float(np.median(snr)), "min": float(np.min(snr)),
                                         "max": float(np.max(snr))},
           "note": "* keynet-speechaug (provisional) was trained on attack_under_speech.load_keys's split, which "
                   "differs from harrison_split: part of this TEST split was in its training set",
           "eval_seconds": round(time.perf_counter() - t0)}
    names = list(nets)
    print("\ntop-1 on harrison TEST presses (n=%d, chance %.1f %%); WHITE-BOX for the 4 training attackers" %
          (n, 100 / N_KEYS))
    print(f"{'condition':<34}" + "".join(f"{nm[:22]:>24}" for nm in names))
    for c, r in res.items():
        print(f"{c:<34}" + "".join(f"{100 * r[nm]:>23.1f}%" for nm in names))
    print(f"stream adversarial: key-window RMS over added-delta RMS per stroke (dB): {out['stream_adversarial_snr_db']}")
    print(f"eval {out['eval_seconds']} s")
    (RUNS / "adversarial_eval.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.keystroke_guard.adversarial", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--budget-db", type=float, default=-18.0, help="delta L2 budget vs key-window RMS (Keyguard: -18)")
    ap.add_argument("--k", type=int, default=8, help="number of universal deltas (one picked per stroke)")
    ap.add_argument("--steps", type=int, default=500, help="ascent steps per phase (2 phases)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=2.0, help="pairwise-cosine diversity weight")
    ap.add_argument("--kappa", type=float, default=5.0, help="margin-loss clip (logits)")
    ap.add_argument("--epochs", type=int, default=30, help="epochs for the attackers trained here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(DELTAS))
    ap.add_argument("--attacker", default="keynet", choices=["keynet", "ctc"],
                    help="train: optimize vs the provisional KeyNets on harrison (default) or vs Keyguard's CTC attacker "
                         "on Keyguard's bank (CallGuard's default attacker)")
    ap.add_argument("--speech", default="hearsay", choices=["hearsay", "keyguard"],
                    help="--attacker ctc: speech for EOT, Hearsay's pools (HEARSAY_ROOT) or Keyguard's data/speech clips")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        (train_ctc if a.attacker == "ctc" else train)(a)
    else:
        evaluate(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
