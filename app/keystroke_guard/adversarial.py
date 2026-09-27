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

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
DELTAS = RUNS / "adversarial_deltas.pt"
KEY_WIN = 4800      # keyguard.config.KEY_WIN (0.30 s at 16 kHz); asserted when training
PRE = 320           # keyguard.config.PRE_S * SR: Keyguard's window starts 20 ms before the onset
EST = 960           # level window: [onset - PRE, onset - PRE + EST), 60 ms
JITTER = 640        # EOT onset misalignment, +/-40 ms
N_KEYS = 36         # harrison: A-Z0-9 (Keyguard's CLASSES later appended space)

log = logging.getLogger(__name__)


# --- runtime ----------------------------------------------------------------------------------------------------
class DeltaStage:
    """Adds one of K deltas per key event on the absolute sample clock. No torch at runtime: plain numpy adds."""

    def __init__(self, deltas: np.ndarray, level_gain: float, seed: int = 0, meta: dict | None = None):
        self.deltas = np.ascontiguousarray(deltas, np.float32)          # (K, KEY_WIN), level-relative units
        assert self.deltas.ndim == 2 and self.deltas.shape[1] == KEY_WIN, self.deltas.shape
        self.level_gain, self.seed, self.meta = float(level_gain), seed, meta or {}
        self.reset()

    @classmethod
    def load(cls, path: str | Path | None = None, seed: int = 0) -> "DeltaStage":
        path = Path(path or DELTAS)
        if not path.exists():
            raise FileNotFoundError(f"adversarial deltas not found at {path}; train them with "
                                    f"`python -m app.keystroke_guard.adversarial train`")
        import torch
        d = torch.load(path, map_location="cpu")
        return cls(d["deltas"].numpy(), d["meta"]["level_gain"], seed, d["meta"])

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._pending: list[int] = []
        self._active: list[tuple[int, np.ndarray]] = []      # (absolute start, scaled delta)
        self.history: deque = deque(maxlen=64)                  # (start, k, level) per stroke: tests + debugging

    def add(self, events) -> None:
        self._pending.extend(int(e) for e in events)

    def apply(self, out: np.ndarray, start: int, buf: np.ndarray, t: int) -> np.ndarray:
        """out = the outgoing block [start, start + len(out)); buf = raw input, buf[-1] is sample t - 1."""
        base = t - len(buf)
        for e in [e for e in self._pending if e - PRE + EST <= t]:   # its level window has arrived
            self._pending.remove(e)
            a = e - PRE
            seg = buf[max(a - base, 0):max(a + EST - base, 0)]
            level = self.level_gain * float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) if len(seg) else 0.0
            k = int(self.rng.integers(len(self.deltas)))
            self._active.append((a, level * self.deltas[k]))
            self.history.append((a, k, level))
        n = len(out)
        self._active = [(a, d) for a, d in self._active if a + KEY_WIN > start]
        for a, d in self._active:
            lo, hi = max(a, start), min(a + KEY_WIN, start + n)
            if lo < hi:
                out[lo - start:hi - start] += d[lo - a:hi - a]
        return out


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
        feats = torch_logmel(g * _perturb(torch, m, U, gain, k, j))
        attack = sum(_margin(torch, net(feats), y[i], kappa) for net in nets.values())
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
    a = ap.parse_args(argv)
    (train if a.cmd == "train" else evaluate)(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
