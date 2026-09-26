"""Real Keyguard drivers: the keystroke attacker (KeyNet) and the streaming DSP shield.

Keyguard (KEYGUARD_ROOT) is the teammate's repo and is read-only: we import it via sys.path with bytecode writing off,
so nothing (not even __pycache__) lands inside it. What we use from it:
- keyguard.config: SR, KEY_WIN, PRE_S, CLASSES, HOP, N_FFT
- keyguard.attackers.supervised.KeyNet, keyguard.shield.adversarial.torch_logmel / train_attacker (attacker)
- keyguard.segment.windows (window cutting, identical to Keyguard training)
- keyguard.shield.shield.Shield / ShieldConfig (the DSP shield, streamed here with a lookahead buffer)
- data/pool/harrison.npz (provisional attacker training data)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from callguard.types import BLOCK, SR, KeyGuess

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROVISIONAL = REPO / "runs" / "provisional_keynet.pt"
PROVISIONAL_AUG = REPO / "runs" / "provisional_keynet_speechaug.pt"  # plan 04's adaptive attacker (experiments/)
SPLIT_SEED = 0          # plan 04: per-key seeded 60/40 split of harrison presses
TRAIN_FRAC = 0.6


def keyguard_root() -> Path:
    return Path(os.environ.get("KEYGUARD_ROOT") or REPO.parent / "keyboard-acoustic-shield")


def _import_keyguard(root: Path | None = None) -> None:
    """Put KEYGUARD_ROOT on sys.path without letting Python write caches into it."""
    root = Path(root or keyguard_root())
    if not (root / "keyguard").is_dir():
        raise FileNotFoundError(f"Keyguard repo not found at {root} (set KEYGUARD_ROOT)")
    sys.dont_write_bytecode = True  # read-only upstream: no __pycache__ inside it
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def harrison_split(root: Path | None = None, seed: int = SPLIT_SEED):
    """Harrison press bank split per key, seeded: (Xtr, ytr, Xte, yte), labels as class indices."""
    _import_keyguard(root)
    from keyguard.config import CLS_IDX
    d = np.load(Path(root or keyguard_root()) / "data" / "pool" / "harrison.npz")
    wins, labels = d["wins"].astype(np.float32), d["labels"]
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for k in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == k))
        cut = int(round(TRAIN_FRAC * len(idx)))
        tr += list(idx[:cut]); te += list(idx[cut:])
    y = np.array([CLS_IDX[str(k)] for k in labels])
    return wins[tr], y[tr], wins[te], y[te]


class KeyguardAttacker:
    """KeystrokeAttackerDriver around Keyguard's KeyNet.

    Weights: `weights` arg, else CALLGUARD_ATTACKER_WEIGHTS, else the provisional speech-augmented KeyNet saved by
    experiments/attack_under_speech.py (the adaptive attacker: it has heard keys under speech), else a clean
    provisional KeyNet that CallGuard trains once on the harrison train split with Keyguard's own train_attacker and
    caches in runs/. The clean one reads keys well alone but not under speech. Provisional = ours, not the
    teammate's tuned attacker; it's replaced when their weights ship (plan 05).
    """
    name = "keyguard-keynet"

    def __init__(self, weights: str | Path | None = None, root: Path | None = None, epochs: int = 40):
        import torch
        _import_keyguard(root)
        from keyguard.attackers.supervised import KeyNet
        from keyguard.config import CLASSES
        self.classes = list(CLASSES)
        self.net = KeyNet(len(self.classes)).eval()
        path = weights or os.environ.get("CALLGUARD_ATTACKER_WEIGHTS")
        self.provisional = not path
        path = Path(path) if path else (PROVISIONAL_AUG if PROVISIONAL_AUG.exists() else PROVISIONAL)
        if not path.exists():
            if not self.provisional:
                raise FileNotFoundError(f"attacker weights not found: {path}")
            self._train_provisional(path, root, epochs)
        state = torch.load(path, map_location="cpu")
        self.net.load_state_dict(state.get("state_dict", state))  # cache dict or a bare Keyguard state_dict
        kind = ", speech-aug" if path == PROVISIONAL_AUG else ""
        self.name = f"keyguard-keynet{f' (provisional{kind})' if self.provisional else ''}"
        if self.provisional:
            log.warning("attacker: using PROVISIONAL KeyNet %s (CallGuard-trained, not the teammate's)", path)

    def _train_provisional(self, path: Path, root, epochs: int) -> None:
        import torch
        from keyguard.shield.adversarial import train_attacker
        log.warning("attacker: training provisional KeyNet on harrison (CPU, ~1-3 min, cached at %s)", path)
        t0 = time.perf_counter()
        torch.manual_seed(SPLIT_SEED)
        Xtr, ytr, _, _ = harrison_split(root)
        self.net.train()
        train_attacker(self.net, torch.from_numpy(Xtr), torch.from_numpy(ytr).long(), epochs=epochs)
        self.net.eval()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "provisional": True, "split_seed": SPLIT_SEED,
                    "train_frac": TRAIN_FRAC, "epochs": epochs, "data": "harrison.npz"}, path)
        log.warning("attacker: provisional KeyNet trained in %.0f s", time.perf_counter() - t0)

    def probs(self, wins: np.ndarray) -> np.ndarray:
        """(n, KEY_WIN) windows -> (n, classes) softmax, same features as training (torch_logmel)."""
        import torch
        from keyguard.shield.adversarial import torch_logmel
        if len(wins) == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(wins, dtype=np.float32))
            return torch.softmax(self.net(torch_logmel(x)), dim=1).numpy()

    def read(self, audio: np.ndarray, onsets: np.ndarray, k: int = 3) -> list[KeyGuess]:
        from keyguard.segment import windows  # cuts at onset - PRE_S*SR, KEY_WIN long, zero-padded
        onsets = np.asarray(onsets, dtype=int)
        p = self.probs(windows(np.asarray(audio, np.float32), onsets))
        top = np.argsort(-p, axis=1)[:, :k]
        return [KeyGuess(int(o), [(self.classes[j], float(p[i, j])) for j in top[i]]) for i, o in enumerate(onsets)]


class KeyguardShield:
    """ShieldDriver: Keyguard's offline DSP Shield made streaming.

    Why not keyguard.realtime._process_block: it multiplies every block by a Hann window even with no key active
    (irfft(spec(b*w)) / w * w = b*w), so speech gets 50 Hz amplitude modulation. Instead we keep a short history,
    delay the output by `lookahead` samples, and run Shield.apply on history+lookahead only when a key event can
    touch the outgoing block; otherwise the block passes through untouched (exact identity).

    Latency = lookahead (default 1280 samples = 80 ms). Why that much: Shield's STFT frames are N_FFT=1024 wide, so a
    frame touching the outgoing block needs 512 samples of future to be seen whole, and the key region starts 2 hops
    (256 samples) before the onset. With 80 ms, OS key events may arrive up to ~50 ms after the sound and still be
    shielded in full. Key events are absolute sample indices on the same clock as the blocks (sum of block lengths
    since reset).
    """

    def __init__(self, mode: str = "dsp", root: Path | None = None, lookahead: int = 4 * BLOCK,
                 history: int = 16 * BLOCK, seed: int = 0, **shield_cfg):
        if mode == "adversarial":
            raise NotImplementedError("shield mode 'adversarial' waits for the teammate's streaming adversarial "
                                      "shield D (plan 05 blockers); use mode='dsp'")
        if mode != "dsp":
            raise ValueError(f"unknown shield mode {mode!r} (dsp | adversarial)")
        _import_keyguard(root)
        from keyguard.config import HOP, N_FFT
        from keyguard.shield.shield import Shield, ShieldConfig
        self.name = "keyguard-dsp"
        self.mode = mode
        self.shield = Shield(ShieldConfig(**shield_cfg), seed=seed)
        self.lookahead, self.history = int(lookahead), int(history)
        cfg = self.shield.cfg
        # sample span a key event at e can change: [e - before, e + after)
        self._before = 2 * HOP + N_FFT // 2
        self._after = cfg.key_frames * HOP + N_FFT // 2
        self.latency = self.lookahead    # samples of output delay (ShieldDriver contract)
        self.latency_ms = self.lookahead / SR * 1000
        self.shield.apply(np.zeros(4 * N_FFT, np.float32), np.array([N_FFT]))  # warm librosa (~3 s cold) off the audio thread
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(self.history + self.lookahead, np.float32)
        self._t = 0                      # absolute index one past the newest sample
        self._events: list[int] = []

    def process(self, block: np.ndarray, key_events: list[int]) -> np.ndarray:
        block = np.asarray(block, np.float32).ravel()
        n = len(block)
        self._buf = np.concatenate([self._buf, block])[-(self.history + self.lookahead + n):]
        self._t += n
        self._events.extend(int(e) for e in key_events)
        start = self._t - self.lookahead - n            # outgoing block = [start, start + n)
        self._events = [e for e in self._events if e + self._after > start - self.history]
        out = self._buf[-self.lookahead - n:len(self._buf) - self.lookahead]
        if not any(e - self._before < start + n and e + self._after > start for e in self._events):
            return out.copy()
        base = self._t - len(self._buf)                 # absolute index of self._buf[0]
        onsets = np.array([e - base for e in self._events], dtype=int)
        # ponytail: re-runs Shield.apply on the ~0.4 s buffer for each key-touched block (~11 blocks per stroke, ~10-15 ms
        # each on this laptop, 0 ms otherwise); upgrade = run once per stroke and cache. Also, the per-stroke random
        # residue gain differs between consecutive blocks, which only matters inside the already-destroyed key region.
        shielded = self.shield.apply(self._buf.copy(), onsets)
        return shielded[-self.lookahead - n:len(shielded) - self.lookahead]
