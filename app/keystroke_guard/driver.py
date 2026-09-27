"""Real Keyguard drivers: the keystroke attacker (KeyNet) and the streaming DSP shield.

Keyguard is vendored as the top-level `keyguard` package; its weights/data live in runs/keyguard and data/keyguard
(gitignored, copied in by app.keystroke_guard.get_assets).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from app.source.types import BLOCK, SR, KeyGuess
from keyguard.config import DATA as KEYGUARD_DATA, RUNS as KEYGUARD_RUNS  # runs/keyguard, data/keyguard

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROVISIONAL = REPO / "runs" / "provisional_keynet.pt"
PROVISIONAL_AUG = REPO / "runs" / "provisional_keynet_speechaug.pt"  # adaptive attacker
# 26 frames (208 ms) covers press + release; wider than Keyguard's default 14 and reads keys under speech much better
KEY_FRAMES = 26
SPLIT_SEED = 0          # per-key seeded 60/40 split of harrison presses
TRAIN_FRAC = 0.6
N_KEYS = 36             # harrison A-Z0-9 = keyguard CLASSES[:36]
SHIELD_MODES = ("dsp", "adversarial", "dsp+adversarial")
DASHBOARD_ADVERSARIAL = "dsp+adversarial"   # what the dashboard's "adversarial" runs


BANK = KEYGUARD_DATA / "live_bank_rich.npz"   # Keyguard's per-key press bank
HARRISON = KEYGUARD_DATA / "pool" / "harrison.npz"


def keyguard_bank(path: Path | None = None) -> dict[str, np.ndarray]:
    """{key: (n, clip) presses} from Keyguard's bank, each clip starting PRE_S before its onset. It's the CTC
    attacker's training domain, so reads on it are optimistic."""
    d = np.load(Path(path or BANK))
    return {k: d[k].astype(np.float32) for k in d.files}


def harrison_split(seed: int = SPLIT_SEED):
    """Harrison press bank split per key, seeded: (Xtr, ytr, Xte, yte), labels as class indices."""
    from keyguard.config import CLS_IDX
    d = np.load(HARRISON)
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

    Weights come from the `weights` arg, else ATHENA_ATTACKER_WEIGHTS, else the provisional speech-aug KeyNet, else a
    clean provisional KeyNet trained once on the harrison train split and cached in runs/. Provisional = ours, not
    the teammate's tuned attacker.
    """
    name = "keyguard-keynet"

    def __init__(self, weights: str | Path | None = None, epochs: int = 40):
        import torch
        from keyguard.attackers.supervised import KeyNet
        from keyguard.config import CLASSES
        # harrison has A-Z0-9 = CLASSES[:36]; the head gets resized from the weights below
        self.classes = list(CLASSES)[:N_KEYS]
        self.net = KeyNet(len(self.classes)).eval()
        path = weights or os.environ.get("ATHENA_ATTACKER_WEIGHTS")
        self.provisional = not path
        path = Path(path) if path else (PROVISIONAL_AUG if PROVISIONAL_AUG.exists() else PROVISIONAL)
        if not path.exists():
            if not self.provisional:
                raise FileNotFoundError(f"attacker weights not found: {path}")
            self._train_provisional(path, epochs)
        state = torch.load(path, map_location="cpu")
        state = state.get("state_dict", state)  # cache dict or a bare Keyguard state_dict
        if state["head.weight"].shape[0] != len(self.classes):
            self.classes = list(CLASSES)[:state["head.weight"].shape[0]]
            self.net = KeyNet(len(self.classes)).eval()
        self.net.load_state_dict(state)
        kind = ", speech-aug" if path == PROVISIONAL_AUG else ""
        self.name = f"keyguard-keynet{f' (provisional{kind})' if self.provisional else ''}"
        if self.provisional:
            log.warning("attacker: using PROVISIONAL KeyNet %s (Athena-trained, not the teammate's)", path)

    def _train_provisional(self, path: Path, epochs: int) -> None:
        import torch
        from keyguard.shield.adversarial import train_attacker
        log.warning("attacker: training provisional KeyNet on harrison (CPU, ~1-3 min, cached at %s)", path)
        t0 = time.perf_counter()
        torch.manual_seed(SPLIT_SEED)
        Xtr, ytr, _, _ = harrison_split()
        self.net.train()
        train_attacker(self.net, torch.from_numpy(Xtr), torch.from_numpy(ytr).long(), epochs=epochs)
        self.net.eval()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "provisional": True, "split_seed": SPLIT_SEED,
                    "train_frac": TRAIN_FRAC, "epochs": epochs, "data": "harrison.npz"}, path)
        log.warning("attacker: provisional KeyNet trained in %.0f s", time.perf_counter() - t0)

    def probs(self, wins: np.ndarray) -> np.ndarray:
        """(n, KEY_WIN) windows -> (n, classes) softmax, same features as training."""
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


CTC_WEIGHTS = KEYGUARD_RUNS / "ctc_rich_ft.pt"  # Keyguard's current attacker (Ares)
CTC_CTX = SR // 2       # samples of context each side of an onset (0.5 s before / after)
CTC_WIN = 1             # +/- frames averaged at the onset frame


class KeyguardCTCAttacker:
    """KeystrokeAttackerDriver around Keyguard's current attacker: MtlCRNN (CNN -> BiGRU -> CTC + onset head),
    weights runs/keyguard/ctc_rich_ft.pt (or `weights` / ATHENA_ATTACKER_WEIGHTS).

    Onsets are given, so each is read the way onset_gated_decode reads a peak: non-blank logits averaged over +/-1
    frame at frame onset // HOP, softmaxed over the 37 keys (A-Z, 0-9, space), in its own 1 s window.
    """

    def __init__(self, weights: str | Path | None = None):
        import torch
        from keyguard.ctc.data import VOCAB
        from keyguard.ctc.train_overlap import MtlCRNN
        path = Path(weights or os.environ.get("ATHENA_ATTACKER_WEIGHTS") or CTC_WEIGHTS)
        if not path.exists():
            raise FileNotFoundError(f"CTC attacker weights not found: {path} (run python -m app.keystroke_guard.get_assets)")
        state = torch.load(path, map_location="cpu")
        self.net = MtlCRNN(n_sym=len(VOCAB)).eval()
        self.net.load_state_dict(state.get("state_dict", state))
        self.classes = list(VOCAB[1:])
        self.name = f"keyguard-ctc ({path.stem})"

    def read(self, audio: np.ndarray, onsets: np.ndarray, k: int = 3) -> list[KeyGuess]:
        import torch
        from keyguard.ctc.model import HOP, logmel
        onsets = np.asarray(onsets, dtype=int)
        if len(onsets) == 0:
            return []
        padded = np.pad(np.asarray(audio, np.float32), CTC_CTX)
        mels = np.stack([logmel(padded[o:o + 2 * CTC_CTX]) for o in onsets])  # onset at CTC_CTX in each window
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(mels))
        f = CTC_CTX // HOP
        z = logits[:, f - CTC_WIN:f + CTC_WIN + 1, 1:].mean(1)  # drop blank (index 0)
        p = torch.softmax(z, dim=1).numpy()
        top = np.argsort(-p, axis=1)[:, :k]
        return [KeyGuess(int(o), [(self.classes[j], float(p[i, j])) for j in top[i]]) for i, o in enumerate(onsets)]


class KeyguardShield:
    """ShieldDriver: Keyguard's offline DSP Shield made streaming.

    We keep a short history, delay the output by `lookahead` samples, and run Shield.apply on history+lookahead only
    when a key event can touch the outgoing block; otherwise the block passes through untouched. (Keyguard's own
    realtime path windows every block, which amplitude-modulates speech even with no key active.)

    Latency = lookahead (default 80 ms): enough for an STFT frame to be seen whole plus slack for OS key events that
    arrive up to ~50 ms late. Key events are absolute sample indices on the block clock (samples since reset).

    Modes (set_mode switches at runtime, same output delay in all): "dsp" = the above; "adversarial" = no
    inpainting, one of K trained deltas per key event; "dsp+adversarial" = DSP inpainting, then the delta. The
    delta's level comes from the raw input, before the DSP stage.
    """

    def __init__(self, mode: str = "dsp", lookahead: int = 4 * BLOCK,
                 history: int = 24 * BLOCK, seed: int = 0, deltas: str | Path | None = None, **shield_cfg):
        from keyguard.config import HOP, N_FFT
        from keyguard.shield.shield import Shield, ShieldConfig
        self.name = "keyguard-shield"
        self.mode = "dsp"
        self.adv, self._deltas, self._seed = None, deltas, seed
        self.shield = Shield(ShieldConfig(**{"key_frames": KEY_FRAMES, **shield_cfg}), seed=seed)
        self.lookahead, self.history = int(lookahead), int(history)
        cfg = self.shield.cfg
        # sample span a key event at e can change: [e - before, e + after)
        self._before = 2 * HOP + N_FFT // 2
        self._after = cfg.key_frames * HOP + N_FFT // 2
        self.latency = self.lookahead    # samples of output delay
        self.latency_ms = self.lookahead / SR * 1000
        self.shield.apply(np.zeros(4 * N_FFT, np.float32), np.array([N_FFT]))  # warm librosa off the audio thread
        self.reset()
        self.set_mode(mode)

    def set_mode(self, mode: str) -> str:
        """Switch dsp | adversarial | dsp+adversarial. Raises FileNotFoundError if the deltas aren't trained."""
        if mode not in SHIELD_MODES:
            raise ValueError(f"unknown shield mode {mode!r} ({' | '.join(SHIELD_MODES)})")
        if "adversarial" in mode and self.adv is None:
            from app.keystroke_guard.adversarial import DeltaStage
            self.adv = DeltaStage.load(self._deltas, seed=self._seed)
        if "adversarial" in mode and "adversarial" not in self.mode:
            self.adv.reset()  # no stale strokes from before the switch
        self.mode = mode
        return mode

    def reset(self) -> None:
        self._buf = np.zeros(self.history + self.lookahead, np.float32)
        self._t = 0  # absolute index one past the newest sample
        self._events: list[int] = []
        if self.adv is not None:
            self.adv.reset()

    def process(self, block: np.ndarray, key_events: list[int]) -> np.ndarray:
        block = np.asarray(block, np.float32).ravel()
        n = len(block)
        mode = self.mode                                # one read: set_mode may run on another thread
        self._buf = np.concatenate([self._buf, block])[-(self.history + self.lookahead + n):]
        self._t += n
        start = self._t - self.lookahead - n            # outgoing block = [start, start + n)
        if mode != "adversarial":
            self._events.extend(int(e) for e in key_events)
        self._events = [e for e in self._events if e + self._after > start - self.history]
        out = self._buf[-self.lookahead - n:len(self._buf) - self.lookahead].copy()
        if any(e - self._before < start + n and e + self._after > start for e in self._events):
            base = self._t - len(self._buf)             # absolute index of self._buf[0]
            onsets = np.array([e - base for e in self._events], dtype=int)
            # re-runs Shield.apply on the whole buffer per key-touched block; upgrade = run once per stroke
            # and cache. Only matters inside the already-destroyed key region.
            shielded = self.shield.apply(self._buf.copy(), onsets)
            out = shielded[-self.lookahead - n:len(shielded) - self.lookahead]
        if "adversarial" in mode:
            self.adv.add(key_events)
            out = self.adv.apply(out, start, self._buf, self._t)
        return out
