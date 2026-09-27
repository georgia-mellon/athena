"""Plan 04: can a keystroke attacker read keys *under speech*, and does the Keyguard shield stop it?

Hearsay showed keystrokes don't break voice detection. This is the reverse direction: with someone talking over the
typing, is keystroke leakage still a threat on a call? If yes, the shield has something real to defend against.

Protocol (fixed seeds, CPU, 8 threads, ~12 min):
- Keys: Keyguard's harrison bank (36 keys x 25 presses, one MacBook), per-key 60/40 split.
- Attackers: KeyNet + torch_logmel trained here (provisional, in-domain). `clean` on clean presses; `speech-aug`
  on presses mixed with speech at +0..+20 dB (the adaptive attacker, which knows calls have speech).
- Speech: real LibriSpeech/LJSpeech bona fide clips from Hearsay's test_internal split, speakers split between the
  attacker's training noise and the evaluation mixtures.
- Each test press sits at a random spot in a 1.5 s speech excerpt at a speech-to-key power ratio (speech excerpt
  power over key-window power). Attack with the oracle onset and with Keyguard's onset detector on the mixture.
- Shield: Keyguard DSP Shield with ShieldConfig(key_frames=26) (as CallGuard ships it; Keyguard's default 14 as a
  secondary row at keys only / +10 dB) given the true onset (the victim has OS key events).
- Hearsay check (pass criterion 3): full real clips (>= 3 s) from the test speakers, with test presses at +10 dB
  speech-to-key, scored clean / keys unshielded / keys shielded by CallGuard's Hearsay driver (r4ft, CPU).

Run: .venv/Scripts/python app/keystroke_guard/eval/attack_under_speech.py   (reads KEYGUARD_ROOT, HEARSAY_ROOT; writes docs/reports/)
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

os.environ.setdefault("KEYGUARD_DEVICE", "cpu")
KEYGUARD_ROOT = Path(os.environ.get("KEYGUARD_ROOT", r"C:\Users\danma\Documents\Dan\Projects\keyboard-acoustic-shield"))
HEARSAY_ROOT = Path(os.environ.get("HEARSAY_ROOT", r"C:\Users\danma\Documents\Dan\Projects\Hearsay"))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(KEYGUARD_ROOT))
sys.dont_write_bytecode = True  # never leave __pycache__ inside the read-only Keyguard checkout
try:
    import keyguard.memory  # noqa: F401  (pulled in by adversarial; may need pymongo/dotenv)
except Exception:
    sys.modules["keyguard.memory"] = types.ModuleType("keyguard.memory")
from keyguard import segment  # noqa: E402
from keyguard.attackers.supervised import KeyNet  # noqa: E402
from keyguard.config import CLASSES, PRE_S, SR  # noqa: E402
from keyguard.shield.adversarial import torch_logmel, train_attacker  # noqa: E402
from keyguard.shield.shield import Shield, ShieldConfig  # noqa: E402

SEED = 0
EXCERPT = int(1.5 * SR)
PRE = int(PRE_S * SR)
MATCH_TOL = int(0.030 * SR)
LEVELS = [None, -10, -5, 0, 5, 10, 20]  # speech-to-key dB; None = keys only
SHIELD_CFG = ShieldConfig(key_frames=26)  # as app/keyguard_real.KeyguardShield runs it (press + release)
SHIELD_CFG_14 = ShieldConfig()  # Keyguard's default key_frames=14: secondary comparison only
LEVELS_14 = {None, 10}
DEMO_SPEAKERS = {"100", "2803"}  # voices in the demo scenario: the attacker must never have heard them
N_SPEECH = 200
EPOCHS = 40
N_HEARSAY = 100          # real clips for criterion 3
KEYS_PER_CLIP = 5
HEARSAY_LEVEL = 10       # dB speech-to-key
torch.set_num_threads(8)  # leave cores for other work on this machine
CHANCE = 1 / len(CLASSES)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval: honest at small n and near 0, where the normal approximation lies."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2)) + 1e-12


def load_keys(rng):
    d = np.load(KEYGUARD_ROOT / "data" / "pool" / "harrison.npz")
    wins, labels = d["wins"].astype(np.float32), d["labels"]
    y = np.array([CLASSES.index(str(c)) for c in labels])
    tr, te = [], []
    for k in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == k))
        cut = int(round(0.6 * len(idx)))
        tr += list(idx[:cut]); te += list(idx[cut:])
    return wins[tr], y[tr], wins[te], y[te]


def real_clips() -> pd.DataFrame:
    """Hearsay's held-out bona fide LibriSpeech/LJSpeech clips (all >= 3 s), minus the demo voices."""
    m = pd.read_parquet(HEARSAY_ROOT / "data" / "processed" / "manifest.parquet")
    return m[(m.label == "bonafide") & m.source.isin(["librispeech", "ljspeech"]) & (m.split == "test_internal")
             & (m.duration >= 3.0) & ~m.speaker.astype(str).isin(DEMO_SPEAKERS)]


def load_speech(rng):
    """~200 real clips (>= 3 s), split by speaker: half feed attacker training, half the evaluation."""
    m = real_clips()
    m = m.sample(n=N_SPEECH, random_state=SEED)
    spk = np.array(sorted(m.speaker.unique()))
    train_spk = set(rng.permutation(spk)[: len(spk) // 2])
    pools = {"train": [], "test": []}
    for r in m.itertuples():
        x, sr = sf.read(HEARSAY_ROOT / r.path, dtype="float32")
        assert sr == SR, (r.path, sr)
        pools["train" if r.speaker in train_spk else "test"].append(x if x.ndim == 1 else x.mean(1))
    return pools, set(spk) - train_spk


def excerpt(pool, rng) -> np.ndarray:
    x = pool[rng.integers(len(pool))]
    a = rng.integers(0, len(x) - EXCERPT + 1)
    return x[a:a + EXCERPT]


def mix(key: np.ndarray, sp: np.ndarray, level_db, pos: int) -> np.ndarray:
    """Key window at `pos` inside the speech excerpt, speech scaled to `level_db` over the key window's power."""
    out = np.zeros(EXCERPT, np.float32) if level_db is None else \
        sp * np.sqrt(power(key) * 10 ** (level_db / 10) / power(sp))
    out = out.astype(np.float32)
    out[pos:pos + len(key)] += key
    return out


def cut(y: np.ndarray, onset: int) -> np.ndarray:
    return segment.windows(y, np.array([onset]))[0]


def speech_aug_set(X, y, pool, rng, copies=4):
    """Adaptive attacker's training data: clean presses + `copies` fresh speech mixes each at +0..+20 dB."""
    Xs, ys = [X], [y]
    for _ in range(copies):
        for_this = np.empty_like(X)
        for i, k in enumerate(X):
            sp = excerpt(pool, rng)
            pos = rng.integers(0, EXCERPT - len(k))
            for_this[i] = cut(mix(k, sp, rng.uniform(0, 20), pos), pos + PRE)
        Xs.append(for_this); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def train(X, y) -> KeyNet:
    torch.manual_seed(SEED)
    net = KeyNet()
    train_attacker(net, torch.tensor(X), torch.tensor(y), epochs=EPOCHS)
    return net.eval()


def top_ranks(net, W: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank of the true key in the attacker's output (0 = top-1)."""
    with torch.no_grad():
        logits = torch.cat([net(torch_logmel(torch.tensor(W[i:i + 256]))) for i in range(0, len(W), 256)])
    true = logits[torch.arange(len(y)), torch.tensor(y)]
    return (logits > true[:, None]).sum(1).numpy()


def detect(y: np.ndarray, true_onset: int):
    """Attacker's own segmentation: nearest Keyguard onset within 30 ms, else a miss."""
    on = segment.onsets(y)
    if len(on) == 0:
        return None
    o = int(on[np.argmin(np.abs(on - true_onset))])
    return o if abs(o - true_onset) <= MATCH_TOL else None


def quality(ref: np.ndarray, deg: np.ndarray) -> tuple[float, float]:
    from pesq import pesq
    from pystoi import stoi
    s = stoi(ref, deg, SR, extended=False)
    try:
        p = pesq(SR, ref, deg, "wb")
    except Exception:  # PESQ refuses some very short / silent references
        p = np.nan
    return s, p


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    Xtr, ytr, Xte, yte = load_keys(rng)
    pools, test_spk = load_speech(rng)
    print(f"keys train {len(Xtr)} test {len(Xte)}; speech clips train {len(pools['train'])} "
          f"test {len(pools['test'])}", flush=True)

    Xa, ya = speech_aug_set(Xtr, ytr, pools["train"], rng)
    nets = {"clean": train(Xtr, ytr), "speech-aug": train(Xa, ya)}
    (REPO / "runs").mkdir(exist_ok=True)  # the adaptive attacker the live pipeline loads (gitignored)
    torch.save({"state_dict": nets["speech-aug"].state_dict(), "provisional": True, "speech_aug": True,
                "split_seed": SEED, "data": "harrison.npz + Hearsay test_internal train-speaker speech"},
               REPO / "runs" / "provisional_keynet_speechaug.pt")
    print(f"attackers trained ({time.time() - t0:.0f}s)", flush=True)

    # One speech excerpt + position per test press, reused at every level (paired design).
    sps = [excerpt(pools["test"], rng) for _ in Xte]
    poss = [int(rng.integers(0, EXCERPT - Xte.shape[1])) for _ in Xte]
    rows, qual = [], []
    for level in LEVELS:
        lvl = "keys only" if level is None else f"{level:+d} dB"
        shields = ("off", "on", "on14") if level in LEVELS_14 else ("off", "on")
        W = {(s, o): np.zeros_like(Xte) for s in shields for o in ("oracle", "detected")}
        hit = {(s, "detected"): np.zeros(len(Xte), bool) for s in shields}
        for i, (k, sp, pos) in enumerate(zip(Xte, sps, poss)):
            onset = pos + PRE
            raw = mix(k, sp, level, pos)
            shielded = Shield(SHIELD_CFG, seed=SEED + i).apply(raw, onsets=np.array([onset]))
            audio = {"off": raw, "on": shielded}
            if "on14" in shields:
                audio["on14"] = Shield(SHIELD_CFG_14, seed=SEED + i).apply(raw, onsets=np.array([onset]))
            for s, a in audio.items():
                W[s, "oracle"][i] = cut(a, onset)
                d = detect(a, onset)
                if d is not None:
                    W[s, "detected"][i] = cut(a, d); hit[s, "detected"][i] = True
            if level is not None:
                q14 = quality(raw, audio["on14"]) if "on14" in audio else (np.nan, np.nan)
                qual.append((lvl, *quality(raw, shielded), *quality(sp, raw), *quality(sp, shielded), *q14))
        for name, net in nets.items():
            for (s, o), w in W.items():
                r = top_ranks(net, w, yte)
                ok = hit.get((s, o), np.ones(len(yte), bool))
                n, k1, k3 = len(yte), int(((r == 0) & ok).sum()), int(((r < 3) & ok).sum())
                rows.append(dict(level=lvl, attacker=name, shield=s, onset=o, n=n,
                                 top1=k1 / n, top1_lo=wilson(k1, n)[0], top1_hi=wilson(k1, n)[1],
                                 top3=k3 / n, top3_lo=wilson(k3, n)[0], top3_hi=wilson(k3, n)[1],
                                 onset_found=ok.mean()))
        print(f"{lvl}: done ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    q = pd.DataFrame(qual, columns=["level", "stoi_shield_vs_mix", "pesq_shield_vs_mix", "stoi_mix_vs_speech",
                                    "pesq_mix_vs_speech", "stoi_shield_vs_speech", "pesq_shield_vs_speech",
                                    "stoi_shield14_vs_mix", "pesq_shield14_vs_mix"])
    q = q.groupby("level", sort=False).mean().reset_index()
    res = res.merge(q, on="level", how="left")
    out = REPO / "docs" / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "attack_under_speech.csv", index=False, float_format="%.4f")
    plot(res, out / "figures" / "attack_under_speech.png")
    print(res.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"attack done ({time.time() - t0:.0f}s)", flush=True)

    hs = hearsay_check(test_spk, Xte, rng)
    hs.to_csv(out / "attack_under_speech_hearsay.csv", index=False, float_format="%.4f")
    print(hs.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"total {time.time() - t0:.0f}s")


def hearsay_check(test_spk: set, Xte: np.ndarray, rng) -> pd.DataFrame:
    """Criterion 3: does the shield make a real voice look fake? Full test-speaker clips, test presses only."""
    sys.path.insert(0, str(REPO))
    from app.hearsay.driver import HearsayDriver
    drv = HearsayDriver(mode="r4ft", threads=8, device="cpu")
    m = real_clips()
    m = m[m.speaker.isin(test_spk)]
    m = m.sample(n=min(N_HEARSAY, len(m)), random_state=SEED)
    klen = Xte.shape[1]
    p = {c: [] for c in ("clean", "keys, shield off", "keys, shield on")}
    for i, r in enumerate(m.itertuples()):
        x, sr = sf.read(HEARSAY_ROOT / r.path, dtype="float32")
        assert sr == SR, (r.path, sr)
        x = x if x.ndim == 1 else x.mean(1)
        slot = len(x) // KEYS_PER_CLIP  # one press per slot, so presses never overlap
        keyed, onsets = x.copy(), []
        for j, ki in enumerate(rng.choice(len(Xte), KEYS_PER_CLIP, replace=False)):
            k = Xte[ki] * np.sqrt(power(x) / (power(Xte[ki]) * 10 ** (HEARSAY_LEVEL / 10)))
            pos = j * slot + int(rng.integers(0, slot - klen + 1))
            keyed[pos:pos + klen] += k
            onsets.append(pos + PRE)
        shielded = Shield(SHIELD_CFG, seed=SEED + i).apply(keyed, onsets=np.array(onsets))
        for c, a in zip(p, (x, keyed, shielded)):
            p[c].append(drv.score(a).p_synthetic)
        if i % 20 == 19:
            print(f"hearsay {i + 1}/{len(m)}", flush=True)
    rows = []
    for c, v in p.items():
        v = np.array(v); k, n = int((v > 0.5).sum()), len(v)
        lo, hi = wilson(k, n)
        rows.append(dict(condition=c, n=n, flagged=k, flag_rate=k / n, flag_lo=lo, flag_hi=hi,
                         p_synthetic_median=float(np.median(v)), p_synthetic_max=float(v.max())))
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = list(dict.fromkeys(res.level))
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, onset in zip(axes, ("oracle", "detected")):
        for (att, sh), color, ls in ((("speech-aug", "off"), "#c0392b", "-"), (("clean", "off"), "#e67e22", "--"),
                                     (("speech-aug", "on"), "#2471a3", "-"), (("clean", "on"), "#5dade2", "--")):
            d = res[(res.attacker == att) & (res.shield == sh) & (res.onset == onset)].set_index("level").loc[levels]
            ax.errorbar(x, d.top1 * 100, yerr=[(d.top1 - d.top1_lo) * 100, (d.top1_hi - d.top1) * 100],
                        color=color, ls=ls, marker="o", capsize=3, label=f"{att}, shield {sh}")
        ax.axhline(CHANCE * 100, color="gray", ls=":", label="chance (2.8 %)")
        ax.set_xticks(x, levels, rotation=30)
        ax.set_xlabel("speech-to-key level")
        ax.set_title(f"{onset} onset")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("top-1 key accuracy (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Keystroke attacker under speech, with and without Keyguard's shield (harrison, 95 % Wilson CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
