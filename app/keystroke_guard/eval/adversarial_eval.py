"""Does the adversarial keystroke shield beat the DSP shield, and against which attacker?

Held-out design of attack_under_speech.py (whose helpers this imports): harrison_split TEST presses (360, never
trained on), each at a random spot in a 1.5 s test-speaker speech excerpt (paired: same excerpt, position and OS
onset error under every shield), top-1 with 95 % Wilson CIs, chance 1/36.

What is new here: every shielded condition goes through the RUNTIME streaming driver (KeyguardShield.process, 20 ms
blocks, 80 ms lookahead), fed OS-style key events = true onset + uniform +/-30 ms, delivered one block late.
Shields: none | dsp | adversarial | dsp+adversarial. Conditions: keys only | +10 dB speech (attack_under_speech's
level: excerpt power over key-window power). Onsets: oracle | detected (Keyguard's segment.onsets, 30 ms match).

Attackers (all trained on harrison_split TRAIN presses only; speech from the train-speaker pool):
  (a) white-box: the ensemble the deltas were optimized against (provisional clean, s1-speechaug, s2-advretrain;
      the provisional speech-aug KeyNet is reported but flagged: its split leaked TEST presses).
  (b) transfer: held out of the delta optimization: a fresh speech-aug KeyNet (seed 5) and the builder's
      speech-aug WideCNN (seed 3, keyguard.attackers.population).
  (c) adaptive: a fresh KeyNet per shield, trained on train presses PASSED THROUGH that shield's live driver
      (clean + 3 speech copies at +0..+20 dB, same OS onset error). Its "none" member is (b)'s KeyNet-s5.
Plus speech quality (STOI, wideband PESQ) of shielded vs unshielded mixtures at +10 and 0 dB, and Hearsay false
flags (r4ft, 100 test-speaker clips, 5 test presses each at +10 dB) for every shield.

Run: .venv/Scripts/python -m app.keystroke_guard.eval.adversarial_eval  (KEYGUARD_ROOT, HEARSAY_ROOT; ~25 min, 8 cpu)
Writes docs/reports/adversarial_shield.csv, _quality.csv, _hearsay.csv, figures/adversarial_shield.png.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf

from app.keystroke_guard.eval import attack_under_speech as aus   # sets up the read-only Keyguard import

os.environ.setdefault("KEYGUARD_ROOT", str(aus.KEYGUARD_ROOT))       # workers inherit it
os.environ.setdefault("HEARSAY_ROOT", str(aus.HEARSAY_ROOT))
import torch  # noqa: E402

from app.keystroke_guard import adversarial as A  # noqa: E402
from app.keystroke_guard.driver import harrison_split  # noqa: E402
from app.source.types import BLOCK  # noqa: E402

SHIELDS = ("none", "dsp", "adversarial", "dsp+adversarial")
OS_JITTER = int(0.030 * aus.SR)      # OS key event error, uniform +/-30 ms
TRAIN_COPIES = 3                      # adaptive attackers: clean + 3 speech copies, all through the shield
SMOKE = bool(os.environ.get("ADV_EVAL_SMOKE"))
EPOCHS = 1 if SMOKE else 30           # 30 as the builder's attackers
N_HEARSAY = 3 if SMOKE else aus.N_HEARSAY
CHANCE = 1 / A.N_KEYS
WORKERS = 8
CHUNK = 10 if SMOKE else 270                          # strokes per streaming job
OUT = aus.REPO / "docs" / "reports"


# --- streaming (runs in worker processes) -------------------------------------------------------------------------
def _init_worker():
    sys.dont_write_bytecode = True
    torch.set_num_threads(1)


def stream(x: np.ndarray, events: np.ndarray, mode: str, seed: int, cut_at: np.ndarray | None = None) -> np.ndarray:
    """x through the live KeyguardShield in 20 ms blocks, each OS event handed over with the block after it.
    Returns the aligned output, or the attacker windows cut at `cut_at` (to keep pickles small)."""
    if mode != "none":
        from app.keystroke_guard.driver import KeyguardShield
        s = KeyguardShield(mode=mode, seed=seed)
        ev = np.sort(np.clip(events, 0, None))
        xp = np.concatenate([x, np.zeros(s.latency + 2 * BLOCK - len(x) % BLOCK, np.float32)])
        out, j = [], 0
        for i in range(0, len(xp), BLOCK):
            k = int(np.searchsorted(ev, i))
            out.append(s.process(xp[i:i + BLOCK], ev[j:k].tolist()))
            j = k
        x = np.concatenate(out)[s.latency:s.latency + len(x)]
    return x if cut_at is None else np.stack([aus.cut(x, int(o)) for o in cut_at])


def _quality_job(args):
    raws, outs, sps = args
    return [(*aus.quality(r, o), *aus.quality(s, r), *aus.quality(s, o)) for r, o, s in zip(raws, outs, sps)]


# --- data ---------------------------------------------------------------------------------------------------------
def speech_pools():
    """attack_under_speech's speaker split, rng replayed exactly as adversarial.speech_bank does."""
    rng = np.random.default_rng(aus.SEED)
    aus.load_keys(rng)
    return aus.load_speech(rng)


def build(X, pool, levels, rng):
    """One 1.5 s excerpt per press, concatenated: (audio, true onsets, OS events, per-press raw excerpts, speech)."""
    n = len(X)
    sps = [aus.excerpt(pool, rng) for _ in range(n)]
    poss = rng.integers(0, aus.EXCERPT - X.shape[1], n)
    raws = [aus.mix(k, sp, lv, int(p)) for k, sp, lv, p in zip(X, sps, levels, poss)]
    onsets = np.arange(n) * aus.EXCERPT + poss + aus.PRE
    events = onsets + rng.integers(-OS_JITTER, OS_JITTER + 1, n)
    return np.concatenate(raws), onsets, events, sps


def chunked(pool, x, onsets, events, mode, seed, cut=True):
    """Stream x in CHUNK-stroke jobs (each its own driver, like separate calls); strokes never straddle chunks.
    Returns futures, in order."""
    L = CHUNK * aus.EXCERPT
    futs = []
    for c in range(0, len(x), L):
        m = (onsets >= c) & (onsets < c + L)
        futs.append(pool.submit(stream, x[c:c + L], events[m] - c, mode, seed + c // L, onsets[m] - c if cut else None))
    return futs


def gather(futs):
    return np.concatenate([f.result() for f in futs])


# --- attackers ----------------------------------------------------------------------------------------------------
def fit(net, W, y, seed):
    from keyguard.shield.adversarial import train_attacker
    torch.manual_seed(seed)
    train_attacker(net.requires_grad_(True), torch.from_numpy(W), torch.from_numpy(y).long(), epochs=EPOCHS)
    return net.eval()


def ranks(net, W, y):
    return aus.top_ranks(net, W.astype(np.float32), y)


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar p on paired hits."""
    from scipy.stats import binomtest
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return 1.0 if n01 + n10 == 0 else float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


# --- main ---------------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    lap = lambda s: print(f"[{time.time() - t0:5.0f} s] {s}", flush=True)  # noqa: E731
    torch.set_num_threads(8)
    Xtr, ytr, Xte, yte = harrison_split()
    if SMOKE:   # pipeline check only: tiny subsets, numbers meaningless
        Xtr, ytr, Xte, yte = Xtr[::20], ytr[::20], Xte[::20], yte[::20]
    pools, test_spk = speech_pools()
    rng = np.random.default_rng(1234)
    lap(f"train {len(Xtr)} / test {len(Xte)} presses; speech clips train {len(pools['train'])} "
        f"test {len(pools['test'])}")

    # Test mixtures: same excerpts/positions/OS errors at every level (the rng is re-seeded per level).
    tests = {lv: build(Xte, pools["test"], [lv] * len(Xte), np.random.default_rng(7)) for lv in (None, 10, 0)}
    # Adaptive-attacker training mixtures: clean copy + TRAIN_COPIES speech copies at +0..+20 dB (train speakers).
    Xa = np.concatenate([Xtr] * (TRAIN_COPIES + 1))
    ya = np.concatenate([ytr] * (TRAIN_COPIES + 1))
    lv_tr = [None] * len(Xtr) + list(rng.uniform(0, 20, len(Xtr) * TRAIN_COPIES))
    tr_x, tr_on, tr_ev, _ = build(Xa, pools["train"], lv_tr, rng)

    with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as pool:
        test_f = {(lv, mode): chunked(pool, x, on, ev, mode, 200, cut=False)
                  for lv, (x, on, ev, _) in tests.items() for mode in SHIELDS}
        train_f = {mode: chunked(pool, tr_x, tr_on, tr_ev, mode, 100) for mode in SHIELDS[1:]}
        tr_w = {"none": np.stack([aus.cut(tr_x, int(o)) for o in tr_on])}
        outs = {k: gather(f) for k, f in test_f.items()}
        lap("test streams done")
        tr_w.update({mode: gather(f) for mode, f in train_f.items()})
        del tr_x
        lap("adaptive-attacker training streams done")

        # Speech quality in the pool while the attackers train.
        qfut = {}
        for lv in (10, 0):
            x, on, ev, sps = tests[lv]
            E = aus.EXCERPT
            per = lambda a: [a[i * E:(i + 1) * E] for i in range(len(on))]  # noqa: E731
            for mode in SHIELDS[1:]:
                qfut[lv, mode] = [pool.submit(_quality_job, (per(x)[c:c + 45], per(outs[lv, mode])[c:c + 45],
                                                             sps[c:c + 45])) for c in range(0, len(on), 45)]

        A._torch(8)
        from keyguard.attackers.population import WideCNN
        nets = {("white-box", "keynet-clean (provisional)"): A._keynet(torch, A.RUNS / "provisional_keynet.pt"),
                ("white-box", "keynet-s1-speechaug"): A._keynet(torch, A.RUNS / "adv_attacker_keynet_s1_speechaug.pt"),
                ("white-box", "keynet-s2-advretrain"): A._keynet(torch, A.RUNS / "adv_attacker_keynet_s2_advretrain.pt"),
                ("white-box*", "keynet-speechaug (provisional, split leak)"):
                    A._keynet(torch, A.RUNS / "provisional_keynet_speechaug.pt")}
        wide = WideCNN(A.N_KEYS)
        wide.load_state_dict(torch.load(A.RUNS / "adv_attacker_widecnn_s3_speechaug.pt", map_location="cpu")["state_dict"])
        nets["transfer", "widecnn-s3-speechaug"] = wide.eval()
        for seed, mode in zip((5, 6, 7, 8), SHIELDS):
            nets["adaptive", f"keynet-s{seed} through {mode}"] = fit(A._keynet(torch), tr_w[mode], ya, seed)
            lap(f"adaptive attacker through {mode} trained")
        nets["transfer", "keynet-s5-speechaug"] = nets["adaptive", "keynet-s5 through none"]
        qual = []
        for (lv, mode), fs in qfut.items():
            q = np.array([r for f in fs for r in f.result()], float)
            qual.append(dict(level=f"{lv:+d} dB", shield=mode, n=len(q),
                             **{c: np.nanmean(q[:, j]) for j, c in enumerate(
                                 ["stoi_shield_vs_mix", "pesq_shield_vs_mix", "stoi_mix_vs_speech",
                                  "pesq_mix_vs_speech", "stoi_shield_vs_speech", "pesq_shield_vs_speech"])},
                             stoi_shield_vs_mix_p10=np.nanpercentile(q[:, 0], 10)))
        lap("quality done")

    # Attack grid.
    rows, hits = [], {}
    for lv in (None, 10):
        lvl = "keys only" if lv is None else f"+{lv} dB"
        x, on, ev, _ = tests[lv]
        E = aus.EXCERPT
        for mode in SHIELDS:
            y = outs[lv, mode]
            Wo = np.stack([aus.cut(y, int(o)) for o in on])
            Wd, found = np.zeros_like(Wo), np.zeros(len(on), bool)
            for i, o in enumerate(on):
                d = aus.detect(y[i * E:(i + 1) * E], int(o) - i * E)
                if d is not None:
                    Wd[i], found[i] = aus.cut(y[i * E:(i + 1) * E], d), True
            if lv is None and mode == "adversarial":
                added = Wo - np.stack([aus.cut(x, int(o)) for o in on])
                ratio = 10 * np.log10(np.mean(Xte.astype(np.float64) ** 2, 1) / (np.mean(added ** 2, 1) + 1e-20))
            for (cls, name), net in nets.items():
                for onset, W, ok in (("oracle", Wo, np.ones(len(on), bool)), ("detected", Wd, found)):
                    r = ranks(net, W, yte)
                    h = (r == 0) & ok
                    hits[lvl, mode, onset, name] = h
                    n, k1, k3 = len(yte), int(h.sum()), int(((r < 3) & ok).sum())
                    rows.append(dict(level=lvl, shield=mode, onset=onset, attacker_class=cls, attacker=name, n=n,
                                     top1=k1 / n, top1_lo=aus.wilson(k1, n)[0], top1_hi=aus.wilson(k1, n)[1],
                                     top3=k3 / n, onset_found=ok.mean()))
    # Paired tests: each adversarial mode vs dsp, same presses, same attacker.
    for r in rows:
        if r["shield"] in ("adversarial", "dsp+adversarial"):
            r["p_vs_dsp"] = mcnemar(hits[r["level"], "dsp", r["onset"], r["attacker"]],
                                    hits[r["level"], r["shield"], r["onset"], r["attacker"]])
    res = pd.DataFrame(rows)
    lap("attack grid done")

    hs = hearsay_check(test_spk, Xte, np.random.default_rng(11))
    lap("hearsay done")

    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "adversarial_shield.csv", index=False, float_format="%.4f")
    q = pd.DataFrame(qual)
    q.to_csv(OUT / "adversarial_shield_quality.csv", index=False, float_format="%.4f")
    hs.to_csv(OUT / "adversarial_shield_hearsay.csv", index=False, float_format="%.4f")
    plot(res, OUT / "figures" / "adversarial_shield.png")
    pd.set_option("display.width", 250)
    head = res[res.attacker_class != "white-box*"]
    for onset in ("oracle", "detected"):
        h = head[head.onset == onset]
        t = h.loc[h.groupby(["level", "shield", "attacker_class"], sort=False).top1.idxmax()]
        print(f"\n== {onset}: worst case (max top-1) per attacker class ==")
        print(t[["level", "shield", "attacker_class", "attacker", "top1", "top1_lo", "top1_hi", "onset_found",
                 "p_vs_dsp"]].to_string(float_format=lambda v: f"{v:.3f}"))
    adap = res[(res.attacker_class == "adaptive")]
    print("\n== adaptive attackers x every shield (top-1, oracle) ==")
    print(adap[adap.onset == "oracle"].pivot_table(index=["level", "attacker"], columns="shield", values="top1",
                                                   sort=False).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n== all attackers (top-1, oracle) ==")
    print(res[res.onset == "oracle"].pivot_table(index=["level", "attacker"], columns="shield", values="top1",
                                                 sort=False).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nadversarial mode, keys only: key-window power over added-perturbation power per stroke (dB): "
          f"median {np.median(ratio):.1f}, p5 {np.percentile(ratio, 5):.1f}, min {ratio.min():.1f}")
    print("\n" + q.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + hs.to_string(float_format=lambda v: f"{v:.3f}"))
    lap("total")


def hearsay_check(test_spk, Xte, rng) -> pd.DataFrame:
    """attack_under_speech's criterion-3 layout (5 test presses per clip at +10 dB), shields run live."""
    from app.hearsay.driver import HearsayDriver
    drv = HearsayDriver(mode="r4ft", threads=8, device="cpu")
    m = aus.real_clips()
    m = m[m.speaker.isin(test_spk)].sample(n=N_HEARSAY, random_state=aus.SEED)
    klen = Xte.shape[1]
    clean, keyed, onsets = [], [], []
    base = 0
    for r in m.itertuples():
        x, sr = sf.read(aus.HEARSAY_ROOT / r.path, dtype="float32")
        assert sr == aus.SR, (r.path, sr)
        x = x if x.ndim == 1 else x.mean(1)
        slot = len(x) // aus.KEYS_PER_CLIP
        k_ = x.copy()
        for j, ki in enumerate(rng.choice(len(Xte), aus.KEYS_PER_CLIP, replace=False)):
            k = Xte[ki] * np.sqrt(aus.power(x) / (aus.power(Xte[ki]) * 10 ** (aus.HEARSAY_LEVEL / 10)))
            pos = j * slot + int(rng.integers(0, slot - klen + 1))
            k_[pos:pos + klen] += k
            onsets.append(base + pos + aus.PRE)
        clean.append(x); keyed.append(k_); base += len(x)
    onsets = np.array(onsets)
    events = onsets + rng.integers(-OS_JITTER, OS_JITTER + 1, len(onsets))
    edges = np.cumsum([0] + [len(c) for c in clean])
    conds = {"clean": clean, "keys, no shield": keyed}
    allk = np.concatenate(keyed)
    for mode in SHIELDS[1:]:
        y = stream(allk, events, mode, seed=300)
        conds[f"keys, {mode}"] = [y[a:b] for a, b in zip(edges[:-1], edges[1:])]
    rows = []
    for c, clips in conds.items():
        v = np.array([drv.score(a).p_synthetic for a in clips])
        k, n = int((v > 0.5).sum()), len(v)
        lo, hi = aus.wilson(k, n)
        rows.append(dict(condition=c, n=n, flagged=k, flag_rate=k / n, flag_lo=lo, flag_hi=hi,
                         p_synthetic_median=float(np.median(v)), p_synthetic_max=float(v.max())))
        print(f"hearsay {c}: {k}/{n} flagged", flush=True)
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"none": "#7f7f7f", "dsp": "#2471a3", "adversarial": "#e67e22", "dsp+adversarial": "#c0392b"}
    d = res[(res.attacker_class != "white-box*")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    groups = ("white-box", "transfer", "adaptive")
    for row, lvl in enumerate(("keys only", "+10 dB")):
        for col, onset in enumerate(("oracle", "detected")):
            ax = axes[row, col]
            for si, sh in enumerate(SHIELDS):
                vals, lo, hi = [], [], []
                for g in groups:
                    s = d[(d.level == lvl) & (d.onset == onset) & (d.shield == sh) & (d.attacker_class == g)]
                    if g == "adaptive":     # the attacker trained through THIS shield
                        s = s[s.attacker.str.endswith(f"through {sh}")]
                    b = s.loc[s.top1.idxmax()]
                    vals.append(b.top1 * 100); lo.append((b.top1 - b.top1_lo) * 100); hi.append((b.top1_hi - b.top1) * 100)
                x = np.arange(len(groups)) + (si - 1.5) * 0.2
                ax.bar(x, vals, 0.2, yerr=[lo, hi], capsize=2, color=colors[sh], label=sh)
            ax.axhline(CHANCE * 100, color="black", ls=":", lw=1, label="chance 2.8 %")
            ax.set_xticks(range(len(groups)), ["white-box\n(worst of 3)", "transfer\n(worst of 2)",
                                               "adaptive\n(trained through shield)"])
            ax.set_title(f"{lvl}, {onset} onset")
            ax.grid(axis="y", alpha=0.3)
    for a in axes[:, 0]:
        a.set_ylabel("top-1 key accuracy (%)")
    axes[0, 1].legend(fontsize=8, title="shield (live driver, OS onsets +/-30 ms)")
    fig.suptitle("Keystroke attackers vs CallGuard shields, 360 held-out harrison presses (95 % Wilson CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
