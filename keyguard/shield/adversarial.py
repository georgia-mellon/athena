"""Adversarial keystroke shield -- the AI-essential defense (min-max).

The defense is NOT "detect and delete" (that's deterministic and garble-able). It
solves a constrained adversarial game:

    min_D  max_A  Leakage(A(D(x)))   s.t.   Perceptual(D(x), x) <= eps

D adds a BOUNDED perturbation to the audio so a keystroke attacker A can't read
keys, while the change stays under a perceptual budget (inaudible / speech
preserved). This is provably not a fixed DSP rule: the optimal D is defined
THROUGH A's neural network (an adversarial-example optimization), and A retrains
against D, so D must co-adapt. "Just garble it" is the trivial max-perturbation
solution that violates the perceptual budget.

This module implements the core with a differentiable torch log-mel front-end
(no torchaudio needed), the KeyNet attacker, a learned universal adversarial
perturbation, and GAN-style co-training rounds. Runs on Harrison isolated keys
(the attacker that actually works) so the min-max is demonstrable.
"""
from __future__ import annotations
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from ..config import SR, KEY_WIN, N_MELS, N_FFT, HOP, CLASSES, CLS_IDX, RUNS, DATA  # CallGuard: + DATA
from ..attackers.supervised import KeyNet, DEVICE
from .. import audio, segment
from .. import memory

_MEL_FB = torch.tensor(
    librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS), dtype=torch.float32)


def torch_logmel(x: torch.Tensor) -> torch.Tensor:
    """Differentiable log-mel that numerically matches features.mel (librosa).

    Two settings are load-bearing for the match (corr 1.0 vs ~0.96 without):
    librosa 1.0 pads the STFT with zeros (pad_mode='constant'; torch defaults to
    'reflect'), and librosa.power_to_db floors the dB at max-80 (top_db=80.0).
    Both are pure-torch so the perturbation gradient stays intact.
    x:(B,samples)->(B,1,mel,frames)."""
    fb = _MEL_FB.to(x.device)
    win = torch.hann_window(N_FFT, device=x.device)
    spec = torch.stft(x, N_FFT, hop_length=HOP, window=win, return_complex=True,
                      center=True, pad_mode="constant")       # librosa 1.0 default
    power = (spec.real ** 2 + spec.imag ** 2)                 # (B,freq,frames)
    mel = torch.matmul(fb, power)                             # (B,mel,frames)
    logmel = 10 * torch.log10(torch.clamp(mel, min=1e-10))
    logmel = logmel - logmel.amax(dim=(1, 2), keepdim=True)   # ref=max, like librosa
    logmel = torch.maximum(logmel, logmel.amax(dim=(1, 2), keepdim=True) - 80.0)  # top_db=80
    m = logmel.mean(dim=(1, 2), keepdim=True)
    s = logmel.std(dim=(1, 2), keepdim=True) + 1e-6
    return ((logmel - m) / s).unsqueeze(1)                    # (B,1,mel,frames)


def harrison_windows(root=str(DATA / "harrison" / "MBPWavs"), n_test=5):  # CallGuard: under config.DATA, not cwd
    """Audio windows per key -> (Xtr, ytr, Xte, yte) float32 (n, KEY_WIN)."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for k in CLASSES:
        try:
            y = audio.load(f"{root}/{k}.wav")
        except Exception:
            continue
        on = segment.onsets_n(y, 25)
        w = segment.windows(y, on)                            # (n, KEY_WIN)
        Xtr.append(w[:-n_test]); ytr += [CLS_IDX[k]] * (len(w) - n_test)
        Xte.append(w[-n_test:]); yte += [CLS_IDX[k]] * n_test
    return (np.concatenate(Xtr), np.array(ytr),
            np.concatenate(Xte), np.array(yte))


def _acc(net, X, y):
    net.eval()
    with torch.no_grad():
        pred = net(torch_logmel(X)).argmax(1)
    return (pred == y).float().mean().item()


def train_attacker(net, X, y, epochs=40, lr=1e-3, bs=64):
    """Minibatch SGD -- full-batch (one step/epoch) badly undertrains the attacker
    (~37% on Harrison); minibatches get it to ~88%, the honest clean baseline."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(net(torch_logmel(X[idx])), y[idx])
            loss.backward(); opt.step()
    return net


def optimize_perturbation(net, X, y, eps, steps=150, lr=5e-3):
    """Universal audio perturbation delta (KEY_WIN,) that fools net, ||delta||_2<=eps.
    eps is set relative to keystroke RMS (perceptual budget)."""
    delta = torch.zeros(KEY_WIN, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    net.eval()
    for _ in range(steps):
        opt.zero_grad()
        logits = net(torch_logmel(X + delta))
        loss = -F.cross_entropy(logits, y)          # maximize attacker error
        loss.backward(); opt.step()
        with torch.no_grad():                        # project to eps L2 ball
            nrm = delta.norm()
            if nrm > eps:
                delta.mul_(eps / nrm)
    return delta.detach()


def _write_arena(run_dir, state):
    """Persist arena state after every round so the dashboard can stream it live."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "arena.json").write_text(json.dumps(state, indent=2))


def co_train(rounds=4, eps_snr_db=-18.0, epochs=30):
    """GAN-style min-max: alternate (optimize delta vs frozen A) and (retrain A on
    perturbed audio). Reports the arms race + the perceptual cost of delta, and
    persists every round to runs/arena/adv-<ts>/arena.json (streamed to the
    dashboard) plus a cross-session memory of the run (keyguard.memory)."""
    Xtr_np, ytr_np, Xte_np, yte_np = harrison_windows()
    Xtr = torch.tensor(Xtr_np, device=DEVICE); ytr = torch.tensor(ytr_np, device=DEVICE)
    Xte = torch.tensor(Xte_np, device=DEVICE); yte = torch.tensor(yte_np, device=DEVICE)
    key_rms = Xtr.pow(2).mean().sqrt().item()
    eps = key_rms * (10 ** (eps_snr_db / 20)) * np.sqrt(KEY_WIN)   # L2 budget at SNR
    net = KeyNet().to(DEVICE)
    ckpt = RUNS / "supervised_mbp.pt"
    if ckpt.exists():   # warm-start: the production 87% attacker (features now match librosa)
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"warm-started attacker from {ckpt.name}")
    else:
        train_attacker(net, Xtr, ytr, epochs=60)
    clean = _acc(net, Xte, yte)
    chance = 1 / len(CLASSES)
    created = memory.now_iso()
    run_dir = RUNS / "arena" / f"adv-{time.strftime('%Y%m%d-%H%M%S')}"
    state = {"created": created, "budget_snr_db": eps_snr_db, "chance": chance,
             "clean_acc": clean, "status": "running", "rounds": []}
    _write_arena(run_dir, state)
    print(f"perturbation budget: {eps_snr_db} dB SNR (inaudible-ish)")
    print(f"{'round':>6}{'attack_acc(perturbed)':>24}{'after_retrain':>16}", flush=True)
    delta = optimize_perturbation(net, Xtr, ytr, eps)
    for r in range(rounds):
        shielded = _acc(net, Xte + delta, yte)               # A vs current delta
        train_attacker(net, Xtr + delta, ytr, epochs=epochs)  # A adapts to delta
        adapted = _acc(net, Xte + delta, yte)
        print(f"{r:>6}{shielded:>23.1%}{adapted:>16.1%}", flush=True)
        state["rounds"].append({"round": r, "shielded_acc": shielded,
                                "retrained_acc": adapted})
        _write_arena(run_dir, state)
        delta = optimize_perturbation(net, Xtr, ytr, eps)     # D re-optimizes vs adapted A
    final_shielded = _acc(net, Xte + delta, yte)
    d_snr = 20 * np.log10(key_rms / (delta.norm().item() / np.sqrt(KEY_WIN) + 1e-12))
    state["status"] = "done"
    state["final"] = {"final_shielded": final_shielded, "snr_db": d_snr}
    _write_arena(run_dir, state)
    memory.remember_run({"created": created, "clean_acc": clean,
                         "final_shielded": final_shielded, "snr_db": d_snr,
                         "rounds": rounds, "chance": chance})
    print(f"\nclean attack acc: {clean:.1%}  (chance {chance:.1%})")
    print(f"final shielded acc vs adapted attacker: {final_shielded:.1%}  "
          f"at perturbation SNR {d_snr:.1f} dB")
    return {"clean": clean, "final": final_shielded, "snr_db": d_snr}


def _pop_perturbation(nets, X, y, eps, steps=150, lr=5e-3):
    """One universal perturbation delta vs a WHOLE population: maximize the SUM of
    every attacker's cross-entropy at once (worst-case-ish multi-adversary robust).
    Backprops through all nets simultaneously, same eps L2 ball as the single case."""
    delta = torch.zeros(KEY_WIN, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for net in nets:
        net.eval()
    for _ in range(steps):
        opt.zero_grad()
        feats = torch_logmel(X + delta)                 # shared front-end, computed once
        loss = -sum(F.cross_entropy(net(feats), y) for net in nets)  # maximize total error
        loss.backward(); opt.step()
        with torch.no_grad():
            nrm = delta.norm()
            if nrm > eps:
                delta.mul_(eps / nrm)
    return delta.detach()


def co_train_population(rounds=4, eps_snr_db=-18.0, epochs=30, pop=None,
                        warm_epochs=45, pert_steps=150):
    """Multi-adversary min-max: D optimizes ONE inaudible perturbation against a
    whole POPULATION of diverse attacker agents. Each round D re-solves delta vs
    the summed loss of every (retrained) attacker, so no single architecture's
    blind spot can be exploited -- if all N are held at chance, the acoustic
    signal itself is dead. Persists the extended population schema per round and
    remembers the run (worst-case) in Backboard."""
    from ..attackers.population import population
    Xtr_np, ytr_np, Xte_np, yte_np = harrison_windows()
    Xtr = torch.tensor(Xtr_np, device=DEVICE); ytr = torch.tensor(ytr_np, device=DEVICE)
    Xte = torch.tensor(Xte_np, device=DEVICE); yte = torch.tensor(yte_np, device=DEVICE)
    key_rms = Xtr.pow(2).mean().sqrt().item()
    eps = key_rms * (10 ** (eps_snr_db / 20)) * np.sqrt(KEY_WIN)
    agents = [(name, net.to(DEVICE)) for name, net in (pop or population())]
    chance = 1 / len(CLASSES)
    created = memory.now_iso()
    run_dir = RUNS / "arena" / f"adv-{time.strftime('%Y%m%d-%H%M%S')}"

    ckpt = RUNS / "supervised_mbp.pt"
    clean = {}
    for name, net in agents:
        if name == "keynet" and ckpt.exists():   # warm-start the production 87% attacker
            net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            train_attacker(net, Xtr, ytr, epochs=warm_epochs)   # others to their clean baseline
        clean[name] = _acc(net, Xte, yte)
        print(f"agent {name:>9}: clean acc {clean[name]:.1%}", flush=True)

    state = {"created": created, "budget_snr_db": eps_snr_db, "chance": chance,
             "mode": "population", "status": "running",
             "attackers": [{"name": n, "clean_acc": clean[n]} for n, _ in agents],
             "clean_acc": float(np.mean(list(clean.values()))), "rounds": []}
    _write_arena(run_dir, state)
    print(f"perturbation budget: {eps_snr_db} dB SNR vs population of {len(agents)}")

    nets = [net for _, net in agents]
    delta = _pop_perturbation(nets, Xtr, ytr, eps, steps=pert_steps)
    for r in range(rounds):
        per = {}
        for name, net in agents:
            shielded = _acc(net, Xte + delta, yte)                  # A vs current delta
            train_attacker(net, Xtr + delta, ytr, epochs=epochs)    # A adapts to delta
            retrained = _acc(net, Xte + delta, yte)
            per[name] = {"shielded_acc": shielded, "retrained_acc": retrained}
        worst_re = max(v["retrained_acc"] for v in per.values())
        print(f"round {r}: worst retrained {worst_re:.1%}  " +
              "  ".join(f"{n}={per[n]['shielded_acc']:.1%}->{per[n]['retrained_acc']:.1%}"
                        for n in clean), flush=True)
        state["rounds"].append({"round": r, "per_attacker": per})
        _write_arena(run_dir, state)
        delta = _pop_perturbation(nets, Xtr, ytr, eps, steps=pert_steps)  # D re-solves vs whole pop

    final = {name: _acc(net, Xte + delta, yte) for name, net in agents}
    worst_final = max(final.values())
    d_snr = 20 * np.log10(key_rms / (delta.norm().item() / np.sqrt(KEY_WIN) + 1e-12))
    state["status"] = "done"
    state["final"] = {"per_attacker": final, "worst_final_shielded": worst_final,
                      "snr_db": d_snr}
    _write_arena(run_dir, state)
    memory.remember_run({"created": created, "clean_acc": state["clean_acc"],
                         "final_shielded": worst_final, "snr_db": d_snr,
                         "rounds": rounds, "chance": chance, "mode": "population",
                         "attackers": list(clean.keys())})
    print(f"\nclean (mean) {state['clean_acc']:.1%}  "
          f"worst final shielded {worst_final:.1%}  (chance {chance:.1%})  "
          f"at SNR {d_snr:.1f} dB")
    return {"clean": clean, "final": final, "worst_final": worst_final, "snr_db": d_snr}


def demo_population():
    """Fast assert-based self-check for the population path (tiny epochs)."""
    from ..attackers.population import population
    Xtr, ytr, _, _ = harrison_windows()
    X = torch.tensor(Xtr[:180], device=DEVICE); y = torch.tensor(ytr[:180], device=DEVICE)
    nets = [net.to(DEVICE) for _, net in population()]
    for net in nets:
        train_attacker(net, X, y, epochs=12)
    a0 = [_acc(net, X, y) for net in nets]
    key_rms = X.pow(2).mean().sqrt().item()
    eps = key_rms * (10 ** (-18 / 20)) * np.sqrt(KEY_WIN)
    d = _pop_perturbation(nets, X, y, eps, steps=60)
    a1 = [_acc(net, X + d, y) for net in nets]
    # _pop_perturbation maximizes the SUMMED loss, so the guaranteed property is an
    # aggregate (mean) drop, not per-agent monotonicity on a tiny undertrained subset.
    assert np.mean(a1) < np.mean(a0), (a0, a1)
    print(f"population co-train ok: one -18dB perturbation lowers the population "
          f"mean {np.mean(a0):.0%}->{np.mean(a1):.0%} across {len(nets)} agents "
          f"({[f'{a:.0%}->{b:.0%}' for a, b in zip(a0, a1)]}).")


def demo():
    """Fast smoke: differentiable mel + a few adversarial steps drop accuracy."""
    Xtr, ytr, _, _ = harrison_windows()
    X = torch.tensor(Xtr[:72], device=DEVICE); y = torch.tensor(ytr[:72], device=DEVICE)
    net = KeyNet().to(DEVICE)
    train_attacker(net, X, y, epochs=40)
    a0 = _acc(net, X, y)
    key_rms = X.pow(2).mean().sqrt().item()
    eps = key_rms * (10 ** (-18 / 20)) * np.sqrt(KEY_WIN)
    d = optimize_perturbation(net, X, y, eps, steps=60)
    a1 = _acc(net, X + d, y)
    assert a1 < a0, (a0, a1)
    print(f"adversarial demo ok: train acc {a0:.1%} -> {a1:.1%} under a bounded "
          f"(-18dB) universal perturbation. AI is doing the work.")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "cotrain":
        co_train()
    elif arg == "cotrain_pop":
        co_train_population()
    elif arg == "poptest":
        demo_population()
    else:
        demo()
