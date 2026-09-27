"""Keyguard's attacker population as Athena readers (Ares' per-key sub-agents).

Four attackers with different inductive biases (keynet SE-CNN, widecnn, resnet, framegru Bi-GRU), all on the same
log-mel of KEY_WIN windows cut at onsets. Athena trains them on the harrison press bank (36 keys A-Z0-9) and loads
them as PopulationReader agents.

Run:
  uv run python -m app.keystroke_guard.population train [--epochs N]      -> runs/keyguard/population/<name>.pt + .json
  uv run python -m app.keystroke_guard.population arena [--rounds N --epochs N]   (keyguard co_train_population)
  uv run python -m app.keystroke_guard.population burst                    (accuracy on the arms-race demo burst)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from app.keystroke_guard import driver
from keyguard.attackers.population import population
from keyguard.config import CLASSES, RUNS as KEYGUARD_RUNS
from keyguard.segment import windows
from keyguard.shield import adversarial
from keyguard.shield.adversarial import torch_logmel, train_attacker

log = logging.getLogger(__name__)

POP_DIR = KEYGUARD_RUNS / "population"
WARM = KEYGUARD_RUNS / "supervised_mbp.pt"   # Keyguard's production KeyNet (36-way head)
TOP_K = 3


def architectures(n_classes: int = driver.N_KEYS) -> list[tuple[str, torch.nn.Module]]:
    """Keyguard's population, rebuilt with an n-way head (harrison = 36 keys; Keyguard's default adds space = 37)."""
    return [(name, type(net)(n_classes=n_classes)) for name, net in population()]


def _state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu")
    return state.get("state_dict", state)


class PopulationReader:
    """One population attacker: top-3 keys per onset. Space (a 37th logit) is masked: harrison has no space."""

    def __init__(self, name: str, net: torch.nn.Module):
        self.name, self.net = name, net.eval()
        self.classes = list(CLASSES)[:driver.N_KEYS]

    def logits(self, wins: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(wins, dtype=np.float32))
            x = x.to(next(self.net.parameters()).device)
            return self.net(torch_logmel(x))[:, :len(self.classes)].cpu()

    def read(self, audio: np.ndarray, onsets: np.ndarray) -> list[list[str]]:
        onsets = np.asarray(onsets, dtype=int)
        if len(onsets) == 0:
            return []
        top = self.logits(windows(np.asarray(audio, np.float32), onsets)).topk(TOP_K, dim=1).indices
        return [[self.classes[j] for j in row] for row in top.tolist()]


def load_population(device: str = "cpu", root: Path = POP_DIR) -> list[PopulationReader]:
    """Readers for every architecture with trained weights in root/<name>.pt; skipped ones are logged."""
    readers = []
    for name, _ in architectures():
        path = Path(root) / f"{name}.pt"
        if not path.exists():
            log.warning("population: skipping %s: no weights at %s (run `population train`)", name, path)
            continue
        try:
            state = _state(path)
            net = dict(architectures(state["head.weight"].shape[0]))[name]
            net.load_state_dict(state)
        except Exception as e:
            log.warning("population: skipping %s: cannot load %s (%s)", name, path, e)
            continue
        readers.append(PopulationReader(name, net.to(device)))
    return readers


def _warm_start(net: torch.nn.Module, path: Path) -> bool:
    """Load path into net when every tensor shape matches (keynet <- supervised_mbp.pt)."""
    if not path.exists():
        return False
    state, own = _state(path), net.state_dict()
    if state.keys() != own.keys() or any(state[k].shape != own[k].shape for k in own):
        return False
    net.load_state_dict(state)
    return True


def held_out(net: torch.nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    top = PopulationReader("", net).logits(X).topk(TOP_K, dim=1).indices.numpy()
    return {"top1": float((top[:, 0] == y).mean()), "top3": float((top == y[:, None]).any(1).mean()),
            "n_test": int(len(y))}


def train(epochs: int = 30, root: Path = POP_DIR, warm: Path = WARM, seed: int = driver.SPLIT_SEED) -> dict:
    """Train every architecture on the harrison train split; save <name>.pt and population.json (held-out top-1/3)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = driver.harrison_split(seed)
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    report = {"data": "harrison.npz", "split_seed": seed, "train_frac": driver.TRAIN_FRAC, "epochs": epochs,
              "n_train": int(len(ytr)), "chance_top1": 1 / driver.N_KEYS, "agents": {}}
    for name, net in architectures():
        torch.manual_seed(seed)
        warmed = name == "keynet" and _warm_start(net, Path(warm))
        # supervised_mbp.pt saw these test presses, so a warm keynet's held-out is optimistic; `before` shows how much
        before = held_out(net, Xte, yte)["top1"] if warmed else None
        t0 = time.perf_counter()
        train_attacker(net, X, y, epochs=epochs)
        acc = {**held_out(net, Xte, yte), "warm_start": Path(warm).name if warmed else None,
               "warm_ckpt_top1_before_training": before, "train_s": round(time.perf_counter() - t0, 1)}
        torch.save({"state_dict": net.state_dict(), **acc, "epochs": epochs, "split_seed": seed}, root / f"{name}.pt")
        report["agents"][name] = acc
        print(f"{name:>9}: held-out top1 {acc['top1']:.1%} top3 {acc['top3']:.1%} "
              f"({acc['train_s']} s{', warm' if warmed else ''})", flush=True)
    (root / "population.json").write_text(json.dumps(report, indent=2))
    return report


def arena(rounds: int = 2, epochs: int = 5, warm_epochs: int = 10, pert_steps: int = 60,
          device: str = "cpu", root: Path = POP_DIR) -> dict:
    """Keyguard's multi-adversary min-max (co_train_population) on the harrison MBPWavs; writes
    runs/keyguard/arena/adv-*/arena.json and remembers the worst case. Trained population weights are the starting
    point when present; keynet is re-warmed from supervised_mbp.pt."""
    trained = {r.name: r.net for r in load_population(device, root)}
    pop = [(name, trained.get(name, net)) for name, net in architectures()]
    adversarial.DEVICE = device   # MPS training is flaky here; co_train reads this global
    return adversarial.co_train_population(rounds=rounds, epochs=epochs, pop=pop, warm_epochs=warm_epochs,
                                           pert_steps=pert_steps)


def burst(line: str = "hey meet me at noon my password is hunter2 thanks", root: Path = POP_DIR) -> dict:
    """Each reader's accuracy on the arms-race demo burst (live_bank_rich synth, known onsets): a different keyboard
    than the harrison training presses."""
    from keyguard.agents.arms_race_demo import build_utterance
    y, _, onsets, kstr = build_utterance(line)
    truth = list(kstr.replace(" ", ""))
    out = {"line": line, "n_keys": len(truth), "agents": {}}
    for r in load_population(root=root):
        reads = r.read(y, onsets)
        top1 = float(np.mean([g[0] == t for g, t in zip(reads, truth)]))
        top3 = float(np.mean([t in g for g, t in zip(reads, truth)]))
        out["agents"][r.name] = {"top1": top1, "top3": top3, "read": "".join(g[0] for g in reads)}
        print(f"{r.name:>9}: top1 {top1:.1%} top3 {top3:.1%}  read {out['agents'][r.name]['read']}")
    print(f"{'truth':>9}: {''.join(truth)}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["train", "arena", "burst"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warm-epochs", type=int, default=10)
    ap.add_argument("--pert-steps", type=int, default=60)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.cmd == "train":
        train(epochs=a.epochs or 30)
    elif a.cmd == "arena":
        arena(rounds=a.rounds, epochs=a.epochs or 5, warm_epochs=a.warm_epochs, pert_steps=a.pert_steps)
    else:
        burst()


if __name__ == "__main__":
    main()
