"""The arena: BLUE evolves the shield, RED always retrains against it.

Every generation the proposer (LLM or mutation) suggests shield configs. Each is
scored by the harness with an *adaptive* attacker (RED retrains on the shielded
audio), plus speech quality and a timing-leak check. Fitness rewards low attack
accuracy while holding speech quality above a floor. Winners seed the next
generation; every attempt is logged to a lineage tree the dashboard replays.

The harness is the sole judge -- the headline is whatever survives an attacker
that has adapted to the final shield, not a static one.
"""
from __future__ import annotations
import json
import time
import dataclasses
from pathlib import Path

from ..config import RUNS
from ..shield.shield import Shield, ShieldConfig
from ..eval import harness
from ..attackers.timing import detects_timing
from .. import audio, segment
from .proposer import get_proposer, BOUNDS

SPEECH_FLOOR = 0.80          # STOI must stay above this or the config is penalized
PENALTY = 1.0


def _fitness(attack_acc: float, stoi: float | None) -> float:
    """Blue wants low attack_acc. Below the speech floor gets penalized hard."""
    base = 1.0 - attack_acc
    if stoi is not None and stoi < SPEECH_FLOOR:
        base -= PENALTY * (SPEECH_FLOOR - stoi) * 5
    return base


def _timing_leak(cfg: ShieldConfig) -> float:
    """How much the shield still exposes typing rhythm (0=safe, 1=leaky)."""
    y = audio.load("data/harrison/MBPWavs/A.wav")
    clean_on = segment.onsets_n(y, 25)
    shielded = Shield(cfg).apply(y, onsets=clean_on)
    shielded_on = segment.onsets(shielded)
    return float(detects_timing(clean_on, shielded_on))


def _score(cfg_dict: dict, epochs: int) -> dict:
    cfg = ShieldConfig(**cfg_dict)
    sh = Shield(cfg)
    m = harness.evaluate(protect=sh.apply, adaptive=True, epochs=epochs)
    q = harness.speech_quality(sh.apply)
    stoi = q.get("stoi")
    return {
        "config": cfg_dict,
        "attack_acc": m["attack_acc"],
        "mi_bits": m["mi_bits"],
        "stoi": stoi,
        "pesq": q.get("pesq"),
        "timing_leak": _timing_leak(cfg),
        "fitness": _fitness(m["attack_acc"], stoi),
    }


def run(generations=3, pop=3, epochs=60, proposer="auto", seed=0, out=None):
    prop = get_proposer(proposer, seed)
    prop_name = type(prop).__name__
    out = Path(out) if out else RUNS / "arena" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    nodes, nid = [], 0

    def log(node):
        nonlocal nid
        node["id"] = nid
        nid += 1
        nodes.append(node)
        (out / "lineage.json").write_text(json.dumps({
            "created": time.time(), "proposer": prop_name,
            "config_space": BOUNDS, "generations": generations, "nodes": nodes,
        }, indent=2))
        print(f"  node {node['id']} gen{node['gen']} "
              f"acc={node['attack_acc']:.1%} stoi={node['stoi']} "
              f"leak={node['timing_leak']:.2f} fit={node['fitness']:.3f}"
              + ("  <- best" if node.get("is_best") else ""))

    # gen 0: default shield baseline
    print(f"[arena] proposer={prop_name} gens={generations} pop={pop}")
    base = _score(dataclasses.asdict(ShieldConfig()), epochs)
    base.update(gen=0, parent=None, survived=True, is_best=True)
    log(base)
    best = base

    for g in range(1, generations + 1):
        cands = prop.propose(best["config"], nodes, pop)
        gen_best = None
        for c in cands:
            s = _score(c, epochs)
            s.update(gen=g, parent=best["id"], survived=False, is_best=False)
            if gen_best is None or s["fitness"] > gen_best["fitness"]:
                gen_best = s
            log(s)
        if gen_best["fitness"] > best["fitness"]:
            gen_best["survived"] = True
            gen_best["is_best"] = True
            best = gen_best
            for n in nodes:
                if n["id"] == gen_best["id"]:
                    n["survived"] = True
                    n["is_best"] = True
        print(f"[arena] gen{g} best fitness so far {best['fitness']:.3f} "
              f"(attack {best['attack_acc']:.1%})")

    summary = {
        "best": best,
        "baseline": base,
        "drop": base["attack_acc"] - best["attack_acc"],
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[arena] DONE. attack {base['attack_acc']:.1%} -> {best['attack_acc']:.1%} "
          f"(speech STOI {best['stoi']}). lineage: {out}")
    return summary


if __name__ == "__main__":
    import sys
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run(generations=g, pop=2, epochs=40)
