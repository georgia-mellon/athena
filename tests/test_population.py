"""Population readers (app/keystroke_guard/population.py): offline, tiny, tmp dirs only."""
import json
import logging

import numpy as np
import pytest

from app.keystroke_guard import driver
from app.keystroke_guard import population as P
from keyguard.config import CLASSES, KEY_WIN


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BACKBOARD_API_KEY", raising=False)


@pytest.fixture
def tiny_harrison(tmp_path, monkeypatch):
    """3 random presses per key (2 train / 1 test under the 60/40 split)."""
    rng = np.random.default_rng(0)
    keys = CLASSES[:driver.N_KEYS]
    path = tmp_path / "harrison.npz"
    np.savez(path, wins=rng.normal(0, 0.05, (3 * len(keys), KEY_WIN)).astype(np.float32),
             labels=np.repeat(keys, 3))
    monkeypatch.setattr(driver, "HARRISON", path)
    return path


def test_load_population_skips_missing_weights(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert P.load_population(root=tmp_path) == []
    assert all(name in caplog.text for name, _ in P.architectures())


def test_train_writes_weights_and_report_and_readers_read(tmp_path, tiny_harrison):
    root = tmp_path / "pop"
    report = P.train(epochs=1, root=root, warm=tmp_path / "none.pt")
    names = [n for n, _ in P.architectures()]
    assert all((root / f"{n}.pt").exists() for n in names)
    saved = json.loads((root / "population.json").read_text())
    assert set(saved["agents"]) == set(names) == set(report["agents"])
    assert all(0 <= a["top1"] <= a["top3"] <= 1 for a in saved["agents"].values())

    readers = P.load_population(root=root)
    assert [r.name for r in readers] == names
    audio = np.random.default_rng(1).normal(0, 0.05, 16000).astype(np.float32)
    onsets = np.array([1000, 6000, 12000])
    for r in readers:
        reads = r.read(audio, onsets)
        assert len(reads) == len(onsets)
        assert all(len(g) == 3 and set(g) <= set(CLASSES[:36]) for g in reads)
    assert readers[0].read(audio, np.array([], int)) == []


def test_reader_masks_space_on_37_way_head(tmp_path):
    import torch
    name, net = P.architectures(n_classes=len(CLASSES))[0]      # 37 = with space
    torch.save(net.state_dict(), tmp_path / f"{name}.pt")
    (r,) = P.load_population(root=tmp_path)
    reads = r.read(np.zeros(8000, np.float32), np.array([2000]))
    assert " " not in reads[0] and len(reads[0]) == 3


@pytest.mark.skipif(not (P.adversarial.DATA / "harrison" / "MBPWavs").exists(),
                    reason="data/keyguard/harrison/MBPWavs missing (get_assets)")
def test_arena_tiny_writes_population_arena(tmp_path, monkeypatch):
    from keyguard import memory
    monkeypatch.setattr(P.adversarial, "RUNS", tmp_path)            # arena dir + no warm checkpoint
    monkeypatch.setattr(P.adversarial, "DEVICE", P.adversarial.DEVICE)   # arena() pins cpu; restore after
    monkeypatch.setattr(memory, "LOCAL", tmp_path / "arena_memory.jsonl")
    out = P.arena(rounds=1, epochs=1, warm_epochs=1, pert_steps=2, root=tmp_path / "nopop")
    (arena_json,) = (tmp_path / "arena").glob("adv-*/arena.json")
    state = json.loads(arena_json.read_text())
    assert state["mode"] == "population" and state["status"] == "done"
    assert set(state["final"]["per_attacker"]) == {n for n, _ in P.architectures()}
    assert 0 <= out["worst_final"] <= 1
    assert (tmp_path / "arena_memory.jsonl").exists()
