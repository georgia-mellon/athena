"""Keyguard's Ares-vs-Athena arms race, run live on the user's own typing bursts.

The pipeline cuts each typing burst (raw mic, before any CallGuard shield) and hands it to an AgentWorker; the worker
runs one ArmsRace match per burst off the audio thread and streams the moves on the bus:

  keyguard.burst      {t_audio, n_keys, shield}                        a burst was queued
  keyguard.move       {t_audio, agent, title, reasoning, action, result, tag}
  keyguard.arms_race  the full match log (keyguard's arms_race_data.js shape + source/t_audio/shield/route/agents/...)

The match is keyguard.agents.arms_race_demo.main's loop, reusing its functions, with one change: Ares' opening read
is a team. The CTC reader (the differentiable one Athena crafts against) plus every trained per-key PopulationReader
(app.keystroke_guard.population) read the keystrokes, and a Gemini language agent (llm.ask agent="ares") fuses their
top-3 lattices into Ares' reading. After Athena's final shield every sub-agent reads the secret again: the per-agent
before/after reads are the honest transfer check (the shield was crafted against the CTC net only).

The live CTC attacker is never touched: every match deep-copies its net.

CLI (one match on keyguard's demo utterance, synthesized from the rich key bank):
    uv run python -m app.keystroke_guard.agents [--line "..." --rounds 2 --device auto]
"""
from __future__ import annotations

import copy
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

CTX_S = 0.5            # audio kept before the first / after the last key of a burst (pipeline side)


@dataclass
class Burst:
    audio: np.ndarray      # raw mic clip, float32 16 kHz, before any CallGuard shield
    onsets: np.ndarray     # sample index into `audio` of each char of `keys` (spaces included)
    keys: str              # what was typed (victim-side truth; never leaves this process except in the match log)
    t_audio: float         # stream time of the first key (s)
    shield: str            # CallGuard shield mode while it was typed


def _device(device: str) -> str:
    """"auto" = Apple GPU (mps) when present, else cpu. ponytail: owner call, no cpu-vs-mps benchmark."""
    if device != "auto":
        return device
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _top1(lattice: list[list[str]]) -> str:
    return "".join(r[0] for r in lattice if r)


class ArmsRace:
    """One Ares-vs-Athena match per burst (see module doc). `emit(topic, **data)` is EventBus.emit."""

    def __init__(self, net, emit: Callable[..., None], rounds: int = 2, snr_db: float = 16.0, floor: float = 0.85,
                 steps: int = 250, adapt_steps: int = 120, device: str = "auto", population: list | None = None):
        self.net, self.emit = net, emit
        self.rounds, self.snr_db, self.floor = rounds, snr_db, floor
        self.steps, self.adapt_steps, self.device = steps, adapt_steps, _device(device)
        self._population = population       # None = load lazily on the first match (worker thread, not startup)
        self.population_note = ""

    # --- the team ---------------------------------------------------------------------------------------------
    def population(self) -> list:
        if self._population is None:
            try:
                from app.keystroke_guard.population import load_population
                self._population = list(load_population(device="cpu"))
                self.population_note = f"{len(self._population)} population readers"
            except Exception as e:  # noqa: BLE001 - no module / weights: Ares plays with the CTC reader only
                log.warning("keyguard population unavailable (%s); Ares reads with the CTC reader only", e)
                self._population = []
                self.population_note = f"population unavailable: {type(e).__name__}: {e}"
            if not self._population and not self.population_note:
                self.population_note = "no trained population readers"
        return self._population

    def _ctc_lattice(self, net, y: np.ndarray, onsets: np.ndarray) -> list[list[str]]:
        """Per-onset top-3 keys from the CTC net (non-blank, non-space logits averaged over +/-1 frame)."""
        import torch
        from keyguard.agents.defense_audio import torch_logmel
        from keyguard.ctc.data import VOCAB
        from keyguard.ctc.model import HOP
        dev = next(net.parameters()).device
        with torch.no_grad():
            lg, _ = net(torch_logmel(torch.tensor(y, dtype=torch.float32, device=dev))[None])
        lg = lg[0].cpu().numpy()
        out = []
        for o in onsets:
            f = min(int(o) // HOP, len(lg) - 1)
            z = lg[max(0, f - 1):f + 2, 1:-1].mean(0)              # drop blank (0) and space (last)
            out.append([VOCAB[1 + j] for j in np.argsort(-z)[:3]])
        return out

    def _read_team(self, net, y, onsets) -> dict[str, list[list[str]]]:
        team = {"ctc": self._ctc_lattice(net, y, onsets)}
        for r in self.population():
            try:
                team[r.name] = [list(row) for row in r.read(y, onsets)]
            except Exception as e:  # noqa: BLE001 - one broken reader must not end the match
                log.warning("population reader %s failed: %s", getattr(r, "name", r), e)
        return team

    # --- the match --------------------------------------------------------------------------------------------
    def run(self, burst: Burst) -> dict:
        from keyguard import memory
        from keyguard.agents import arms_race_demo as A, llm, smart_dict as SD
        from keyguard.agents import defense_audio as D
        from keyguard.ctc.data import SYM_OF_KEY

        t0 = time.monotonic()
        # keyguard's convention: space is a keystroke too (its own onset + CTC symbol); the per-key readers have no
        # space class, so they read the non-space onsets only
        keep = [(c, int(o)) for c, o in zip(burst.keys.upper(), burst.onsets) if c in SYM_OF_KEY]
        text = "".join(c for c, _ in keep)
        on_all = np.array([o for _, o in keep], dtype=int)
        key_ids = np.array([SYM_OF_KEY[c] for c in text], dtype=int)
        on = np.array([o for c, o in keep if c != " "], dtype=int)
        y = np.asarray(burst.audio, np.float32)
        net = copy.deepcopy(self.net).to(self.device).eval()     # never train the live attacker
        moves: list[dict] = []

        def mv(agent, title, reasoning="", action="", result="", tag=""):
            m = {"agent": agent, "title": title, "reasoning": reasoning, "action": action, "result": result,
                 "tag": tag}
            moves.append(m)
            self.emit("keyguard.move", t_audio=burst.t_audio, **m)

        base = {"line": text, "attacker": "Ares", "defender": "Athena", "backend": llm.backend(),
                "source": "callguard", "t_audio": burst.t_audio, "shield": burst.shield,
                "moves": moves}

        def finish(log_: dict) -> dict:
            log_ = {**base, **log_, "route": llm.last_route, "population": self.population_note or None,
                    "seconds": round(time.monotonic() - t0, 2)}   # note is set by the lazy load in _read_team
            self.emit("keyguard.arms_race", **log_)
            return log_

        # ---- ROUND 0: Ares' team reads the burst ----
        raw0 = D.decode(net, y)
        team = self._read_team(net, y, on)
        for name, lat in team.items():
            mv(f"⚔️ ARES·{name}", "Acoustic read (no defense)",
               action=("CTC transcript + per-key top-3 at each onset" if name == "ctc"
                       else "Per-key classifier top-3 at each onset"),
               result=(f"`{raw0}` · top-1 `{_top1(lat)}`" if name == "ctc" else f"top-1 `{_top1(lat)}`"),
               tag="👂 listening")
        lattice = "\n".join(f"pos {i + 1}: " + " | ".join(f"{n}: {' '.join(team[n][i])}" for n in team
                                                           if i < len(team[n]))
                            for i in range(len(on)))
        reply = (llm.ask(A.ATK_SYS, f'Noisy transcription: "{raw0}"\nPer-keystroke top-3 guesses from '
                                    f'{len(team)} acoustic readers (best first):\n{lattice}\nMost likely typed text:',
                         agent="ares") or "").strip().strip('"').strip()
        llm0 = reply.splitlines()[0][:90] if len(reply) >= 6 else raw0
        mv("⚔️ ARES", "Opening read (no defense)",
           reasoning=f"Language agent fuses {len(team)} readers' lattices ({', '.join(team)}).",
           action="Transcribe the keystroke audio, then reconstruct with the LLM.",
           result=f'acoustic: `{raw0}`  →  LLM: “{llm0.strip()[:80]}”', tag="🔓 secret region exposed")

        # ---- ATHENA: triage + deception plan ----
        plan = A.defender_plan(text)
        secret, decoy = plan["sensitive"], plan["decoy"]
        if secret and secret not in text:                     # the LLM normalized it: fall back to the rule
            r = llm.rule_triage(text)
            secret, decoy = r["sensitive"], llm.rule_decoy(r["sensitive"])
        if not secret or secret not in text:
            mv("🦉 ATHENA", "Triage (LLM)", reasoning=plan.get("reason", ""),
               result="Nothing sensitive in this burst; no shield needed.", tag="✅ nothing to protect")
            return finish({"secret": "", "decoy": "", "kind": "none", "rounds": [], "agents": {},
                           "protected": None, "final_read": raw0, "clean_span_read": "",
                           "smart_dict_clean": None, "smart_dict_defended": None})
        mv("🦉 ATHENA", "Triage + deception plan (LLM)",
           reasoning=plan.get("reason", "") or f"{secret} looks like a {plan['kind']}.",
           action=f"Mark `{secret}` sensitive; fabricate a coherent decoy `{decoy}`.",
           result=f"Plan: steer the attacker to read `{decoy}` instead of `{secret}`.", tag="🎭 lie prepared")

        lo, hi = A.span_bounds(text, on_all, secret)
        hi = min(hi, len(y))
        i0 = text.find(secret)
        sec_on = on_all[i0:i0 + len(secret)]               # the secret has no spaces: contiguous in on_all
        clean_span = D.span_read(net, y, lo, hi)
        sd0 = SD.summarize(net, y, lo, hi, secret, decoy)
        rk, top5 = sd0["secret_rank"], ", ".join(c["guess"] for c in sd0["top5"])
        mv("⚔️ ARES", "Smart-dictionary attack",
           action=f"Rank candidate secrets from the per-key acoustics over the span "
                  f"(search space {A._num(sd0['search_space'])}).",
           result=(f"true secret `{secret}` is Ares' guess **#{rk}** of {sd0['shortlist']} (top: {top5})" if rk
                   else f"true secret not in Ares' top {sd0['shortlist']} (top: {top5})"),
           tag=(f"🔓 password in top {rk}" if rk and rk <= 10 else "🔓 shortlisted"))

        # ---- THE ARMS RACE (arms_race_demo.main's loop) ----
        snr, rounds, y_def, info = self.snr_db, [], y, {}
        for r in range(self.rounds):
            y_def, info = D.craft(net, y, lo, hi, mode="deceive", target=decoy, snr_db=snr, steps=self.steps)
            span = D.span_read(net, y_def, lo, hi)
            reads_true = A._match(span, secret)
            rounds.append({"round": r, "mode": "deceive", "span_read": span, "reads_true_secret": reads_true,
                           "stoi": info["stoi"], "snr_db": info["snr_db"], "decoy": decoy})
            mv("🦉 ATHENA", f"Deploy shield (round {r})",
               action=f"Optimize an inaudible perturbation over the secret's span to steer it toward `{decoy}` "
                      f"(budget {snr:.0f} dB, ‖δ‖≤ε).",
               result=f"attacker now reads `{A._clean(span)}` · STOI **{info['stoi']:.3f}** (speech intact)",
               tag="🎭 attacker fooled" if not reads_true else "⚠️ leak")
            if r == self.rounds - 1:
                break
            A.adapt_attacker(net, y_def, key_ids, on_all, steps=self.adapt_steps)
            broke = D.span_read(net, y_def, lo, hi)
            broke_true = A._match(broke, secret)
            rounds[-1].update(after_retrain_read=broke, after_retrain_reads_true=broke_true)
            mv("⚔️ ARES", f"Evolve — retrain on the shielded audio (round {r})",
               action="Fine-tune on the defended clip to learn through this exact shield.",
               result=f"now reads `{A._clean(broke)}`", tag="🔓 broke through!" if broke_true else "🎭 still fooled")
            if broke_true and info["stoi"] > self.floor:
                snr = max(6.0, snr - 3.0)
                nd = llm.ask(A.DEF_SYS, agent="athena",
                             prompt=f'The attacker broke through and read "{broke}". Give ONE new same-length '
                                    f'same-shape decoy for "{secret}". Reply with only the decoy token.')
                nd = "".join(c for c in nd.strip().upper() if c in SYM_OF_KEY and c != " ")
                prev, decoy = decoy, (nd if len(nd) == len(secret) else decoy)
                mv("🦉 ATHENA", "Escalate (LLM)",
                   reasoning="Attacker adapted to the last shield; spend more budget and switch the lie so the "
                             "stale one can't be trusted.",
                   action=f"Lower SNR budget to {snr:.0f} dB" + (f"; new decoy `{prev}`→`{decoy}`"
                                                                 if decoy != prev else ""),
                   result="re-optimize next round against the adapted attacker.", tag="🔁 counter-move")

        # ---- verdict ----
        final_read = D.decode(net, y_def)
        final_span = D.span_read(net, y_def, lo, hi)
        sdD = SD.summarize(net, y_def, lo, hi, secret, decoy)
        rkD, rkDecoy = sdD["secret_rank"], sdD["decoy_rank"]
        mv("⚔️ ARES", "Smart-dictionary attack (under shield)",
           action="Re-rank candidate secrets from the defended audio.",
           result=(f"true secret `{secret}` " + (f"crashed to **#{rkD}**" if rkD else
                                                 f"**fell out of the top {sdD['shortlist']}**")
                   + (f"; the decoy `{decoy}` is now #{rkDecoy}" if rkDecoy else "") + f". (was #{rk} before)"),
           tag="🛡️ guess list poisoned")

        # every sub-agent re-reads the secret under Athena's final shield (the transfer check)
        agents = {"ctc": {"before": clean_span, "after": final_span}}
        after = self._read_team(net, y_def, sec_on)
        before = self._read_team(net, y, sec_on) if len(after) > 1 else {}
        for name in after:
            if name != "ctc":
                agents[name] = {"before": _top1(before.get(name, [])), "after": _top1(after[name])}
        leaked = [n for n, a in agents.items() if A._match(a["after"], secret)]
        protected = not leaked
        for name, a in agents.items():
            if name != "ctc":
                mv(f"⚔️ ARES·{name}", "Re-read under Athena's shield",
                   action="Read the secret's keystrokes again (shield crafted against the CTC reader).",
                   result=f"`{a['before']}` → `{a['after']}`",
                   tag="🔓 still reads it" if name in leaked else "🛡️ fooled too")
        mv("🏁 OUTCOME", "Final state",
           result=f"attacker's last read of the secret span: `{A._clean(final_span)}` (aiming for decoy `{decoy}`, "
                  f"true secret `{secret}`); speech STOI **{rounds[-1]['stoi']:.3f}**"
                  + (f"; leaked to {', '.join(leaked)}" if leaked else "") + ".",
           tag="🛡️ secret protected" if protected else "⚠️ leaked")

        rs = rounds
        memory.remember(
            f"Arms race ({plan['kind']}, {len(rs)} rounds, live CallGuard burst): Athena planted decoy {decoy!r}; "
            f"Ares broke through {sum(bool(r.get('after_retrain_reads_true')) for r in rs)} time(s); final read "
            f"{A._clean(final_span)!r}, secret {'protected' if protected else 'LEAKED'} at STOI "
            f"{rs[-1]['stoi']:.3f}, budget {rs[-1]['snr_db']:.1f} dB; team {', '.join(agents)}.",
            metadata={"leaked": not protected, "rounds": len(rs), "decoy": decoy, "stoi": float(rs[-1]["stoi"]),
                      "source": "callguard", "agents": list(agents)}, kind="arms_race")
        return finish({"secret": secret, "decoy": decoy, "kind": plan["kind"], "clean_span_read": clean_span,
                       "smart_dict_clean": sd0, "smart_dict_defended": sdD, "rounds": rounds,
                       "final_read": final_read, "agents": agents, "protected": protected})


class AgentWorker:
    """Runs ArmsRace matches on one daemon thread. Newest burst wins (queue of 1); never blocks the caller."""

    def __init__(self, race: ArmsRace, bus):
        self.race, self.bus = race, bus
        self.latest: dict = {}
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, burst: Burst) -> None:
        self.bus.emit("keyguard.burst", t_audio=burst.t_audio, n_keys=len(burst.keys), shield=burst.shield)
        with self._lock:
            try:
                self._q.get_nowait()                    # drop the older burst still waiting
            except queue.Empty:
                pass
            self._q.put_nowait(burst)
            if self._thread is None or not self._thread.is_alive():   # (re)started lazily after stop()
                self._stop.clear()
                self._thread = threading.Thread(target=self._loop, name="callguard-keyguard-agents", daemon=True)
                self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                burst = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.latest = self.race.run(burst)
            except Exception as e:  # noqa: BLE001 - a failed match is reported, the next burst still runs
                log.exception("keyguard arms race failed")
                self.bus.emit("driver.error", driver="keyguard agents", kind="agents",
                              error=f"{type(e).__name__}: {e}", quarantined=False)

    def stop(self) -> None:
        """Stop after the current match (it can't be interrupted); a later submit restarts the thread."""
        self._stop.set()
        with self._lock:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass


# --- CLI ------------------------------------------------------------------------------------------------------
def demo_burst(line: str) -> Burst:
    """keyguard's demo utterance as a Burst: synthesized from the rich key bank (one onset per key, spaces included)."""
    from keyguard.agents import arms_race_demo as A
    y, _, onsets, kstr = A.build_utterance(line)
    return Burst(y, np.asarray(onsets, dtype=int), kstr, 0.0, "off")


def load_net(device: str = "cpu"):
    import torch
    from app.keystroke_guard.driver import CTC_WEIGHTS
    from keyguard.ctc.data import VOCAB
    from keyguard.ctc.train_overlap import MtlCRNN
    net = MtlCRNN(n_sym=len(VOCAB))
    state = torch.load(CTC_WEIGHTS, map_location="cpu")
    net.load_state_dict(state.get("state_dict", state))
    return net.to(device).eval()


def main(argv: list[str] | None = None) -> dict:
    import argparse
    ap = argparse.ArgumentParser(description="One Ares-vs-Athena match on keyguard's demo utterance.")
    ap.add_argument("--line", default="hey meet me at noon my password is hunter2 thanks")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=80, help="craft steps per round (keyguard's offline demo: 250)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    def emit(topic: str, **d: Any) -> None:
        if topic == "keyguard.move":
            print(f"{d['agent']:<22} {d['title']}\n{'':<22} {d['result']}  {d['tag']}")

    race = ArmsRace(load_net(), emit, rounds=a.rounds, steps=a.steps, device=a.device)
    out = race.run(demo_burst(a.line))
    keys = ("secret", "decoy", "kind", "protected", "agents", "route", "backend", "seconds", "population")
    print(json.dumps({k: out.get(k) for k in keys}
                     | {"rounds": [{k: r[k] for k in ("round", "span_read", "reads_true_secret", "stoi")}
                                   for r in out["rounds"]]}, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
