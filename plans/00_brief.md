# 00: Owner brief (the source of truth for intent)

Recorded 2026-09-26 from the owner's request, so no context is lost. When a later plan conflicts with this file,
this file wins unless the owner says otherwise.

## What the owner asked for
- **New private repo, `callguard`**: our **primary HackGT 13 submission**. It must be **demoable and presentable** at
  the expo, **with a dashboard**, and **hooked into a real meeting platform** (Zoom, Microsoft Teams or Google Meet;
  pick the best one).
- CallGuard **houses two models** behind proper **hooks and drivers**:
  - **Hearsay** (ours, `danmano411/hearsay`): real vs. synthetic voice. Our NSA HEARSAY challenge submission.
  - **Keyguard** (teammate's, `LordKarV/keyboard-acoustic-shield`): an acoustic keystroke **attacker** (reads keys
    from call audio) and a **defender/shield** (hides keystrokes, keeps speech intact).
- **Demo:** during a call where **an AI agent is speaking**, run **both models at the same time** and produce a
  **threat score**.
- **Prove the reverse direction.** Hearsay was shown to work with keystrokes as background noise
  (`hearsay/reports/generalization.md` §2). Now show that the **attacker still picks up keystrokes with speech in
  the background**. That makes keystroke leakage a real threat on a call, so the **defender** has something real to
  defend against, and we show that it does.
- **Plans in `.md` first**, then build with **distributed workflows and parallel, task-scoped subagents**, and **stop
  when further work genuinely needs the other two repos to be finished**.
- After both repos are done: **merge process** (plan 05).

## Hard constraints
1. **Hearsay repo is frozen.** Don't edit it. We are waiting for the NSA contact's green or red light on the
   submission (minDCF was reported as 1.0, most likely a score-direction mismatch; see hearsay `docs/scoring.md`).
   CallGuard *reads* Hearsay's code and frozen model files from `HEARSAY_ROOT`; it never writes there.
2. **Keyguard is the teammate's repo.** Read-only: nothing is pushed there. CallGuard reads it from `KEYGUARD_ROOT`.
   Its final attacker and shield weights are still being developed.
3. The repo is **private**. No secrets, audio, weights or other people's recordings get committed.
4. **Ethics** (carried over from Keyguard): only our own devices, consenting teammates and judges, and **fake
   passwords** in every demo. The AI "caller" clones only a consenting teammate's voice.

## Stop criterion for this phase
Stop when everything that does not depend on unfinished upstream work is built, tested and demoable with the current
upstream code and mocks. The remaining items, listed in `plans/05_workplan_and_merge.md` §4, should be only:
- Hearsay model changes, which wait for the NSA verdict;
- Keyguard's final attacker/shield weights and APIs, which wait for the teammate.
