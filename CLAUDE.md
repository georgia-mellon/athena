# CallGuard: notes for Claude

Primary HackGT 13 submission (team GeorgiaMellon). A desktop app that protects a Google Meet call with three
pillars: **Hearsay** (real vs. synthetic voice), **Keystroke Guard** (Keyguard's acoustic keystroke attacker + shield)
and **Secret Shield** (redacts codes you read aloud to an unverified caller), feeding one threat score and a dashboard.

## Reading order
1. `docs/plans/00_brief.md`: the owner's intent; it wins every conflict.
2. `HANDOFF.md`: current status and owner to-dos (latest section at the bottom).
3. `docs/plans/01-06` (06 = the Secret Shield), then `README.md`.
4. Before touching a pillar: its `app/<pillar>/README.md` (contract, placeholder vs real, harness).

## Layout
- `app/source/`: runtime (types.py contracts, pipeline, threat, bus, hooks, config, registry, cli, desktop.py,
  harness.py, audio/, connectors/meet/ = Meet bridge + launcher + /meet router + test room).
- `app/hearsay/`, `app/keystroke_guard/`, `app/secret_shield/`: driver around the real model, mock, harness
  (`python -m app.<pillar>.harness`), eval, README.
- `dashboard/` (FastAPI + static UI), `demo/` (replay scenario `ai_caller`), `docs/` (plans, reports,
  experiments.md, demo_runbook.md, meeting_setup.md), `tests/`.
- Run: `uv run callguard app` (desktop + Meet), `callguard run --mode meet|replay|live`, `/meet/testroom`.

## Hard rules
- **Never write to the upstream repos.** Hearsay (`HEARSAY_ROOT`, default `../Hearsay`) and Keyguard (`KEYGUARD_ROOT`,
  the teammate's) are imported read-only. The Hearsay freeze is lifted (the NSA review confirmed the scorer: Pspoof 0.3,
  Cfa 4, higher score = real); CallGuard uses Hearsay's final model E5 (R4ft + R6 + R1 fusion).
- Contracts live in `app/source/types.py`; change them only deliberately, and update every driver and test.
- Commit no audio, weights, recordings, `.env`, or webhook URLs. The repo is private but stays clean.
- Demos use fake passwords and consenting voices only.
- The audio path must never block or drop because of a model: drivers run off the audio thread; failures quarantine
  the driver and pass audio through.

## Conventions
- Python 3.12, `uv` (`uv sync`, `uv run pytest -q`). Tests run without models or audio devices (mock drivers).
- Branch `wp<N>-<slug>` + PR; plans before new phases. No attribution trailers (no Co-Authored-By or
  "Generated with" lines) in commits or PRs: owner rule.
