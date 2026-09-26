# CallGuard: notes for Claude

Primary HackGT 13 submission (team GeorgiaMellon). It fuses **Hearsay** (real vs. synthetic voice) and **Keyguard**
(acoustic keystroke attacker + shield) into one live call-security layer with a dashboard. Read `docs/plans/00_brief.md`
first: it records the owner's intent and wins every conflict. Then read plans 01-06 (plan 06 = the third pillar, the spoken-secret shield).

## Hard rules
- **Never write to the upstream repos.** Hearsay (`HEARSAY_ROOT`, default `../Hearsay`) is frozen until the NSA review
  answers. Keyguard (`KEYGUARD_ROOT`) is the teammate's. Import them read-only.
- Contracts live in `app/source/types.py`; change them only deliberately, and update every driver and test.
- Commit no audio, weights, recordings, `.env`, or webhook URLs. The repo is private but stays clean.
- Demos use fake passwords and consenting voices only.
- The audio path must never block or drop because of a model: drivers run off the audio thread; failures quarantine
  the driver and pass audio through.

## Conventions
- Python 3.12, `uv` (`uv sync`, `uv run pytest -q`). Tests run without models or audio devices (mock drivers).
- Branch `wp<N>-<slug>` + PR; plans before new phases; commits end with a `Co-Authored-By: Claude ...` line.
