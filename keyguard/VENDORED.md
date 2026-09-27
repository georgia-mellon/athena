# Vendored Keyguard

Source: the teammate's Keyguard repo, [LordKarV/keyboard-acoustic-shield](https://github.com/LordKarV/keyboard-acoustic-shield)
(local checkout `../../keyboard`), commit `55bb112` ("feat: Ares and Athena reason and remember through Backboard").
Author: LordKarV (Keyguard team). Copied: the whole `keyguard/` package (code + `web/` assets), no `__pycache__`.
CallGuard imports it as the top-level `keyguard` package; nothing references the teammate's checkout.

Weights and data are not committed. `uv run python -m app.keystroke_guard.get_assets` copies them from a Keyguard
checkout (`KEYGUARD_ROOT`, else `upstream/keyguard`, `../../keyboard`, `../keyboard-acoustic-shield`) into
`runs/keyguard/` and `data/keyguard/` (both gitignored).

## Local edits (all marked `# CallGuard:`)
- `__init__.py`: added (upstream is a namespace package) so hatch packages it.
- `config.py`: `ROOT` = the callguard repo (so `.env` is callguard's), `DATA` = `data/keyguard`, `RUNS` = `runs/keyguard`.
- `agents/arms_race_demo.py`: no forced `KEYGUARD_DEVICE=cuda` at import; `CKPT`/`BANK` default under `RUNS`/`DATA`;
  `adapt_attacker` trains on the net's own device; `write_replay`/`write_web` write into `RUNS`, not the repo/package.
- `agents/backboard_agent.py`: loads `.env` via `keyguard.config`; Backboard agents default to
  `BACKBOARD_PROVIDER=google`, `BACKBOARD_MODEL=gemini-2.5-flash` (Gemini reasoning, Backboard memory).
- `agents/llm.py`: `last_route` records which path answered the last `ask()` (`backboard:google/gemini-2.5-flash`,
  `gemini-direct:<model>`, or `none`).
- `web/index.html`: `/api/...` and `/static/...` made relative so the UI works mounted under a prefix.

Not changed: some of Keyguard's own training/eval CLIs (`agents/pipeline.py`, `finetune_live.py`, `defense_demo.py`,
`shield/separator.py`, `smart_dict.demo`, `shield/adversarial.harrison_windows`) still default to paths relative to
the working directory (`runs/...`, `data/...`) as upstream; pass explicit paths when running them from CallGuard.
