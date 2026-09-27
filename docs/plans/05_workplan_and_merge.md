# 05: Work plan, parallel packages, stop point, merge

## 1. Rules for every package
- Code only against `app/source/types.py` (the contracts). A contract change goes through the integrator (WP8), never
  a silent edit.
- **Own only your files** (the table below). The packages run in parallel in one checkout, so touching someone
  else's files breaks their work.
- **Hearsay and Keyguard are read-only** (`HEARSAY_ROOT`, `KEYGUARD_ROOT`). Import them; never write to them.
- Every package leaves tests that run **without models or audio devices** (mock drivers, synthetic signals).
- Env: `uv` venv at `.venv` (Python 3.12). Run `uv run pytest -q`.

## 2. Packages
| WP | scope | owns |
|---|---|---|
| WP1 core | EventBus, ThreatEngine (plan 02 §4), hooks/sinks, config | `app/source/bus.py`, `threat.py`, `hooks.py`, `config.py`, `tests/test_bus.py`, `tests/test_threat.py`, `tests/test_hooks.py`, `callguard.example.toml` |
| WP2 audio I/O | device listing/selection, mic in → virtual mic out (duplex), WASAPI loopback capture, ring buffers, energy VAD, pynput KeyClock, replay file sources/sinks | `app/source/audio/*`, `tests/test_audio.py`, `docs/zoom_setup.md` |
| WP3 Hearsay driver | `hearsay_real` (r4ft + r5 modes, sha check, prep, deployment threshold), latency bench | `app/hearsay/driver.py`, `tests/test_hearsay_driver.py` (skips if HEARSAY_ROOT is missing) |
| WP4 Keyguard drivers | attacker (KeyNet, weights path or provisional), DSP shield streaming adapter (block-wise with key events), `adversarial` placeholder | `app/keystroke_guard/driver.py`, `tests/test_keyguard_driver.py` |
| WP5 mocks + registry | `mock` drivers with realistic timing and scripted outputs, driver registry/factory from config, quarantine wrapper | `app/source/registry.py`, `app/*/mock.py`, `tests/test_drivers_mock.py` |
| WP6 server + dashboard | FastAPI app, `/ws` push, controls API, static dashboard | `dashboard/*`, `tests/test_server.py` |
| WP7 attack proof | plan 04 experiment + report | `experiments/*`, `docs/reports/attack_under_speech.*` |
| WP8 integrate | `pipeline.py`, `cli.py`, demo scenarios + agent renderer/caller, end-to-end replay test, README, runbook | `app/source/pipeline.py`, `app/source/cli.py`, `demo/*`, `tests/test_e2e_replay.py`, `README.md`, `docs/demo_runbook.md` |

| WP9 secret shield (plan 06) | Vosk grammar spotter, delay-line redactor, `SecretSpotterDriver` contract (additive), pipeline + threat + dashboard wiring, eval + report | `app/secret_shield/spotter.py`, `app/secret_shield/redactor.py`, `app/secret_shield/eval/secret_shield_eval.py`, `docs/reports/secret_shield.md`, `tests/test_secret_shield.py` (+ coordinated edits to types/mock/pipeline/threat/dashboard) |

Order: WP1-WP7 in parallel (WP7 is independent), then WP8 (needs all of them), then an independent review pass.

## 3. Done for this phase
- `uv run pytest -q` passes (mock drivers). The real-driver tests pass on this laptop (HEARSAY_ROOT and KEYGUARD_ROOT
  set).
- `callguard run --mode replay --scenario ai_caller` works end to end with the **real** drivers and shows the
  story on the dashboard.
- `callguard devices` finds VB-CABLE when installed, and prints the setup steps when it's missing.
- The attack-proof report exists, with numbers.
- WP9: the secret shield is in the `ai_caller` demo, and `docs/reports/secret_shield.md` has leak and false-redaction numbers.

## 4. Blocked on upstream (the stop list)
| item | waits for |
|---|---|
| Swap Hearsay model/threshold if the NSA review changes anything | NSA green/red light on the Hearsay submission |
| Keyguard's trained attacker weights (replaces the provisional KeyNet) | teammate |
| Keyguard adversarial shield `D` (streaming form) + its perceptual checks | teammate |
| Keyguard CTC free-typing attacker + Gemini LM lattice in the readout | teammate |
| Final merge (below) | both repos done |

## 5. Merge process (after both upstream repos are done)
1. Freeze upstream versions: record the Hearsay commit + model sha256, and the Keyguard commit + weight hashes, in
   `UPSTREAM.lock`.
2. Pick the packaging: **git submodules** under `third_party/` (default: keeps authorship and history) or
   vendored wheels. Keep the `HEARSAY_ROOT`/`KEYGUARD_ROOT` override for development.
3. Replace provisional pieces (attacker weights, shield) and re-run plan 04 + the e2e replay with the final models.
4. Merge the dashboards: bring Keyguard's arena view into CallGuard as a tab (or link it); no duplicate servers.
5. Joint README/Devpost: one story, credits for both repos, and a licences section.
