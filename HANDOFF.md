# Handoff: CallGuard phase 1 (paused 2026-09-26 ~14:20 EDT)

Paused by the owner so that a dedicated Claude session can continue CallGuard here, while the original session works
only on Hearsay. **Start by reading `CLAUDE.md`, then `plans/00_brief.md`** (owner intent), then plans 01-05.

## Where things stand
Phase 1 ran as a workflow (script saved in `plans/phase1_workflow.js`) with 7 parallel work packages (plan 05 §2),
then integration (WP8), then review. It was stopped during the build phase. **`pytest`: 52 passed** (mock drivers,
no devices), at the commit that added this file.

| WP | status | notes from its report |
|---|---|---|
| WP1 core (bus, threat, hooks, config) | **done**, 19 tests | synthetic voice alone → 64 (WARN) after ~6 s; synthetic voice + readable typing, shield off → 97 (CRITICAL); shield on → 64. Payload contract is in the `threat.py` docstring. "Reset V on new speaker" not built. |
| WP2 audio I/O | **done**, 7 tests | live check on this laptop: mic opens at 16 kHz WASAPI, duplex to speakers 0 xruns, loopback of the Realtek speakers works. **VB-CABLE not installed** (routing check reports MISSING). |
| WP4 Keyguard drivers | **done**, 4 tests | provisional KeyNet (harrison held-out n=360): **top-1 70.8 %, top-3 95.3 %** (chance 2.8 %), cached at `runs/provisional_keynet.pt`; streaming DSP shield adds 80 ms; Keyguard commit b0349c0. |
| WP5 mocks + registry + quarantine | **done**, 11 tests | |
| WP6 server + dashboard | **done**, 6 tests | FastAPI `create_app(bus, state_provider, controls)`, `/ws`, static dashboard with no CDNs. |
| WP3 Hearsay driver | **partial**: `callguard/drivers/hearsay_real.py` written, no report | calibration cached in `runs/hearsay_calibration.json`: r4ft thr −2.82, s 6.27; r5 thr 0.61, s 3.89 (from Hearsay `val_testlike`). Verify it, then write `tests/test_hearsay_driver.py`. |
| WP7 attack proof | **partial**: `experiments/attack_under_speech.py` written, not run to a report | finish, run, and write `reports/attack_under_speech.md` (plan 04). |
| WP8 integrate | **not started** | `pipeline.py`, `cli.py`, demo scenario, e2e replay, README, runbook. The prompt is in `plans/phase1_workflow.js`. |
| Review | not started | |

## Contract mismatches WP8 must reconcile (reported by the packages)
1. **Key-event sample clock.** WP5's mocks assume `ShieldDriver.process(block, key_events)` gets offsets *within the
   block*. WP4's real shield wants **absolute sample indices since `reset()`**. WP2's `MicShieldStream` calls
   `hook(block, start_sample)`. Pick one convention (absolute, since WP4 and WP2 already use it), and adapt the mocks
   and `types.py` docs.
2. **Attacker truth.** The Protocol has no `truth` argument, but `MockAttacker.read(..., truths=)` takes one. The
   pipeline should attach truth to `KeyGuess.truth` after the call, not pass it in.
3. **Error callback.** Quarantine's `on_error` receives an `Event`; the bus is `publish(Event)`. Wire it as
   `on_error=bus.publish`.
4. The **controls** object for the server needs `set_shield(mode)` and `scenario(action, name)`.
5. The **KeyClock acoustic offset** (OS timestamp → mic sample) is set from the stream's reported latency. Calibrate
   it with real typing (`offset_s`).
6. **Shield CPU:** 10-15 ms per key-touched 20 ms block. Watch the budget when Hearsay runs on the same CPU (set
   Hearsay threads to 2-4).

## Environment
- `uv sync` → `.venv` (Python 3.12, torch 2.14 CPU, transformers 5.17: the same versions as Hearsay, so its
  checkpoint loads).
- Upstream (read-only): `HEARSAY_ROOT=../Hearsay`, `KEYGUARD_ROOT=../keyboard-acoustic-shield` (a clone of
  `LordKarV/keyboard-acoustic-shield`; `git pull` it for the teammate's latest).
- `runs/` (gitignored) holds the provisional attacker and the Hearsay calibration cache.

## Hearsay status the CallGuard session should know
- The organizers' scoring is now confirmed as **Pspoof = 0.3, Cfa = 4** (moderator message, 2026-09-26). Hearsay's
  numbers are being recomputed by the Hearsay session. If the Hearsay deployment threshold changes, re-run the
  calibration (delete `runs/hearsay_calibration.json`).
- Hearsay stays frozen until the NSA review answers. CallGuard only reads it.

## Owner to-dos
- Install VB-CABLE (vb-audio.com/Cable, admin, reboot) for live Zoom mode.
