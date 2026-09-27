# Handoff: CallGuard phase 1 (paused 2026-09-26 ~14:20 EDT)

Paused by the owner so that a dedicated Claude session can continue CallGuard here, while the original session works
only on Hearsay. **Start by reading `CLAUDE.md`, then `docs/plans/00_brief.md`** (owner intent), then plans 01-05.

## Where things stand
Phase 1 ran as a workflow (script saved in `docs/plans/phase1_workflow.js`) with 7 parallel work packages (plan 05 §2),
then integration (WP8), then review. It was stopped during the build phase. **`pytest`: 52 passed** (mock drivers,
no devices), at the commit that added this file.

| WP | status | notes from its report |
|---|---|---|
| WP1 core (bus, threat, hooks, config) | **done**, 19 tests | synthetic voice alone → 64 (WARN) after ~6 s; synthetic voice + readable typing, shield off → 97 (CRITICAL); shield on → 64. Payload contract is in the `threat.py` docstring. "Reset V on new speaker" not built. |
| WP2 audio I/O | **done**, 7 tests | live check on this laptop: mic opens at 16 kHz WASAPI, duplex to speakers 0 xruns, loopback of the Realtek speakers works. **VB-CABLE not installed** (routing check reports MISSING). |
| WP4 Keyguard drivers | **done**, 4 tests | provisional KeyNet (harrison held-out n=360): **top-1 70.8 %, top-3 95.3 %** (chance 2.8 %), cached at `runs/provisional_keynet.pt`; streaming DSP shield adds 80 ms; Keyguard commit b0349c0. |
| WP5 mocks + registry + quarantine | **done**, 11 tests | |
| WP6 server + dashboard | **done**, 6 tests | FastAPI `create_app(bus, state_provider, controls)`, `/ws`, static dashboard with no CDNs. |
| WP3 Hearsay driver | **partial**: `app/hearsay/driver.py` written, no report | calibration cached in `runs/hearsay_calibration.json`: r4ft thr −2.82, s 6.27; r5 thr 0.61, s 3.89 (from Hearsay `val_testlike`). Verify it, then write `tests/test_hearsay_driver.py`. |
| WP7 attack proof | **partial**: `app/keystroke_guard/eval/attack_under_speech.py` written, not run to a report | finish, run, and write `docs/reports/attack_under_speech.md` (plan 04). |
| WP8 integrate | **not started** | `pipeline.py`, `cli.py`, demo scenario, e2e replay, README, runbook. The prompt is in `docs/plans/phase1_workflow.js`. |
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
- (Superseded 2026-09-26: the freeze is lifted; see the latest update below.) CallGuard only reads Hearsay.

## Owner to-dos
- Install VB-CABLE (vb-audio.com/Cable, admin, reboot) for live Zoom mode.

## Update 2026-09-26: third pillar added (WP9)
The owner approved the **spoken-secret shield**: redact codes, passwords and card numbers from your outbound voice
while the caller is unverified. Spec: `docs/plans/06_spoken_secret_shield.md`; work package WP9 in `docs/plans/05`. It is **in
scope for this phase and not blocked on upstream**. Build it after WP8 lands (it plugs into the pipeline's outbound
chain after the Keyguard shield, with a constant delay line like the existing shield).

## Update 2026-09-26 evening: phase 1 complete up to the upstream blockers
Built by the CallGuard session. PRs (open, for the owner to merge in order): **#1** WP3 Hearsay driver, **#2** WP7
attack proof, **#3** WP8 integration, **#4** WP9 secret shield (stacked on #3). `uv run pytest -q`: 72 passed with
the upstream repos + Vosk model present.
- `callguard run --mode replay --scenario ai_caller --drivers real` plays the full story on the dashboard:
  SAFE → WATCH → WARN → CRITICAL (~26-29 s) → WARN (shield on) → WATCH → SAFE; the secret shield arms on the
  synthetic voice and hears the agent's "read me the code".
- The six contract mismatches above are reconciled (see PR #3). An independent review's findings are fixed
  (PR #4 description lists them).
- Honest results: docs/reports/attack_under_speech.md (plan 04: criterion 1 pass with known key timing only, 2 marginal
  fail, 3 pass at the boundary) and docs/reports/secret_shield.md (0.19 s/min false redaction on normal speech; 1.32
  digits leaked per sequence vs the ≤ 1 acceptance).
- Owner to-dos: install VB-CABLE; merge #1 before the others (it removes a hard-coded path and stops __pycache__
  in Hearsay); record a consenting teammate reading the fake code to `demo/audio/recorded/victim_code.wav` (the
  secret-shield demo beat and a real eval set need it; never TTS); decide on the agent voice (currently a dataset
  ElevenLabs clone of a LibriSpeech speaker, brief constraint 4 wants a consenting teammate's clone); confirm
  whether CallGuard should keep Hearsay's current 4:1 threshold or follow Pspoof=0.3/Cfa=4 once the Hearsay
  session recomputes it (delete runs/hearsay_calibration.json after it changes).

## Update 2026-09-26 night: desktop app + Google Meet, repo restructured
Branch `restructure-app-layout` (docs on `docs-desktop-meet`); not merged to `main` yet.
- **Layout:** `app/source` (runtime), `app/hearsay` | `app/keystroke_guard` | `app/secret_shield` (driver around the
  real model, mock, harness, eval, README), `dashboard/`, `demo/`, `docs/` (plans, reports, experiments,
  demo_runbook, meeting_setup), `tests/`. Console script: `callguard = app.source.cli:main`.
- **Desktop app:** `uv run callguard app` (pywebview window; system browser fallback). **Google Meet connector**
  (`app/source/connectors/meet`): CallGuard opens Meet in its own Chrome/Edge profile and injects an audio bridge
  over DevTools, so Meet needs no VB-CABLE. `callguard run --mode meet` is the same engine with the dashboard in the
  browser. Local test room at `/meet/testroom`. VB-CABLE (`--mode live`) is now only for Zoom/Teams: see the
  appendix of `docs/meeting_setup.md` (renamed from `docs/zoom_setup.md`).
- **Pillar harnesses:** `python -m app.<pillar>.harness` checks any driver (mock, real, `module:Class`) against its
  contract and real-time budget; the real runs are recorded in each pillar README.
- **Measured:** full-stack meeting-room test (headless Chrome test room → bridge → meet mode, real drivers) blocked
  a spoken 6-digit code with ~1 s leaked at its start. `uv run pytest -q`: 90 passed (84 passed + 7 skipped in a
  checkout without the Vosk model).
- **Hearsay for the main track:** the model repository is [danmano411/hearsay](https://github.com/danmano411/hearsay); CallGuard itself lives in [georgia-mellon/callguard](https://github.com/georgia-mellon/callguard), a fork of it, and
  a dedicated README section, per the NSA organizers' guidance. The NSA challenge submission stays separate and frozen.
- **Still blocked upstream:** Keyguard's final attacker weights, the adversarial shield stage, the CTC free-typing
  attacker (teammate); the owner's consenting victim recording of a fake code (Secret Shield eval + demo beat).
- **Code follow-up (not docs):** `app/source/audio/devices.py` still points at `docs/zoom_setup.md`; it should say
  `docs/meeting_setup.md`.

## Update 2026-09-26 late: repository move
CallGuard now lives at https://github.com/georgia-mellon/callguard, a fork of danmano411/hearsay whose main branch
carries the CallGuard application (both histories kept). danmano411/callguard is no longer used. No attribution
trailers in commits or PRs (owner rule, same as Hearsay). The root README and the Hearsay overview are templates to be
written.

## Update 2026-09-26 night: Hearsay E5 is the default
The Hearsay freeze is lifted: the NSA review confirmed the scorer (Pspoof 0.3, Cfa 4, higher score = real). Official
HGT test score for R5: minDCF 0.0584, EER 2.5 % (organizers); E5 is Hearsay's final submission (score pending).
CallGuard's default voice model is now **E5** (LR fusion of R4ft + R6 + R1, `data/models/e5/e5_fusion.json`), with r5 /
r4ft selectable. E5 on CPU, 4 threads: ~1.4 s per 4 s window (keeps the 2 s hop, no skipped windows in the realtime
replay), 20/20 on held-out clips, threshold 0.97 calibrated on E5's val_testlike cross-fitted scores (cache keyed on
both checkpoint shas). The ai_caller demo was retimed (agent 12-44 s, typing at 24 s, shield at 31 s) so the voice
risk has built up when the typing starts.

## Update 2026-09-26 night: test voices, judge-my-voice, stricter threshold, smoother graph
- **Test room:** "Play the 10 test voices" (5 real + 5 AI held-out Hearsay clips in `demo/audio/testclips/`, picked by
  scoring 66 held-out clips through an emulation of the room's path: 48 kHz playback, the bridge's resampler, Opus;
  that path scores within ~0.03 of the files fed directly). Live run, headless Chrome + real E5: 10/10 (real 0.03-0.22,
  AI 0.94-0.98). "Judge my voice" = your raw mic as the caller (real voice on a fake mic: real, p 0.02; AI clip: AI,
  p 0.99). Rebuild the gitignored clip set with `python demo/build_testclips.py`.
- **Threshold:** `drivers.hearsay_ai_p = 0.7` (was Hearsay's 0.5). On the 66 clips: real voices peaked at 0.40, so no
  real is flagged either way; 6/38 fakes fall under 0.7 (2/38 under 0.5). 0.8 broke the ai_caller demo (agent median
  p 0.59, V never passed the 0.5 gate); 0.7 keeps it (real replay test passes).
- **Threat rule:** the social-engineering boost is 60 + 40*leak once V >= 0.5 (was 60 + 40*V*leak), so a stricter voice
  threshold does not also weaken the alarm (plan 02 updated).
- **Dashboard:** the voice label and curve are the mean of the last 3 windows (a gap > 6 s starts over).
