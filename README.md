# CallGuard

**CallGuard protects what you hear, what you type, and what you say.**

A desktop app that guards a Google Meet call with three pillars: **Hearsay** (is the voice you hear real?),
**Keystroke Guard** (can the call hear your keys?) and **Secret Shield** (don't read the code to a fake caller). All
three feed one live threat score on a dashboard. HackGT 13, team GeorgiaMellon.

## The threat
An AI voice agent joins your Google Meet posing as IT: *"please type the reset code while we're on the line"*, then
*"just read me the code"*. Three things go wrong at once:

1. **What you hear:** the voice is synthetic, and you can't tell.
2. **What you type:** your keystrokes are audible in your mic stream. Anyone recording the call can run a keystroke
   classifier on it and read the code. No malware needed.
3. **What you say:** you read the code out loud.

CallGuard, live: the voice light goes red; the keyboard panel shows what an eavesdropper reads from your raw mic vs.
the shielded mic; the digits you start reading are cut from your outgoing audio (the dashboard shows "6-digit code
blocked", never the digits); the threat score climbs to **CRITICAL** (synthetic caller + sensitive typing = social
engineering in progress).

## How it works
```
 CallGuard's own Chrome window (Google Meet, bridge.js injected over DevTools)
   your mic ─► 16 kHz blocks ─► ws /meet/mic ─┐                    ┌─► processed blocks ─► the track Meet sends
   remote participants' audio ─► ws /meet/far ─┐                   │
                                               │  pipeline         │
                                               │   mic: Keystroke Guard shield ─► Secret Shield delay line (500 ms)
                                               │         └─ attacker reads raw vs shielded at each OS key event
                                               └─► far: VAD ─► Hearsay (4 s windows every 2 s)
                                                        └─► "read me the code" listener
 events (voice.verdict, keys.readout, secret.*) ─► threat engine (0-100: SAFE/WATCH/WARN/CRITICAL)
                                                ─► dashboard (native window, WebSocket) + hooks (console/JSONL/webhook)
```
- **No plugin, no bot, no virtual cable.** `callguard app` opens Meet in a dedicated Chrome (or Edge) profile and
  injects an audio bridge before Meet's scripts run. The bridge swaps the mic track Meet sends for CallGuard's
  processed audio and taps the remote audio. Details: [app/source/connectors/meet](app/source/connectors/meet/README.md).
- **Fails open.** If CallGuard stops answering, the bridge sends your raw mic within the jitter buffer. A driver
  that raises is quarantined and the audio passes through. Your mic is never muted by a bug.
- **Threat-aware Secret Shield.** It arms only while the caller is unverified (Hearsay p ≥ 0.5), after the caller
  asks for a code, or by hand. With a verified colleague the same sentence passes untouched. The recognized words
  never leave the spotter; events carry category and length only.
- **Swappable drivers** behind the Protocols in `app/source/types.py`: `real` (the models) or `mock` (deterministic,
  no weights). Each pillar has a harness that checks a new driver against the contract and the real-time budget.
- **Hooks** scrub key identities and attacker guesses before anything leaves the process (webhook URL via
  `CALLGUARD_WEBHOOK_URL`).

## Repository layout
| path | what's there |
|---|---|
| `app/source/` | the runtime: pipeline, threat engine, event bus, hooks, config, driver registry, CLI, desktop shell, shared harness helpers |
| `app/source/audio/` | device I/O, ring buffers, VAD, key timing, replay (for replay and virtual-device modes) |
| `app/source/connectors/meet/` | the Google Meet bridge: `bridge.js`, the Chrome launcher, the `/meet` router, the local test room |
| `app/hearsay/` | Hearsay pillar: driver around the frozen model, mock, harness, README |
| `app/keystroke_guard/` | Keystroke Guard pillar: attacker + shield drivers, mocks, harness, the attack-under-speech eval |
| `app/secret_shield/` | Secret Shield pillar: Vosk spotter, delay-line redactor, mock, harness, eval, model fetcher |
| `dashboard/` | FastAPI server + static UI (no CDNs) |
| `demo/` | the `ai_caller` replay scenario and its audio builders, the second-device agent player |
| `docs/` | plans, reports (all measured numbers), experiments, demo runbook, meeting setup |
| `tests/` | pytest; runs on mock drivers without models or audio devices |

## Quickstart (Windows, Python 3.12, [uv](https://docs.astral.sh/uv/), Chrome or Edge)
```
uv sync
uv run pytest -q                                   # no models or audio devices needed
uv run python -m app.secret_shield.get_model       # Vosk model for the Secret Shield (40 MB, sha256-checked, once)
uv run callguard app                               # desktop window; click Join, paste a Meet link
uv run callguard app --meet-url abc-defg-hij       # or open that meeting at start
```
The real Hearsay and Keystroke Guard drivers read the upstream repos read-only from sibling checkouts: `../Hearsay`
and `../keyboard-acoustic-shield` (override with `HEARSAY_ROOT` / `KEYGUARD_ROOT`). Config: copy
`callguard.example.toml` to `callguard.toml`.

The first time, sign in to Google in CallGuard's Chrome window (its own profile in `%LOCALAPPDATA%\CallGuard\meet-profile`, outside the repo) or join as a
guest. Turn Meet's noise cancellation off (it removes key clicks and hides the shield). Setup and troubleshooting:
[docs/meeting_setup.md](docs/meeting_setup.md).

| command | what it does |
|---|---|
| `uv run callguard app [--meet-url URL]` | desktop app: dashboard in a native window + CallGuard's Meet window |
| `uv run callguard run --mode meet [--meet-url URL]` | the same engine, dashboard in your browser |
| `uv run callguard run --mode replay --scenario ai_caller` | offline demo: the scripted 60 s call through the same pipeline (build its audio first, see [demo/](demo/README.md)) |
| `uv run callguard run --mode replay --drivers mock` | UI work without models (numbers are meaningless) |
| `uv run callguard bench` | per-driver latency |
| `python -m app.hearsay.harness` (also `keystroke_guard`, `secret_shield`) | check a driver against its contract and budget |

Dashboard: <http://127.0.0.1:8765/>. Test room: <http://127.0.0.1:8765/meet/testroom>. The expo script is
[docs/demo_runbook.md](docs/demo_runbook.md).

## Testing the Secret Shield in a meeting
**Local test room** (no second device, no network). A local page with one fake participant: your mic goes through
the same bridge as in Meet, then over a real WebRTC connection to "the other side", which you can hear and record.
1. `uv run callguard run --mode meet`, then open <http://127.0.0.1:8765/meet/testroom> (the dashboard's *Test room*
   link; the page loads the bridge itself, so any Chrome or Edge works). Headphones on.
2. **Join with mic**, tick **listen to what the room hears**.
3. **Arm** the Secret Shield (or pick a demo clip and **Play as caller**: *Auto* arms on a synthetic voice or a
   "read me the code" request).
4. Read a fake code aloud, e.g. "the code is four eight two one nine three". You hear a tone where the digits were;
   the dashboard logs "6-digit code blocked". **Record** saves what the room heard.

**Real Google Meet** (a second device as the caller):
1. On the laptop: `uv run callguard app`, Join, start a meeting.
2. On a phone or second laptop, join the same meeting. Play the agent's lines from it
   (`demo/agent_caller.py`, see [demo/](demo/README.md)) or just speak.
3. On the laptop, read a fake code. On the second device you hear the tone, not the digits. Type a fake code into
   any text box to see the keystroke readout.

## Results
All numbers are CPU-only on the dev laptop; sources are linked. The keystroke attacker is **provisional** (a KeyNet
CallGuard trained on Keyguard's public harrison bank, one keyboard) until the teammate's weights ship.

**Full stack in a meeting.** Headless Chrome test room → bridge → meet mode with the real drivers: a spoken 6-digit
code was blocked, with about 1 s leaked at its start (manual run, not yet a report). The bridge round-trip is
asserted < 200 ms in `tests/test_meet_connector.py`. `uv run pytest -q`: 90 passed with the upstream repos and
models present; without the Vosk model 84 pass and 7 skip.

**End-to-end replay** (`ai_caller`, real drivers, `tests/test_e2e_replay.py`): SAFE → WATCH/WARN (synthetic voice)
→ CRITICAL (typing the code while the agent speaks, shield off) → not CRITICAL once the shield is on.

**Latency** (pillar harnesses): Hearsay R4ft median 545 ms per 4 s window (budget 2 s, off the audio thread);
attacker 1.2 ms per keystroke; DSP shield 3.1 ms median per 20 ms block with a key active, plus a constant 80 ms
lookahead; Secret Shield spotter 5.46 ms mean per 20 ms block (p99 79 ms) plus the constant 500 ms delay line.

**Keystroke Guard** ([attack_under_speech.md](docs/reports/attack_under_speech.md), 360 held-out presses, top-1,
adaptive speech-trained attacker, chance 2.8 %):

| condition | no shield | Keyguard DSP shield |
|---|---|---|
| quiet typing, attacker knows key timing | 53.6 % | 10.8 % |
| quiet typing, attacker detects keys itself | 47.5 % | 11.4 % |
| speech +10 dB over the keys, known timing | 15.8 % | 5.8 % |
| speech +10 dB, attacker detects keys itself | 5.8 % | 2.5 % |

Keys are clearly readable on a call when you type while quiet. The shield cuts reads about 5x but does not reach
chance against this adaptive attacker: plan 04 criterion 2 misses narrowly (5.8 % vs a 5.6 % bar; STOI 0.897 vs
0.9). The fix is Keyguard's adversarial shield stage.

**Secret Shield** ([secret_shield.md](docs/reports/secret_shield.md), 40 TTS sequences):

| metric | target | result |
|---|---|---|
| secret words leaked per sequence | 0 (acceptance ≤ 1) | 1.32 |
| sequences fully blocked | - | 36 % (45 % with a trigger phrase first) |
| false redaction, LibriSpeech | < 1 s/min | 0.19 s/min |
| false redaction, casual-number sentences | < 1 s/min | 4.85 s/min |
| inbound "read me the code" requests caught | - | 7 / 8 |

It misses both acceptance bars, and every test utterance is one synthetic voice. Without a trigger phrase the first
digit passes by design (the delay line cannot wait for a whole sequence).

**Hearsay in a call.** With the Keyguard shield on, Hearsay flags 2 / 100 real voices (0 / 100 unshielded). With
typing as loud as the voice it flags ≤ 1.7 % of real speakers (Hearsay `reports/generalization.md` §2). It was never
evaluated on a meeting codec with echo cancellation.

## Hearsay model
Hearsay is our submission to the NSA "HEARSAY" challenge at HackGT: score each clip from 0.0 (real) to 1.0
(synthetic), judged by ASVspoof 5 Track-1 minDCF. CallGuard uses it as one component, read-only. The model-creation
repository, with every plan, report and script, is **[swail-labs/hearsay](https://github.com/swail-labs/hearsay)**
(a fork, per the organizers' guidance for main-track use). **The NSA challenge submission itself is separate and
frozen**; CallGuard does not change it.

- **Data.** The given data (70k DiffSSD fakes vs 242 clips of one real speaker) was a trap: a depth-3 tree on
  trivial cues separated it perfectly. Hearsay added 57k real clips from 10 corpora, 11 external corpora in all, and
  5.2k of its own TTS fakes: 185,915 clips with grouped splits and frozen evaluation sets.
- **Preprocessing.** `prep()` (DC removal, silence trim, 7 kHz low-pass, RMS normalization) removes the channel
  shortcuts: a trivial-cue model drops from minDCF 0.54 to 0.92 (near chance). Class-symmetric augmentation (MP3,
  noise, resampler).
- **Models.** R1: LightGBM on 228 spectral + speech-biology features (jitter, shimmer, HNR, formants). R4ft:
  XLS-R-300M cut to 12 blocks, fine-tuned end to end with a light back end (2 h 24 min on an 8 GB laptop GPU).
  R5: logistic fusion of R4ft + R1 (the submitted model).

| model (held-out `test_internal_testlike`, 9,747 clips, half the fakes from unseen generators) | minDCF | EER |
|---|---|---|
| R1 classic features | 0.253 | 6.5 % |
| **R4ft** XLS-R fine-tune (the honest estimate; CallGuard's default) | **0.0282** | 0.76 % |
| R5 fusion, submitted (optimistic: the headline informed the switch) | 0.0218 | 0.54 % |

In CallGuard, `p_synthetic = 0.5` sits at Hearsay's own deployment threshold (a real voice flagged costs 4x). More:
[app/hearsay/README.md](app/hearsay/README.md).

## Keystroke Guard credits
The acoustic keystroke attacker architecture (KeyNet), its features and the DSP shield come from our teammate's repo
**[LordKarV/keyboard-acoustic-shield](https://github.com/LordKarV/keyboard-acoustic-shield)**, used read-only.
CallGuard streams the shield in 20 ms blocks and, until the teammate's trained weights ship, uses a provisional
attacker trained on that repo's public harrison bank. More: [app/keystroke_guard/README.md](app/keystroke_guard/README.md).

## Ethics
Our own devices, consenting teammates and judges only. **Fake codes and passwords** in every demo. The AI caller uses
public research clips or a consenting teammate's cloned voice. No audio, weights, recordings or webhook URLs are
committed. CallGuard's analysis stays on the laptop; an optional webhook gets scrubbed events only.

## What's next
- **Keyguard's final attacker weights** (teammate): replace the provisional attacker; no code change
  (`CALLGUARD_ATTACKER_WEIGHTS`).
- **Adversarial shield stage** (teammate): the dashboard's *adversarial* mode waits for it; it is what should take
  the attacker to chance under speech.
- **CTC free-typing attacker** (teammate): reads continuous typing instead of isolated presses.
- **A real victim recording** (owner): consenting teammates reading fake codes, to replace the TTS-only Secret Shield
  evaluation and complete the demo's spoken-code beat.
