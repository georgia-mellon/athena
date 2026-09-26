# CallGuard

**Live call security for the AI-voice era.** CallGuard sits between you and your meeting app. It checks whether the
voice you're hearing is a real person (**Hearsay**), keeps the keys you type from leaking through your microphone
(**Keyguard**), and turns both into one live **threat score** on a dashboard. HackGT 13, team GeorgiaMellon.

## The threat
An AI voice agent joins a Zoom call posing as IT: *"please type the reset code while we're on the line."* Two things
go wrong at once:
1. **Inbound:** the voice is synthetic, and you can't tell.
2. **Outbound:** your keystrokes are audible in your mic stream. Anyone recording the call can run a keystroke
   classifier and read the code. No malware needed.

CallGuard's answer, live: the voice light goes red, the keystroke panel shows what an eavesdropper reads from your
raw mic vs. from the shielded mic, the **shield** makes the eavesdropper read noise while your speech stays intact,
and the threat score escalates to **CRITICAL** (synthetic caller + sensitive typing = social engineering in progress).

## How it works
```
 physical mic ─► [20 ms blocks] ─► Keyguard shield ─► virtual mic (VB-CABLE) ─► Zoom / Meet / Teams
                     │ raw copy          │ shielded copy
                     ▼                   ▼
               raw ring buffer    shielded ring buffer ──► keystroke attacker (at each OS key event) ─► keys.readout
 key timing (pynput) ─┘

 meeting speaker ─► WASAPI loopback ─► far-end ring ─► VAD ─► Hearsay (4 s windows every 2 s) ─► voice.verdict

 all events ─► EventBus ─► ThreatEngine (0-100, SAFE/WATCH/WARN/CRITICAL) ─► dashboard (WebSocket) + hooks
```
- **Works with any meeting app**: integration is at the audio-device layer, so there's no plugin or bot, and the
  shield can act on your outgoing audio before the app encodes it ([plans/03](plans/03_meeting_platform.md)).
- **Swappable drivers** (`callguard/drivers/`): `real` (Hearsay's frozen models; Keyguard's attacker + DSP shield)
  or `mock` (deterministic, no models) behind the Protocols in `callguard/types.py`.
- **Fail-safe audio**: models never run on the audio thread except the shield, and a driver that raises is
  quarantined: the audio passes through, and the dashboard raises an alarm. Your mic is never muted by a bug.
- **Threat score** ([plans/02 §4](plans/02_architecture.md)): voice risk V (EMA of p_synthetic), keystroke exposure
  E/L (attacker accuracy above chance on the raw/shielded mic), typing activity T, plus a social-engineering rule:
  typing while an unverified voice speaks.
- **Hooks**: console, JSONL and webhook sinks (e.g. a Slack incoming webhook via `CALLGUARD_WEBHOOK_URL`). Key
  identities and attacker guesses are scrubbed before anything leaves the process.

## Quickstart (Windows, Python 3.12, [uv](https://docs.astral.sh/uv/))
```
uv sync
uv run pytest -q                                   # mock drivers: no models, no audio devices needed
```
The real drivers read the two upstream repos, read-only, from sibling checkouts (override with `HEARSAY_ROOT` /
`KEYGUARD_ROOT`): `../Hearsay` (its frozen `R4ft_xlsr_light` checkpoint) and `../keyboard-acoustic-shield`.
```
uv run python demo/build_scenario_audio.py         # builds demo/audio/ (gitignored) from the upstream data
uv run callguard run --mode replay --scenario ai_caller          # the full story, real models, dashboard opens
uv run callguard run --mode replay --drivers mock                # UI work without models
uv run callguard devices                           # checks the Zoom routing (VB-CABLE, loopback)
uv run callguard run --mode live                   # real call: see docs/zoom_setup.md
uv run callguard bench                             # per-driver latency
```
Dashboard: <http://127.0.0.1:8765/>. Config: copy `callguard.example.toml` to `callguard.toml`.

## Demo modes
| mode | what it needs | use |
|---|---|---|
| `replay` | nothing but this laptop | the expo demo: a scripted 60 s call (real colleague → AI agent asks for the code → you type it → shield on → agent leaves) through the same pipeline as live |
| `live` | VB-CABLE + Zoom + a second device as the "agent" | the real thing: `demo/agent_caller.py` plays TTS lines into the call from the second device |

The runbook for the expo is [docs/demo_runbook.md](docs/demo_runbook.md).

## Results
All numbers from this laptop's CPU (AMD, 16 threads). The keystroke attacker is **provisional** (a KeyNet we trained on
Keyguard's public harrison bank, one keyboard) until the teammate's weights ship.

**End-to-end replay** (`ai_caller`, real drivers, `tests/test_e2e_replay.py`): SAFE → WATCH (synthetic voice) →
CRITICAL (typing the code while the agent speaks, shield off) → WARN (shield on) → WATCH → SAFE after the agent
hangs up. Hearsay median p_synthetic per segment: real colleague 0.19, ElevenLabs clone of that colleague 0.83,
colleague again 0.11. The attacker's top 3 holds the true key for ~89 % of the unshielded keystrokes and ~33 % of the
shielded ones (chance 8 %).

**Latency** (`callguard bench`): Hearsay R4ft ~0.65 s per 4 s window (every 2 s, off the audio thread); attacker
2 ms per keystroke; DSP shield 2.8 ms median (5 ms max) per 20 ms block while a key is active, 0 ms otherwise, plus
80 ms constant lookahead.

**Attack proof** ([reports/attack_under_speech.md](reports/attack_under_speech.md), 360 held-out presses, 95 % CIs):
can an eavesdropper read keys with someone talking over them, and does the shield stop it? Top-1, adaptive
(speech-trained) attacker, chance 2.8 %:

| condition | no shield | Keyguard DSP shield (208 ms key region) |
|---|---|---|
| quiet typing, attacker knows key timing | 53.6 % | 10.8 % |
| quiet typing, attacker detects keys itself | 47.5 % | 11.4 % |
| speech +10 dB over the keys, known timing | 15.8 % | 5.8 % |
| speech +10 dB, attacker detects keys itself | 5.8 % | 2.5 % |

Honest reading: keystrokes are clearly readable on a call when you type while quiet, and still 5.7x chance under
speech if the attacker knows when you typed; the attacker's own key detection is what fails under speech. The
shield cuts quiet-typing reads ~5x but doesn't reach chance against this adaptive attacker (plan 04 criterion 2
misses narrowly: 5.8 % vs a 5.6 % bar, STOI 0.897 vs 0.9 on a pessimistic one-key-per-1.5 s test). The fix is
Keyguard's adversarial shield stage (teammate). Hearsay flags 2/100 real voices on shielded speech (0/100 unshielded).

**Hearsay under keystrokes** (the reverse direction, `hearsay/reports/generalization.md` §2): with typing as loud as
the voice, the submitted model flags ≤ 1.7 % of real speakers; with the Keyguard shield on, 1.3 %.

## Repository
```
callguard/   types.py (contracts) · pipeline.py · cli.py · bus.py · threat.py · hooks.py · config.py
             audio/ (devices, streams, ring, vad, keys, replay) · drivers/ (hearsay_real, keyguard_real, mock, base)
             server/ (FastAPI + static dashboard, no CDNs)
demo/        scenarios/ai_caller.toml · build_scenario_audio.py · agent_caller.py · render_agent.py
experiments/ attack_under_speech.py      reports/ attack_under_speech.md
plans/       00 brief · 01 spec · 02 architecture · 03 meeting platform · 04 attack proof · 05 work plan and merge
```

## Ethics
Our own devices, consenting teammates and judges only. **Fake passwords** in every demo. The AI "caller" uses
public research clips or a consenting teammate's cloned voice. No audio, weights or recordings are committed.

## Credits
- **Hearsay** (ours, `danmano411/hearsay`): real vs. synthetic speech. Our NSA HEARSAY challenge submission, used
  here read-only and frozen.
- **Keyguard** (`LordKarV/keyboard-acoustic-shield`, by our teammate): the acoustic keystroke attacker and the
  shield. Used here read-only; CallGuard's attacker is a *provisional* KeyNet trained on Keyguard's public harrison
  bank until the teammate's trained weights ship.
