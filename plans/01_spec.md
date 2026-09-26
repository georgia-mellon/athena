# 01: CallGuard product spec

## 1. One sentence
CallGuard sits between you and your meeting app. It checks whether the voice you are hearing is a real person
(**Hearsay**), keeps the keys you type from leaking through your microphone (**Keyguard**), and turns both into
one live **threat score** on a dashboard.

## 2. The threat (demo story)
An **AI voice agent** joins a Zoom call posing as a manager or IT. It asks the victim to *"type the reset code while
we're on the line"*. Two things go wrong at once:
1. **Inbound:** the voice is synthetic and the victim can't tell.
2. **Outbound:** the victim's keystrokes are audible in their mic stream. The agent (or anyone recording the call)
   runs a keystroke classifier and reads the code, with no malware needed.

CallGuard's answer, live on the dashboard: the **voice-authenticity light goes red**, the **keystroke-exposure meter**
shows what an eavesdropper would read, the **shield** kicks in so the eavesdropper reads noise, and the **threat
score** escalates to CRITICAL: *synthetic caller + sensitive typing = social-engineering attack in progress*.

## 3. Users and surfaces
| surface | who | what |
|---|---|---|
| Dashboard (browser, `localhost:8765`) | the person on the call; judges at the expo | live threat score, voice light, attacker readout (raw vs. shielded), timeline, event log |
| Meeting app (Zoom primary; Meet/Teams work the same way) | everyone on the call | unchanged app; CallGuard is a virtual microphone + loopback listener (`plans/03`) |
| Hooks | integrators | webhook / console / file sinks for threat events (`plans/02` §5) |
| CLI | us | `callguard run --mode live|replay`, `callguard devices`, `callguard bench` |

## 4. Functional requirements
**F1 Inbound voice authenticity (Hearsay).** Capture the far-end audio (meeting output) and keep a rolling 4 s
buffer. Every 2 s of *speech* (energy VAD; skip silence), score it with the Hearsay driver. The driver returns
`p_synthetic ∈ [0,1]` plus the raw margin. Smooth over the last N windows (EMA). Latency target: a verdict within
≤ 4 s of speech onset on CPU.

**F2 Outbound keystroke exposure (Keyguard attacker).** Take the local mic audio *before* the shield, plus OS
key-event timestamps (`pynput`, timing only; key identity is used only locally, for the demo's ground truth, and is
never logged or sent). Around each key event, run the attacker driver and show what an eavesdropper would read:
top-1/top-3 keys, running text, and accuracy against the true keys.

**F3 Outbound shield (Keyguard defender).** Every mic block passes through the shield driver before it goes to the
virtual mic. Speech must stay intact. The attacker is also run on the shielded stream, and the dashboard shows raw vs.
shielded readout side by side. Shield modes: `off | dsp | adversarial` (adversarial when the teammate ships weights).

**F4 Threat score** (`plans/02` §4): one number 0-100 and a level `SAFE | WATCH | WARN | CRITICAL`, updated at
≥ 2 Hz, with reasons ("synthetic voice 0.93 for 12 s", "typing while an unverified voice is speaking").

**F5 Dashboard:** live over WebSocket; works from a projector at the expo; has a scenario/replay control for
the offline demo.

**F6 Hooks:** events `voice.window`, `voice.verdict`, `keys.stroke`, `keys.readout`, `shield.state`,
`threat.update`, `threat.level_change`; sinks: console, JSONL file, HTTP webhook (e.g. Slack incoming webhook).

**F7 Two run modes:**
- `live`: real devices, real meeting.
- `replay`: scripted scenario from audio files (AI-agent voice, typing, human voice) through the *same pipeline*.
  This is the expo fallback when Wi-Fi or Zoom misbehaves. The demo must never depend on the venue network.

**F8 Drivers are swappable** (`plans/02` §3): `real` (Hearsay frozen models; Keyguard current code) or `mock`
(deterministic, no models) for tests and UI work. Chosen by config, never by code edits.

## 5. Non-functional
- **Runs on one Windows laptop, CPU only.** CUDA is used if present. Hearsay live inference uses 1 window, ≈ 0.2 s
  on CPU.
- **Privacy:** no audio leaves the machine except the (shielded) mic into the meeting app. Key identities stay in
  memory for the demo readout only.
- **Fail safe:** if a driver crashes, the audio path keeps flowing (shield failure → pass-through + dashboard alarm).
  The mic is never muted by a bug.
- **Reproducible:** `uv` env, pinned deps; tests run without models (mock drivers).

## 6. Acceptance (definition of done for the expo)
1. `callguard run --mode replay --scenario ai_caller` plays the full story end to end: the dashboard goes
   SAFE → WARN → CRITICAL and back; the attacker readout reads the typed fake code without the shield and noise with it.
2. `callguard run --mode live` with Zoom: CallGuard's virtual mic is selectable in Zoom; far-end audio is scored;
   a TTS "agent" joining from a second device triggers the red voice light.
3. The attack proof (`plans/04`): attacker accuracy with speech in the background, well above chance, and near
   chance with the shield on; speech quality numbers for the shielded audio.
4. Tests pass with mock drivers; a `bench` command reports per-driver latency.
5. README + a demo runbook that a judge-facing teammate can follow.
