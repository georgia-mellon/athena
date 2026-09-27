# Demo runbook (expo)

For the teammate at the table. Two modes: **replay** (always works, no network) and **live** (a real Google Meet
through CallGuard's desktop app). Start with replay. Switch to live only if the room and Wi-Fi allow. The local
**test room** is the fallback for the Secret Shield beat when Wi-Fi is down.

## Before the expo (once, at home)
1. `uv sync`, then `uv run pytest -q`. Expect everything to pass (about 10 real-driver tests skip when the upstream repos or the Vosk model are missing).
2. Check that `../Hearsay` and `../keyboard-acoustic-shield` are present (sibling folders), then run
   `uv run python demo/build_scenario_audio.py` (it writes `demo/audio/`).
3. Dry run: `uv run callguard run --mode replay --scenario ai_caller --exit-at-end`. The first start takes
   ~20-40 s (Hearsay model load, plus a one-time sha256 check of the checkpoint). Watch the dashboard go
   SAFE → WARN → CRITICAL → back.
4. For live: Chrome or Edge installed. Run `uv run callguard app` once, sign in to Google in CallGuard's Chrome
   window (its own profile) and turn Meet's noise cancellation off. Details: `docs/meeting_setup.md`.
5. Laptop on power, notifications off, headphones in, display mirrored to the projector at 100 % zoom.
6. Spoken-secret beat: `uv run python -m app.secret_shield.get_model` (40 MB, once). Record a teammate (consenting) reading
   the **fake** code, e.g. "four eight two one nine three", as a 16 kHz mono WAV at
   `demo/audio/recorded/victim_code.wav`, then rebuild the demo audio. Without the recording the replay still arms
   the secret shield but has nothing to cut (plan 06: never fake the victim with TTS).

## Replay demo (the default, ~90 s)
Start: `uv run callguard run --mode replay --scenario ai_caller`. The browser opens the dashboard, and the story
starts 3 s later with its audio on the speakers (`--mute` to stay silent). **Start** on the dashboard restarts it.

| time | on screen | say |
|---|---|---|
| 0-12 s | a colleague talks; voice light **green (real)**; score SAFE | "CallGuard listens to the call. Right now it's a real colleague: Hearsay says real." |
| 12-20 s | the "IT agent" takes over; voice light turns **red (synthetic)**; score climbs to WATCH/WARN | "This is an AI voice. You can't hear the difference; Hearsay can." |
| 20-29 s | the user types the reset code; *Eavesdropper reads, no shield* fills in (green = read exactly, amber = true key in its top 3); score WARN, then **CRITICAL** once most of the code has been read (about 25 s) | "The agent asks for the reset code. Anyone recording the call can run a keystroke classifier on it. The true key is in its top 3 almost every time: a 9-character code drops to about 20,000 guesses." |
| 29 s | shield switches on (scripted) | "Now Keyguard's shield turns on. It only touches the few milliseconds around each key press." |
| 29-34 s | code typed again; the *shielded* row reads mostly wrong keys and its bar drops toward the chance tick; the raw row still reads it; score back to WARN | "Same typing. The meeting now gets the shielded mic: the eavesdropper's hit rate collapses. Your voice between key presses is untouched. (Honest caveat if asked: against this adaptive attacker the DSP shield cuts reads about 5x but not to chance; the adversarial shield stage is the teammate's next piece.)" |
| 34-40 s | the agent says *"just read me the verification code"*; *What you're saying* shows **armed** (unverified caller); if the victim recording is built in, the digits are cut from the outgoing audio (you hear a tone) and the panel logs "6-digit code blocked" | "Third pillar: what you say. The caller asked for the code, so CallGuard cuts the digits out of your voice before they reach the call. It only ever shows 'six-digit code', never the digits. With a real colleague it wouldn't touch a thing." |
| 40-60 s | agent hangs up, colleague returns; score decays to SAFE/WATCH | "The alert decays when the synthetic voice leaves." |

Point at: the gauge and its **reasons** list, the p_synthetic sparkline, the two readout rows with their accuracy
bars against the chance line, and the event log. The shield buttons (off / dsp) work at any time. *adversarial*
shows an error banner until the teammate's model ships. Don't click it on stage.

Talking points: "No plugin and no bot: CallGuard opens Meet in its own window and processes the audio before Meet sends it." "Nothing leaves the laptop."
"Fake password, consenting voices." The attack proof: `docs/reports/attack_under_speech.md` (keys are still readable
with someone talking over them, and the shield brings the attacker down toward chance).

## Live demo (Google Meet)
1. Laptop A (victim): `uv run callguard app`. On the dashboard's Meeting bar, paste the meeting link and **Join**;
   CallGuard's Chrome window opens Meet. Headphones on. Wait for the pill `in meeting: mic ✓ far ✓`.
2. Laptop/phone B ("IT Support") joins the same meeting. On B: `uv run python demo/agent_caller.py
   demo/audio/agent_lines --device "<B's speaker or virtual mic>"` (render the lines once with
   `demo/render_agent.py`, or play any consenting-voice clips).
3. A human on B speaks first (green light), then B plays the agent lines (red light). On A, type the fake code in
   any text box: the readout rows fill. Toggle the shield on the dashboard. Then read the fake code aloud: B hears
   a tone instead of the digits.
4. The outgoing mic is delayed by ~0.6 s in total (bridge jitter buffer + 80 ms Keyguard lookahead + 500 ms
   secret-shield delay line). Set `[secret] enabled = false` in `callguard.toml` if the call feels laggy; the
   other two pillars keep working.
5. If key timing looks off (readouts wrong even with the shield off), set `[devices] meet_key_offset_s` (Meet) or `key_offset_s` (virtual devices) in
   `callguard.toml` (try +0.02 to +0.08 s).

## Test room (Secret Shield without a second device)
`uv run callguard run --mode meet`, open <http://127.0.0.1:8765/meet/testroom>, **Join with mic**, tick *listen to
what the room hears*, **Arm**, read the fake code: the room hears a tone. **Record** keeps what it heard.

## When something breaks
| symptom | do this |
|---|---|
| Wi-Fi or Meet down | Use replay mode (no network), and the test room for the spoken-code beat. |
| Meet pill never shows `mic ✓ far ✓` | Leave and Join again from the dashboard; see `docs/meeting_setup.md` troubleshooting. If CallGuard dies, Meet keeps your raw mic (fail open). |
| Dashboard blank / "reconnecting…" | The server isn't running, or the port is taken: re-run with `--port 8766` and open that URL. |
| Red alarm banner "Driver … failed" | The audio keeps flowing (the driver is quarantined). Restart CallGuard. `--drivers mock` only proves the UI works: its numbers don't tell the story, so don't present them. |
| `scenario 'ai_caller' audio missing` | Run `uv run python demo/build_scenario_audio.py` (needs the sibling repos). |
| First start is slow | Model load + the one-time checkpoint hash. Start CallGuard before the judges arrive. |
| No sound in replay | Check the default Windows output device; `--mute` runs it silently. |
