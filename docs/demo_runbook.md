# Demo runbook (expo)

For the teammate at the table. Two modes: **replay** (always works, no network) and **live** (a real Zoom call).
Start with replay. Switch to live only if the room and Wi-Fi allow.

## Before the expo (once, at home)
1. `uv sync`, then `uv run pytest -q`. Expect everything to pass (1-2 real-driver tests skip without the upstream repos).
2. Check that `../Hearsay` and `../keyboard-acoustic-shield` are present (sibling folders), then run
   `uv run python demo/build_scenario_audio.py` (it writes `demo/audio/`).
3. Dry run: `uv run callguard run --mode replay --scenario ai_caller --exit-at-end`. The first start takes
   ~20-40 s (Hearsay model load, plus a one-time sha256 check of the checkpoint). Watch the dashboard go
   SAFE → WARN → CRITICAL → back.
4. For live: install **VB-CABLE** (https://vb-audio.com/Cable/, run as admin, reboot), then follow
   `docs/zoom_setup.md`. `uv run callguard devices` must show `virtual mic out: CABLE Input`.
5. Laptop on power, notifications off, headphones in, display mirrored to the projector at 100 % zoom.

## Replay demo (the default, ~90 s)
Start: `uv run callguard run --mode replay --scenario ai_caller`. The browser opens the dashboard, and the story
starts 3 s later with its audio on the speakers (`--mute` to stay silent). **Start** on the dashboard restarts it.

| time | on screen | say |
|---|---|---|
| 0-12 s | a colleague talks; voice light **green (real)**; score SAFE | "CallGuard listens to the call. Right now it's a real colleague: Hearsay says real." |
| 12-22 s | the "IT agent" takes over; voice light turns **red (synthetic)**; score climbs to WATCH/WARN | "This is an AI voice. You can't hear the difference; Hearsay can." |
| 22-27 s | the user types the reset code; *Eavesdropper reads, no shield* fills in (green = read exactly, amber = true key in its top 3); score WARN, then **CRITICAL** once a few keys are read | "The agent asks for the reset code. Anyone recording the call can run a keystroke classifier on it. The true key is in its top 3 almost every time: a 9-character code drops to about 20,000 guesses." |
| 28 s | shield switches on (scripted) | "Now Keyguard's shield turns on. It only touches the few milliseconds around each key press." |
| 29-34 s | code typed again; the *shielded* row reads noise; the raw row still reads it | "Same typing. The meeting now gets the shielded mic, and the eavesdropper reads garbage. Your voice is untouched." |
| 40-60 s | agent hangs up, colleague returns; score decays to SAFE/WATCH | "The alert decays when the synthetic voice leaves." |

Point at: the gauge and its **reasons** list, the p_synthetic sparkline, the two readout rows with their accuracy
bars against the chance line, and the event log. The shield buttons (off / dsp) work at any time. *adversarial*
shows an error banner until the teammate's model ships. Don't click it on stage.

Talking points: "Works with any meeting app: it's a virtual mic, not a plugin." "Nothing leaves the laptop."
"Fake password, consenting voices." The attack proof: `reports/attack_under_speech.md` (keys are still readable
with someone talking over them, and the shield brings the attacker down toward chance).

## Live demo (Zoom)
1. Laptop A (victim): `uv run callguard run --mode live`. In Zoom, Microphone = **CABLE Output**, Speaker =
   headphones, Suppress background noise = **Low**.
2. Laptop/phone B ("IT Support") joins the same meeting. On B: `uv run python demo/agent_caller.py
   demo/audio/agent_lines --device "<B's virtual mic or speaker>"` (render the lines once with
   `demo/render_agent.py`, or play any consenting-voice clips).
3. A human on B speaks first (green light), then B plays the agent lines (red light). On A, type the fake code in
   any text box: the readout rows fill. Toggle the shield on the dashboard.
4. If key timing looks off (readouts wrong even with the shield off), set `[devices] key_offset_s` in
   `callguard.toml` (try +0.02 to +0.08 s).

## When something breaks
| symptom | do this |
|---|---|
| Wi-Fi or Zoom down | Use replay mode. It needs no network. |
| Dashboard blank / "reconnecting…" | The server isn't running, or the port is taken: re-run with `--port 8766` and open that URL. |
| Red alarm banner "Driver … failed" | The audio keeps flowing (the driver is quarantined). Restart CallGuard; switch to `--drivers mock` for a UI-only demo. |
| `scenario 'ai_caller' audio missing` | Run `uv run python demo/build_scenario_audio.py` (needs the sibling repos). |
| First start is slow | Model load + the one-time checkpoint hash. Start CallGuard before the judges arrive. |
| No sound in replay | Check the default Windows output device; `--mute` runs it silently. |
