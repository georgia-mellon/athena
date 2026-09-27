# Meeting setup: Google Meet

Athena joins Google Meet in **your own Chrome**, through a small Chrome extension that connects Meet's audio to the
Athena app on this computer: no bot, no virtual audio cable ([app/source/connectors/meet](../app/source/connectors/meet/README.md)).

```
your Chrome tab on meet.google.com (the Athena extension puts bridge.js in the page before Meet's scripts)
  your mic ─► ws://127.0.0.1:8765/meet/mic ─► Athena (Keystroke Guard shield + Secret Shield delay line) ─► the track Meet sends
  remote participants ─► ws /meet/far ─► Athena (Hearsay + "read me the code" listener)
```

## 1. Install the extension (once)
1. Chrome (or Edge) → `chrome://extensions` → turn on **Developer mode**.
2. **Load unpacked** → pick `app/source/connectors/meet/extension` in this repo (the dashboard's Meeting bar can open
   the folder for you: `POST /api/control/meet {"action": "extension"}`).
3. The first time a Meet page connects, Chrome may ask to let **meet.google.com access devices on your local network**:
   allow it (that is Athena on 127.0.0.1).
4. Extension **Details → Site access: "On specific sites" (meet.google.com)**, not "On click": with "On click" the
   bridge only starts after you click the extension icon, too late to see the call. Reload the Meet tab after
   installing or changing this.

The extension only runs on `https://meet.google.com/*`; it removes Meet's Content-Security-Policy header there (so the
bridge may open its local socket and audio worklet), the same thing the old automated window did.

## 2. Start
```
uv run athena app                           # desktop window; Join opens the Meet link in your Chrome
uv run athena app --meet-url abc-defg-hij   # or open this meeting at start
uv run athena run --mode meet               # same engine, dashboard in your normal browser
```
- Athena must run on port **8765** (the default): that is where the extension connects.
- The window opens at once and shows what it is loading; the dashboard appears when the models are warm (the header
  pill then reads **ready**). A Meet tab opened earlier connects within ~2 s of that.
- The **meeting pill** is what the extension reports from the Meet tab (`/meet/status`), not what the buttons did:
  `no Meet tab` → `Meet open · not in a call` → `in call · mic ✓ caller ✓` (green once both streams flow).
- **Join** opens the link (or sends your open Meet tab there). **Leave** leaves the call: the extension clicks Meet's
  own "Leave call" button. It is disabled while no Meet tab is connected.
- Without the extension, Meet runs unprotected and the pill reads `no Meet tab`.
- The **Events** log records what the extension reports: `Meet tab connected`, `joined the call (Meet)`,
  `left the call (Meet)`, `Meet tab closed or disconnected` (bus topic `meet.call`, so hooks can log it too).
  "In a call" = a live WebRTC connection or live caller audio in the tab, re-checked every second.
- **Whose voice is judged** (Caller voice panel: Caller | My mic): Caller is the product (the other people in the
  call). My mic judges your own mic instead, to test alone: e.g. play an AI voice from your phone into your laptop's
  mic during a Meet. Switching starts a fresh voice history.
- After updating Athena, reload the extension (`chrome://extensions` → reload) and the Meet tab.
- **Audio in** (Caller voice panel): live mic and caller levels (dBFS). The line on the caller meter is the **speech
  gate**: a 4 s window is judged only when at least half of it is louder (default -45 dBFS). Drag it down if a quiet
  caller never gets judged; the line under the meter says what happened to the last window.

## 3. Meet settings
⋮ → Settings → Audio:
- **Microphone:** your real mic (the bridge sits behind Meet's device choice).
- **Speaker:** headphones. A speaker bleeding into the mic puts the caller's voice in the attacker's view too.
- **Noise cancellation: off.** It removes key clicks and hides what the shield does.

## 3. Check it
- The Meeting bar pill: `mic ✓ far ✓`. The Pipeline panel shows the outgoing latency (shield + secret delay).
- The local test room, <http://127.0.0.1:8765/meet/testroom> (meet mode only): the same bridge, a local WebRTC
  "room" with one fake caller. Join with mic, tick *listen to what the room hears*, Arm the Secret Shield, read a
  fake code: you hear a tone where the digits were.
- With the second device in a real meeting: what it hears is what Meet received from you.

## 4. Latency
Your outgoing voice is delayed by about 0.6 s in total: a ~60 ms jitter buffer in the bridge, the 80 ms Keystroke
Guard lookahead, and the 500 ms Secret Shield delay line. Set `[secret] enabled = false` in `athena.toml` if a
call feels laggy; the other two pillars keep working.

## 5. Troubleshooting
| symptom | fix |
|---|---|
| "Chrome or Edge not found" | Install Chrome, or set `ATHENA_BROWSER` to the browser's `.exe`. |
| "didn't open a DevTools port" | An Athena Meet window is already open on that profile. Close it (or Leave) and Join again. |
| Pill stays on `joining…` or shows `✗` | The bridge isn't connected. Use Join from the dashboard (not a Meet tab in your normal browser); reload the Meet tab. |
| Meet shows a marketing page | You're signed out. Open a meeting link or code directly, or sign in once in Athena's window. |
| Voice light stays grey | No far-end speech yet. Someone else has to talk in the meeting. |
| Keystroke readouts look shifted | Set `[devices] meet_key_offset_s` (Meet; `key_offset_s` for virtual devices) in `athena.toml` (+0.02 to +0.08 s; positive = clicks land later than their OS timestamp). |
| Keystrokes not timed | pynput needs a desktop session; elevated (admin) windows hide keys from a non-admin listener. |
| Athena crashes mid-call | Meet keeps working: the bridge falls back to your raw mic (fail open) and reconnects when Athena is back. |

---

## Appendix: other apps (Zoom, Teams) via virtual devices
For meeting apps other than Meet, Athena works at the audio-device layer (`athena run --mode live`, plan 03):

```
physical mic ─► Athena (shield + delay line) ─► "CABLE Input"  ══ VB-CABLE ══►  "CABLE Output" = the app's microphone
the app's speaker (your headphones) ─► WASAPI loopback ─► Athena (Hearsay)
```

**Install VB-CABLE (Windows, once)**
1. Download the **VBCABLE_Driver_Pack** zip from https://vb-audio.com/Cable/ and unzip it (don't run it from inside the zip).
2. Right-click **VBCABLE_Setup_x64.exe** → **Run as administrator** → **Install Driver**.
3. **Reboot.** The devices don't appear to other apps until you do.
4. Check: `uv run athena devices` shows `virtual mic out: CABLE Input (VB-Audio Virtual Cable)`.
5. Keep CABLE Input a *non-default* output. If Windows makes it the default speaker, set your headphones back, or
   you'll hear nothing.

**Run:** `uv run athena run --mode live [--mic "<name>"] [--speaker "<name>"]`.

**App settings**
- **Zoom:** Settings → Audio → Microphone = `CABLE Output`, Speaker = headphones; uncheck *Automatically adjust
  microphone volume*; Suppress background noise = **Low**.
- **Teams:** Settings → Devices → Microphone = `CABLE Output`, Speaker = headphones; Noise suppression = **Low** or **Off**.
- **macOS:** BlackHole 2ch (`brew install blackhole-2ch`) plays the role of CABLE. macOS has no WASAPI loopback:
  route the meeting speaker through a Multi-Output Device with a second BlackHole and point `--speaker` at it. Grant
  the terminal Microphone and Input Monitoring permissions.

| symptom | fix |
|---|---|
| `virtual mic out: MISSING` | You didn't reboot, or the installer wasn't run as admin. |
| The app hears nothing | Its mic must be **CABLE Output** (not Input), and Athena must be running. |
| `loopback speakers: NONE` | Update the audio driver, or use a second virtual cable as the meeting speaker and pass it as `--speaker`. |
| Echo | Headphones; don't monitor CABLE Output via "Listen to this device". |
