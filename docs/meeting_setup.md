# Meeting setup: Google Meet

CallGuard joins Google Meet through **its own Chrome window**: no plugin, no bot, no virtual audio cable
([app/source/connectors/meet](../app/source/connectors/meet/README.md)).

```
CallGuard's Chrome window (profile runs/meet-profile, bridge.js injected before Meet's scripts)
  your mic ─► ws /meet/mic ─► CallGuard (Keystroke Guard shield + Secret Shield delay line) ─► the track Meet sends
  remote participants ─► ws /meet/far ─► CallGuard (Hearsay + "read me the code" listener)
```

## 1. Start
```
uv run callguard app                           # desktop window; Join opens Meet
uv run callguard app --meet-url abc-defg-hij   # or open this meeting at start
uv run callguard run --mode meet               # same engine, dashboard in your normal browser
```
- Needs **Chrome or Edge** installed (Chrome first). Another path: set `CALLGUARD_BROWSER` to the browser's `.exe`.
- The window uses a dedicated profile (`runs/meet-profile`, gitignored). The first time, **sign in to Google** there
  or join as a guest. Your everyday Chrome profile is never touched.
- The window only opens Meet links (`https://meet.google.com/...` or a code like `abc-defg-hij`) and local pages.
- Join / Leave and the link box are on the dashboard's **Meeting** bar. Its pill reads `in meeting: mic ✓ far ✓`
  and turns green once both bridge streams are connected (hover it for the bridge round trip).

## 2. Meet settings (in CallGuard's window)
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
Guard lookahead, and the 500 ms Secret Shield delay line. Set `[secret] enabled = false` in `callguard.toml` if a
call feels laggy; the other two pillars keep working.

## 5. Troubleshooting
| symptom | fix |
|---|---|
| "Chrome or Edge not found" | Install Chrome, or set `CALLGUARD_BROWSER` to the browser's `.exe`. |
| "didn't open a DevTools port" | A CallGuard Meet window is already open on that profile. Close it (or Leave) and Join again. |
| Pill stays on `joining…` or shows `✗` | The bridge isn't connected. Use Join from the dashboard (not a Meet tab in your normal browser); reload the Meet tab. |
| Meet shows a marketing page | You're signed out. Open a meeting link or code directly, or sign in once in CallGuard's window. |
| Voice light stays grey | No far-end speech yet. Someone else has to talk in the meeting. |
| Keystroke readouts look shifted | Set `[devices] key_offset_s` in `callguard.toml` (+0.02 to +0.08 s; positive = clicks land later than their OS timestamp). |
| Keystrokes not timed | pynput needs a desktop session; elevated (admin) windows hide keys from a non-admin listener. |
| CallGuard crashes mid-call | Meet keeps working: the bridge falls back to your raw mic (fail open) and reconnects when CallGuard is back. |

---

## Appendix: other apps (Zoom, Teams) via virtual devices
For meeting apps other than Meet, CallGuard works at the audio-device layer (`callguard run --mode live`, plan 03):

```
physical mic ─► CallGuard (shield + delay line) ─► "CABLE Input"  ══ VB-CABLE ══►  "CABLE Output" = the app's microphone
the app's speaker (your headphones) ─► WASAPI loopback ─► CallGuard (Hearsay)
```

**Install VB-CABLE (Windows, once)**
1. Download the **VBCABLE_Driver_Pack** zip from https://vb-audio.com/Cable/ and unzip it (don't run it from inside the zip).
2. Right-click **VBCABLE_Setup_x64.exe** → **Run as administrator** → **Install Driver**.
3. **Reboot.** The devices don't appear to other apps until you do.
4. Check: `uv run callguard devices` shows `virtual mic out: CABLE Input (VB-Audio Virtual Cable)`.
5. Keep CABLE Input a *non-default* output. If Windows makes it the default speaker, set your headphones back, or
   you'll hear nothing.

**Run:** `uv run callguard run --mode live [--mic "<name>"] [--speaker "<name>"]`.

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
| The app hears nothing | Its mic must be **CABLE Output** (not Input), and CallGuard must be running. |
| `loopback speakers: NONE` | Update the audio driver, or use a second virtual cable as the meeting speaker and pass it as `--speaker`. |
| Echo | Headphones; don't monitor CABLE Output via "Listen to this device". |
