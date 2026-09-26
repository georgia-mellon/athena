# Meeting setup: Zoom (and Meet / Teams)

CallGuard works at the audio-device layer (plan 03), so the meeting app needs no plugin:

```
physical mic ─► CallGuard (shield) ─► "CABLE Input"  ══ VB-CABLE ══►  "CABLE Output" = Zoom microphone
Zoom speaker (your headphones) ─► WASAPI loopback ─► CallGuard (voice check)
```

Run `callguard devices` (or `python -m callguard.audio.devices`) at any point. It prints the mic it will use, the
virtual mic, and the loopback speakers, and tells you what's missing.

## 1. Install VB-CABLE (Windows, once)
1. Download the **VBCABLE_Driver_Pack** zip from https://vb-audio.com/Cable/ and unzip it (don't run it from inside the zip).
2. Right-click **VBCABLE_Setup_x64.exe** → **Run as administrator** → **Install Driver**.
3. **Reboot.** The devices don't appear to other apps until you do.
4. Check: Sound settings now lists **CABLE Input** (output) and **CABLE Output** (input). `callguard devices`
   should show `virtual mic out: CABLE Input (VB-Audio Virtual Cable)`.
5. Leave CABLE Input as a *non-default* output. If Windows makes it the default speaker, set your real
   headphones/speakers back as the default, or you'll hear nothing.

Without VB-CABLE, CallGuard still runs: it analyses the mic and drives the dashboard, but nothing reaches the meeting
(it prints the install steps on start).

## 2. Zoom
Settings → Audio:
- **Microphone:** `CABLE Output (VB-Audio Virtual Cable)`.
- **Speaker:** your real headphones or speakers. This is the device CallGuard loopback-captures, so pass
  `--speaker "<name>"` if it isn't the Windows default.
- Uncheck **Automatically adjust microphone volume**.
- **Suppress background noise → Low.** On Auto/High, Zoom's own denoiser removes keystrokes and masks the shield
  demo.
- Optional, for the cleanest demo audio: Advanced → **Show in-meeting option to enable "Original sound for
  musicians"** and turn it on during the call.
- Use **Test Mic**. When you type, the level meter should move, and with the shield on the recording should
  play back without crisp key clicks.

Use headphones. A speaker bleeding into the mic makes the far-end voice show up in the attacker's view too.

## 3. Google Meet / Microsoft Teams
It's the same routing, so only the settings screen differs:
- **Meet** (browser): ⋮ → Settings → Audio → Microphone = `CABLE Output`, Speaker = your headphones. Chrome may
  need a page reload after the driver install. Meet's noise cancellation (Settings → Audio → Noise cancellation) should be **off**.
- **Teams:** Settings → Devices → Microphone = `CABLE Output`, Speaker = your headphones; Noise suppression =
  **Low** or **Off**.

## 4. macOS
Install **BlackHole 2ch** (`brew install blackhole-2ch`, reboot or restart coreaudiod). BlackHole plays the role
of CABLE, and the meeting app's mic = `BlackHole 2ch`. macOS has no WASAPI loopback. For inbound capture, create a
**Multi-Output Device** (Audio MIDI Setup) of your headphones + a second BlackHole (16ch), set the meeting's
speaker to it, and point CallGuard at that BlackHole as the far-end source. Give the terminal **Microphone** and
**Input Monitoring** (for key timing) permissions in System Settings → Privacy & Security.

## 5. Troubleshooting
| symptom | fix |
|---|---|
| `virtual mic out: MISSING` after install | You didn't reboot, or the installer wasn't run as admin. Reinstall as admin, then reboot. |
| Zoom hears nothing | Zoom mic must be **CABLE Output** (not Input). Check that CallGuard is running and that `callguard devices` shows the cable. |
| You hear nothing | CABLE Input became the default Windows speaker. Set your headphones back as the default. |
| Robotic or choppy outgoing audio | Close other audio apps. Check the audio-device warnings in the `callguard` console. Try another host API with `--mic "<name>"` (MME is the most tolerant). |
| `loopback speakers: NONE` | `soundcard` can't see WASAPI loopback. Update the audio driver, or set the meeting speaker to a second virtual cable and pass it as `--speaker`. |
| Voice light stays grey | There's no far-end speech in the loopback. Make sure the meeting plays to the speaker CallGuard captures (`--speaker`). |
| Keystrokes not timed (no key events) | pynput needs a desktop session. Some elevated apps (admin windows) hide keys from a non-admin listener, so run from a normal terminal and type into a normal window. |
| Attacker offsets look shifted | Tune `KeyClock.offset_s` (seconds; positive = the clicks land later in the audio than their OS timestamp). |
| Echo or feedback | Use headphones, and don't also monitor CABLE Output through "Listen to this device". |
