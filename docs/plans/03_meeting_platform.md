# 03: Meeting platform: Zoom, through virtual audio devices

## Decision
**Primary: Zoom, integrated at the audio-device layer** (a virtual microphone plus loopback capture of the meeting's
output). The same setup works unchanged in Google Meet and Microsoft Teams. That is the demo line: "works with any
meeting app".

## Why not the platforms' bot/media APIs (for the hackathon)
| option | real-time audio access | blocker for us |
|---|---|---|
| Zoom RTMS (Realtime Media Streams) | server-side audio streams per meeting | app must be approved/enabled on the account; needs a public HTTPS endpoint; venue network risk |
| Zoom Meeting SDK bot | raw audio in a headless bot | Linux bot build, SDK credentials, a second participant in the call |
| Google Meet Media API | real-time audio | developer-preview enrollment only |
| Teams bots (Graph Communications media) | raw audio | Azure bot registration, C#/.NET media SDK on Windows Server |
| **Virtual devices (chosen)** | everything local, both directions | a one-time driver install (VB-CABLE) |

The device layer is also the only place the **shield** can act: it must modify *your outgoing* audio before the
meeting app encodes it. A server-side bot can listen, but it can't clean your mic.

## Routing on Windows (this laptop)
- **Outbound:** physical mic → Athena (shield) → **VB-CABLE "CABLE Input"** (playback device). In Zoom:
  Microphone = **"CABLE Output (VB-Audio Virtual Cable)"**, and turn **off** Zoom's "Suppress background noise"
  (set it to Low) so Zoom's own processing doesn't mask the effect.
- **Inbound:** Zoom Speaker = the normal headphones/speakers. Athena captures that device through **WASAPI
  loopback** (the `soundcard` library, `include_loopback=True`). Fallback: set the Zoom speaker to a second virtual
  cable, and Athena captures it and plays it through to the headphones.
- macOS: BlackHole 2ch replaces VB-CABLE (Keyguard's `realtime.py` already targets it); loopback = a Multi-Output
  Device.
- `athena devices` lists the devices and checks the routing; `docs/zoom_setup.md` has screenshots and steps.

## The "AI agent" caller in the demo
A second device (phone or laptop) joins the Zoom call as "IT Support". `demo/agent_caller.py` plays pre-rendered
lines from an open-source TTS into that device's virtual mic. The lines can use a clone of a *consenting*
teammate's voice (Hearsay's sim TTS stack). Pre-rendering keeps the demo offline-safe. A live LLM-driven agent is a
stretch goal. The same WAV lines feed `replay` mode.

## Later (after the hackathon)
Zoom RTMS / Meet Media API as a **listen-only** cloud tier: Hearsay scores all participants for an organization.
The shield always stays on the device.
