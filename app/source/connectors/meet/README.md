# Google Meet connector

Gets a Google Meet call's audio into CallGuard and the processed mic back out, without a plugin, a bot or a virtual
audio cable. Used by `callguard app` and `callguard run --mode meet`. User setup: [docs/meeting_setup.md](../../../../docs/meeting_setup.md).

| file | what it does |
|---|---|
| `launcher.py` | Starts Chrome (or Edge; `CALLGUARD_BROWSER` overrides) with a dedicated profile `runs/meet-profile` and `--remote-debugging-port=0`. Over the DevTools protocol, every page gets `Page.setBypassCSP` and `bridge.js` via `addScriptToEvaluateOnNewDocument`, before Meet's own scripts run; auto-attach covers tabs opened later. Only Meet links and local pages are opened. |
| `bridge.js` | In the page: wraps `getUserMedia` and `RTCPeerConnection`. Mic → AudioWorklet (16 kHz, 320-sample float32 blocks) → `ws /meet/mic` → processed blocks back → ~60 ms jitter buffer → the track Meet sends. Every remote audio track → mixed → `ws /meet/far`. Video untouched. Activates only on meet.google.com and local pages. |
| `router.py` | FastAPI side: `/meet/mic` (answers each block with exactly one processed block, in order), `/meet/far`, `/meet/testroom`, `/meet/static/{bridge.js,testroom.html}`, `/meet/audio` (the demo WAVs for the test room). |
| `testroom.html` | A local "meeting": your mic through the same bridge, then over a real WebRTC loopback to a fake remote participant who can play demo clips. Listen to and record what the room hears; Arm / Disarm / Auto / Allow the Secret Shield. |

## Guarantees
- **Fail open.** Whenever CallGuard isn't answering (socket down, stalled, audio context blocked) the bridge sends the
  raw mic and reconnects with backoff. A block of the wrong size is echoed unchanged.
- **Origin check.** `/meet/*` sockets accept only `https://meet.google.com`, local pages, or no Origin (a local
  process). Another website in the same browser cannot listen to or inject into your mic.
- **One tab per stream.** A second tab is closed with 1013 and passes its mic through raw until the first leaves,
  so two tabs never interleave blocks in the shield's timeline.
- The bridge logs only `console.debug` stats, never audio or text.

Wire format both ways: raw little-endian float32 mono 16 kHz, one 20 ms block per binary message.

## Tests
`tests/test_meet_connector.py`: the router over TestClient (order, size, origins, second tab, redaction with a
manual Arm), and headless Chrome end to end against the test room (skips without Chrome/Edge or with
`CALLGUARD_SKIP_BROWSER=1`). The real meet.google.com check is opt-in: `CALLGUARD_MEET_ONLINE=1`.
