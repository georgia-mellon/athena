# Google Meet connector

Gets a Google Meet call's audio into Athena and the processed mic back out, without a plugin, a bot or a virtual
audio cable. Used by `athena app` and `athena run --mode meet`. User setup: [docs/meeting_setup.md](../../../../docs/meeting_setup.md).

| file | what it does |
|---|---|
| `launcher.py` | `open_tab(url)` (Join), `show_extension()`. Also, for the headless end-to-end tests only: starts Chrome (or Edge; `ATHENA_BROWSER` overrides) with a dedicated profile outside the repo (`%LOCALAPPDATA%\Athena\meet-profile`; `~/Library/Application Support/...` on macOS, `~/.local/share/...` on Linux; `profile_dir` overrides) and DevTools on `127.0.0.1`, port 0. Every page gets `bridge.js` via `addScriptToEvaluateOnNewDocument`, before Meet's own scripts run; auto-attach covers tabs opened later. Main-frame navigations are held (Fetch) and classified: Meet and local pages get `Page.setBypassCSP`, Google sign-in pages load without it, anything else (a link from Meet's chat) opens in the system browser. Local Network Access is granted to `https://meet.google.com` only; a browser that can't grant it per origin is relaunched with LNA checks off (`MeetSession.lna` = `grant` / `flag`, logged). |
| `extension/` | Chrome MV3 extension (load unpacked, once): injects `extension/bridge.js` into `https://meet.google.com/*` at `document_start` in the page's world and removes Meet's CSP header (declarativeNetRequest) so the bridge can reach `ws://127.0.0.1:8765`. **Join** opens the Meet link as a normal tab in the user's own Chrome (`launcher.open_tab`). |
| `extension/bridge.js` | In the page: wraps `getUserMedia` and `RTCPeerConnection`. Mic → AudioWorklet (16 kHz, 320-sample float32 blocks) → `ws /meet/mic` → processed blocks back → jitter buffer (60 ms, +20 ms after each underflow up to 150 ms, easing back after 10 s clean) → the track Meet sends. With several mics open (Meet's preview + the call), the one a peer connection sends is processed. Every remote audio track → mixed → `ws /meet/far`. Video untouched. Activates only on meet.google.com and local pages. |
| `router.py` | FastAPI side: `/meet/mic` (answers each block with exactly one processed block, in order), `/meet/far`, `/meet/testroom`, `/meet/static/{bridge.js,testroom.html}`, `/meet/audio` (the demo WAVs for the test room). `make_router(pipeline, port=None)`: `port` is the server's own (default `pipeline.meet_port`, else `cfg.server.port`). |
| `testroom.html` | A local "meeting": your mic through the same bridge, then over a real WebRTC loopback to a fake remote participant who can play demo clips. Listen to and record what the room hears; Arm / Disarm / Auto / Allow the Secret Shield. **Play the 10 test voices** plays `demo/audio/testclips/{real,ai}_N_*.wav` (5 real + 5 AI held-out Hearsay clips, gitignored) as the caller over one open WebRTC line with 5 s of silence between them and tables Hearsay's verdict per clip (mean p over its windows, from `/ws`). **Judge my voice** sends your raw mic (no echo cancellation / noise suppression / AGC, not through the shield) as the caller; verdict = mean of the last 3 windows. |

## Guarantees
- **Fail open.** Whenever Athena isn't answering (socket down, more than 15 % of the last 500 ms missing, audio
  context blocked) the bridge sends the raw mic and reconnects with backoff. If the audio context is suspended
  mid-call and won't resume, the call's RTP sender is switched to the raw mic until it runs again
  (`stats.ctxSuspended`, `stats.rawSwaps`). A block of the wrong size is echoed unchanged.
- **No jumps without a fade.** Every raw ↔ processed switch is a 10 ms crossfade. The processed stream runs ~0.5 s
  behind the raw one (the secret delay line), so a switch still skips or repeats that much audio.
- **No stale audio.** Each new `/meet/mic` owner flushes the delay line and resets the shield, so audio from before a
  reconnect never comes out after it. The price: when protection starts (first connect, or after a reconnect) the
  call hears a one-off ~0.5 s hole (the empty delay line) and repeats the ~0.1 s it already sent raw.
- **Origin check.** `/meet/*` sockets accept only `https://meet.google.com`, pages this server serves (a local
  host on its own port), or no Origin (a local process). Another website, or another local server, cannot listen to
  or inject into your mic.
- **One tab per stream.** A second tab is closed with 1013 and passes its mic through raw (backing off, up to
  10 s) until the first leaves, so two tabs never interleave blocks in the shield's timeline. Exception: Meet takes a
  stream over from a local page (a real meeting beats a stray test-room tab). `meet.state.owner` is the origin of the
  page being protected.
- **Key timing.** Keystrokes are placed in the mic stream from the blocks' arrival times, jitter removed (lower
  envelope over 3 s). The constant capture + socket latency left over is `[devices] meet_key_offset_s`
  (default 0; + if clicks land later than their OS timestamps, typically 0.02-0.08).
- The bridge logs only `console.debug` stats, never audio or text.

Wire format both ways: raw little-endian float32 mono 16 kHz, one 20 ms block per binary message.

## Tests
`tests/test_meet_connector.py`: the router over TestClient (order, size, origins and ports, second tab, Meet
takeover, delay-line flush on reconnect, owner in meet.state, redaction with a manual Arm), the key-anchor fit, and
headless Chrome end to end against the test room (skips without Chrome/Edge or with `ATHENA_SKIP_BROWSER=1`).
The real meet.google.com check is opt-in: `ATHENA_MEET_ONLINE=1`.
