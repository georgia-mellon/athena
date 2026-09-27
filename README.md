# Athena

## Inspiration (the problem)
Two scary things happened in 2023.
Voice cloning got good enough to fool your ears.
And Harrison et al. (*A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*) showed a phone next to a MacBook reads 93 % of keystrokes solely from audio over a Zoom call, because every key resonates differently on the plate.
We built solutions to both of these problems this weekend. Then, we realized their common mission (to act as a sort of bodyguard for audio/video calls), integrated them into the same product, and Athena was born.
Imagine you get the modern social-engineering call: a fake AI voice clone of a colleague joins your Google Meet, asks you to log in to "confirm your identity", and then says *"just read me the reset code"*. As you type, your keyboard leaks the password over the call audio thanks to an agent on the other end recording your keystrokes' audio. You don't realize that you're speaking with a bot, so you verbally leak the code. Nothing on the call stops any of it.
We wanted one system that closes all three channels at once, on your own machine, on a real meeting platform, with a clear metric that says how much trouble you are in. So we built the ultimate AI bodyguard system, fit for this age of deepfake calls and acoustic keylogging algorithms.

### What it does
Athena is a desktop agent that sits between you and Google Meet. It opens Meet in its own Chrome window and quietly injects an audio bridge, so the mic Meet publishes is the one Athena has already shielded and redacted. Three pillars run on the two audio streams:

**What you hear:** Our NSA HEARSAY challenge model scores every 4 s of the caller's voice real-vs-synthetic: two fine-tuned XLS-R models plus a gradient-boosted model on classic spectral features, fused. Official organizer score: minDCF 0.058, EER 2.5 %; 10 of 10 held-out voices in our Meet test room. Because Meet's gain control and Opus shift scores, we calibrate through an emulation of that path. The smoothed output is the voice risk V that arms the other two pillars.

**What you type:** Harrison published a dataset but not their attacker model, and the only known defences play noise over the call. We wanted a defence we could *prove*, which needs a live adversary that keeps adapting, so we built the attacker algorithm from scratch in less than 12 hours: public recordings, our own keyboard through a laptop mic, and a synthesis engine that overlap-adds real clicks with exact labels.
Isolated keys were easy (87 % top-1). Fast, overlapping typing is what the paper couldn't do, so we treated typing as speech recognition: a CNN + BiGRU transcriber with CTC decoding, trained through blank-collapse with a per-frame onset head, an entropy bonus, and a slow-to-fast curriculum. It reads real free typing at 80 % per key, and Gemini turns the noisy read `MY PASSPLRS IS HUNT EFI2` into "my password is hunter2". We're proud of it, but it is a means to an end: the defender, **Athena**, whose LLM picks the sensitive span, invents a coherent decoy, and optimizes an inaudible perturbation that makes the attacker read the decoy while speech stays intact (STOI 0.98). The attacker, **Ares**, retrains on the shielded audio and breaks through; Athena escalates; both remember rounds through Backboard.
A frozen shield gets clawed back to ~56 %; a re-solving one blinds the attacker to chance (2.8 %) every round. In Athena's dashboard, each burst of your real typing becomes a live match on the dashboard, while a streaming 80 ms shield protects the audio Meet is sending. The attacker on your raw mic is your exposure; on the shielded stream it is the residual leak L.

**What you say:** While the caller is unverified, or right after they ask for a code, your outgoing voice passes through a 500 ms delay line and a streaming recognizer (Vosk) spots digit runs, passwords and card numbers; the redactor tones them out before Meet hears them. It reports *where*, never *what*: the recognized words never leave the spotter. A full-stack test blocked a spoken 6-digit code with about 1 s leaked at its start.

These three signals are fused into one threat score:

$$\text{threat} = 100\,\bigl(1 - (1 - w_v V)(1 - w_l\, L\, T_{on})\bigr)$$

with a boost when an unverified voice speaks *while* you type. The dashboard shows SAFE / WATCH / WARN / CRITICAL, the attacker's guesses next to what you really typed, and a live **Ares vs Athena** panel where KeyGuard's attacker and defender agents fight over each burst of your typing.

## How we built it
- **Runtime.** Python 3.12, one audio pipeline with a lock-free event bus. Every model runs as a *driver* behind a small contract, off the audio thread; if a driver throws or falls behind it is quarantined and the audio passes through unchanged. A call must never glitch because a model hiccupped.
- **Google Meet connector.** Athena launches Chrome with a dedicated profile and, through the DevTools protocol, injects a script before Meet's own scripts load. That script wraps `getUserMedia` and the incoming WebRTC tracks, and streams your mic and the far end to Athena over two local WebSockets. The processed mic (shield + secret delay) is what Meet sends. A local WebRTC test room lets you hear exactly what the room hears.
- **Voice detector.** Two fine-tuned XLS-R passes plus a LightGBM on classic spectral features, fused by logistic regression. Calibrated on held-out data so 0.5 means "at the threshold"; Athena runs it stricter (AI when $p \ge 0.7$). ~1.4 s per 4 s window on a laptop CPU.
- **Keystroke attacker and shield.** The attacker is Gemini armed with a CNN + BiGRU with CTC decoding over 8 ms log-mel frames, 37 keys, with a per-frame onset head for reasoning; the streaming shield is also Gemini and a DSP defence with a constant 80 ms lookahead; the adversarial shield is a bounded perturbation optimized against the attacker network under a perceptual constraint. Key timing comes from the OS keyboard hook and a clock that maps it onto mic samples.
- **Secret Shield.** Vosk small English with low-latency partial timings, a span protocol, and a delay-line redactor that counts leaked samples honestly.
- **Agents.** Ares and Athena reason with Gemini 2.5 Flash and remember across rounds through Backboard; a rule-based offline route keeps the demo running without keys.
- **Dashboard.** FastAPI + WebSocket + a static UI with no CDNs, wrapped in a pywebview desktop window.
- **Evidence.** Every claim on the dashboard has an offline, seeded experiment behind it, and every pillar has a harness that checks any driver (mock or real) against its contract and its real-time budget. 90 tests run with mock drivers and no audio devices.

## Challenges we ran into
**Building the attacker from nothing.** Harrison's paper came with a dataset and no model, and the moment we trained on it we hit the wall: a keystroke attacker does not transfer. Zero-shot on any other keyboard it reads 0 %. We pulled every labeled keystroke-audio set we could find (Harrison, MKA's four laptops, two Zenodo sets, SKAID's IRB-approved free-typing sessions), pooled eight keyboards and trained one attacker. Held-in it hit 43.7 % top-3; leave-one-keyboard-out it fell to 6.4 %, below chance. Per-key acoustics are device-specific, and a keyboard-agnostic attacker is an open research problem, so we faded it and banked our own keyboard instead. That meant writing a capture tool that records the mic while a keylogger stamps every press with its identity and time (our own machine, consenting typists, fake passwords), tapping all 36 keys about twenty times each, and discovering that the default mic was an AirPod, which gave us a bank of silence. On the laptop mic the isolated-key attacker landed at 38 % top-1: signal, not code.

Free typing was harder, and we made significant progress, beyond Harrison et al., on the overlapping typing problem. We recorded 33 continuous sessions of ourselves typing while talking in the background, and found the real failure was onset detection, not classification: at speed, clicks land 24 ms apart on a 120 ms ring-down, hand-tuned detectors got either 56 % recall or 31 % precision, and every missed onset is a guaranteed deletion. Plain CTC then mode-collapsed and emitted the same letters for any input. The fix was a per-frame onset head trained jointly with CTC, an entropy bonus, a slow-to-fast curriculum, and adding space to the vocabulary, which broke every checkpoint. Since the isolated bank had no space clips, we built a "rich bank" by cutting per-key clips out of the real free-typing sessions at their labeled onsets and overlap-adding them into synthetic phrases in the Mac's own key sounds. The result reads novel text it never saw at 80 % per letter, `PASSWORD IDEAS` as `PASSWORD IDEAS`, and `HUNTER2 SUNFLOWER99` verbatim at the onsets.
- **Proving the reverse direction.** Our voice detector had shown that typing in the background doesn't break voice detection. We had to show the opposite: that an attacker still reads keys with someone talking over them, otherwise the shield defends nothing. A naive attacker collapses to chance under speech. An attacker trained *with* speech mixed in does not: at speech 10 dB louder than the keys it still gets 15.8 % top-1 (chance 2.8 %), and the shield pushes it back to 5.8 % while speech intelligibility stays at STOI 0.90. That was the experiment that made the project real.
- **Getting into Meet without a bot.** Meet has no audio API. Virtual cables need a driver install and a reboot. Injecting a bridge before Meet's scripts, and making the processed track the one Meet publishes, took most of a night, and the wrong order of `getUserMedia` hooks silently gives you an unshielded mic.
- **Latency budget.** Your outgoing voice is delayed about 0.6 s in total (jitter buffer + 80 ms shield + 500 ms secret delay). The secret spotter has to place a span inside that 500 ms window or the first digit leaks. At 100 ms of worker lag the leak rate rises measurably, so the pipeline feeds it every 20 ms while armed.
- **Sample clocks.** Three components each assumed a different key-time convention (offsets in the block, absolute since reset, OS timestamps). Reconciling them was the single largest source of bugs; spaces turned out to be keystrokes too, which shifted every secret span until we found it.
- **Compute.** A full Ares-vs-Athena match with default settings takes over ten minutes on a laptop. We exposed the step count so a live match takes seconds.

## What we learned
- Detectors that are great on datasets need calibration for the room: Meet's AGC, Opus, and 48 kHz playback shift scores, so we score through an emulation of that path.
- A fixed filter can't beat an adapting attacker, hence the need for LLMs here. The optimal defence depends on the current attacker, and only optimization finds it.
- Signal strength is the attack's wall, not code: a laptop mic and quiet keys, and the transcriber falls to chance on keyboards it wasn't tuned for.
- Redaction should say *where*, never *what*. Designing the spotter so the secret never leaves it made the privacy story simple.

## Quickstart
<!-- TEMPLATE: prerequisites (Windows, Python 3.12, uv, Chrome/Edge), install, get the Vosk model, run the desktop app, join a Meet, the test room, the offline replay demo. Commands in one code block. -->

## Demo
<!-- TEMPLATE: the replay story timeline and the live Meet demo; link docs/demo_runbook.md. Screenshot/GIF of the dashboard. -->

## Results
<!-- TEMPLATE: one small table per pillar, numbers only from docs/reports/*.md, with honest limits. -->

## Repository layout
<!-- TEMPLATE: app/source, app/hearsay, app/keystroke_guard, app/secret_shield, dashboard, demo, docs, tests: one line each. -->

## Team and credits
<!-- TEMPLATE: team GeorgiaMellon members; Hearsay (model repository link); Keystroke Guard author; third-party models (Vosk, XLS-R, ...) and licences. -->

## Ethics
<!-- TEMPLATE: fake codes only, consenting voices, nothing leaves the device, no recordings committed. -->
