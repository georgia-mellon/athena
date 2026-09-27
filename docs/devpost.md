# Devpost submission: CallGuard

## Elevator pitch
CallGuard protects what you hear, what you type, and what you say on a video call: it spots AI voices, hides your keystrokes from the microphone, and bleeps the codes you're about to read to a fake caller.

## About the project

## Inspiration
Two scary things happened in 2023. 
Voice cloning got good enough to fool your ears. 
And Harrison et al. (*A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*) showed a phone next to a MacBook reads 93 % of keystrokes solely from audio over a Zoom call, because every key resonates differently on the plate.
We built solutions to both of these problems this weekend. Then, we realized their common mission (to act as a sort of bodyguard for audio/video calls), integrated them into the same product, and Athena was born. 
Imagine you get the modern social-engineering call: a fake AI voice clone of a colleague joins your Google Meet, asks you to log in to "confirm your identity", and then says *"just read me the reset code"*. As you type, your keyboard leaks the password over the call audio thanks to an agent on the other end recording your keystrokes' audio. You don't realize that you're speaking with a bot, so you verbally leak the code. Nothing on the call stops any of it.
We wanted one system that closes all three channels at once, on your own machine, on a real meeting platform, with a clear metric that says how much trouble you are in. So we built the ultimate AI bodyguard system, fit for this age of deepfake calls and acoustic keylogging algorithms.

## What it does
Athena is a desktop agent that sits between you and Google Meet. It opens Meet in its own Chrome window and quietly injects an audio bridge, so the mic Meet publishes is the one Athena has already shielded and redacted. Three pillars run on the two audio streams:

**What you hear:** Our NSA HEARSAY challenge model scores every 4 s of the caller's voice real-vs-synthetic: two fine-tuned XLS-R models plus a gradient-boosted model on classic spectral features, fused. Official organizer score: minDCF 0.058, EER 2.5 %; 10 of 10 held-out voices in our Meet test room. Because Meet's gain control and Opus shift scores, we calibrate through an emulation of that path. The smoothed output is the voice risk $V$ that arms the other two pillars.

**What you type:** Harrison published a dataset but not their attacker model, and the only known defences play noise over the call. We wanted a defence we could *prove*, which needs a live adversary that keeps adapting, so we built the attacker algorithm from scratch in less than 12 hours: public recordings, our own keyboard through a laptop mic, and a synthesis engine that overlap-adds real clicks with exact labels. 
Isolated keys were easy (87 % top-1). Fast, overlapping typing is what the paper couldn't do, so we treated typing as speech recognition: a CNN + BiGRU transcriber with CTC decoding, trained through blank-collapse with a per-frame onset head, an entropy bonus, and a slow-to-fast curriculum. It reads real free typing at 80 % per key, and Gemini turns the noisy read `MY PASSPLRS IS HUNT EFI2` into "my password is hunter2". We're proud of it, but it is a means to an end: the defender, **Athena**, whose LLM picks the sensitive span, invents a coherent decoy, and optimizes an inaudible perturbation that makes the attacker read the decoy while speech stays intact (STOI 0.98). The attacker, **Ares**, retrains on the shielded audio and breaks through; Athena escalates; both remember rounds through Backboard. 
A frozen shield gets clawed back to ~56 %; a re-solving one blinds the attacker to chance (2.8 %) every round. In Athena's dashboard, each burst of your real typing becomes a live match on the dashboard, while a streaming 80 ms shield protects the audio Meet is sending. The attacker on your raw mic is your exposure; on the shielded stream it is the residual leak $L$.

**What you say: Secret Shield.** While the caller is unverified, or right after they ask for a code, your outgoing voice passes through a 500 ms delay line and a streaming recognizer (Vosk) spots digit runs, passwords and card numbers; the redactor tones them out before Meet hears them. It reports *where*, never *what*: the recognized words never leave the spotter. A full-stack test blocked a spoken 6-digit code with about 1 s leaked at its start.

These three signals are fused into one threat score:

$$\text{threat} = 100\,\bigl(1 - (1 - w_v V)(1 - w_l\, L\, T_{on})\bigr)$$

with a boost when an unverified voice speaks *while* you type. The dashboard shows SAFE / WATCH / WARN / CRITICAL, the attacker's guesses next to what you really typed, and a live **Ares vs Athena** panel where KeyGuard's attacker and defender agents (Gemini + Backboard memory) fight over each burst of your typing.

## How we built it
- **Runtime.** Python 3.12, one audio pipeline with a lock-free event bus. Every model runs as a *driver* behind a small contract, off the audio thread; if a driver throws or falls behind it is quarantined and the audio passes through unchanged. A call must never glitch because a model hiccupped.
- **Google Meet connector.** Athena launches Chrome with a dedicated profile and, through the DevTools protocol, injects a script before Meet's own scripts load. That script wraps `getUserMedia` and the incoming WebRTC tracks, and streams your mic and the far end to Athena over two local WebSockets. The processed mic (shield + secret delay) is what Meet sends. A local WebRTC test room lets you hear exactly what the room hears.
- **Voice detector.** Two fine-tuned XLS-R passes plus a LightGBM on classic spectral features, fused by logistic regression. Calibrated on held-out data so 0.5 means "at the threshold"; Athena runs it stricter (AI when $p \ge 0.7$). ~1.4 s per 4 s window on a laptop CPU.
- **Keystroke attacker and shield.** The attacker is a CNN + BiGRU with CTC decoding over 8 ms log-mel frames, 37 keys, with a per-frame onset head; the streaming shield is a DSP defence with a constant 80 ms lookahead; the adversarial shield is a bounded perturbation optimized against the attacker network under a perceptual constraint. Key timing comes from the OS keyboard hook and a clock that maps it onto mic samples.
- **Secret Shield.** Built from scratch this weekend: Vosk small English with low-latency partial timings, a span protocol, and a delay-line redactor that counts leaked samples honestly.
- **Agents.** Ares and Athena reason with Gemini 2.5 Flash and remember across rounds through Backboard; a rule-based offline route keeps the demo running without keys.
- **Dashboard.** FastAPI + WebSocket + a static UI with no CDNs, wrapped in a pywebview desktop window.
- **Evidence.** Every claim on the dashboard has an offline, seeded experiment behind it, and every pillar has a harness that checks any driver (mock or real) against its contract and its real-time budget. 90 tests run with mock drivers and no audio devices.

## Challenges we ran into
**Building the attacker from nothing.** Harrison's paper came with a dataset and no model, and the moment we trained on it we hit the wall: a keystroke attacker does not transfer. Zero-shot on any other keyboard it reads 0 %. We pulled every labeled keystroke-audio set we could find (Harrison, MKA's four laptops, two Zenodo sets, SKAID's IRB-approved free-typing sessions), pooled eight keyboards and trained one attacker. Held-in it hit 43.7 % top-3; leave-one-keyboard-out it fell to 6.4 %, below chance. Per-key acoustics are device-specific, and a keyboard-agnostic attacker is an open research problem, so we faded it and banked our own keyboard instead. That meant writing a capture tool that records the mic while a keylogger stamps every press with its identity and time (our own machine, consenting typists, fake passwords), tapping all 36 keys about twenty times each, and discovering that the default mic was an AirPod, which gave us a bank of silence. On the laptop mic the isolated-key attacker landed at 38 % top-1: signal, not code.

Free typing was harder. We recorded 33 continuous sessions of ourselves typing while talking and coughing in the background, and found the real failure was onset detection, not classification: at speed, clicks land 24 ms apart on a 120 ms ring-down, hand-tuned detectors got either 56 % recall or 31 % precision, and every missed onset is a guaranteed deletion. Plain CTC then mode-collapsed and emitted the same letters for any input. The fix was a per-frame onset head trained jointly with CTC, an entropy bonus, a slow-to-fast curriculum, and adding space to the vocabulary, which broke every checkpoint. Since the isolated bank had no space clips, we built a "rich bank" by cutting per-key clips out of the real free-typing sessions at their labeled onsets and overlap-adding them into synthetic phrases in the Mac's own key sounds. Our Mac could not train it (MPS crashes, 6 s per step under load), so the bank went to a teammate's CUDA laptop over a git force-add of gitignored data and came back at 20 ms per step. The result reads novel text it never saw at 80 % per letter, `PASSWORD IDEAS` as `PASSWORD IDEAS`, and `HUNTER2 SUNFLOWER99` verbatim at the onsets.

**Proving keystrokes leak on a call.** An attacker trained on quiet typing collapses once someone talks over it. One trained with speech mixed in still reads 15.8 % top-1 with speech 10 dB louder than the keys; the shield brings it to 5.8 % with STOI 0.90.

**Getting into Meet without a bot.** Meet has no audio API and virtual cables need a driver install. Injecting a bridge before Meet's scripts and making the processed track the one Meet publishes took a night; the wrong hook order silently gives you an unshielded mic.

**Latency and clocks.** Your voice is delayed ~0.6 s in total; the secret spotter must place a span inside the 500 ms window or the first digit leaks. Three components used three key-timing conventions, and spaces turned out to be keystrokes too, which shifted every secret span until we found it.

**Compute.** A full Ares-vs-Athena match with default settings took over ten minutes on a laptop, so we exposed the step count and a live match now takes seconds.

## Accomplishments that we're proud of
- One app, one threat score, three live defences, in a real Google Meet, with no driver install.
- The attack-under-speech result: keystroke leakage on a call is a measured threat, not a slide.
- An attacker that reads fast, overlapping typing, which the paper we started from could not do, built from scratch in under 12 hours.
- A learned, speech-preserving shield that re-blinds a retraining attacker to chance every round. The closest prior art (EveGuard, IEEE S&P 2025) protects spoken content from vibration sensors; nobody had applied this to typed text over VoIP.
- A full-stack meeting-room test with real models blocked a spoken 6-digit code with about 1 s leaked at its start.
- Our voice detector's official test score from the HEARSAY organizers: minDCF 0.058, EER 2.5 %; 10 out of 10 held-out test voices in the Meet test room.
- Honest numbers. The Secret Shield misses its own acceptance bar on synthetic test sentences (1.32 words leaked per sequence versus the target of at most 1). We kept the miss in the report rather than tuning it away.

## What we learned
- Detectors that are great on datasets need calibration for the room: Meet's AGC, Opus, and 48 kHz playback shift scores, so we score through an emulation of that path.
- A side channel you can't demonstrate live isn't taken seriously. Showing the attacker's guesses next to what you actually typed convinced people in five seconds.
- A fixed filter can't beat an adapting attacker. The optimal defence depends on the current attacker, and only optimization finds it.
- Signal strength is the attack's wall, not code: a laptop mic and quiet keys, and the transcriber falls to chance on keyboards it wasn't tuned for.
- Redaction should say *where*, never *what*. Designing the spotter so the secret never leaves it made the privacy story simple.
- Write the plan first. Every pillar had a one-page contract before any code, and that let parallel work land without a merge war.

## What's next for Athena
Move the adversarial shield into the live audio path, transfer the transcriber across keyboards with a phone-mic rig, on-device speaker enrolment so a *verified* colleague disarms the secret shield, Zoom and Teams through the same bridge, and a proper user study of the false-redaction rate.

## Built with
Python 3.12, PyTorch, Hugging Face Transformers (XLS-R / wav2vec 2.0), LightGBM, librosa, SciPy, NumPy, Vosk, FastAPI, uvicorn, WebSockets, WebRTC, Chrome DevTools Protocol, Google Meet, pywebview, sounddevice, pynput, Gemini 2.5 Flash, Backboard, uv, pytest

## "Try it out" links
- https://github.com/georgia-mellon/callguard (CallGuard app and docs)
- https://github.com/danmano411/hearsay (Hearsay voice model)
- https://github.com/LordKarV/keyboard-acoustic-shield (Keyguard attacker and shield)

## Project media
- Image gallery: dashboard at CRITICAL (AI voice + readable typing), the Ares vs Athena panel, the Meet test room with a redacted code, the attack-under-speech figure (`docs/reports/figures/attack_under_speech.png`), the engine map (`keyguard/web/engine_map.html`).
- Video demo link: (add after recording; runbook in `docs/demo_runbook.md`)

## Schools
Georgia Institute of Technology; Carnegie Mellon University

## .Tech domains
None

## Feedback on technology
- **Google Meet.** No audio API for participants, so we bridged through Chrome DevTools script injection. It works well and needs no install, but a supported "processed microphone" hook would make tools like this far easier to build.
- **Gemini 2.5 Flash + Backboard.** Fast enough to sit inside an interactive attacker/defender loop; Backboard's memory made the "Athena remembers your last decoy" behaviour a few lines. A rule-based offline route was needed for when keys aren't set, and we'd like clearer latency guarantees per call.
- **Vosk.** The only streaming recognizer we found that gives partial word timings with sub-100 ms lag on CPU; the macOS wheel lags a version behind.
- **GitHub.** Stacked PRs and a private fork kept two teams' repos separate without pushing to each other.

## Generative AI
Yes, in two roles. Gemini 2.5 Flash, routed through Backboard for cross-round memory, is the reasoning core of both agents in the keystroke arms race. On the attack side it turns the transcriber's weak per-key guesses into readable text, because English carries about one bit per character and a language model recovers what the acoustics can't. On the defence side it triages which span of your typing is sensitive and invents a coherent decoy password, which the adversarial shield then steers the attacker to read. Crafting a believable false secret is inherently a language task that a fixed filter can't do. Generative voice is also the adversary we defend against: the demo's AI caller is a pre-generated voice clone from a public deepfake dataset (DiffSSD), and an open-source TTS model (VITS, facebook/mms-tts-eng) renders the agent's lines for live tests. Hearsay, our voice detector, is a discriminative model whose whole job is to catch those generators.
