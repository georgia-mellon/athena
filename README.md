# Athena

## Inspiration (the problem)
In 2023, voice cloning got good enough to fool your ears.
The same year, Harrison et al. (*A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*) showed a phone next to a MacBook, when paired with an LLM, reads 93 % of keystrokes solely from audio over a Zoom call, because every key resonates differently on the plate.
Put them together and you get the modern social-engineering call: an AI deepfake colleague joins your Google Meet, asks you to log in to "confirm your identity", then says "just read me the reset code". Your keyboard leaks the password through the call audio. Your mouth leaks the code. Nothing on the call stops any of it.
We built solutions to all of these problems this weekend. Then, we realized their common mission (to act as a sort of bodyguard for audio/video calls), integrated them into the same product, and Athena, the first real-time AI call-security engine, was born.

### What it does
Athena is a desktop agent that sits between you and Google Meet. It opens Meet in its own Chrome window and quietly injects an audio bridge, so the mic Meet publishes is the one Athena has already shielded and redacted. Three pillars run on the two audio streams:

**What you hear:** Our NSA HEARSAY challenge model scores every 4 s of the caller's voice real-vs-synthetic: two fine-tuned XLS-R models plus a gradient-boosted model on classic spectral features, fused. Official organizer score: minDCF 0.058, EER 2.5 %; 10 of 10 held-out voices in our Meet test room. Because Meet's gain control and Opus shift scores, we calibrate through an emulation of that path. The smoothed output is the voice risk V that arms the other two pillars. In Athena we run Hearsay's **E5** model, its final fusion (R4ft + R6 + R1). The full Hearsay design flow and models live in a separate repo: <https://github.com/danmano411/hearsay>.

**What you type:** This is what we're most proud of. Harrison published a dataset but not their attacker model, and the only known defences play noise over the call. We wanted a defence we could *prove*, which needs a live adversary that keeps adapting, so we built the attacker algorithm ourselves, from scratch, in less than a day: public recordings, our own keyboard through a laptop mic, and a synthesis engine that overlap-adds real clicks with exact labels.
Isolated keys were easy (87 % top-1). Fast, overlapping typing is what the paper couldn't do, so we treated typing as speech recognition: a CNN + BiGRU transcriber with CTC decoding, trained through blank-collapse with a per-frame onset head, an entropy bonus, and a slow-to-fast curriculum. It reads real free typing at 80 % per key, and Gemini turns the noisy read `MY PASSPLRS IS HUNT EFI2` into "my password is hunter2". We are proud of the progress we made here, but it is still a means to an end: the defender, our agent **Athena**, whose LLM picks the sensitive span, invents a coherent decoy, and optimizes an inaudible perturbation that makes the attacker read the decoy while speech stays intact (STOI 0.98). The attacker, **Ares**, retrains on the shielded audio and breaks through; Athena escalates; both remember rounds through Backboard.
A frozen shield gets clawed back to ~56 %; a re-solving one blinds the attacker to chance (2.8 %) every round. In Athena's dashboard, each burst of your real typing becomes a live match on the dashboard, while a streaming 80 ms shield protects the audio Meet is sending. The attacker on your raw mic is your exposure; on the shielded stream it is the residual leak L.

**What you say:** While the caller is unverified, or right after they ask for a code, your outgoing voice passes through a 500 ms delay line and a streaming recognizer (Vosk) spots digit runs, passwords and card numbers; the redactor tones them out before Meet hears them. It reports *where*, never *what*: the recognized words never leave the spotter. A full-stack test blocked a spoken 6-digit code with about 1 s leaked at its start.

These are fused into one number. Voice risk, residual keystroke leak, and typing activity fuse into a 0 to 100 threat score, with a boost when an unverified voice speaks while you type. The dashboard shows SAFE / WATCH / WARN / CRITICAL, the attacker's guesses next to what you really typed, and the live match between our agents Ares and Athena.

## How we built it
- Runtime. Python 3.12, one audio pipeline, lock-free event bus. Every model runs as a driver off the audio thread; if it throws or falls behind it is quarantined and audio passes through. A call must never glitch because a model hiccupped.
- Meet connector. Chrome with a dedicated profile; a DevTools-injected script wraps getUserMedia and incoming WebRTC tracks before Meet's own scripts load, streaming both sides to Athena over local WebSockets. The processed mic is what Meet publishes. No bot, no virtual cable, no driver install.
- Voice detector. Two XLS-R passes + LightGBM on spectral features, fused by logistic regression, calibrated so 0.5 is the threshold. ~1.4 s per 4 s window on a laptop CPU.
- Keystroke attacker and shields. CTC transcriber over 8 ms log-mel frames, 37 keys. Streaming shield: DSP with 80 ms lookahead. Adversarial shield: bounded perturbation optimized against the attacker under a perceptual constraint. Key timing from the OS keyboard hook mapped onto mic samples.
- Agents. Ares and Athena reason with Gemini 2.5 Flash, remember through Backboard, and fall back to a rule-based route so the demo runs without keys.
- Dashboard. FastAPI + WebSocket + static UI, wrapped in pywebview.

## Challenges we ran into
- Keystroke attackers don't transfer across different keyboards. Trained on Harrison's data, our attacker read 0% on any other keyboard. We pooled eight public keyboards (Harrison, MKA, two Zenodo sets, SKAID): 43.7% top-3 held-in, 6.4% leave-one-keyboard-out. Keyboard-agnostic attack is an open problem, so we recorded our own machine instead, with consenting typists and fake passwords.
- Fast typing is an onset problem. At speed, clicks land 24 ms apart on a 120 ms ring-down. Hand-tuned detectors got 56% recall or 31% precision, and every missed onset is a guaranteed deletion. Plain CTC mode-collapsed. The fix was a jointly trained onset head, an entropy bonus, a slow-to-fast curriculum, and a synthetic "rich bank" built by cutting real clicks at labeled onsets and overlap-adding them into phrases.
- Proving the attack survives speech. If talking breaks the attacker, the shield defends nothing. A naive attacker collapses to chance under speech; one trained with speech mixed in still reads 15.8% top-1 with speech 10 dB louder than the keys (chance 2.8%). The shield pushes it back to 5.8% at STOI 0.90. That experiment made the project real.
- Getting into Meet without a bot. Meet has no audio API. Injecting the bridge before Meet's scripts and making the processed track the one Meet publishes took a night, and the wrong hook order silently gives you an unshielded mic.
- Latency and clocks. Your voice is delayed ~0.6 s total. The secret spotter must place a span inside the 500 ms window or the first digit leaks. Three components each assumed a different key-time convention; reconciling them was our biggest bug source. Spaces are keystrokes too.

## What we learned
- Detectors that are great on datasets need calibration for the room: Meet's AGC, Opus, and 48 kHz playback shift scores, so we score through an emulation of that path.
- A fixed filter can't beat an adapting attacker, hence the need for LLMs here. The optimal defence depends on the current attacker, and only optimization finds it.
- Signal strength is the attack's wall, not code: a laptop mic and quiet keys, and the transcriber falls to chance on keyboards it wasn't tuned for.
- Redaction should say *where*, never *what*. Designing the spotter so the secret never leaves it made the privacy story simple.

## Running it
Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Heads up: the model weights and audio datasets are large and gitignored, so **they are not on GitHub**. Running the full stack needs the sibling Hearsay checkout (`../Hearsay`) with its models, a Keyguard checkout for the keystroke weights, and the built demo audio — without them the drivers won't load.

```
uv sync
uv run python -m app.keystroke_guard.get_assets    # copy Keyguard weights/data (needs a Keyguard checkout)
uv run python app/secret_shield/get_model.py        # Vosk model for the spoken-secret shield
uv run athena run --mode replay --scenario ai_caller  # scripted demo + dashboard
uv run athena app                                   # desktop app over Google Meet
```

Open the dashboard, then the **⚔️ Live battle** tab in the Ares-vs-Athena panel to watch a match in real time.

## References
- Harrison et al., *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards* (2023) — the keystroke acoustic attack Keyguard is based on.

This repo has few commits because it combines several separate repos (the Hearsay voice model, the Keyguard keystroke shield, and the Athena app) into one.

Made for HackGT 2026.
