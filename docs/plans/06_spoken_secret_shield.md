# 06: Third pillar: the spoken-secret shield

Owner-approved 2026-09-26 ("I love this"). It joins Hearsay and Keyguard as Athena's third core function.

## 1. Why this pillar
| pillar | question | protects |
|---|---|---|
| Hearsay | who am I hearing? | you, from a fake voice |
| Keyguard | what am I typing? | your keystrokes, from being overheard |
| **Spoken-secret shield** | **what am I saying?** | **your data, from being handed over** |

Scammers rarely need to hack anything; they get the victim to *read out* the code. This pillar is data-loss
prevention for your voice, and it is **threat-aware**. The same sentence passes untouched to a verified colleague
and gets redacted when the caller is unverified or synthetic. It bridges privacy (your data never leaves your mouth
to the wrong party) and security (the scam fails).

Pitch line: **Athena protects what you hear, what you type, and what you say.**

## 2. Behaviour
1. **Arm** when any of these holds (config toggles):
   - Hearsay voice risk `V ≥ 0.5` (unverified or synthetic caller);
   - a **request trigger** heard on the *inbound* audio: the same small recognizer with a trigger grammar ("code",
     "verification", "password", "PIN", "read me", "one-time", "security number", "card number"). This is a cheap
     intent listener, and a later full intent pillar can replace it;
   - manual toggle on the dashboard.
   Disarm: manual, after `disarm_after_s` (default 60 s) with no trigger and `V < 0.3`, or (future) a verified caller
   (Voice Passport).
2. While armed, the recognizer runs on **your outbound mic**, after the Keyguard shield (plan 02 §1 chain:
   mic → Keyguard shield → **secret shield** → virtual mic).
3. **Detect:** a sensitive sequence = ≥ `min_digits` (default 3) digit/letter tokens within `gap_s` (1.2 s) of each
   other, *or* a trigger phrase on your side ("my password is", "the code is", "card number"). Single numbers in
   normal speech ("in five minutes", "at two") don't match.
4. **Redact before it leaves:** mute or bleep (config: `mute | tone | noise`) from the first sensitive token's start
   to `tail_s` (0.3 s) after the last. The goal is **zero digits leaked**; the acceptance ceiling is ≤ 1 digit per
   sequence (§5).
5. **Show:** `secret.blocked {category: "digits"|"password"|"card", length, armed_by, t}` on the bus. The dashboard
   reads "6-digit code blocked from an unverified caller" and offers **Allow** (one click, 30 s bypass).
   **The recognized text is never logged or sent**, not even to the dashboard: only the category and the length.

## 3. Design constraints (from what the pipeline already does)
- **Constant delay.** The redactor needs lookahead: a delay line of `delay_ms` (default 500 ms) between recognition
  and output. `pipeline._hook` already keeps the Keyguard shield's delay constant even in mode `off`, so the stream
  never jumps. Do the same here: when the feature is enabled, the delay is always present; arming changes only
  whether redaction applies. Report the added latency in `shield.state`. (Alternative, if 500 ms feels bad in
  calls: insert the delay only while armed, and switch during VAD silence. Stretch goal.)
- **Audio thread does only the delay line + gain envelope** (O(block)). Recognition runs on a worker thread fed from a
  ring. The worker marks sample ranges to redact (absolute indices); the audio thread applies the marks when those
  samples reach the delay-line output. A mark that arrives too late (sample already sent) is counted as
  `leaked_samples`. That count is the honest metric.
- **Fail open.** If the recognizer fails it is quarantined, as with the other drivers. The dashboard shows "secret
  shield offline" and audio keeps flowing. Never mute the mic because of a bug.
- **Contract:** a new Protocol in `app/source/types.py` (additive; the existing drivers are untouched):
  ```python
  class SecretSpotterDriver(Protocol):
      name: str
      def feed(self, block: np.ndarray, start: int) -> list[SecretSpan]: ...   # streaming, absolute samples
      def reset(self) -> None: ...
  # SecretSpan(start: int, end: int, category: str, length: int)   (no text field, by design)
  ```
  Plus a mock with scripted spans for tests and replay.

## 4. Recognizer choice
**Vosk** (Apache-2.0, offline, CPU) small English model (~40 MB), `KaldiRecognizer(model, 16000, grammar)` with a
**restricted grammar**: digits (zero/oh … nine), the letters A-Z (NATO words optional), and the trigger phrases, plus
`[unk]`. Word timings come from `SetWords(True)`. Partial results (`PartialResult`) give early detection; the
final result gives exact spans. A restricted grammar is faster and far more accurate than open-vocabulary
transcription.
Fallback: sherpa-onnx keyword spotting. Whisper is too slow to stream within a 500 ms budget on this CPU.
Model files go in `runs/models/` (gitignored), with a download script and sha256.

## 5. Evaluation (`app/secret_shield/eval/secret_shield_eval.py` → `docs/reports/secret_shield.md`)
- **Sensitive set:** our own recorded utterances ("the code is four eight two one nine three", card-style
  4-4-4-4 groups, "my password is …") from consenting teammates, plus TTS renders for volume, over the Keyguard
  shielded path and with keyboard noise mixed in.
- **Innocent set:** normal speech containing numbers and times ("see you at two", "about five minutes", "room
  three"), plus LibriSpeech clips (read speech has few digit strings).
- **Metrics:** digits leaked per sequence (target 0, acceptance ≤ 1), sequences fully blocked %, false-redaction
  seconds per minute of innocent speech (target < 1 s/min), end-to-end added latency, and CPU per 20 ms block. Run
  with Hearsay and the Keyguard shield on the same CPU (the budget matters).

## 6. Threat engine + dashboard
- New input `S` = blocked-secret attempts in the last 60 s. A blocked secret while `V ≥ 0.5` → at least WARN;
  combined with a request trigger → **CRITICAL** with the reason "caller asked for a code and you started reading it".
- Dashboard: a third panel **"What you're saying"** with the armed/disarmed state, what armed it, a redaction log
  (category + length, e.g. `••••••  6-digit code  blocked 14:02:11`), and the Allow button. Show the mute visually
  on the outbound waveform.

## 7. Demo beat (add to the `ai_caller` scenario)
The AI caller says "just read me the verification code". The victim starts "four, eight, two, one…". The far-end
laptop hears bleeps. The dashboard shows the secret blocked, the voice flagged synthetic and the keys shielded: all
three pillars in one ten-second moment. Replay mode needs a recorded victim line with the digits. Record our own
voices; don't fake the victim with TTS.

## 8. Privacy (it must practice what it preaches)
Everything is on-device. The recognized words live only inside the worker, and each result is discarded after
matching. Events carry the category and length only. Nothing is recorded. The recognizer listens to *your* mic only
while armed, plus the inbound trigger grammar, which matches a fixed phrase list and doesn't transcribe the call.

## 9. Work package
**WP9 secret shield**: owns `app/secret_shield/spotter.py`, `app/secret_shield/redactor.py` (delay line +
envelope), mock additions in `app/*/mock.py` (coordinate with WP5's file), the `types.py` additions
(additive), the pipeline wiring (coordinate with WP8), the dashboard panel (coordinate with WP6), threat input `S`,
`app/secret_shield/eval/secret_shield_eval.py`, `docs/reports/secret_shield.md`, and tests (a redaction timing test with synthetic
spans, a leak counter, fail-open). **Not blocked on upstream.** It is in scope for this phase.
