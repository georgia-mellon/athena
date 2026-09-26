# Secret Shield pillar: don't read the code to a fake caller

While the caller is unverified (Hearsay p >= 0.5), after they ask for a code, or when armed by hand, CallGuard
delays your outgoing voice by 500 ms and cuts out digit runs, passwords and card numbers before the meeting hears
them ([`docs/plans/06_spoken_secret_shield.md`](../../docs/plans/06_spoken_secret_shield.md)). Two parts: the
**spotter** (a streaming recognizer that says *where* a secret is, never *what*) and the **redactor** (the delay line
that tones those samples out on the audio thread).

## Contract

`SecretSpotterDriver` and `SecretSpan` in [`app/source/types.py`](../source/types.py):

- `name`, `reset()`, `feed(block, start) -> list[SecretSpan]`: consecutive 20 ms blocks with the absolute index of
  their first sample; spans come out as soon as they can be placed, each at most once.
- `SecretSpan(start, end, category, length)`: absolute samples, `category` in digits | password | card (outbound) |
  request (inbound "read me the code"), `length` = tokens so far. **No text field, by design**: the recognized words
  never leave the spotter.
- Deterministic after `reset()`; budget: mean < 10 ms per 20 ms block (it runs on a worker thread, and a span
  must arrive within the 500 ms delay line or the word leaks).

## Placeholder vs real

| | class | what it does |
|---|---|---|
| placeholder | `app.secret_shield.mock:MockSpotter` | scripted spans `[(t_start, t_end, category, length)]`, each released 0.3 s after its start (like a recognizer's lag) |
| real | `app.secret_shield.spotter:VoskSpotter` | Vosk small en-us, open vocabulary, low-latency partial word timings; `mode="outbound"` (your mic) or `"inbound"` (the caller) |
| redactor | `app.secret_shield.redactor:Redactor` | the delay line (`delay`, `style` = mute \| tone \| noise); counts `leaked_samples` honestly |

Built here, no upstream repo. Get the model once: `python -m app.secret_shield.get_model` (downloads to
`runs/models/`, checks its sha256, adds the low-latency decoder options). Config
([`callguard.example.toml`](../../callguard.example.toml)): `[drivers] secret = "real"|"mock"`, and the `[secret]`
section (`delay_ms`, `style`, `arm_on_voice`, `arm_voice`, `arm_on_request`, `min_digits`, `gap_s`, ...).

## Plug in a new spotter

Implement the Protocol (a constructor taking `mode=` is used if present), then
`python -m app.secret_shield.harness --spotter mypkg.mod:MySpotter [--mode inbound]`. Fix every FAIL, then add it to
`make_spotter` in [`app/source/registry.py`](../source/registry.py) (integrator change).

## Harness

```
python -m app.secret_shield.harness [--spotter mock|real|module:Class] [--mode outbound|inbound] [--no-quality]
```

Contract checks and per-block latency on a TTS utterance (`facebook/mms-tts-eng` from the local HF cache; a synthetic
probe without it), then informational rows: spans on silence, and for each test utterance (fake codes only) how many
digit words end up > 80 % cut when spans must arrive within the 500 ms delay line (word times from Vosk forced
alignment). `mock` is scripted with one span so the span checks run. Exit code 1 on any FAIL. Real run (CPU,
2026-09-26):

```
Secret Shield harness: real (outbound)
  load      build spotter                           PASS  VoskSpotter in 0.6 s
  contract  Protocol + attributes                   PASS  name='vosk-spotter-outbound'
  contract  feed() -> [SecretSpan]                  PASS  6 span(s) on the TTS utterance; fields ok, no text, categories valid, none repeated
  contract  reset() + deterministic                 PASS  reset() + same audio -> same spans (within 1 ms)
  latency   feed 20 ms block                        PASS  mean 5.46 ms, p99 79.4 ms, max 146.0 ms per 20 ms block
  quality   spans on 3 s of silence                 INFO  0 span(s) (want 0; the scripted mock emits its script regardless)
  quality   TTS: 'the code is four seven two nine'  INFO  4/4 digit words redacted (> 80 % cut through a 500 ms delay line); 6 span(s)
  quality   TTS: 'five eight one six three'         INFO  4/5 digit words redacted (> 80 % cut through a 500 ms delay line); 7 span(s)
  => FITS  (5 PASS, 3 INFO)

Secret Shield harness: real (inbound)
  ...
  latency   feed 20 ms block                                  PASS  mean 4.64 ms, p99 54.6 ms, max 190.4 ms per 20 ms block
  quality   TTS: 'just read me the verification code please'  INFO  6 request span(s) (want >= 1)
  quality   TTS: 'can you tell me your pin number'            INFO  4 request span(s) (want >= 1)
  => FITS  (5 PASS, 3 INFO)
```

Without a trigger phrase the first digit of a run passes by design (the delay line can't wait for a whole sequence).
Vosk's word times jitter by a sample between runs, so determinism is checked to within 1 ms.

## Measured numbers

From [`docs/reports/secret_shield.md`](../../docs/reports/secret_shield.md) (40 TTS sequences, clean and with
shielded typing):

| metric | target | result |
|---|---|---|
| secret words leaked per sequence | 0 (acceptance <= 1) | 1.32 (1.65 with 100 ms worker lag) |
| sequences fully blocked | - | 36 % (45 % with a trigger phrase first) |
| false redaction, LibriSpeech | < 1 s/min | 0.19 s/min |
| false redaction, casual-number sentences | < 1 s/min | 4.85 s/min |
| inbound requests caught | - | 7 / 8 |
| CPU per 20 ms block, outbound / inbound | << 20 ms | 4.42 / 6.25 ms mean (p99 152 ms) |
| added latency | - | 500 ms, constant while on |

## Known limits

- Misses both acceptance bars on TTS; **every test utterance is synthetic (one TTS voice)**. Owner to-do: record
  consenting teammates reading fake codes and rerun `app/secret_shield/eval/secret_shield_eval.py`.
- CPU spikes when Vosk finishes an utterance (p99 ~ 150 ms): the spotter must run on its own worker, fed every
  20-40 ms while armed, or spans arrive after their words left the delay line.
- No letters A-Z (NATO words would be needed); English only.
