# Hearsay pillar: is the voice real?

CallGuard scores the far-end voice of a call (4 s windows, every 2 s, on a worker thread) and publishes
`voice.verdict`. A high `p_synthetic` raises the threat score and arms the Secret Shield.

## Contract

`VoiceAuthenticityDriver` in [`app/source/types.py`](../source/types.py):

- attributes `name: str`, `sample_rate = 16000`
- `score(audio: np.ndarray) -> VoiceScore`: mono float32, 16 kHz, a 4 s window in the pipeline
- `VoiceScore.p_synthetic` in [0, 1], and **0.5 = the deployment threshold** (`p > 0.5` exactly when
  `margin > threshold`); `margin` is the raw model score (higher = more synthetic); `latency_ms`; `detail` dict
- deterministic for the same audio; one worker thread calls it, never the audio thread
- budget: a 4 s window in < 2 s on CPU

## Placeholder vs real

| | class | what it does |
|---|---|---|
| placeholder | `app.hearsay.mock:MockVoice` | deterministic: `1 - spectral flatness`, or a scripted schedule. No weights. |
| real | `app.hearsay.driver:HearsayDriver` | Hearsay's final E5 fusion (`mode="e5"`, default: R4ft + R6 XLS-R + R1 LightGBM), the R5 fusion (`mode="r5"`: R4ft + R1), or R4ft alone (`mode="r4ft"`), read-only from `HEARSAY_ROOT` |

Pick one in `callguard.toml` ([`callguard.example.toml`](../../callguard.example.toml)): `[drivers] voice = "real"|"mock"`,
`hearsay_mode = "e5"|"r5"|"r4ft"` (default `e5`), `hearsay_ai_p` (decision threshold, default 0.7: a window is AI when its
calibrated p >= 0.7 instead of Hearsay's 0.5; p_synthetic is re-centred so 0.5 still means "at the threshold"), `device`, `threads`; `HEARSAY_ROOT` defaults to `../Hearsay`. The real driver checks
`best.pth` against the sha256 frozen in its `config.json` and caches the threshold calibration in
`runs/hearsay_calibration.json`.

## Plug in a new model

1. Write a class with `name`, `sample_rate = 16000` and `score(audio) -> VoiceScore` (map your margin so that
   `p_synthetic = 0.5` at your operating threshold; `app.hearsay.driver.p_from_margin` does that with a logistic).
2. Run the harness on it: `python -m app.hearsay.harness --driver mypkg.mymodule:MyDetector` (built with no
   arguments). Fix every FAIL.
3. To ship it, add it to `make_voice` in [`app/source/registry.py`](../source/registry.py) (integrator change).

## Harness

```
python -m app.hearsay.harness [--driver mock|real|module.path:ClassName] [--clips N] [--no-quality]
```

Contract checks, latency against the 2 s budget, and accuracy at `p = 0.5` on N clips of Hearsay's held-out
`test_internal_testlike` set (half real, half fake, seed 0; skipped when `HEARSAY_ROOT` has no manifest). Exit
code 1 on any FAIL. Real run on the dev laptop (CPU, 2026-09-26):

```
Hearsay harness: real
  load      build driver              PASS  HearsayDriver in 11.7 s
  contract  Protocol + attributes     PASS  name='hearsay_r4ft', sample_rate=16000
  contract  score(4 s) -> VoiceScore  PASS  VoiceScore, p in [0,1], finite, p>0.5 <=> margin>threshold; p = 0.789, 0.826
  contract  p = 0.5 at the threshold  PASS  p(threshold=-2.824) = 0.5
  contract  deterministic             PASS  same window twice: 0.826032 = 0.826032
  latency   score 4 s window          PASS  median 545 ms, max 551 ms per 4 s window (budget 2000 ms)
  quality   held-out clips (n=20)     INFO  accuracy at p=0.5: 20/20 = 100 % (reals flagged 0, fakes missed 0); from test_internal_testlike, seed 0
  => FITS  (6 PASS, 1 INFO)
```

(The probe signals are a noise burst and a synthetic harmonic tone, so p near 0.8 on them is expected: neither is
human speech.)

## Measured in CallGuard

- Latency per 4 s window on CPU (4 threads): ~0.4-0.65 s (r4ft), ~0.5-0.55 s (r5), ~1.4 s (e5: two XLS-R passes + R1).
  The pipeline scores every 2 s, so all three keep up (e5 with ~0.6 s to spare).
- R5 vs R4ft in the pipeline (40 real + 40 fake simulated callers from Hearsay's held-out set): both arm the Secret
  Shield on 37/40 fakes (R5 median 13.2 s after the caller starts, R4ft 12.0 s) and on the same 10/40 reals; no
  window skipped even at 100 % CPU. See [`docs/reports/hearsay_r5_in_callguard.md`](../../docs/reports/hearsay_r5_in_callguard.md).
- Keystrokes under the voice, with the Keyguard shield on: Hearsay flags 2 / 100 real voices (0 / 100 without the
  shield; median p_synthetic 0.14 -> 0.18). See [`docs/reports/attack_under_speech.md`](../../docs/reports/attack_under_speech.md).

## Known limits

- E5 (Hearsay's final model) runs by default: R4ft and R6 XLS-R passes plus the R1 LightGBM, fused with the frozen E5
  weights. `hearsay_mode = "r5"` drops R6, `"r4ft"` runs one XLS-R alone. A new mode = an entry in `driver.SCORE_FILES` and `config.HEARSAY_MODES` plus its scoring branch in
  `HearsayDriver`.
- Trained and calibrated on clean 16 kHz clips; a meeting codec, echo cancellation and noise suppression were not in
  Hearsay's evaluation. Short windows (< 1 s) are rejected; 3-4 s is what it was built for.
- Models trained on Hearsay's data are for non-commercial use (DiffSSD, SONAR, MLAAD-tiny are CC BY-NC; Hearsay
  README, "Credits and licenses").

---

## The Hearsay model
<!-- TEMPLATE: a short overview of the Hearsay model as used inside CallGuard (what it is, the checkpoint CallGuard loads, the deployment threshold, one or two headline numbers) and a link to the Hearsay model repository, https://github.com/danmano411/hearsay, which holds the full model documentation. -->
