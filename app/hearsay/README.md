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
| real | `app.hearsay.driver:HearsayDriver` | Hearsay's frozen R4ft model (`mode="r4ft"`), or the R5 fusion (`mode="r5"`), read-only from `HEARSAY_ROOT` |

Pick one in `callguard.toml` ([`callguard.example.toml`](../../callguard.example.toml)): `[drivers] voice = "real"|"mock"`,
`hearsay_mode = "r4ft"|"r5"`, `device`, `threads`; `HEARSAY_ROOT` defaults to `../Hearsay`. The real driver checks
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

- Latency: ~0.55-0.65 s per 4 s window on CPU (harness above; `callguard bench`).
- Keystrokes under the voice, with the Keyguard shield on: Hearsay flags 2 / 100 real voices (0 / 100 without the
  shield; median p_synthetic 0.14 -> 0.18). See [`docs/reports/attack_under_speech.md`](../../docs/reports/attack_under_speech.md).

## Known limits

- R4ft alone runs by default (fast); the submitted R5 fusion adds the R1 LightGBM on the centre 4 s.
- Trained and calibrated on clean 16 kHz clips; a meeting codec, echo cancellation and noise suppression were not in
  Hearsay's evaluation. Short windows (< 1 s) are rejected; 3-4 s is what it was built for.
- Models trained on Hearsay's data are for non-commercial use (DiffSSD, SONAR, MLAAD-tiny are CC BY-NC; Hearsay
  README, "Credits and licenses").

---

## The Hearsay model (NSA HEARSAY challenge)

Hearsay is our submission to the NSA "HEARSAY" challenge at HackGT: score each test clip from 0.0 (confident real) to
1.0 (confident synthetic), judged by the organizers' ASVspoof 5 Track-1 minDCF (lower is better, 1 = trivial).
CallGuard uses the model as one component. The model-creation repository, with every plan, report and script, is
**[swail-labs/hearsay](https://github.com/swail-labs/hearsay)** (a fork of the original). Facts below come from that
repo's own documents; paths are relative to it.

**Data** (`README.md`, `docs/dataset.md`, `docs/external_data.md`). The given data was 70k DiffSSD fakes against 242
real clips of one speaker, a trap: a depth-3 tree on trivial cues (bandwidth, silence, level) separated them perfectly
(`reports/data_audit.md`). Hearsay added 57k real clips from 10 corpora (including the LibriSpeech speakers DiffSSD
clones), 11 external corpora in all, and 5.2k of its own TTS fakes (`docs/synthetic_speech.md`): 185,915 clips with
grouped splits and frozen evaluation subsets.

**Preprocessing** (`src/hearsay/preprocess.py`). `prep()`: DC removal, silence trim, 7 kHz low-pass, RMS
normalization. It removes the channel shortcuts: a trivial-cue LightGBM goes from minDCF 0.54 on raw audio to 0.92
(near chance) after `prep()`. Training adds class-symmetric `augment()` (MP3 round trip, noise, resampler) to real and
fake alike.

**Models** (`reports/final_results.md`, `reports/r4ft_xlsr.md`, `reports/r1_classic.md`):

- **R1**: LightGBM on 228 features: 180 spectral (LFCC and MFCC with deltas, centroid, bandwidth, rolloff, flatness,
  band contrast; all on 0-7 kHz) plus 48 speech-biology features (jitter, shimmer, HNR, formant dynamics,
  micro-prosody; `docs/speech_biology.md`).
- **R4ft**: `facebook/wav2vec2-xls-r-300m` cut to its first 12 transformer blocks, fine-tuned end to end with a light
  back end (learned weights over the hidden states, linear 1024 -> 256, attentive statistics pooling, one spoof
  logit). bf16, gradient checkpointing, 2 h 24 min on an 8 GB laptop GPU. Inference: `prep()` the whole clip, mean
  logit over up to 3 evenly spaced 4 s windows.
- **R5**: logistic-regression fusion of R4ft and R1, fit on `val_testlike` (group cross-fitted); the submitted model
  (an owner amendment over the pre-registered choice of R4ft alone, recorded in `plans/06_modeling_ladder.md`).

**Key results** (headline set `test_internal_testlike`: 9,747 held-out clips, 70/30 real/fake, about half the fakes
from three generators never seen in training; `reports/final_results.md`):

| model | headline combined minDCF | EER |
|---|---|---|
| R1 (classic features) | 0.253 | 6.5 % |
| **R4ft** (XLS-R fine-tune) | **0.0282** (the honest estimate) | 0.76 % |
| R5 (R4ft + R1 fusion, submitted) | 0.0218 (optimistic: the headline informed the switch) | 0.54 % |

R4ft is about 9x better than the best classic model, and fixed the two failure modes the error analysis found:
unseen generators and real speech from unusual recording chains (`reports/error_analysis.md`). At the brief's
operating point R4ft flags 0.22 % of real clips as fake and misses 1.74 % of fakes; best plain accuracy 99.40 %.

**Deployment threshold in CallGuard** (`app/hearsay/driver.py`). The margin threshold is Hearsay's own
`bench_score.threshold()` on `val_testlike`, minimizing 4 x P(real flagged) + P(fake passed) (a real voice flagged as
fake costs 4x, as in the challenge brief). `p_synthetic` is a logistic centred on it (p = 0.5 at the threshold),
scaled so the median `val_testlike` fake maps to p = 0.95. For the frozen R4ft checkpoint: threshold -2.82, scale 6.27
(R5: 0.61, 3.89), from `runs/hearsay_calibration.json`.
