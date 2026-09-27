# Hearsay R5 vs R4ft inside CallGuard

`app/hearsay/eval/r5_in_callguard.py`, 2026-09-26, dev laptop CPU (16 logical cores), Hearsay driver on 4 torch threads, 26 min. CPU load from other processes when the run started: 20 %.

## Question

CallGuard switches its voice pillar from R4ft (the XLS-R fine-tune alone) to R5 (Hearsay's submitted fusion: R4ft + the R1 LightGBM on classic features, frozen weights). The Secret Shield arms when the smoothed voice risk V >= `secret.arm_voice` (0.5; stays armed while V >= `keep_voice` 0.3; V = EMA of p_synthetic, half-life 6 s). So the model decides when your spoken codes get bleeped. Does R5 arm sooner on fake callers, arm less on real ones, and keep up with the 2 s hop?

## Method

- 40 real and 40 fake simulated callers from Hearsay's held-out `test_internal_testlike` rows (never used for training or for the threshold, which comes from `val_testlike`). A call is 20-30 s of one speaker's clips with 0.4 s pauses: a real call is one bona fide speaker, round-robin over the 11 bona fide sources; each clip is levelled to -26 dBFS active-speech RMS (a stand-in for Meet's AGC: dataset levels vary and the pipeline's VAD has a fixed -45 dBFS floor; Hearsay's prep() normalizes, so this only decides which windows get scored); a fake call is one (generator, speaker), one per generator, modern ones first (DiffSSD's ElevenLabs, PlayHT, OpenVoice v2, XTTS v2, YourTTS...; MLAAD's FishTTS, Llasa, MegaTTS3, Dia, OuteTTS; SONAR's OpenAI, VoiceBox, xTTS; DFADD's NaturalSpeech 2, StyleTTS 2; ASVspoof5). LibriSpeech speakers 100 and 2803 excluded. Seed 0.
- Each call is the far end of `Pipeline.replay(realtime=False)` with the real `HearsayDriver` and mock attacker/shield/spotters (Secret Shield enabled, default config). The voice worker runs on the audio clock: a verdict lands `latency_ms` (measured live on this CPU) after its window was due, and the worker can't start the next window before then, so `voice_step`'s catch-up (jump to the newest window when > 2 s behind) skips windows exactly as it would live. Arm time = audio seconds from the caller's first sample until `armed_by == "voice"`. The first window is due at 4 s.
- Without typing, the score is 100 * 0.7 * V, so a voice alone tops out at WARN (70); CRITICAL needs typing or a blocked secret. Levels below are for voice alone.

## Results: simulated calls

| index | r4ft | r5 |
| --- | --- | --- |
| fakes armed | 37/40 | 37/40 |
| arm time median (s) | 12.0 | 13.2 |
| arm time p90 (s) | 15.3 | 17.4 |
| first flagged window median (s) | 4.8 | 5.8 |
| real false arms | 10/40 | 10/40 |
| real V max (median / max) | 0.22 / 0.91 | 0.23 / 0.92 |
| fake V median (median) | 0.47 | 0.42 |
| windows flagged, fake / real | 98% / 27% | 97% / 27% |
| WARN reached, fake / real | 27 / 4 | 27 / 3 |
| latency median / p95 (ms) | 477 / 769 | 1684 / 2133 |
| windows skipped | 0 of 854 | 0 of 830 |

Arm time and V are per call; latency is the median of the per-call medians and the 95th percentile of the per-call p95s. The windows count includes silence-skipped windows (VAD < 0.5).

### By source

| mode | kind | source | calls | armed | t_arm | p_median |
| --- | --- | --- | --- | --- | --- | --- |
| r4ft | fake | asvspoof2019_la | 1 | 1 | 15.88 | 0.79 |
| r4ft | fake | asvspoof5 | 3 | 3 | 11.32 | 1.0 |
| r4ft | fake | cvoicefake_en | 5 | 4 | 16.36 | 0.96 |
| r4ft | fake | dfadd | 2 | 2 | 14.56 | 0.84 |
| r4ft | fake | diffssd | 10 | 10 | 12.04 | 0.98 |
| r4ft | fake | in_the_wild | 1 | 1 | 10.84 | 0.99 |
| r4ft | fake | librisevoc | 6 | 6 | 10.96 | 0.94 |
| r4ft | fake | mlaad_tiny | 6 | 4 | 14.8 | 0.87 |
| r4ft | fake | sonar | 3 | 3 | 12.04 | 0.95 |
| r4ft | fake | wavefake | 3 | 3 | 13.48 | 0.81 |
| r4ft | real | asvspoof2019_la | 7 | 0 |  | 0.76 |
| r4ft | real | asvspoof5 | 7 | 2 | 13.84 | 0.25 |
| r4ft | real | cvoicefake_en | 1 | 0 |  | 1.0 |
| r4ft | real | dfadd | 2 | 1 | 13.24 | 0.87 |
| r4ft | real | in_the_wild | 7 | 2 | 22.84 | 0.17 |
| r4ft | real | librisevoc | 6 | 4 | 16.24 | 0.24 |
| r4ft | real | librispeech | 6 | 0 |  | 0.13 |
| r4ft | real | lj_real | 1 | 0 |  | 0.21 |
| r4ft | real | ljspeech | 1 | 0 |  | 0.16 |
| r4ft | real | mlaad_tiny | 1 | 1 | 15.64 | 0.24 |
| r4ft | real | sonar | 1 | 0 |  | 0.17 |
| r5 | fake | asvspoof2019_la | 1 | 1 | 17.32 | 0.77 |
| r5 | fake | asvspoof5 | 3 | 3 | 12.28 | 1.0 |
| r5 | fake | cvoicefake_en | 5 | 4 | 17.92 | 0.93 |
| r5 | fake | dfadd | 2 | 2 | 15.28 | 0.85 |
| r5 | fake | diffssd | 10 | 10 | 12.64 | 0.97 |
| r5 | fake | in_the_wild | 1 | 1 | 12.52 | 0.98 |
| r5 | fake | librisevoc | 6 | 6 | 12.28 | 0.94 |
| r5 | fake | mlaad_tiny | 6 | 4 | 15.88 | 0.86 |
| r5 | fake | sonar | 3 | 3 | 13.0 | 0.93 |
| r5 | fake | wavefake | 3 | 3 | 14.68 | 0.78 |
| r5 | real | asvspoof2019_la | 7 | 0 |  | 0.77 |
| r5 | real | asvspoof5 | 7 | 2 | 13.84 | 0.35 |
| r5 | real | cvoicefake_en | 1 | 0 |  | 0.99 |
| r5 | real | dfadd | 2 | 1 | 13.96 | 0.87 |
| r5 | real | in_the_wild | 7 | 2 | 23.2 | 0.19 |
| r5 | real | librisevoc | 6 | 4 | 16.72 | 0.26 |
| r5 | real | librispeech | 6 | 0 |  | 0.12 |
| r5 | real | lj_real | 1 | 0 |  | 0.17 |
| r5 | real | ljspeech | 1 | 0 |  | 0.15 |
| r5 | real | mlaad_tiny | 1 | 1 | 15.88 | 0.19 |
| r5 | real | sonar | 1 | 0 |  | 0.13 |

### Calls the arming got wrong (fake never armed, or real armed)

| mode | call | kind | source | generator | speaker | p_median | frac_flagged | V_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r4ft | real09 | real | mlaad_tiny | bonafide |  | 0.242 | 0.231 | 0.555 |
| r4ft | real13 | real | dfadd | bonafide | p227 | 0.985 | 0.8 | 0.649 |
| r4ft | real14 | real | in_the_wild | bonafide | George W. Bush | 0.185 | 0.333 | 0.529 |
| r4ft | real15 | real | librisevoc | bonafide | 4018 | 0.272 | 0.444 | 0.549 |
| r4ft | real20 | real | librisevoc | bonafide | 4267 | 0.981 | 0.625 | 0.718 |
| r4ft | real23 | real | asvspoof5 | bonafide | E_4580 | 0.998 | 1.0 | 0.906 |
| r4ft | real25 | real | librisevoc | bonafide | 6437 | 0.183 | 0.3 | 0.529 |
| r4ft | real30 | real | librisevoc | bonafide | 8108 | 0.967 | 0.917 | 0.883 |
| r4ft | real33 | real | asvspoof5 | bonafide | E_1991 | 0.409 | 0.4 | 0.512 |
| r4ft | real34 | real | in_the_wild | bonafide | Norm MacDonald | 0.942 | 0.636 | 0.752 |
| r4ft | fake13 | fake | mlaad_tiny | mlaad_MegaTTS3 |  | 0.474 | 0.375 | 0.396 |
| r4ft | fake14 | fake | mlaad_tiny | mlaad_Nari_Dia_1.6B |  | 0.615 | 1.0 | 0.449 |
| r4ft | fake31 | fake | cvoicefake_en | cvoicefake_en_griffin_lim_generated |  | 0.75 | 0.875 | 0.498 |
| r5 | real09 | real | mlaad_tiny | bonafide |  | 0.186 | 0.231 | 0.55 |
| r5 | real13 | real | dfadd | bonafide | p227 | 0.982 | 0.8 | 0.648 |
| r5 | real14 | real | in_the_wild | bonafide | George W. Bush | 0.204 | 0.333 | 0.536 |
| r5 | real15 | real | librisevoc | bonafide | 4018 | 0.255 | 0.444 | 0.535 |
| r5 | real20 | real | librisevoc | bonafide | 4267 | 0.97 | 0.625 | 0.707 |
| r5 | real23 | real | asvspoof5 | bonafide | E_4580 | 0.998 | 1.0 | 0.917 |
| r5 | real25 | real | librisevoc | bonafide | 6437 | 0.219 | 0.3 | 0.513 |
| r5 | real30 | real | librisevoc | bonafide | 8108 | 0.982 | 1.0 | 0.884 |
| r5 | real33 | real | asvspoof5 | bonafide | E_1991 | 0.426 | 0.4 | 0.528 |
| r5 | real34 | real | in_the_wild | bonafide | Norm MacDonald | 0.908 | 0.6 | 0.721 |
| r5 | fake13 | fake | mlaad_tiny | mlaad_MegaTTS3 |  | 0.392 | 0.375 | 0.369 |
| r5 | fake14 | fake | mlaad_tiny | mlaad_Nari_Dia_1.6B |  | 0.612 | 0.833 | 0.416 |
| r5 | fake31 | fake | cvoicefake_en | cvoicefake_en_griffin_lim_generated |  | 0.602 | 0.714 | 0.405 |

## Latency per 4 s window (back to back, no other CallGuard pillar)

| mode | when | load_before | load_after | median | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| r4ft | 20:42 | 19.0 | 27.0 | 636.0 | 676.0 | 691.0 |
| r5 | 20:50 | 24.0 | 30.0 | 542.0 | 602.0 | 629.0 |

`load_before` = system CPU % from other processes just before the bench (a Keyguard `adversarial_eval` job with 8 worker processes was running during parts of this experiment); `load_after` includes the bench.

## ai_caller demo scenario, realtime, every pillar real

`demo/scenarios/ai_caller`: colleague (real, 0-12 s), cloned-voice AI agent (12-40 s) asking for the reset code, typed at 20.5 s (shield off) and 29.5 s (shield on from 29 s), colleague again (40-60 s). Realtime replay with the real Keyguard attacker + DSP shield and the Vosk spotters on their worker threads, so the voice latency here is with the other pillars running.

### r4ft

- drivers: attacker `mock_attacker`, shield `keyguard-dsp`, spotter `vosk-spotter-outbound`; the real attacker failed to load, so the mock stood in: `RuntimeError: Error(s) in loading state_dict for KeyNet:`
- Secret Shield armed by voice at **23.6 s**
- first arming of any kind: voice at 23.6 s
- voice windows: 29 scored of 29, 0 skipped by catch-up; latency median 438 ms, p95 522 ms, max 638 ms (CPU load before: 16 %)
- verdicts (t_audio s, p): 4:0.15, 6:0.17, 8:0.19, 10:0.26, 12:0.20, 14:0.39, 16:0.62, 18:0.69, 20:0.84, 22:0.84, 24:0.82, 26:0.83, 28:0.74, 30:0.77, 32:0.83, 34:0.88, 36:0.93, 38:0.87, 40:0.79, 42:0.40, 44:0.14, 46:0.06, 48:0.08, 50:0.11, 52:0.26, 54:0.19, 56:0.14, 58:0.09, 60:0.10

| t (s) | level | score | reasons |
|---|---|---|---|
| 20.3 | WATCH | 25 | unverified voice 0.36 |
| 22.1 | WARN | 56 | unverified voice 0.45; keystrokes 40% readable by an eavesdropper (shield off) |
| 24.5 | CRITICAL | 75 | synthetic voice 0.54 for 8 s; typing while an unverified voice is speaking; keystrokes 67% readable by an eavesdropper (shield off) |
| 29.2 | WARN | 60 | synthetic voice 0.65 for 13 s; typing while an unverified voice is speaking; shield (dsp) blocking a 75% readable keyboard; residual leak 0% |
| 33.1 | CRITICAL | 75 | synthetic voice 0.70 for 17 s; typing while an unverified voice is speaking; shield (dsp) blocking a 83% readable keyboard; residual leak 55% |
| 43.3 | WARN | 54 | synthetic voice 0.77 |
| 45.9 | WATCH | 44 | synthetic voice 0.63 |
| 57.3 | SAFE | 20 | unverified voice 0.28 |

### r5

- drivers: attacker `mock_attacker`, shield `keyguard-dsp`, spotter `vosk-spotter-outbound`; the real attacker failed to load, so the mock stood in: `RuntimeError: Error(s) in loading state_dict for KeyNet:`
- Secret Shield armed by voice at **24.8 s**
- first arming of any kind: voice at 24.8 s
- voice windows: 29 scored of 29, 0 skipped by catch-up; latency median 1657 ms, p95 1798 ms, max 1808 ms (CPU load before: 100 %)
- verdicts (t_audio s, p): 4:0.15, 6:0.15, 8:0.15, 10:0.21, 12:0.14, 14:0.33, 16:0.69, 18:0.76, 20:0.89, 22:0.87, 24:0.86, 26:0.87, 28:0.81, 30:0.84, 32:0.85, 34:0.89, 36:0.88, 38:0.87, 40:0.81, 42:0.47, 44:0.13, 46:0.06, 48:0.10, 50:0.11, 52:0.29, 54:0.24, 56:0.11, 58:0.09, 60:0.10

| t (s) | level | score | reasons |
|---|---|---|---|
| 21.4 | WATCH | 42 | unverified voice 0.35; keystrokes 25% readable by an eavesdropper (shield off) |
| 22.2 | WARN | 54 | unverified voice 0.40; keystrokes 40% readable by an eavesdropper (shield off) |
| 24.8 | CRITICAL | 76 | synthetic voice 0.52 for 7 s; typing while an unverified voice is speaking; keystrokes 70% readable by an eavesdropper (shield off) |
| 29.0 | WARN | 60 | synthetic voice 0.65 for 11 s; typing while an unverified voice is speaking; shield (dsp) blocking a 75% readable keyboard; residual leak 0% |
| 33.1 | CRITICAL | 76 | synthetic voice 0.72 for 15 s; typing while an unverified voice is speaking; shield (dsp) blocking a 83% readable keyboard; residual leak 55% |
| 43.3 | WARN | 57 | synthetic voice 0.81 for 26 s |
| 47.1 | WATCH | 45 | synthetic voice 0.64 |
| 59.2 | SAFE | 20 | unverified voice 0.28 |

## Threshold sensitivity

First arming recomputed with the real `ThreatEngine` on each call's recorded verdicts (the time each verdict landed), for other half-lives and `arm_voice`. `keep_voice` only matters once armed, so it doesn't change these. Row half_life 6 / arm 0.5 is the shipped config and should match the pipeline run above (to the 0.25 s tick).

| mode | half_life_s | arm_voice | fakes_armed | n_fake | t_arm_median | t_arm_p90 | real_false_arms | n_real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r4ft | 3.0 | 0.4 | 40 | 40 | 7.5 | 8.83 | 18 | 40 |
| r4ft | 3.0 | 0.5 | 40 | 40 | 8.25 | 10.1 | 16 | 40 |
| r4ft | 3.0 | 0.6 | 37 | 40 | 9.25 | 13.25 | 14 | 40 |
| r4ft | 4.0 | 0.4 | 40 | 40 | 8.25 | 9.83 | 17 | 40 |
| r4ft | 4.0 | 0.5 | 39 | 40 | 9.75 | 13.05 | 12 | 40 |
| r4ft | 4.0 | 0.6 | 37 | 40 | 11.5 | 14.8 | 11 | 40 |
| r4ft | 6.0 | 0.4 | 39 | 40 | 10.25 | 13.05 | 13 | 40 |
| r4ft | 6.0 | 0.5 | 37 | 40 | 12.25 | 15.45 | 10 | 40 |
| r4ft | 6.0 | 0.6 | 36 | 40 | 15.62 | 20.0 | 5 | 40 |
| r5 | 3.0 | 0.4 | 40 | 40 | 8.5 | 10.02 | 19 | 40 |
| r5 | 3.0 | 0.5 | 40 | 40 | 9.5 | 11.75 | 15 | 40 |
| r5 | 3.0 | 0.6 | 37 | 40 | 10.5 | 14.7 | 15 | 40 |
| r5 | 4.0 | 0.4 | 40 | 40 | 9.25 | 11.25 | 16 | 40 |
| r5 | 4.0 | 0.5 | 38 | 40 | 10.75 | 13.65 | 13 | 40 |
| r5 | 4.0 | 0.6 | 37 | 40 | 12.25 | 16.6 | 11 | 40 |
| r5 | 6.0 | 0.4 | 39 | 40 | 11.25 | 14.0 | 13 | 40 |
| r5 | 6.0 | 0.5 | 37 | 40 | 13.25 | 17.75 | 10 | 40 |
| r5 | 6.0 | 0.6 | 36 | 40 | 16.12 | 22.12 | 5 | 40 |

## Recommendation

**Ship R5 as the default (done: `drivers.hearsay_mode = "r5"`, `r4ft` still selectable); leave the arming
thresholds alone.** Inside the pipeline the two modes behave almost the same:

- Arming: both arm on 37/40 fake callers (misses: MLAAD MegaTTS3 and Nari Dia, CVoiceFake Griffin-Lim, the same 3
  for both). R5 arms about 1 s later (median 13.2 s vs 12.0 s from the caller's first word, p90 17.4 vs 15.3 s),
  because the fusion is a little less extreme on fakes (fake V median 0.42 vs 0.47).
- False arms: both arm on the same 10/40 real callers (LibriSeVoc x4, ASVspoof5 x2, In-the-Wild x2, DFADD, MLAAD
  bona fide): R5 doesn't fix them. 4 of them are confident errors (p near 1 on most windows); the rest are single
  high windows that push V just past 0.5. LibriSpeech, LJSpeech, SONAR and ASVspoof2019 bona fide never arm. A false
  arm costs little here: the shield then listens and bleeps codes you read out, it doesn't cut the call.
- Latency: R5 costs about +90 ms per 4 s window on a quiet CPU (smoke run at 6-8 % load: r4ft 392-399 ms, r5 483 ms
  median; the benches above were taken with the Keyguard adversarial_eval job at 19-24 % load). It is fast enough for
  the 2 s hop: no window was ever skipped by the catch-up in any run, including the realtime ai_caller replay next to
  the Keyguard DSP shield and both Vosk spotters with the CPU at 100 % (another test suite was running on top of the
  Keyguard job): r5 median 1.66 s, max 1.81 s per window there. The r5 call-run latencies (median ~1.7 s) were
  measured under the same load, so they are an upper bound; the arming times include that latency.
- Thresholds: the data doesn't clearly support a change. A 3-4 s half-life arms 3-4 s sooner but adds 3-5 false
  arms out of 40; `arm_voice` 0.6 halves false arms (10 -> 5) but loses a fake and adds ~3 s. The honest fix for
  false arms is the model (R6) or calibration on meeting audio, not the EMA. Kept 0.5 / 0.3 / 6 s.

## Caveats

- Clean 16 kHz dataset audio, not a Meet call: no Opus codec, echo cancellation or noise suppression, which Hearsay never saw. The meeting-path numbers still need a real call.
- One speaker per call and no silence at the start; a real call has hellos and gaps (V decays toward 0 in silence after `voice_stale_s` 4 s).
- 3 of the 7 ASVspoof2019 LA bona fide calls had no window pass the pipeline's VAD (energy + zero-crossing gate, even after levelling), so they were never scored: their 'no false arm' says nothing about the model.
- 40 + 40 calls, one seed; the 10/40 false arms carry a wide interval (95 % CI roughly 13-41 %).
- Latency depends on what else runs: a Keyguard `adversarial_eval` job (8 processes) ran throughout, and a test suite overlapped the r5 call runs and the r5 demo. No quiet re-measurement was possible in the time box; the quiet numbers are from the smoke run (see the recommendation).
- The real Keyguard attacker failed to load in the demo (provisional KeyNet weights have 36 classes, Keyguard's CLASSES now has 37 with space), so the mock attacker stood in there.
