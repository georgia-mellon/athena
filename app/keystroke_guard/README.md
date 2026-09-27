# Keystroke Guard pillar: can the call hear your keys?

Two drivers. The **attacker** plays the eavesdropper: it reads keys from your outgoing call audio at the OS key
timestamps, and its readout drives the threat score (`keys.readout`). The **shield** runs on the audio thread in
20 ms blocks and destroys the keystroke sound before it leaves your mic, keeping speech intact.

## Contract

In [`app/source/types.py`](../source/types.py):

- `KeystrokeAttackerDriver`: `name`, `classes: list[str]`, `read(audio, onsets) -> list[KeyGuess]`, one guess per
  onset (sample indices in `audio`, the edges included), `top` = `[(key, prob), ...]` best first, keys in `classes`.
  The attacker never gets the true key (`truth` stays None; only a mock may opt in with `wants_truth`).
  Deterministic. Budget: < 50 ms per keystroke.
- `ShieldDriver`: `name`, `process(block, key_events) -> block` (same length, finite), `reset()`, optional
  `latency` (output delay in samples). `key_events` are absolute sample indices since `reset()`, each passed once,
  possibly a block or two late. With no key events the output must be the input, delayed by `latency`, exactly.
  Budget: < 20 ms per 20 ms block with a key active.

## Placeholder vs real

| | class | what it does |
|---|---|---|
| placeholder attacker | `app.keystroke_guard.mock:MockAttacker` | reads the true key with a set accuracy when the pipeline hands it the truth, chance otherwise and on shielded onsets |
| placeholder shield | `app.keystroke_guard.mock:MockShield` | adds a quiet 7.6 kHz pilot tone for 100 ms after each key event (the mock attacker listens for it) |
| real attacker | `app.keystroke_guard.driver:KeyguardCTCAttacker` | Keyguard's current attacker ("Ares"): `MtlCRNN` (CNN + BiGRU + CTC + per-frame onset head) on Keyguard's `ctc.model.logmel`, 37 keys (A-Z, 0-9, space). Weights: `ATHENA_ATTACKER_WEIGHTS` / `attacker_weights`, else `runs/keyguard/ctc_rich_ft.pt`. Each onset is read like Keyguard's `onset_gated_decode` reads a peak (non-blank logits, +/-1 frame at the onset frame, 1 s window) |
| older attacker | `app.keystroke_guard.driver:KeyguardAttacker` | the **provisional** KeyNet Athena trained on harrison presses; no longer the default, kept because `adversarial.py`'s deltas are trained against it |
| real shield | `app.keystroke_guard.driver:KeyguardShield` | Keyguard's DSP `Shield`, streamed with an 80 ms lookahead; `key_frames=26` (press + release) |

Real code is the teammate's Keyguard
([LordKarV/keyboard-acoustic-shield](https://github.com/LordKarV/keyboard-acoustic-shield), commit `55bb112`),
**vendored** as the top-level `keyguard/` package (`import keyguard...`; local edits listed in
[`keyguard/VENDORED.md`](../../keyguard/VENDORED.md)). Athena never imports from the teammate's checkout.

**Weights and data** (gitignored): `uv run python -m app.keystroke_guard.get_assets` copies them once from a Keyguard
checkout (`KEYGUARD_ROOT`, else `upstream/keyguard`, `../../keyboard`, `../keyboard-acoustic-shield`; `--root PATH`)
into `runs/keyguard/` (`ctc_rich_ft.pt`, `demo_attacker.pt`, `supervised_mbp.pt`, `arena/`, `arena_memory.jsonl`, ...)
and `data/keyguard/` (`live_bank_rich.npz`, `pool/harrison.npz`, `speech/`, `harrison/MBPWavs/`, ~300 MB), which is
where `keyguard.config.RUNS` / `DATA` point. `.env` (`GEMINI_API_KEY`, `BACKBOARD_API_KEY`) is athena's own, loaded by
`keyguard.config`. Keyguard's tools run from here, e.g. `uv run python -m keyguard.agents.arms_race_demo` (Ares vs Athena;
writes `runs/keyguard/arms_race_data.js`), `uv run uvicorn keyguard.server:app --port 8000`, `uv run python -m keyguard.agents.live
record --defend`. Config ([`athena.example.toml`](../../athena.example.toml)):
`[drivers] attacker`, `shield` = `"real"|"mock"`, `shield_mode = "off"|"dsp"|"adversarial"` (see below),
top-level `attacker_weights`.

## Plug in a new model

- **New attacker weights for KeyNet** (the teammate's): no code. Set `ATHENA_ATTACKER_WEIGHTS=path/to.pt` (a
  bare `state_dict` or `{"state_dict": ...}`) and run the harness with `--attacker real`.
- **A new attacker or shield class**: implement the Protocol, then
  `python -m app.keystroke_guard.harness --attacker mypkg.mod:MyAttacker --shield mypkg.mod:MyShield`
  (built with no arguments; `none` skips one). Fix every FAIL, then add it to `make_attacker` / `make_shield` in
  [`app/source/registry.py`](../source/registry.py) (integrator change).

## Harness

```
python -m app.keystroke_guard.harness [--attacker mock|real|module:Class|none] [--shield ...] [--presses N] [--no-quality]
```

Contract and latency checks for both drivers, then two informational rows on Keyguard's harrison TEST-split presses
(per-key seeded 60/40 split, the 40 %: 360 presses; skipped without the Keyguard data): the attacker's keys-only top-1 /
top-3 at the true onsets vs chance, and the same attacker on those presses streamed through the shield (presses
250 ms apart, 20 ms blocks, each key event one block late). Exit code 1 on any FAIL. Real run (CPU, 2026-09-26):

```
Keystroke Guard harness: attacker=real, shield=real
  load      build attacker                     PASS  attacker KeyguardAttacker in 1.8 s
  contract  attacker Protocol + attributes     PASS  name='keyguard-keynet (provisional, speech-aug)', 36 classes
  contract  read() -> [KeyGuess]               PASS  one KeyGuess per onset (edges too), top-3 sorted, keys in classes, truth None, [] for none
  contract  attacker deterministic             PASS  same input twice, same guesses
  latency   attacker read 1 keystroke          PASS  median 1.2 ms, max 2.0 ms per keystroke (budget 50 ms)
  load      build shield                       PASS  shield KeyguardShield in 0.9 s
  contract  shield Protocol + latency          PASS  name='keyguard-shield', latency 1280 samples = 80 ms
  contract  no keys -> exact pass-through      PASS  exact pass-through (delayed 1280 samples) with no key events
  contract  key event -> audio changed         PASS  same length, finite; changed samples -695..+3570 around the key (late event)
  latency   shield 20 ms block, key active     PASS  median 3.1 ms, p95 4.6 ms, max 5.6 ms per 20 ms block, key active; +80 ms constant delay
  quality   attacker on harrison test presses  INFO  keys-only, oracle onsets, n=360: top-1 57.8 %, top-3 89.2 % (chance 2.8 / 8.3 %)
  quality   shield vs that attacker            INFO  shielded, oracle onsets, n=360: top-1 11.4 %, top-3 25.0 % (unshielded 57.8 / 89.2 %)
  => FITS  (10 PASS, 2 INFO)
```

## Measured numbers

From [`docs/reports/attack_under_speech.md`](../../docs/reports/attack_under_speech.md) (speech-augmented provisional
attacker, n = 360 test presses, top-1, chance 2.8 %):

| | keys only | +10 dB speech over the keys |
|---|---|---|
| oracle onsets, shield off | 53.6 % | 15.8 % |
| oracle onsets, shield on (26 frames) | 10.8 % | 5.8 % |
| detected onsets, shield off | 47.5 % | 5.8 % |
| detected onsets, shield on | 11.4 % | 2.5 % |

Speech quality with the shield at +10 dB: STOI 0.897, PESQ 2.86. The harness's own keys-only rows (57.8 % -> 11.4 %)
differ slightly from the report's because it streams the presses 250 ms apart through the live 80 ms-lookahead
shield rather than the report's offline cut.

## Adversarial mode

[`adversarial.py`](adversarial.py) is Keyguard's adversarial shield (`keyguard/shield/adversarial.py`: universal
bounded perturbation through the attacker's differentiable log-mel) made live. `KeyguardShield.set_mode` switches
`dsp` | `adversarial` (delta only) | `dsp+adversarial` (DSP inpainting, then the delta) at runtime on the same
driver, so the 80 ms lookahead, and the stream's delay, never change. The dashboard's **Adversarial** button (and
`shield_mode = "adversarial"`) runs `dsp+adversarial` (`driver.DASHBOARD_ADVERSARIAL`): it measured best (below).

- **Runtime.** At each OS key event `e`, one of K = 8 deltas (picked at random per stroke) is added on
  `[e - 20 ms, e + 280 ms)` (Keyguard's window). Scale: level x delta, where level = RMS of the raw input over
  `[e - 20 ms, e + 40 ms)` x `level_gain` (0.505, the median full-window/first-60-ms RMS ratio of the train presses).
  That window is always inside the lookahead before the delta's first sample is due, so there is no warm-up and no
  EMA: the first stroke is scaled like every other. The budget is Keyguard's, `||delta||_2 <= level * 10^(-18/20) *
  sqrt(KEY_WIN)`, per stroke; on keys alone the level is the key-window RMS, under speech it is the window's RMS
  (speech + key), which masks it. Measured per-stroke key-to-delta ratio on the 360 test presses: median 18.0 dB,
  min 17.0 dB (the level estimate can overshoot the budget by up to 1 dB). CPU: ~0 ms per block for the delta,
  DSP as before (harness p95 3.2 ms per 20 ms block in `dsp+adversarial`). No key events = exact pass-through.
- **Hardening (2026-09-26 review fixes).** The delta is picked with an OS-entropy rng and shifted by a random +/-10 ms
  per stroke (inside the trained jitter), so an attacker holding the 8 deltas can't subtract a predictable pattern.
  Deltas are tapered (3 ms in, 20 ms out) so strokes don't start or end with a step, and a running delta fades out
  over 5 ms when a new stroke or a mode switch cuts it. Repeated events within 30 ms count once. Overlapping strokes
  are capped: every attacker window near a stroke stays within that stroke's budget (probe: 5 presses 120 ms apart,
  worst window exactly 18.0 dB below the key level; before the fix, 15 dB). A late OS event shifts its delta later
  (up to 40 ms) instead of dropping its head. In `dsp+adversarial` the level is taken from the DSP output, so the
  budget holds for what actually goes out.
- **Missing `runs/adversarial_deltas.pt`**: `set_shield("adversarial")` raises a clear 400 on the dashboard and the
  mode stays where it was; a config asking for it starts on dsp and publishes `driver.error`. dsp / off unaffected.
- **Training** (`python -m app.keystroke_guard.adversarial train [--budget-db -18] [--k 8] [--steps 500]`; 457 s on
  8 CPU threads, 2026-09-26): harrison TRAIN split only (`harrison_split`, seed 0, 540 presses). Loss: a clipped
  margin (kappa 5) summed over an ensemble, plus a pairwise-cosine penalty between the K deltas (final mean |cos|
  ~0). Expectation over transformation per draw: onset error uniform +/-40 ms (delta and level window move, the
  attacker's window doesn't), gain +/-6 dB, train-speaker speech (Hearsay test_internal pools of
  `eval/attack_under_speech.py`, LibriSpeech 100 / 2803 excluded) at +0..+20 dB in 70 % of draws, the runtime level
  estimate. Phase 1 (500 steps) vs 3 KeyNets: the clean + speech-aug provisional ones and `keynet-s1-speechaug`
  (trained here, seed 1). Then `keynet-s2-advretrain` (s1 retrained 15 epochs on speech + phase-1 deltas, co_train's
  move) joins and phase 2 (500 steps) runs vs all 4. **Held out of training:** `widecnn-s3-speechaug` (Keyguard's
  WideCNN population architecture, seed 3) and `keynet-s4-retrained-on-deltas` (a fresh KeyNet trained on train
  presses + speech + the FROZEN final deltas: the adaptive attacker). Attackers are cached in `runs/adv_attacker_*`.
- **Eval, measured before the hardening above** (`python -m app.keystroke_guard.adversarial eval`, 297 s; writes `runs/adversarial_eval.json`); * the speech-aug provisional attacker had seen 206 test presses (split bug, fixed since: one `harrison_split` everywhere): top-1 on
  the 360 harrison TEST presses, oracle onsets, chance 2.8 %. The first four columns are **white-box** (the deltas were
  optimized against them). Streamed rows go through the live driver (presses 250 ms apart, events one block late);
  speech rows use the other (test) speaker pool.

| condition | clean prov. | speech-aug prov.* | s1 | s2 advretrain | held-out WideCNN | adaptive retrained |
|---|---|---|---|---|---|---|
| no shield | 70.8 | 57.8 | 46.1 | 54.2 | 43.6 | 37.5 |
| white noise, same L2 budget, +/-40 ms | 2.8 | 26.4 | 8.6 | 29.2 | 13.6 | 24.4 |
| delta, aligned | 2.8 | 3.9 | 2.8 | 4.7 | 5.0 | 21.4 |
| delta, +/-40 ms jitter | 2.8 | 3.6 | 2.8 | 4.4 | 5.0 | 23.9 |
| stream dsp | 5.0 | 11.4 | 7.8 | 11.7 | 5.6 | 7.8 |
| stream adversarial | 2.8 | 3.6 | 2.8 | 3.3 | 5.3 | 19.4 |
| **stream dsp+adversarial** | 2.8 | 2.8 | 2.8 | 2.2 | 3.6 | 4.4 |
| +10 dB speech, no shield | 3.1 | 11.9 | 7.2 | 10.6 | 6.9 | 4.2 |
| stream dsp, +10 dB speech | 2.8 | 5.0 | 3.9 | 5.6 | 4.4 | 3.3 |
| stream adversarial, +10 dB speech | 3.9 | 5.3 | 3.9 | 7.2 | 4.7 | 5.8 |
| stream dsp+adversarial, +10 dB speech | 3.3 | 4.2 | 2.8 | 3.9 | 4.7 | 2.8 |

Reading it honestly: white noise at the same budget already blinds the clean provisional attacker (70.8 -> 2.8 %), so
that column proves nothing; the trained deltas beat noise on every speech-trained attacker (e.g. 29.2 -> 4.4 %). The
delta alone transfers to the held-out architecture (43.6 -> 5.0 %) but **an attacker retrained on the frozen deltas
claws back to 19-24 %** (Keyguard saw 18-56 % in the same situation), which is why the dashboard runs
`dsp+adversarial`: 4.4 % against that attacker. That 4.4 % is not a fully adaptive number: s4 was retrained against
the deltas, not against dsp+deltas. With 360 presses the 95 % interval around 3-5 % is roughly +/-2 points.
\* the speech-aug provisional attacker was trained by `attack_under_speech.load_keys`, whose per-key split iterates
keys in a different order than `harrison_split`: only 334 of its 540 train presses are in `harrison_split`'s train
set, so part of this test split is in its training data and its column (and the harness quality rows, which use
it) is optimistic for the attacker.

Harness (`--shield-mode`, same attacker as above): shielded top-1 11.4 % (dsp), 3.6 % (adversarial), 2.8 %
(dsp+adversarial), from 57.8 % unshielded; all contract and latency checks PASS in every mode.

## Known limits

- **The CTC attacker is trained on the teammate's MacBook** (`ctc_rich_ft`, rich-synth fine-tune), so on harrison
  presses it is near chance (`--data harrison`: top-1 5.0 %, top-3 9.7 %; Keyguard's own `onset_gated_decode` fails the
  same way). The harness's quality rows therefore default to Keyguard's per-key bank (`data/live_bank_rich.npz`, 10
  presses per key, its 0.002 synth noise floor; it is the model's training bank, so optimistic): **raw top-1 54.1 % /
  top-3 74.1 %, DSP shield 14.6 % / 26.2 %** (n = 370, chance 2.7 / 8.1 %, 2026-09-26). On typing synthesized from the
  bank (40-75 wpm, n = 458): raw 77.1 % / 89.3 %, DSP 62.9 % / 78.4 %: with neighbours in context
  the DSP shield dents it far less, as Keyguard's DEMO.md says; the adversarial deltas were trained against KeyNet, not this model.
  Retrain them against it with `adversarial train --attacker ctc`.
- **CTC deltas, first run** (2026-09-27, `train --attacker ctc --speech keyguard`: Keyguard's 10 speech clips, 1000
  steps, 873 s CPU, margin loss 4.16 -> 2.89, still falling). Isolated bank presses (harness, n = 370): top-1 54.1 % raw,
  14.6 % dsp, 21.1 % adversarial, **3.8 % dsp+adversarial** (chance 2.7 %). Typing synthesized from the bank at
  40-75 wpm (neighbours in context, n = 458): 77.1 % raw, 62.9 % dsp, 54.8 % dsp+adversarial: the deltas are trained
  on isolated windows and don't hold up under overlap. Next: rerun with Hearsay speech (`--speech hearsay`), and train
  on typing contexts rather than isolated presses.

- The default attacker is now the teammate's (above). The KeyNet rows and the adversarial table describe the older
  **provisional** KeyNet (Athena-trained, one MacBook keyboard, isolated presses, in-domain).
- The shield misses plan 04's bar against the adaptive attacker at +10 dB (5.8 % vs <= 5.6 %) and its STOI is just
  under 0.9. `key_frames=26` was chosen on the same test presses (mild selection effect).
- Synthetic mixing, no codec or meeting noise suppression; the real-world attack is capped by onset detection
  (49 % of presses found under +10 dB speech). See the report's "Issues for the teammate" and "Limits".
