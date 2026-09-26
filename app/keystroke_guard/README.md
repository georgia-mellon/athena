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
| real attacker | `app.keystroke_guard.driver:KeyguardAttacker` | Keyguard's `KeyNet` on `torch_logmel` features. Weights: `CALLGUARD_ATTACKER_WEIGHTS` / `attacker_weights`, else the **provisional** speech-augmented KeyNet CallGuard trained on Keyguard's harrison presses (`runs/provisional_keynet_speechaug.pt`), else a clean one it trains on first use |
| real shield | `app.keystroke_guard.driver:KeyguardShield` | Keyguard's DSP `Shield`, streamed with an 80 ms lookahead; `key_frames=26` (press + release) |

Real code comes read-only from the teammate's repo
[LordKarV/keyboard-acoustic-shield](https://github.com/LordKarV/keyboard-acoustic-shield) at `KEYGUARD_ROOT`
(default `../keyboard-acoustic-shield`). Config ([`callguard.example.toml`](../../callguard.example.toml)):
`[drivers] attacker`, `shield` = `"real"|"mock"`, `shield_mode = "off"|"dsp"|"adversarial"` (adversarial waits
for the teammate), top-level `attacker_weights`.

## Plug in a new model

- **New attacker weights for KeyNet** (the teammate's): no code. Set `CALLGUARD_ATTACKER_WEIGHTS=path/to.pt` (a
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
(per-key seeded 60/40 split, the 40 %: 360 presses; skipped without `KEYGUARD_ROOT`): the attacker's keys-only top-1 /
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
  contract  shield Protocol + latency          PASS  name='keyguard-dsp', latency 1280 samples = 80 ms
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

## Known limits

- The attacker is **provisional** (CallGuard-trained, one MacBook keyboard, isolated presses, in-domain). It is
  replaced when the teammate's weights ship.
- The shield misses plan 04's bar against the adaptive attacker at +10 dB (5.8 % vs <= 5.6 %) and its STOI is just
  under 0.9. `key_frames=26` was chosen on the same test presses (mild selection effect).
- Synthetic mixing, no codec or meeting noise suppression; the real-world attack is capped by onset detection
  (49 % of presses found under +10 dB speech). See the report's "Issues for the teammate" and "Limits".
