# 04: Attack proof: keystrokes are readable *under speech*, and the shield stops it

## Question
Hearsay already showed that keystrokes in the background don't break voice detection (hearsay
`docs/reports/generalization.md` §2: submitted model 0.037 → 0.083 at 0 dB; reals flagged ≤ 1.7 %). The reverse is what
makes Keyguard's threat real on a call: **with someone talking over the typing, can the attacker still read the
keys?** And if so, does the shield bring it back to chance while the speech stays intact?

## Design (`app/keystroke_guard/eval/attack_under_speech.py`, CPU)
- **Keys:** Keyguard's press banks `KEYGUARD_ROOT/data/pool/*.npz` (`wins` (n, 4800) 16 kHz, labels A-Z/0-9).
  In-domain = `harrison` (36 classes, 25 presses each). Split per key, seeded, 60/40 train/test. Each press's
  test windows are never used in training.
- **Attacker:** Keyguard's `KeyNet` + `torch_logmel` (`keyguard.attackers.supervised`, `keyguard.shield.adversarial`),
  trained with `train_attacker` on the clean train split. This is a **provisional attacker**, marked as such, until the
  teammate ships trained weights. Attacker variants: `clean` (trained on clean keys) and `speech-aug` (trained with
  random speech mixed in). The second is the stronger, adaptive attacker.
- **Speech:** real speech clips (LibriSpeech via `librosa.ex` like Keyguard's synth, or clips from
  `HEARSAY_ROOT/data/processed`, read-only), mixed into each test press at speech-to-key ratios
  **−∞ (keys only), −10, −5, 0, +5, +10, +20 dB**. Realistic calls sit around +5 to +20 dB: the voice is louder
  than the keyboard.
- **Segmentation:** (a) oracle onsets (the key is at `PRE_S`); (b) detected onsets with `keyguard.segment.onsets`
  on the full mixture. The attacker only has (b).
- **Shield:** Keyguard DSP `Shield` with true timestamps (the victim side has OS key events) applied to the mixture;
  then re-run both attackers. `adversarial` shield when available.
- **Metrics:** top-1 / top-3 accuracy vs chance (1/36 = 2.8 %) with 95 % Wilson CIs; for the shield, also speech
  quality of the shielded mixture vs the original: STOI, PESQ (if installable), and Hearsay `p_synthetic` on the
  shielded speech (it must not make a real voice look fake).
- **Output:** `docs/reports/attack_under_speech.md` + `docs/reports/attack_under_speech.csv` + a figure (accuracy vs speech
  level, raw vs shielded, chance line).

## Pass criteria (what we want to be able to say on stage)
1. At a realistic +10 dB speech-over-keys, the adaptive attacker's top-1 is **far above chance** (CI lower bound
   > 3× chance). Otherwise the threat story needs the LM (Gemini lattice) step, which is reported as-is.
2. With the shield on, top-1 falls to **≤ 2× chance**, and STOI stays ≥ 0.9.
3. Hearsay flags ≤ 2 % of real voices on the shielded speech.
Results are reported as they come out, including a failure. A failed criterion becomes an issue for the teammate
(e.g. "the shield needs the adversarial D against the speech-aug attacker").
