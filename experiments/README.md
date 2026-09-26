# experiments

Offline evidence for the claims we make on stage. Each script is standalone, seeded, CPU-only, reads the upstream
repos read-only (`KEYGUARD_ROOT`, `HEARSAY_ROOT`, defaulting to the sibling checkouts) and writes only to `reports/`.

| script | question | output |
|---|---|---|
| `attack_under_speech.py` | Can a keystroke attacker still read keys with someone talking over the typing, and does Keyguard's shield stop it without hurting the speech? (plan 04) | `reports/attack_under_speech.{md,csv}`, `reports/attack_under_speech_hearsay.csv`, `reports/figures/attack_under_speech.png`; also saves the speech-aug attacker to `runs/provisional_keynet_speechaug.pt` (gitignored) for the live pipeline |

```
cd CallGuard
.venv/Scripts/python experiments/attack_under_speech.py     # ~12 min on 8 CPU threads: ~7.5 attack, ~4.5 Hearsay check
```

Needs: Keyguard's `data/pool/harrison.npz`, Hearsay's `data/processed/manifest.parquet` and its processed
LibriSpeech/LJSpeech clips. Python deps beyond the app: `matplotlib`, `pesq` (both in `pyproject.toml`).
The attacker here is trained in the script on Keyguard's public harrison bank (provisional); swap in the
teammate's weights when they ship (plan 05).
