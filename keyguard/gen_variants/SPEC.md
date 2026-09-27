# Variant spec — cross-keyboard generalization study

Context: HackGT defensive research project (KeyGuard). We study whether a keystroke-
audio classifier can generalize to keyboards it never saw during training, so the
DEFENSE can be evaluated against a realistic worst case. Prior pooled model got
leave-one-keyboard-out (LOKO) zero-shot ~6% top-3 (chance = 8.3%) — it failed to
generalize. Each variant tests one hypothesis for closing that gap.

## Contract
Create ONE file `keyguard/gen_variants/<name>.py` exposing `make() -> model` where:
- `model.fit(wins, y, dom)` — wins float32 (n, 4800) raw 0.3s@16kHz windows; y int class
  idx (0..35); dom int domain (keyboard) idx per sample.
- `model.predict_proba(wins) -> (n, 36)` numpy.
- optional `model.finetune(wins, y)` — adapt a pretrained model to a target keyboard's
  few labeled presses. If absent, the harness refits from scratch on pool+shots.

Read but DO NOT edit: `keyguard/gen_bench.py` (harness), `keyguard/gen_variants/baseline.py`
(reference), `keyguard/attackers/supervised.py` (KeyNet + mel), `keyguard/features.py`,
`keyguard/config.py`. Data: `data/pool/*.npz` = {wins, labels}. More keyboards may appear
mid-run; always glob.

## Run (always CPU — MPS is flaky)
```
KEYGUARD_DEVICE=cpu BENCH_THREADS=2 uv run python3 -W ignore -m keyguard.gen_bench keyguard.gen_variants.<name>
```
Writes runs/gen_bench/<name>.json with loko.mean_top1/top3 and fewshot per target@k.
Verify your variant BEATS baseline on LOKO top-3 (primary) and/or few-shot top-3. Keep
each full run under ~8 min on CPU (small nets, <=40 epochs, subsample if needed). Report
final numbers vs baseline and whether the hypothesis held.
