# GPU retrain handoff (2026-09-27)

CallGuard's keystroke attacker is now Keyguard's CTC model (`MtlCRNN`, `runs/ctc_rich_ft.pt`). Two artifacts are
stale against it and need a training run; nothing else does. Run on the GPU machine, then copy the outputs back to
the Mac. Neither job changes any contract or config.

## 0. Setup on the GPU machine

```bash
# CallGuard, this branch
git fetch && git checkout wp11-keyguard-ctc-attacker && uv sync
# Keyguard (the teammate's repo), latest main: has the Backboard-agent changes
cd ../keyboard-acoustic-shield && git pull && uv sync        # or wherever KEYGUARD_ROOT points
```

Data both jobs need, all in Keyguard's checkout (`KEYGUARD_ROOT`):

- `data/live_bank_rich.npz`, `runs/ctc_rich_ft.pt` (tracked in Keyguard's git).
- `data/continuous/skaid/` (SKAID, for job 2). Missing? `GPU_AGENT.md` section 1b downloads and converts it.
- `data/continuous/live/*.wav` + `labels.jsonl` (the real Mac sessions). Commit a369eaa removed them from Keyguard's
  tree; restore with `git checkout 5143272 -- data/continuous/live`.

Job 1 also needs Hearsay's speech pools (`HEARSAY_ROOT`, as for the earlier adversarial training).

## 1. CallGuard's adversarial deltas vs the CTC attacker (the dashboard's Adversarial button)

```bash
cd callguard
KEYGUARD_ROOT=../keyboard-acoustic-shield HEARSAY_ROOT=../Hearsay \
  uv run python -m app.keystroke_guard.adversarial train --attacker ctc --threads 8
```

- Optimizes the K = 8 universal deltas through the CTC attacker's differentiable log-mel on the Keyguard bank (10
  presses per key held out), same -18 dB budget, EOT and runtime format as before. ~2 s/step on 8 CPU threads,
  1000 steps (~30 min); CPU is fine.
- Output: `runs/adversarial_deltas.pt`. Copy it to the Mac's `callguard/runs/`.
- Check: `uv run python -m app.keystroke_guard.harness --attacker real --shield real --shield-mode dsp+adversarial`.
  The shielded top-1 should drop below the dsp-only row (bank presses: raw 54.1 %, dsp 14.6 %).

## 2. The multi-agent pipeline's SKAID model (38 symbols)

`runs/ctc_skaid_crnn.pt` predates the space key (37 symbols) and no longer loads, so `keyguard.agents.pipeline` now
defaults to `ctc_rich_ft.pt`. That model can't read SKAID's other typists (the grounding clip decodes as garbage).
Retrain it at 38 symbols:

```bash
cd ../keyboard-acoustic-shield
KEYGUARD_SKAID=data/continuous/skaid/labels.jsonl KEYGUARD_CKPT=runs/ctc_skaid_crnn.pt \
  uv run python3 -m keyguard.ctc.train_overlap 6000
```

- Output: `runs/ctc_skaid_crnn.pt`. Copy it to the Mac's Keyguard `runs/`, then run the pipeline with
  `uv run python -m keyguard.agents.pipeline --ckpt runs/ctc_skaid_crnn.pt`.

## Optional: a stronger Ares

`ctc_rich_ft.pt` came from the rich-synth recipe (commit 26fa6b0: synth from `live_bank_rich.npz`, then fine-tune on
the real sessions in `data/continuous/live`, `keyguard.agents.finetune_live`). Rerunning it with more steps is Dan's
GPU agent's call (`GPU_AGENT.md`). If Ares changes, rerun job 1 against the new weights.
