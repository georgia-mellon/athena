# GPU retrain handoff (2026-09-27)

CallGuard's keystroke attacker is now Keyguard's CTC model (`MtlCRNN`, `runs/ctc_rich_ft.pt`). Job 1 (shield deltas) runs on the Mac; job 2
(a stronger Ares) runs on the GPU. Run on the GPU machine, then copy the outputs back to
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

## 2. GPU: one combined Ares run (replaces the SKAID-only job)

`ctc_rich_ft.pt` came from mixed data (commit 26fa6b0: synth from the rich bank, then the real Mac sessions, with
source mixing). `train_overlap` takes all three sources in one run; warm-start from the current best so it can only
start from where Ares is now. With SKAID in the mix the same model should also read the pipeline's SKAID clip, so
there is no separate SKAID-only job (`ctc_skaid_crnn.pt` stays retired; the pipeline defaults to `ctc_rich_ft.pt`).

```bash
cd keyboard-acoustic-shield
git checkout 5143272 -- data/continuous/live          # the real Mac sessions (removed from the tree in a369eaa)
KEYGUARD_SKAID=data/continuous/live/labels.jsonl \
KEYGUARD_MIX_SKAID=data/continuous/skaid/labels.jsonl \
KEYGUARD_BANK=data/live_bank_rich.npz KEYGUARD_MIX_SYNTH=2000 \
KEYGUARD_INIT=runs/ctc_rich_ft.pt KEYGUARD_LR=1e-4 KEYGUARD_SPLIT=phrase \
KEYGUARD_CKPT=runs/ctc_mix_ft.pt \
  uv run python3 -m keyguard.ctc.train_overlap 4000
```

- These flags are reconstructed from the code (the exact settings of the original run live in Dan's GPU session, not
  the repo): real Mac sessions as the primary set, SKAID and 2000 bank-synth lines mixed in, warm start, lower LR for
  fine-tuning, phrase-disjoint held-out split.
- Compare against the current best on novel text, same eval as DEMO.md's ~80 %:
  `uv run python scratch/eval_live.py runs/ctc_mix_ft.pt` vs `uv run python scratch/eval_live.py runs/ctc_rich_ft.pt`
  (look at `greedy_acc`; ctc_rich_ft: 0.804).
- **Only if it wins:** copy `runs/ctc_mix_ft.pt` to the Mac as `callguard/upstream/keyguard/runs/ctc_rich_ft.pt` (keep
  the old file as a backup), then rerun job 1 against it. If it loses, keep `ctc_rich_ft.pt`.

## Optional: a stronger Ares

`ctc_rich_ft.pt` came from the rich-synth recipe (commit 26fa6b0: synth from `live_bank_rich.npz`, then fine-tune on
the real sessions in `data/continuous/live`, `keyguard.agents.finetune_live`). Rerunning it with more steps is Dan's
GPU agent's call (`GPU_AGENT.md`). If Ares changes, rerun job 1 against the new weights.
