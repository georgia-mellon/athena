"""Keystroke Guard pillar: can an eavesdropper read your keys from call audio, and does the shield stop it?

- Placeholders (mock.py): MockAttacker (reads the true key with a set accuracy, chance on shielded onsets) and
  MockShield (adds a pilot tone after each key event), which talk through the audio itself. No models.
- Real (driver.py): KeyguardCTCAttacker (Keyguard's current attacker: MtlCRNN, CNN + BiGRU + CTC with an onset head,
  weights KEYGUARD_ROOT/runs/ctc_rich_ft.pt; CALLGUARD_ATTACKER_WEIGHTS overrides) and KeyguardShield (Keyguard's DSP
  shield, streamed in 20 ms blocks with an 80 ms lookahead), from the teammate's LordKarV/keyboard-acoustic-shield,
  loaded read-only from KEYGUARD_ROOT (default ../keyboard-acoustic-shield). KeyguardAttacker (the older PROVISIONAL
  KeyNet CallGuard trained on harrison presses) stays for the adversarial-delta training in adversarial.py.
- Harness (harness.py): `python -m app.keystroke_guard.harness --attacker ... --shield ...`. See README.md.

Imports stay lazy: importing this package never loads torch.
"""


def check(*args, **kwargs):
    """app.keystroke_guard.harness.check: run the harness on an attacker and a shield, return its Report."""
    from app.keystroke_guard.harness import check
    return check(*args, **kwargs)


def main(argv=None) -> int:
    """app.keystroke_guard.harness.main: the harness CLI."""
    from app.keystroke_guard.harness import main
    return main(argv)
