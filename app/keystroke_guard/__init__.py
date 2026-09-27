"""Keystroke Guard pillar: can an eavesdropper read your keys from call audio, and does the shield stop it?

- Placeholders (mock.py): MockAttacker (reads the true key with a set accuracy, chance on shielded onsets) and
  MockShield (adds a pilot tone after each key event), which talk through the audio itself. No models.
- Real (driver.py): KeyguardAttacker (Keyguard's KeyNet) and KeyguardShield (Keyguard's DSP shield, streamed in 20 ms
  blocks with an 80 ms lookahead), from the teammate's LordKarV/keyboard-acoustic-shield, loaded read-only from
  KEYGUARD_ROOT (default ../keyboard-acoustic-shield). Until their weights ship the attacker is a PROVISIONAL KeyNet
  that CallGuard trains on Keyguard's harrison presses (runs/provisional_keynet*.pt; CALLGUARD_ATTACKER_WEIGHTS
  overrides).
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
