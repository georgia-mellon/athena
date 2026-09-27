"""Secret Shield pillar: redact the codes, passwords and card numbers you read aloud while the caller is unverified.

- Placeholder (mock.py): MockSpotter, a scripted spotter (spans at set times, with a recognizer-like lag). No model.
- Real, built here: spotter.py (VoskSpotter, a streaming Vosk recognizer that emits spans, never text) and
  redactor.py (the 500 ms delay line that mutes/tones the spans out). The Vosk model is downloaded with
  `python -m app.secret_shield.get_model` into runs/models/.
- Harness (harness.py): `python -m app.secret_shield.harness --spotter mock|real|module.path:ClassName`. See README.md.

Imports stay lazy: importing this package never loads torch or vosk.
"""


def check(*args, **kwargs):
    """app.secret_shield.harness.check: run the harness on a spotter, return its Report."""
    from app.secret_shield.harness import check
    return check(*args, **kwargs)


def main(argv=None) -> int:
    """app.secret_shield.harness.main: the harness CLI."""
    from app.secret_shield.harness import main
    return main(argv)
