"""Hearsay pillar: is the voice on the call real or synthetic?

- Placeholder (mock.py): MockVoice, a deterministic stand-in (spectral flatness or a scripted schedule), no weights.
- Real (driver.py): HearsayDriver around Hearsay's frozen R4ft XLS-R fine-tune (optionally the R5 fusion with the R1
  LightGBM), from the Hearsay model repository (https://github.com/danmano411/hearsay), loaded
  read-only from HEARSAY_ROOT (default ../Hearsay).
- Harness (harness.py): `python -m app.hearsay.harness --driver mock|real|module.path:ClassName` checks a driver
  against the contract, the latency budget and a few held-out clips. See README.md.

Imports stay lazy: importing this package never loads torch.
"""


def check(*args, **kwargs):
    """app.hearsay.harness.check: run the harness on a driver, return its Report."""
    from app.hearsay.harness import check
    return check(*args, **kwargs)


def main(argv=None) -> int:
    """app.hearsay.harness.main: the harness CLI."""
    from app.hearsay.harness import main
    return main(argv)
