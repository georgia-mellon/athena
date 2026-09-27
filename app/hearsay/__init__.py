"""Hearsay pillar: is the voice on the call real or synthetic?

- mock.py: MockVoice, a deterministic stand-in, no weights.
- driver.py: HearsayDriver around Hearsay's fusion models, loaded read-only from HEARSAY_ROOT (default ../Hearsay);
  from https://github.com/danmano411/hearsay.
- harness.py: `python -m app.hearsay.harness --driver mock|real|...` checks a driver against the contract, latency
  and a few held-out clips.

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
