"""Baseline = the current attacker (KeyNet on per-window log-mel), pooled over keyboards.
finetune continues training on the target's shots (transfer learning)."""
from ..attackers.supervised import SupervisedAttacker
from ..features import mel

EPOCHS = 40
FT_EPOCHS = 30


class Baseline:
    def __init__(self):
        self.atk = SupervisedAttacker()

    def fit(self, wins, y, dom):
        self.atk.fit(mel(wins), y, epochs=EPOCHS)

    def finetune(self, wins, y):
        self.atk.fit(mel(wins), y, epochs=FT_EPOCHS, lr=3e-4, bs=32)

    def predict_proba(self, wins):
        return self.atk.predict_proba(mel(wins))


def make():
    return Baseline()
