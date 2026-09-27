"""Control: NO pooled pretraining -- few-shot trains a fresh KeyNet on the target's
shots only. LOKO for this is identical to baseline, so run with `fewshot` only."""
from ..attackers.supervised import SupervisedAttacker
from ..features import mel
from .baseline import Baseline


class Scratch(Baseline):
    def fit(self, wins, y, dom):
        pass                                   # pooled data deliberately ignored

    def finetune(self, wins, y):
        self.atk = SupervisedAttacker()
        self.atk.fit(mel(wins), y, epochs=80, bs=32)


def make():
    return Scratch()
