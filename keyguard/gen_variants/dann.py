"""DANN = domain-adversarial neural network.

Hypothesis: forcing the feature extractor to be keyboard-agnostic -- unable to tell
which keyboard produced a window -- improves transfer to unseen keyboards. Same KeyNet
conv trunk as baseline extracts features; a key-class head (36-way) classifies the
keystroke; a domain head behind a gradient-reversal layer (GRL) tries to predict which
keyboard (`dom`) the window came from. Backprop through the GRL flips the domain
gradient's sign, so the trunk is pushed to make its features BAD for domain
discrimination while staying good for key classification. Lambda (GRL strength) ramps
0 -> LAMBDA_MAX over training (standard DANN schedule) so the trunk isn't fighting the
domain head before the class head has learned anything useful to protect.

`finetune` targets a single new domain, so there's no second domain to adversarially
train against -- it just fine-tunes trunk+class head on cross-entropy (lam=0).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attackers.supervised import ConvSE, DEVICE
from ..config import N_CLASSES
from ..features import mel

EPOCHS = 40
FT_EPOCHS = 30
LAMBDA_MAX = 0.3   # ponytail: hand-tuned vs LOKO top-3; raise if trunk still leaks domain
GAMMA = 10.0       # DANN paper's ramp steepness


class _GRL(torch.autograd.Function):
    """Identity forward, sign-flipped-and-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None


def _grl(x, lam):
    return _GRL.apply(x, lam)


def _lambda(progress: float) -> float:
    """Standard DANN ramp: 0 at start of training, LAMBDA_MAX by the end."""
    return LAMBDA_MAX * (2.0 / (1.0 + np.exp(-GAMMA * progress)) - 1.0)


class DannNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_domains=2):
        super().__init__()
        self.b1 = ConvSE(1, 32)
        self.b2 = ConvSE(32, 64)
        self.b3 = ConvSE(64, 128)
        self.drop = nn.Dropout(0.3)
        self.cls_head = nn.Linear(128, n_classes)
        self.dom_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_domains))

    def features(self, x):
        x = self.b3(self.b2(self.b1(x)))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x, lam=0.0):
        feat = self.features(x)
        cls_out = self.cls_head(self.drop(feat))
        dom_out = self.dom_head(_grl(feat, lam))
        return cls_out, dom_out


def _to_tensor(wins, dtype=torch.float32):
    return torch.from_numpy(wins).to(DEVICE, dtype=dtype)


class Dann:
    def __init__(self):
        self.net: DannNet | None = None

    def fit(self, wins, y, dom, epochs=EPOCHS, lr=1e-3, bs=64):
        X = _to_tensor(mel(wins)[:, None])
        yb = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
        db = torch.as_tensor(dom, dtype=torch.long, device=DEVICE)
        n_domains = max(2, int(dom.max()) + 1 if len(dom) else 2)
        self.net = DannNet(n_domains=n_domains).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        n = len(X)
        steps_per_epoch = max(1, -(-n // bs))
        total_steps = max(1, epochs * steps_per_epoch - 1)
        step = 0
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = X[idx] + 0.04 * torch.randn_like(X[idx])   # light spec-aug, matches baseline
                lam = _lambda(step / total_steps)
                opt.zero_grad()
                cls_out, dom_out = self.net(xb, lam)
                loss = F.cross_entropy(cls_out, yb[idx]) + F.cross_entropy(dom_out, db[idx])
                loss.backward()
                opt.step()
                step += 1
            sched.step()
        return self

    def finetune(self, wins, y, epochs=FT_EPOCHS, lr=3e-4, bs=32):
        X = _to_tensor(mel(wins)[:, None])
        yb = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        n = len(X)
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                cls_out, _ = self.net(X[idx], 0.0)
                loss = F.cross_entropy(cls_out, yb[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, wins):
        if len(wins) == 0:
            return np.zeros((0, N_CLASSES), np.float32)
        self.net.eval()
        cls_out, _ = self.net(_to_tensor(mel(wins)[:, None]), 0.0)
        return F.softmax(cls_out, dim=1).cpu().numpy()


def make():
    return Dann()


def _demo():
    """ponytail self-check: GRL passes values through forward, flips+scales grad backward."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = _grl(x, 0.5)
    assert torch.allclose(y, x), "GRL forward must be identity"
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5)), "GRL backward must negate+scale grad"
    assert _lambda(0.0) == 0.0 and abs(_lambda(1.0) - LAMBDA_MAX) < 1e-3
    print("ok: GRL identity-forward/negated-backward and lambda ramp 0->LAMBDA_MAX")


if __name__ == "__main__":
    _demo()
