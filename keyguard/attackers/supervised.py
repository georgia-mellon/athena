"""Supervised acoustic keystroke classifier (Harrison et al. 2023 style).

Pipeline: onset-segmented mel-spectrograms -> small CoAtNet-flavored CNN
(conv stem + squeeze-excite blocks) -> 36-way softmax. We keep it compact so
it retrains in seconds inside the arena loop rather than a full CoAtNet.
"""
from __future__ import annotations
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from ..config import N_MELS, N_CLASSES, CLASSES

# KEYGUARD_DEVICE overrides auto-detect; MPS training is flaky here (see HANDOFF),
# so co-training pins cpu for reliable, reproducible numbers.
DEVICE = os.environ.get("KEYGUARD_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu")


class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // 4 or 1, 1)
        self.fc2 = nn.Conv2d(c // 4 or 1, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConvSE(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.se = SEBlock(cout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return F.max_pool2d(x, 2)


class KeyNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.b1 = ConvSE(1, 32)
        self.b2 = ConvSE(32, 64)
        self.b3 = ConvSE(64, 128)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):            # x: (n,1,mel,frames)
        x = self.b3(self.b2(self.b1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.drop(x))


class SupervisedAttacker:
    def __init__(self, n_classes=N_CLASSES, classes=CLASSES):
        self.net = KeyNet(n_classes).to(DEVICE)
        self.classes = classes

    def _batch(self, mels: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(mels[:, None]).float().to(DEVICE)

    def fit(self, mels, labels, epochs=60, lr=1e-3, bs=64, aug=True, log=None):
        X = self._batch(mels)
        y = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        n = len(X)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n, device=DEVICE)
            total = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = X[idx], y[idx]
                if aug:
                    xb = self._specaug(xb)
                opt.zero_grad()
                loss = F.cross_entropy(self.net(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            sched.step()
            if log and (ep % 10 == 0 or ep == epochs - 1):
                log(ep, total / n)
        return self

    def _specaug(self, xb):
        xb = xb + 0.04 * torch.randn_like(xb)                    # light noise only
        xb = torch.roll(xb, shifts=int(torch.randint(-1, 2, (1,))), dims=-1)
        return xb

    @torch.no_grad()
    def predict_proba(self, mels) -> np.ndarray:
        if len(mels) == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        self.net.eval()
        return F.softmax(self.net(self._batch(mels)), dim=1).cpu().numpy()

    def predict(self, mels) -> list[str]:
        p = self.predict_proba(mels)
        return [self.classes[i] for i in p.argmax(1)] if len(p) else []

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        return self
