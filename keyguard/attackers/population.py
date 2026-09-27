"""A POPULATION of diverse attacker agents for the multi-adversary arena.

The single-attacker min-max proves D beats *one* network. That invites the fair
objection: "you just overfit the perturbation to KeyNet's quirks." So here we
field a *population* of attacker agents with genuinely different inductive biases
and make D defend against all of them at once. If one inaudible perturbation
holds every architecture at chance, the shield isn't exploiting one net's blind
spot -- it's killing the acoustic signal itself (Pillar 1).

Each agent takes the SAME differentiable log-mel input (B,1,mel,frames) and emits
36-way logits, so they are drop-in interchangeable inside optimize_perturbation.
They are deliberately small so a whole population retrains on CPU in the arena
loop. The four inductive biases:

- keynet   : SE-CNN (channel attention) -- the production Harrison attacker.
- widecnn  : plain deep/wide CNN, no attention -- pure convolutional capacity.
- resnet   : residual blocks -- gradient-friendly depth, different optimization
             geometry, so D can't rely on one loss landscape.
- framegru : bidirectional GRU reading mel columns as a time series -- a
             recurrent/temporal bias instead of a spatial one, the most different
             agent of the four.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import N_CLASSES
from .supervised import KeyNet


class WideCNN(nn.Module):
    """Plain wide CNN, no channel attention -- convolutional capacity only."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(96, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.drop(x))


class _ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.skip(x))


class ResNetTiny(nn.Module):
    """Tiny residual net -- a different optimization geometry from plain CNNs."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
                                  nn.BatchNorm2d(32), nn.ReLU())
        self.r1 = _ResBlock(32, 32)
        self.r2 = _ResBlock(32, 64)
        self.r3 = _ResBlock(64, 64)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.max_pool2d(self.r1(x), 2)
        x = F.max_pool2d(self.r2(x), 2)
        x = F.max_pool2d(self.r3(x), 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.drop(x))


class FrameGRU(nn.Module):
    """Bi-GRU over mel frames -- a recurrent/temporal bias, not a spatial one.

    Reads the spectrogram as a sequence of per-frame mel vectors, so it attends
    to the *time structure* of a keystroke (press->release) rather than 2D texture.
    """

    def __init__(self, n_mels: int = 64, hidden: int = 64, n_classes: int = N_CLASSES):
        super().__init__()
        self.gru = nn.GRU(n_mels, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,1,mel,frames)
        x = x.squeeze(1).transpose(1, 2)                 # (B, frames, mel)
        out, _ = self.gru(x)                             # (B, frames, 2*hidden)
        x = out.mean(dim=1)                              # temporal average pool
        return self.head(self.drop(x))


def population() -> list[tuple[str, nn.Module]]:
    """The attacker agents D must defeat simultaneously. Each is a fresh module
    with a distinct inductive bias; returned as (name, module) pairs."""
    return [
        ("keynet", KeyNet()),
        ("widecnn", WideCNN()),
        ("resnet", ResNetTiny()),
        ("framegru", FrameGRU()),
    ]


def demo() -> None:
    """Smoke: every agent maps (B,1,mel,frames) -> (B,36) logits and backprops."""
    x = torch.randn(4, 1, 64, 38, requires_grad=True)
    y = torch.randint(0, N_CLASSES, (4,))
    for name, net in population():
        logits = net(x)
        assert logits.shape == (4, N_CLASSES), (name, logits.shape)
        F.cross_entropy(logits, y).backward()
        assert x.grad is not None, name
        x.grad = None
    print(f"population ok: {len(population())} attacker agents, all output "
          f"(B,{N_CLASSES}) logits and pass gradient to the audio.")


if __name__ == "__main__":
    demo()
