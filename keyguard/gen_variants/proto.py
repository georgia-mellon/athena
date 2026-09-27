"""Prototypical / metric-learning variant.

Hypothesis: absolute per-key acoustics are keyboard-specific (LOKO fails because a
softmax head memorizes per-keyboard absolute spectra), but the RELATIVE structure
between keys on any one keyboard -- which keys sound closer/farther to each other --
is more stable across keyboards. So instead of a softmax classifier we learn a KeyNet
trunk -> L2-normalized embedding, trained with a prototypical loss where every episode's
support+query is drawn from a SINGLE keyboard (domain). That forces the embedding to be
useful for *within-keyboard, relative* nearest-prototype classification rather than for
memorizing one global per-class point in absolute feature space.

predict_proba = softmax over cosine similarity (on unit-normalized vectors this is a
monotonic reparam of negative squared Euclidean distance, i.e. exactly the Snell et al.
2017 prototypical loss) to class prototypes.

Zero-shot LOKO has no target labels, so fit() falls back to the nearest sane thing:
prototypes = mean embedding per class over the ENTIRE pool (nearest train-class centroid).
The real test of the hypothesis is finetune(): with a few target-keyboard shots we just
recompute those classes' prototypes from the shots (no gradient steps -- pure metric
calibration), which is cheap and is where the transfer should actually show up.
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
EMB_DIM = 64
N_SUPPORT = 4
N_QUERY = 4
EPISODES_PER_DOMAIN = 2
TEMP = 0.1
UNSEEN_PENALTY = 1e4        # logit penalty for a class with no prototype at all


class EmbedNet(nn.Module):
    """Same KeyNet-style conv-SE trunk as the baseline, but ending in an
    L2-normalized embedding instead of a softmax head."""

    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.b1 = ConvSE(1, 32)
        self.b2 = ConvSE(32, 64)
        self.b3 = ConvSE(64, 128)
        self.proj = nn.Linear(128, dim)

    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return F.normalize(self.proj(x), dim=1)


def _sample_episode(y: np.ndarray, idx: np.ndarray, n_support: int, n_query: int,
                     rng: np.random.Generator):
    """Support/query index split for ONE domain's samples (idx = row ids in that domain)."""
    sup_idx, sup_y, qry_idx, qry_y = [], [], [], []
    for c in rng.permutation(np.unique(y[idx])):
        cidx = idx[y[idx] == c].copy()
        rng.shuffle(cidx)
        n_s = min(n_support, len(cidx) - 1)
        if n_s < 1:
            continue                       # need >=1 support AND >=1 left for query
        n_q = min(n_query, len(cidx) - n_s)
        sup_idx.extend(cidx[:n_s])
        sup_y.extend([c] * n_s)
        qry_idx.extend(cidx[n_s:n_s + n_q])
        qry_y.extend([c] * n_q)
    return (np.array(sup_idx, int), np.array(sup_y, int),
            np.array(qry_idx, int), np.array(qry_y, int))


def _prototypes(embeds: torch.Tensor, labels: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """Mean-embedding prototype per unique label. Returns (classes, protos (k, dim) L2-normed)."""
    classes = np.unique(labels)
    protos = torch.stack([embeds[labels == c].mean(0) for c in classes])
    return classes, F.normalize(protos, dim=1)


def _proto_loss(net: EmbedNet, feats: torch.Tensor, y: np.ndarray, idx: np.ndarray,
                 rng: np.random.Generator):
    sup_i, sup_y, qry_i, qry_y = _sample_episode(y, idx, N_SUPPORT, N_QUERY, rng)
    if len(sup_i) == 0 or len(qry_i) == 0:
        return None
    z_sup = net(feats[sup_i])
    z_qry = net(feats[qry_i])
    classes, protos = _prototypes(z_sup, sup_y)
    logits = (z_qry @ protos.T) / TEMP
    target = torch.as_tensor(np.searchsorted(classes, qry_y), dtype=torch.long, device=DEVICE)
    return F.cross_entropy(logits, target)


class ProtoNet:
    def __init__(self):
        self.net = EmbedNet().to(DEVICE)
        self.protos = torch.zeros(N_CLASSES, EMB_DIM, device=DEVICE)
        self.has_proto = np.zeros(N_CLASSES, bool)

    def _feats(self, wins):
        return torch.from_numpy(mel(wins)[:, None]).float().to(DEVICE)

    def fit(self, wins, y, dom):
        feats = self._feats(wins)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        rng = np.random.default_rng(0)
        domains = np.unique(dom)
        self.net.train()
        for _ in range(EPOCHS):
            for d in rng.permutation(domains):
                idx = np.flatnonzero(dom == d)
                for _ in range(EPISODES_PER_DOMAIN):
                    loss = _proto_loss(self.net, feats, y, idx, rng)
                    if loss is None:
                        continue
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            sched.step()
        self._set_fallback_prototypes(feats, y)
        return self

    @torch.no_grad()
    def _set_fallback_prototypes(self, feats, y):
        """LOKO has no target labels -> nearest train-class centroid over the whole pool."""
        self.net.eval()
        embeds = self.net(feats)
        classes, protos = _prototypes(embeds, y)
        self.protos = self.protos.clone()
        self.has_proto = self.has_proto.copy()
        self.protos[classes] = protos
        self.has_proto[classes] = True

    @torch.no_grad()
    def finetune(self, wins, y):
        """Few-shot calibration: recompute the target's own class prototypes from its
        k labeled shots. No gradient steps -- the trunk is frozen, only the metric's
        reference points move. Classes absent from the shots keep their pool fallback."""
        self.net.eval()
        embeds = self.net(self._feats(wins))
        classes, protos = _prototypes(embeds, y)
        self.protos = self.protos.clone()
        self.has_proto = self.has_proto.copy()
        self.protos[classes] = protos
        self.has_proto[classes] = True
        return self

    @torch.no_grad()
    def predict_proba(self, wins):
        if len(wins) == 0:
            return np.zeros((0, N_CLASSES), np.float32)
        self.net.eval()
        embeds = self.net(self._feats(wins))
        logits = (embeds @ self.protos.T) / TEMP
        logits[:, ~self.has_proto] -= UNSEEN_PENALTY
        return F.softmax(logits, dim=1).cpu().numpy()


def make():
    return ProtoNet()


def _demo():
    """ponytail self-check: episode split is disjoint & covers only requested domain,
    prototype/predict shapes hold, and a class with no prototype gets suppressed."""
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 0, 1, 1, 1, 2])
    idx = np.array([0, 1, 2, 3, 4, 5, 6])
    sup_i, sup_y, qry_i, qry_y = _sample_episode(y, idx, n_support=1, n_query=1, rng=rng)
    assert set(sup_i) & set(qry_i) == set(), "support/query must be disjoint"
    assert len(sup_i) == len(sup_y) and len(qry_i) == len(qry_y)
    assert 2 not in sup_y and 2 not in qry_y, "class with only 1 sample can't form support+query"

    embeds = F.normalize(torch.randn(6, 8), dim=1)
    labels = np.array([0, 0, 1, 1, 2, 2])
    classes, protos = _prototypes(embeds, labels)
    assert list(classes) == [0, 1, 2] and protos.shape == (3, 8)
    assert torch.allclose(protos.norm(dim=1), torch.ones(3), atol=1e-5)

    m = ProtoNet()
    m.has_proto[:] = False
    m.has_proto[0] = True
    m.protos[0] = torch.ones(EMB_DIM) / (EMB_DIM ** 0.5)
    fake = np.zeros((2, 4800), np.float32)
    p = m.predict_proba(fake)
    assert p.shape == (2, N_CLASSES)
    assert np.all(p[:, 0] > p[:, 1]), "seen class should beat unseen-penalized class"
    print("ok: episode split disjoint, prototypes unit-norm, unseen classes suppressed")


if __name__ == "__main__":
    _demo()
