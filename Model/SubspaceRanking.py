"""Per-utterance QR bases and normalized projection-overlap ranking."""
import torch
from torch import nn
import torch.nn.functional as F


def overlap(a, b):
    return (a.transpose(-1, -2) @ b).square().sum(dim=(-2, -1)) / a.shape[-1]


class BasisHead(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.dim, self.rank = dim, rank
        self.linear = nn.Linear(dim, dim * rank)

    def forward(self, x):
        raw = self.linear(x).reshape(-1, self.dim, self.rank)
        return torch.linalg.qr(raw, mode='reduced')[0]


class SubspaceRanking(nn.Module):
    def __init__(self, dim, rank=4, margin=0.1):
        super().__init__()
        if not 1 <= rank < dim:
            raise ValueError('Require 1 <= rank < dim')
        self.margin = margin
        self.shared_head = BasisHead(dim, rank)
        self.private_heads = nn.ModuleList([BasisHead(dim, rank) for _ in range(3)])
        # Nearby full-rank initial bases keep the hinge active initially.
        # Nonzero input weights preserve a gradient path to g/h from the start.
        base = torch.linalg.qr(torch.randn(dim, rank), mode='reduced')[0]
        with torch.no_grad():
            for head in [self.shared_head, *self.private_heads]:
                head.linear.weight.mul_(0.01)
                head.linear.bias.copy_(base.reshape(-1))

    def forward(self, shared, private):
        qs = [self.shared_head(g) for g in shared]
        qp = [head(h) for head, h in zip(self.private_heads, private)]
        terms = []
        for m in range(3):
            within = overlap(qs[m], qp[m])
            # Both other modalities must satisfy the relative relation.
            terms.append(torch.stack([F.relu(self.margin + within - overlap(qs[m], qs[s]))
                                      for s in range(3) if s != m]).mean())
        # Sum modalities, matching the original CSR aggregation convention.
        return torch.stack(terms).sum()
