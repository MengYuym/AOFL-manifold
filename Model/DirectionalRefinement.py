"""Directional OPR alternative; geometric projection, not a manifold optimizer."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class DirectionalRefinement(nn.Module):
    def __init__(self, dim, initial_retention=0.9):
        super().__init__()
        self.gate = nn.Linear(2 * dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(initial_retention / (1-initial_retention)))

    def components(self, shared, private):
        direction = F.normalize(shared, dim=-1, eps=1e-8)
        parallel = (private * direction).sum(dim=-1, keepdim=True) * direction
        alpha = torch.sigmoid(self.gate(torch.cat((shared, private), dim=-1)))
        return parallel, alpha

    def forward(self, shared, private):
        parallel, alpha = self.components(shared, private)
        return private - (1-alpha) * parallel
