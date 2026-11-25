"""Global subspace prototype; differentiable QR, not the R03 optimizer."""
import torch
from torch import nn


class SubspaceRefinement(nn.Module):
    def __init__(self, dim, rank=4, geometry='qr'):
        super().__init__()
        if not 1 <= rank <= dim:
            raise ValueError('rank must be between 1 and feature dimension')
        if geometry not in ('qr', 'free'):
            raise ValueError('geometry must be qr or free')
        self.geometry = geometry
        # Same initial bases in both variants; free is a parameter-matched control.
        self.shared_basis = nn.Parameter(torch.linalg.qr(torch.randn(dim, rank), mode='reduced')[0])
        self.private_basis = nn.Parameter(torch.linalg.qr(torch.randn(dim, rank), mode='reduced')[0])
        self.gates = nn.Linear(dim * 2, 2)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([0.1, 0.2]))

    def bases(self):
        if self.geometry == 'qr':
            return (torch.linalg.qr(self.shared_basis, mode='reduced')[0],
                    torch.linalg.qr(self.private_basis, mode='reduced')[0])
        return self.shared_basis, self.private_basis

    def forward(self, shared, private):
        bs, bp = self.bases()
        q = (private @ bp) @ bp.transpose(-1, -2)
        parallel = (q @ bs) @ bs.transpose(-1, -2)
        perpendicular = q - parallel
        gates = 0.1 * torch.tanh(self.gates(torch.cat((shared, private), dim=-1)))
        return private + gates[..., :1] * parallel + gates[..., 1:] * perpendicular
