"""Basis-invariant spectral gate, avoiding SVD-vector gradients at repeats."""
from torch import nn

def spectral_refine(h,shared_basis,private_basis):
    # If M=P diag(sigma) V^T, this equals
    # Up V diag(1-sigma^2) V^T Up^T h: retention = sin(theta)^2.
    # Direct matrix form remains differentiable at repeated singular values.
    m=shared_basis.transpose(-1,-2)@private_basis
    coefficients=(private_basis.transpose(-1,-2)@h.unsqueeze(-1))
    filtered=coefficients-m.transpose(-1,-2)@(m@coefficients)
    return (private_basis@filtered).squeeze(-1)

class PrincipalSpectrumGate(nn.Module):
    def __init__(self,geometry,modality):
        super().__init__()
        # Geometry is already registered on parent model; do not duplicate weights.
        object.__setattr__(self,'geometry',geometry)
        self.modality=modality

    def forward(self,g,h):
        us=self.geometry.basis(g,self.geometry.shared_head)
        up=self.geometry.basis(h,self.geometry.private_heads[self.modality])
        return spectral_refine(h,us,up)
