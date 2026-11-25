"""Controlled stage-3 change: retain the complement of the private subspace."""
from PrincipalSpectrumGate import PrincipalSpectrumGate

def residual_spectral_refine(h,shared_basis,private_basis):
    m=shared_basis.transpose(-1,-2)@private_basis
    coefficients=private_basis.transpose(-1,-2)@h.unsqueeze(-1)
    removed=private_basis@(m.transpose(-1,-2)@(m@coefficients))
    # h = private projection + its perpendicular complement.
    # Only private principal directions are attenuated by sigma^2.
    return h-removed.squeeze(-1)

class ResidualSpectrumGate(PrincipalSpectrumGate):
    def forward(self,g,h):
        us=self.geometry.basis(g,self.geometry.shared_head)
        up=self.geometry.basis(h,self.geometry.private_heads[self.modality])
        return residual_spectral_refine(h,us,up)
