"""Target-conditioned attenuation; uses a scalar summary of target spectrum."""
import math
import torch
import torch.nn.functional as F
from PrincipalSpectrumGate import PrincipalSpectrumGate

class ContextSpectrumGate(PrincipalSpectrumGate):
    def forward(self,g,h):
        us=self.geometry.basis(g,self.geometry.shared_head)
        up=self.geometry.basis(h,self.geometry.private_heads[self.modality])
        # Conditions cached by the disentanglement pre-hook; detached inputs,
        # trainable target head. Context is shared within each conversation.
        context,all_shared=self.geometry.gate_conditions
        norms=torch.stack([v.norm(dim=-1) for v in all_shared],-1).sum(-1).clamp_min(1e-8)
        reliability=g.norm(dim=-1)/norms
        agreement=torch.stack([F.cosine_similarity(g,v,dim=-1) for i,v in enumerate(all_shared) if i!=self.modality],-1).mean(-1)
        condition=torch.cat([context,reliability[...,None],agreement[...,None]],-1).detach()
        target=(math.pi/2)*torch.sigmoid(self.geometry.target_heads[self.modality](condition))
        strength=torch.cos(target).square().mean(-1,keepdim=True)
        if getattr(self,'fixed_strength',None) is not None:
            strength=torch.full_like(strength,self.fixed_strength)
        m=us.transpose(-1,-2)@up
        coeff=up.transpose(-1,-2)@h.unsqueeze(-1)
        removed=(up@(m.transpose(-1,-2)@(m@coeff))).squeeze(-1)
        self.last_strength=strength.detach()
        return h-strength*removed
