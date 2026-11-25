"""GAPD stage 1: sample-dependent bases, conditional spectrum and geodesic ranking.

QR is a differentiable parameterization, not an R03/Riemannian optimizer.
Original classification/OPR and vector CEN remain for this controlled stage.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

def orthogonal_basis(raw):
    q,r=torch.linalg.qr(raw,mode='reduced')
    signs=torch.where(r.diagonal(dim1=-2,dim2=-1)<0,-1.,1.)
    return q*signs.unsqueeze(-2)

def principal_cosines(u,v):
    # Never add entrywise perturbations: they would break basis invariance.
    return torch.linalg.svdvals(u.transpose(-1,-2)@v).clamp(1e-5,1-1e-5)

def geodesic(u,v):
    return torch.acos(principal_cosines(u,v)).square().sum(-1).clamp_min(1e-10).sqrt()

class PrincipalAngleDisentangle(nn.Module):
    def __init__(self,dim=256,rank=4,margin=.1):
        super().__init__();self.dim=dim;self.rank=rank;self.margin=margin
        self.shared_head=nn.Linear(dim,dim*rank)
        self.private_heads=nn.ModuleList([nn.Linear(dim,dim*rank) for _ in range(3)])
        self.target_heads=nn.ModuleList([nn.Sequential(nn.Linear(dim+2,64),nn.ReLU(),nn.Linear(64,rank)) for _ in range(3)])
        # Independent full-rank initial offsets; avoid common-basis initialization.
        for head in [self.shared_head,*self.private_heads]:
            nn.init.orthogonal_(head.weight)
            with torch.no_grad():head.bias.copy_(orthogonal_basis(torch.randn(dim,rank)).reshape(-1))
        self.context=None;self.last_statistics={}

    def basis(self,x,head):
        return orthogonal_basis(head(x).reshape(*x.shape[:-1],self.dim,self.rank))

    def forward(self,shared,private,context):
        us=[self.basis(g,self.shared_head) for g in shared]
        up=[self.basis(h,head) for h,head in zip(private,self.private_heads)]
        pa=[];ranking=[];spectra=[];targets=[]
        for m in range(3):
            cosine=principal_cosines(us[m],up[m])
            theta=torch.acos(cosine)
            # Context is a masked per-conversation mean of pre-disentanglement x.
            # Reliability: relative feature norm, a proxy, not calibrated confidence.
            norm=shared[m].norm(dim=-1)
            allnorm=torch.stack([g.norm(dim=-1) for g in shared],-1).sum(-1).clamp_min(1e-8)
            reliability=norm/allnorm
            agreement=torch.stack([F.cosine_similarity(shared[m],shared[s],dim=-1) for s in range(3) if s!=m],-1).mean(-1)
            condition=torch.cat([context,reliability[:,None],agreement[:,None]],-1).detach()
            target=(math.pi/2)*torch.sigmoid(self.target_heads[m](condition))
            target=target.sort(dim=-1).values
            pa.append(F.smooth_l1_loss(cosine,torch.cos(target)))
            within=theta.square().sum(-1).clamp_min(1e-10).sqrt()
            ranking.append(torch.stack([F.relu(self.margin+geodesic(us[m],us[s])-within).mean() for s in range(3) if s!=m]).mean())
            spectra.append(theta.detach());targets.append(target.detach())
        pa=torch.stack(pa).mean();rank=torch.stack(ranking).mean()
        self.last_statistics={'actual_angle_radians':torch.stack(spectra),'target_angle_radians':torch.stack(targets)}
        return pa,rank
