"""Independent per-modality decoders for pre-refinement g/h -> detached x."""
import torch
from torch import nn

class FeatureReconstruction(nn.Module):
    def __init__(self,dim=256):
        super().__init__()
        self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(2*dim,dim),nn.ReLU(),nn.Linear(dim,dim)) for _ in range(3)])

    def forward(self,shared,private,targets):
        errors=[];energies=[];predictions=[]
        for decoder,g,h,x in zip(self.decoders,shared,private,targets):
            prediction=decoder(torch.cat([g,h],-1))
            target=x.detach()
            errors.append((prediction-target).square().mean(-1))
            energies.append(target.square().mean(-1))
            predictions.append(prediction)
        per_sample=torch.stack(errors,-1)
        return per_sample.mean(),per_sample,torch.stack(energies,-1),predictions
