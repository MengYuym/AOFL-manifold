"""Stage 4 controlled feasibility run: connect target predictor to classification."""
from pathlib import Path
import os,sys,runpy,textwrap,json
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Model'),str(ROOT/'Train')]
import torch
from torch import nn
from RunGAPDStage1 import install as stage1
from ContextSpectrumGate import ContextSpectrumGate

def install(cls):
    stage1(cls);previous=cls.get_model
    def get_model(self):
        previous(self)
        def condition(module,args):
            mask=self._gapd_mask
            x=torch.stack(args).mean(0)
            c=(x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            shared=[module.shared_encoder(v) for v in args]
            self.model.gapd_stage1.gate_conditions=(c[:,None,:].expand_as(x),shared)
        self.model.multiattn.register_forward_pre_hook(condition)
        self.model.multiattn.subspace_refiners=nn.ModuleDict({n:ContextSpectrumGate(self.model.gapd_stage1,i) for i,n in enumerate(['text','audio','visual'])})
    cls.get_model=get_model

if __name__=='__main__':
    torch.set_num_threads(2)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='stage4');install(scope['AOFL'])
    output=Path(r'D:\codex\shiyan\output\gapd_stage4_seed7_20261112')
    manifest={'stage':'4 target-conditioned attenuation','seed':7,'epochs':30,
      'formula':'h - strength * Up M^T M Up^T h; strength=mean(cos(target_angle)^2)',
      'limitation':'scalar target-spectrum summary, not per-target-principal-angle matching or teacher-exact gate',
      'change':'target head now gets classification gradient from epoch1; overall attenuation also differs from stage3',
      'conditions':'detached conversation mean, relative shared norm, shared agreement',
      'preserved':'outside complement, initialization, losses, data, schedule, parameter count',
      'selection':'validation F1, 30 epochs; no test-based adaptation'}
    (output.parent/'gapd_stage4_plan.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    main=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(v,*a,**kw):
        if isinstance(v,nn.Module):v={'model_state_dict':v.state_dict(),'format':'GAPD stage4; reconstruct with install()'}
        return save(v,*a,**kw)
    torch.save=weights
    try:exec(compile(textwrap.dedent(main),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
