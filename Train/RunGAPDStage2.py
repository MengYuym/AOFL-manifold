"""Stage 2: stage-1 losses plus stable principal-spectrum gating."""
from pathlib import Path
import os,sys,runpy,textwrap,json
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Model'),str(ROOT/'Train')]
import torch
from torch import nn
from RunGAPDStage1 import install as install_stage1
from PrincipalSpectrumGate import PrincipalSpectrumGate

def install(cls):
    install_stage1(cls)
    previous=cls.get_model
    def get_model(self):
        previous(self)
        self.model.multiattn.subspace_refiners=nn.ModuleDict({
            name:PrincipalSpectrumGate(self.model.gapd_stage1,i)
            for i,name in enumerate(['text','audio','visual'])})
    cls.get_model=get_model

if __name__=='__main__':
    torch.set_num_threads(2)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='gapd_stage2')
    install(scope['AOFL'])
    output=Path(r'D:\codex\shiyan\output\gapd_stage2_seed7_20261112')
    manifest={'stage':'GAPD stage 2','seed':7,'epochs':30,'rank':4,
      'retention':'1-sigma^2 = sin(theta)^2',
      'formula':'Up [I - (Us^T Up)^T(Us^T Up)] Up^T h',
      'change_vs_stage1':'replace original OPR h scaling with spectral reconstruction',
      'design_deviation':'teacher suggested 1-cos(theta) or target-relative sigmoid; use smooth 1-cos(theta)^2 to avoid SVD-vector backward singularities',
      'outside_private_subspace':'discarded, following projected reconstruction; no residual',
      'target_spectrum':'auxiliary loss only, not directly used by this gate',
      'preserved':'stage1 initialization, geometry weights, vector CEN, original warmup, classifier, official split',
      'not_implemented':['Grassmann CEN','cross-utterance anti-collapse','R03 optimizer'],
      'selection':'validation weighted F1 only; no test-driven tuning'}
    (output.parent/'gapd_stage2_plan.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    main=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(value,*args,**kwargs):
        if isinstance(value,nn.Module):value={'model_state_dict':value.state_dict(),'format':'GAPD stage2; reconstruct with install()'}
        return save(value,*args,**kwargs)
    torch.save=weights
    try:exec(compile(textwrap.dedent(main),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
