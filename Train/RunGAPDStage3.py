"""GAPD residual reconstruction ablation; not a full or teacher-exact GAPD."""
from pathlib import Path
import os,sys,runpy,textwrap,json
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Model'),str(ROOT/'Train')]
import torch
from torch import nn
from RunGAPDStage1 import install as install_stage1
from ResidualSpectrumGate import ResidualSpectrumGate

def install(cls):
    install_stage1(cls)
    previous=cls.get_model
    def get_model(self):
        previous(self)
        self.model.multiattn.subspace_refiners=nn.ModuleDict({
            name:ResidualSpectrumGate(self.model.gapd_stage1,i)
            for i,name in enumerate(['text','audio','visual'])})
    cls.get_model=get_model

if __name__=='__main__':
    torch.set_num_threads(2)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='gapd_stage3')
    install(scope['AOFL'])
    output=Path(r'D:\codex\shiyan\output\gapd_stage3_seed7_20261112')
    manifest={'stage':'GAPD stage 3: residual ablation','seed':7,'epochs':30,'rank':4,
      'hypothesis':'discarding private-subspace complement in stage2 may suppress useful classification features',
      'single_change_vs_stage2':'add back h - Up Up^T h; all other settings unchanged',
      'formula':'h - Up (Us^T Up)^T(Us^T Up) Up^T h',
      'inside_retention':'sin(theta)^2; same deviation from teacher as stage2',
      'outside_retention':1,'target_spectrum':'auxiliary only; not directly used by gate',
      'preserved':['stage1 heads and initialization','stage2 parameter count','original warmup and schedule','PA weight .01','rank weight .001','vector CEN','official split'],
      'not_implemented':['teacher-exact 1-cos(theta) gate','Grassmann CEN','cross-utterance anti-collapse','R03 optimizer'],
      'selection':'highest validation F1, earliest tie; 30 epoch predetermined budget'}
    (output.parent/'gapd_stage3_plan.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    main=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(value,*args,**kwargs):
        if isinstance(value,nn.Module):value={'model_state_dict':value.state_dict(),'format':'GAPD stage3; reconstruct with install()'}
        return save(value,*args,**kwargs)
    torch.save=weights
    try:exec(compile(textwrap.dedent(main),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
