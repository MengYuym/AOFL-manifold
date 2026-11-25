"""Matched independent control for stage 4: fixed spectral attenuation 0.5."""
from pathlib import Path
import os,sys,runpy,textwrap,json
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Model'),str(ROOT/'Train')]
import torch
from torch import nn
from RunGAPDStage4 import install as adaptive_install

def install(cls):
    adaptive_install(cls)
    previous=cls.get_model
    def get_model(self):
        previous(self)
        for gate in self.model.multiattn.subspace_refiners.values():
            gate.fixed_strength=.5
    cls.get_model=get_model

if __name__=='__main__':
    torch.set_num_threads(2)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='fixed_control')
    install(scope['AOFL'])
    output=Path(r'D:\codex\shiyan\output\gapd_fixed_half_seed7_20261112')
    manifest={'variant':'stage4 matched fixed-strength independent-training control',
      'seed':7,'epochs':30,'strength':.5,
      'formula':'h - 0.5 Up M^T M Up^T h',
      'single_change':'constant replaces predicted strength in classification path',
      'target_heads':'retained; receive PA loss after warmup but no classification gradient',
      'preserved':['parameter count','initialization','conditional-input computation','data split','geometry losses','classification weights','optimizer and scheduler','30 epoch budget'],
      'selection':'highest validation F1, earliest tie; no test-based selection'}
    (output.parent/'gapd_fixed_half_plan.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    main=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(v,*a,**kw):
        if isinstance(v,nn.Module):v={'model_state_dict':v.state_dict(),'format':'GAPD fixed strength .5; reconstruct with install()'}
        return save(v,*a,**kw)
    torch.save=weights
    try:exec(compile(textwrap.dedent(main),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
