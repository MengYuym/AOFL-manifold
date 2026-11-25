from pathlib import Path
import sys,json
ROOT=Path(r'D:/毕设/多模态情感识别/AOFL-main');sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
import torch
from torch import nn
from FeatureReconstruction import FeatureReconstruction
class SingleLayerReconstruction(FeatureReconstruction):
 def __init__(self,dim=256):
  nn.Module.__init__(self);self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(2*dim,dim)) for _ in range(3)])
def patch():
 import RunDecoderComparison as d
 d.FeatureReconstruction=SingleLayerReconstruction
BASE=Path(r'D:/codex/shiyan/output/decoder_structure_20261119')
if __name__=='__main__':
 import ModalDecoderStudy as m
 torch.set_num_threads(2);BASE.mkdir(exist_ok=True,parents=True)
 plan={'model':'original AOFL + decoder','architecture':'per-modality Linear(512,256)','reference':'modal_decoder_study_20261117/audio_visual; two-layer decoder','weights':[.1,1,1],'alignment':0,'seed':7,'decoder_seed':40007,'epochs':30,'selection':'validation F1, earliest tie','new_parameters':394752,'budget':'one new run; reuse existing two-layer reference','initialization_note':'same isolated decoder seed; parameter shapes differ, so decoder and subsequent modality initialization cannot be identical; single-seed structure comparison includes initialization variability'}
 (BASE/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 patch();m.BASE=BASE;m.run('single_layer',[.1,1,1],0)
 p=BASE/'single_layer/effective_variant.json';v=json.loads(p.read_text());v.update(architecture=plan['architecture'],new_parameters=394752);p.write_text(json.dumps(v,indent=2),encoding='utf-8')
