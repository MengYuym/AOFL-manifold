"""Fixed six-seed AOFL decoder comparison; seed7 reused, five pairs fresh."""
from pathlib import Path
import os,sys,runpy,json,textwrap,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
import torch
from torch import nn
from RunDecoderComparison import install
BASE=Path(r'D:/codex/shiyan/output/aofl_decoder_paired_20261113')
NEW=[42,2025,123,1024,10007]
SOURCES=['Train/TrainMultiEMO.py','Train/RunDecoderComparison.py','Model/FeatureReconstruction.py','Model/AOFL_Model.py','Model/Multi_Disentangle.py']
def hashes():return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=NEW);p.add_argument('--decoder',type=int,choices=[0,1],required=True)
    opt=p.parse_args();torch.set_num_threads(2)
    plan=json.loads((BASE/'experiment_plan.json').read_text(encoding='utf-8'))
    assert hashes()==plan['source_sha256']
    output=BASE/f'seed_{opt.seed}'/('with_decoder' if opt.decoder else 'without_decoder')
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='aofl_decoder_pair')
    if opt.decoder:install(scope['AOFL'],'baseline',output,.1)
    manifest={'variant':'original AOFL '+('with' if opt.decoder else 'without')+' decoder','manifold':False,
      'seed':opt.seed,'epochs':30,'weight':.1 if opt.decoder else 0,'decoder_initial_seed':40007 if opt.decoder else None,
      'reconstruction':'same-modality pre-OPR concat(g,h) to detached pre-disentanglement x',
      'selection':'max validation weighted F1, earliest tie; corresponding test','source_sha256':hashes()}
    (output.parent/(output.name+'_plan.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed',str(opt.seed),'--num_epochs','30','--output_dir',str(output)]
    body=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(value,*args,**kwargs):
        if isinstance(value,nn.Module):value={'model_state_dict':value.state_dict(),'format':'AOFL paired; install baseline decoder only when matching variant'}
        return save(value,*args,**kwargs)
    torch.save=weights
    try:exec(compile(textwrap.dedent(body),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
