"""Predeclared seed-paired GAPD stage3 reconstruction verification."""
from pathlib import Path
import os,sys,runpy,json,textwrap,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
import torch
from torch import nn
from RunDecoderComparison import install as install_decoder
from RunGAPDStage3 import install as install_gapd

BASE=Path(r'D:/codex/shiyan/output/gapd_decoder_paired_20261113')
SOURCES=['Train/TrainMultiEMO.py','Train/RunGAPDStage1.py','Train/RunGAPDStage3.py','Train/RunDecoderComparison.py','Model/FeatureReconstruction.py','Model/PrincipalAngleDisentangle.py','Model/ResidualSpectrumGate.py']

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--decoder',type=int,choices=[0,1],required=True)
    opt=p.parse_args();torch.set_num_threads(2)
    output=BASE/f'seed_{opt.seed}'/('with_decoder' if opt.decoder else 'without_decoder')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(f'Preserve existing run: {output}')
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='paired_training')
    if opt.decoder:install_decoder(scope['AOFL'],'gapd',output,.1)
    else:install_gapd(scope['AOFL'])
    manifest={'variant':'GAPD stage3 '+('with' if opt.decoder else 'without')+' decoder',
      'seed':opt.seed,'epochs':30,'decoder_weight':.1 if opt.decoder else 0,
      'gapd_initial_seed':30007,'decoder_initial_seed':40007 if opt.decoder else None,
      'initialization_policy':'Keep the existing fixed additional-module seeds; vary original model and training RNG together within each pair',
      'selection':'maximum validation weighted F1; earliest tie; report corresponding test, never select on test',
      'target':'detached pre-disentanglement x; input pre-gate g,h; exclude label -1 padding',
      'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}}
    (output.parent/(output.name+'_plan.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed',str(opt.seed),'--num_epochs','30','--output_dir',str(output)]
    body=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(value,*args,**kw):
        if isinstance(value,nn.Module):value={'model_state_dict':value.state_dict(),'format':'GAPD stage3 paired; install matching variant before load'}
        return save(value,*args,**kw)
    torch.save=weights
    try:exec(compile(textwrap.dedent(body),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
