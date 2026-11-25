"""Predeclare extension before viewing its results; verify paired initialization."""
from pathlib import Path
import sys,json,runpy,hashlib,gc
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunGAPDDecoderPaired import ROOT,BASE,install_decoder,install_gapd,SOURCES
import torch
torch.set_num_threads(2)
seeds=[123,1024,10007]
planpath=BASE/'extension_plan.json'
if planpath.exists():raise FileExistsError(planpath)
assert all(not (BASE/f'seed_{s}').exists() for s in seeds)
plan={'new_seeds':seeds,'prior_seeds':[7,42,2025],'new_runs':6,'epochs_each':30,
      'reconstruction':'pre-gate same-modality concat(g,h) to detached pre-disentanglement x',
      'weight':.1,'gapd_initial_seed':30007,'decoder_initial_seed':40007,
      'stop_policy':'Complete exactly six runs regardless of interim test performance; no parameter tuning',
      'selection':'validation weighted F1; earliest tie; corresponding test F1',
      'analysis':'Report paired test-F1 differences, mean and sample SD, two-sided 95% Student-t mean-difference CI for all six, follow-up five and new three separately; small-sample approximate inference, no generalization beyond fixed MELD split and initialization policy',
      'interpretation':'Positive mean alone insufficient; interval including zero means uncertain mean benefit; training-seed variability is not test-sample uncertainty',
      'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}}
BASE.mkdir(parents=True,exist_ok=True)
planpath.write_text(json.dumps(plan,indent=2),encoding='utf-8')
config=json.loads(Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113/gapd/config.json').read_text(encoding='utf-8'))
checks={}
for seed in seeds:
    states=[];rngs=[]
    for decoder in [False,True]:
        scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='extension_preflight')
        cls=scope['AOFL']
        if decoder:install_decoder(cls,'gapd')
        else:install_gapd(cls)
        t=cls.__new__(cls)
        for k,v in config.items():setattr(t,k,v)
        t.device=torch.device('cpu');scope['set_seed'](seed);t.get_model()
        states.append({k:v.detach().clone() for k,v in t.model.state_dict().items() if not k.startswith('reconstruction.')})
        rngs.append(torch.get_rng_state().clone())
        del t;gc.collect()
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
    assert torch.equal(*rngs)
    checks[seed]={'matching_base_weights':True,'matching_cpu_rng':True,'base_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in states[0].items() if not k.startswith('gapd_stage1.'))).hexdigest()}
assert len(set(r['base_sha256'] for r in checks.values()))==3
(BASE/'extension_pairing_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('EXTENSION PREFLIGHT PASSED',checks)
