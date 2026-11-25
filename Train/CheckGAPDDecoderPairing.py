"""Verify each seed changes original initialization but not pairing conditions."""
from pathlib import Path
import sys,json,runpy,hashlib,gc
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunGAPDDecoderPaired import ROOT,BASE,install_decoder,install_gapd,SOURCES
import torch
torch.set_num_threads(2)
config=json.loads(Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113/gapd/config.json').read_text(encoding='utf-8'))
results={};fingerprints=[]
for seed in [42,2025]:
    states=[];rngs=[]
    for decoder in [False,True]:
        scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='pair_preflight')
        cls=scope['AOFL']
        if decoder:install_decoder(cls,'gapd')
        else:install_gapd(cls)
        t=cls.__new__(cls)
        for k,v in config.items():setattr(t,k,v)
        t.device=torch.device('cpu');scope['set_seed'](seed);t.get_model()
        states.append({k:v.detach().clone() for k,v in t.model.state_dict().items() if not k.startswith('reconstruction.')})
        rngs.append(torch.get_rng_state().clone())
        assert sum(p.numel() for p in t.model.parameters())==(22872927 if decoder else 22281567)
        del t;gc.collect()
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
    assert torch.equal(*rngs)
    fingerprint=hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in states[0].items() if not k.startswith('gapd_stage1.'))).hexdigest()
    fingerprints.append(fingerprint)
    results[seed]={'matching_base_weights':True,'matching_cpu_rng':True,'base_sha256':fingerprint}
assert fingerprints[0]!=fingerprints[1]
BASE.mkdir(parents=True,exist_ok=True)
(BASE/'pairing_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
plan={'new_seeds':[42,2025],'reuse_seed':7,'runs':4,'epochs_each':30,'lambda_reconstruction':.1,
  'comparison':'GAPD stage3 with decoder versus GAPD stage3 without decoder, not versus original AOFL',
  'stop_policy':'complete all predeclared runs; no interim test-based stopping or tuning',
  'selection':'within each run maximum validation F1, earliest tie',
  'analysis':'per-seed paired difference, arithmetic mean and sample SD; seed7 exploratory, new seeds confirmatory checks; 3 seeds insufficient for strong significance claim',
  'extra_initialization':'geometry seed30007, decoder seed40007 fixed as in original experiment; original model, sampler and training seed varies',
  'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}}
(BASE/'experiment_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print('PAIRING CHECKS PASSED',results)
