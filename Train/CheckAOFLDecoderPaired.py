"""Predeclare all seeds and check original AOFL weights and RNG are paired."""
from pathlib import Path
import sys,json,runpy,hashlib,gc
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunAOFLDecoderPaired import BASE,ROOT,NEW,hashes,install
import torch
torch.set_num_threads(2)
BASE.mkdir(parents=True,exist_ok=True)
planpath=BASE/'experiment_plan.json'
if planpath.exists():raise FileExistsError(planpath)
assert all(not (BASE/f'seed_{s}').exists() for s in NEW)
plan={'new_seeds':NEW,'reuse_seed':7,'runs':10,'epochs_each':30,'weight':.1,
  'reconstruction':'same-modality pre-OPR g,h concatenated, reconstruct detached pre-disentanglement x; exclude padding',
  'initialization':'decoder fixed seed40007, original model and training RNG vary by paired seed',
  'selection':'validation F1, earliest tie; test never used for selecting epochs or changing configuration',
  'stop_policy':'finish all ten runs regardless of interim test results; no tuning or seed replacement',
  'analysis':'all six and new five paired test-F1 differences; means, sample SD, approximate two-sided 95% Student-t CI; compare same seeds with completed GAPD runs descriptively',
  'source_sha256':hashes()}
planpath.write_text(json.dumps(plan,indent=2),encoding='utf-8')
cfg=json.loads(Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113/baseline/config.json').read_text(encoding='utf-8'))
checks={}
for seed in NEW:
    states=[];rngs=[]
    for decoder in [False,True]:
        scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='aofl_preflight');cls=scope['AOFL']
        if decoder:install(cls,'baseline')
        t=cls.__new__(cls)
        for k,v in cfg.items():setattr(t,k,v)
        t.device=torch.device('cpu');scope['set_seed'](seed);t.get_model()
        assert not hasattr(t.model,'gapd_stage1') and not hasattr(t.model.multiattn,'subspace_refiners')
        assert sum(p.numel() for p in t.model.parameters())==(21769747 if decoder else 21178387)
        states.append({k:v.detach().clone() for k,v in t.model.state_dict().items() if not k.startswith('reconstruction.')})
        rngs.append(torch.get_rng_state().clone());del t;gc.collect()
    assert states[0].keys()==states[1].keys() and all(torch.equal(v,states[1][k]) for k,v in states[0].items())
    assert torch.equal(*rngs)
    checks[seed]={'original_aofl_no_manifold':True,'matched_weights_and_cpu_rng':True,'base_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in states[0].values())).hexdigest()}
assert len(set(r['base_sha256'] for r in checks.values()))==len(NEW)
(BASE/'pairing_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('AOFL PAIRING CHECKS PASSED',checks)
