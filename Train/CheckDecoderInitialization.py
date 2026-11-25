"""Ensure decoder addition does not perturb base weights or training RNG."""
from pathlib import Path
import sys,runpy,json,gc
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunDecoderComparison import ROOT,install
import torch
torch.set_num_threads(2)
config=json.loads(Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113/baseline/config.json').read_text(encoding='utf-8'))
results={}
for variant in ['baseline','gapd']:
    snapshots=[];rng=[]
    for with_decoder in [False,True]:
        scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='init_check')
        cls=scope['AOFL']
        if with_decoder:install(cls,variant)
        elif variant=='gapd':
            from RunGAPDStage3 import install as stage3
            stage3(cls)
        t=cls.__new__(cls)
        for k,v in config.items():setattr(t,k,v)
        t.device=torch.device('cpu')
        torch.manual_seed(7);t.get_model()
        snapshots.append({k:v.detach().clone() for k,v in t.model.state_dict().items() if not k.startswith('reconstruction.')})
        rng.append(torch.get_rng_state().clone())
        del t;gc.collect()
    assert snapshots[0].keys()==snapshots[1].keys()
    assert all(torch.equal(v,snapshots[1][k]) for k,v in snapshots[0].items())
    assert torch.equal(*rng)
    results[variant]={'base_state_tensors_identical':len(snapshots[0]),'cpu_rng_after_model_creation_identical':True}
out=Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113/initialization_checks.json')
out.write_text(json.dumps(results,indent=2),encoding='utf-8')
print(results)
