"""Check real training integration before the paired decoder runs."""
from pathlib import Path
import sys,runpy,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunDecoderComparison import install,ROOT
import torch
from torch.utils.data import DataLoader,Subset
from FeatureReconstruction import FeatureReconstruction

torch.set_num_threads(2)
checks={}
d=FeatureReconstruction()
g=[torch.randn(5,256,requires_grad=True) for _ in range(3)]
h=[torch.randn(5,256,requires_grad=True) for _ in range(3)]
x=[torch.randn(5,256,requires_grad=True) for _ in range(3)]
loss,e,energy,p=d(g,h,x)
assert torch.allclose(loss,torch.stack([(p[i]-x[i]).square().mean() for i in range(3)]).mean())
loss.backward()
assert all(v.grad is not None and v.grad.abs().sum()>0 for v in g+h)
assert all(v.grad is None for v in x)
assert sum(p.numel() for p in d.parameters())==591360
checks['synthetic']={'detached_targets':True,'shared_private_gradients':True,'decoder_parameters':591360}
decoder_states=[]
for variant in ['baseline','gapd']:
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='check_'+variant)
    cls=scope['AOFL'];install(cls,variant)
    scope['set_seed'](7)
    t=cls('MELD',64,30,.0001,.0001,4,256,4,1024,.3,0,1,True,torch.device('cuda'))
    decoder_states.append({k:v.detach().cpu().clone() for k,v in t.model.reconstruction.state_dict().items()})
    opt_ids={id(p) for group in t.optimizer.param_groups for p in group['params']}
    assert all(id(p) in opt_ids for p in t.model.reconstruction.parameters())
    loader=DataLoader(Subset(t.train_dataloader.dataset,t.train_indices[:3]),batch_size=3,collate_fn=t.train_dataloader.collate_fn)
    batch=next(iter(loader));n=int((batch[-1]!=-1).sum())
    t.model.train();out=t.model(*[v.to(t.device) for v in batch[:-1]],batch[-1].to(t.device).reshape(-1))
    assert len(out)==10 and all(z.shape==(n,256) for z in t._reconstruction_targets)
    assert all(not z.requires_grad for z in t._reconstruction_targets)
    t.optimizer.zero_grad();t.reconstruction_loss.backward()
    grads={k:float(p.grad.norm()) for k,p in t.model.multiattn.named_parameters() if p.grad is not None and ('encoder' in k)}
    assert grads and all(torch.isfinite(torch.tensor(list(grads.values()))))
    assert any(v>0 for v in grads.values())
    before=next(t.model.reconstruction.parameters()).detach().clone()
    train=t.train_or_eval_model_per_epoch(loader,1,'Train')
    assert not torch.equal(before,next(t.model.reconstruction.parameters()))
    assert t.last_reconstruction_metrics['utterances']==n
    with torch.no_grad():valid=t.train_or_eval_model_per_epoch(loader,1,'Valid')
    checks[variant]={'parameters':sum(p.numel() for p in t.model.parameters()),'valid_utterances':n,'encoder_grad_norms':grads,'train':train[:3],'valid':valid[:3],'metrics':t.last_reconstruction_metrics}
    t.writer.close();del t;torch.cuda.empty_cache()
assert all(torch.equal(v,decoder_states[1][k]) for k,v in decoder_states[0].items())
checks['identical_decoder_initialization']=True
dest=Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113');dest.mkdir(parents=True,exist_ok=True)
(dest/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('ALL DECODER CHECKS PASSED',dest/'checks.json')
