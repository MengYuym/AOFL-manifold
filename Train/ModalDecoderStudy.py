"""Per-modality reconstruction weights and a separate paired shared alignment ablation."""
from pathlib import Path
import os,sys,runpy,json,csv,textwrap,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
import torch
from torch.nn import functional as F
from RunDecoderComparison import install as decoder_install
BASE=Path(r'D:/codex/shiyan/output/modal_decoder_study_20261117')
CANDIDATES={'audio_visual':(.1,1.,1.),'audio':(.1,1.,.1),'visual':(.1,.1,1.)}
def install(cls,weights,alignment,output=None):
    decoder_install(cls,'baseline',output,1.)
    original=cls.get_model
    def get_model(self):
        original(self)
        self._modal_records=[]
        def after(module,args,result):
            gs=list(result[1:4]);hs=list(result[4:7]);xs=self._reconstruction_targets
            _,errors,energy,_=module.reconstruction(gs,hs,xs)
            rec=(errors.mean(0)*errors.new_tensor(weights)).mean()
            align=torch.stack([1-F.cosine_similarity(gs[a],gs[b],dim=-1) for a,b in [(0,1),(0,2),(1,2)]]).mean()
            self.reconstruction_loss=rec+alignment*align
            self._modal_records.append({'reconstruction':float(rec.detach()),'alignment':float(align.detach()),'weighted_alignment':float(alignment*align.detach())})
            return result
        self.model.register_forward_hook(after)
    cls.get_model=get_model
    previous=cls.train_or_eval_model_per_epoch
    def epoch(self,dataloader,epoch,train):
        self._modal_records=[];result=previous(self,dataloader,epoch,train)
        if output:
            p=Path(output)/'auxiliary_metrics.csv';exists=p.exists()
            row={'epoch':epoch,'split':train,**{k:sum(r[k] for r in self._modal_records)/len(self._modal_records) for k in self._modal_records[0]}}
            with p.open('a',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row))
                if not exists:w.writeheader()
                w.writerow(row)
        return result
    cls.train_or_eval_model_per_epoch=epoch
def run(name,weights,alignment):
    output=BASE/name;assert not output.exists(),output
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='modal_decoder');install(scope['AOFL'],weights,alignment,output)
    manifest={'name':name,'seed':7,'epochs':30,'weights_text_audio_visual':weights,'alignment_weight':alignment,'reconstruction':'mean(lambda_m*MSE_m), same pre-OPR g/h to detached x','alignment':'mean same-utterance pairwise (1-cosine), g only, all epochs','selection':'validation F1 only, never test','model':'original AOFL, no manifold','decoder_seed':40007}
    (BASE/(name+'_plan.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    body=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1];save=torch.save
    def weights_save(v,*a,**kw):
        if isinstance(v,torch.nn.Module):v={'model_state_dict':v.state_dict()}
        return save(v,*a,**kw)
    torch.save=weights_save
    try:exec(compile(textwrap.dedent(body),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
def best(p):
    with (p/'metrics.csv').open(encoding='utf-8-sig') as f:return max(list(csv.DictReader(f))[:30],key=lambda r:float(r['valid_f1']))
def prepare():
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'plan.json';assert not p.exists()
    p.write_text(json.dumps({'seed':7,'epochs':30,'candidates':CANDIDATES,'alignment_weights':[.1,1.],'runs':5,'selection':'choose reconstruction candidate using validation F1, then two alignment ablations on that same candidate; no test selection','hypothesis':'audio/visual reconstruction previously weaker; increase their weights individually or together; does not imply global optimum','collapse':'monitor per-utterance variation and effective rank; no claim that overlap alone is useful','old_training':'complete running GAPD job then defer remaining old GAPD queue; preserve existing artifacts'},indent=2),encoding='utf-8')
    # Objective and gradient unit check, no target branch gradients.
    from FeatureReconstruction import FeatureReconstruction
    d=FeatureReconstruction();g=[torch.randn(8,256,requires_grad=True) for _ in range(3)];h=[torch.randn(8,256,requires_grad=True) for _ in range(3)];x=[torch.randn(8,256,requires_grad=True) for _ in range(3)]
    _,e,_,_=d(g,h,x);uniform=(e.mean(0)*.1).mean();assert torch.allclose(uniform,.1*e.mean())
    loss=(e.mean(0)*e.new_tensor([.1,1,1])).mean()+.1*torch.stack([1-F.cosine_similarity(g[a],g[b],dim=-1) for a,b in [(0,1),(0,2),(1,2)]]).mean();loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in g+h)
    assert all(t.grad is None for t in x)
    (BASE/'objective_checks.json').write_text(json.dumps({'uniform_equivalence':True,'finite_nonzero_shared_private_gradients':True,'detached_targets':True}),encoding='utf-8')
if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','choose']);p.add_argument('--name');p.add_argument('--alignment',type=float,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='choose':
        chosen=max(CANDIDATES,key=lambda n:float(best(BASE/n)['valid_f1']));(BASE/'chosen.json').write_text(json.dumps({'name':chosen,'weights':CANDIDATES[chosen],'selection':'validation F1 only'}),encoding='utf-8');print(chosen)
    else:
        weights=CANDIDATES[a.name] if a.name in CANDIDATES else json.loads((BASE/'chosen.json').read_text())['weights']
        run(a.name,weights,a.alignment)
