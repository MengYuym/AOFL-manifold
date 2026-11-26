"""Paired decoder experiment, preserving existing trainer source on disk."""
from pathlib import Path
import os,sys,runpy,textwrap,json,csv,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Model'),str(ROOT/'Train')]
import torch
from torch import nn
from FeatureReconstruction import FeatureReconstruction

def install(cls,variant,output=None,weight=.1):
    if variant != 'baseline':
        raise ValueError('Only baseline + decoder is active; GAPD source is archived.')
    original=cls.get_model
    def get_model(self):
        original(self)
        with torch.random.fork_rng():
            torch.manual_seed(40007)
            self.model.reconstruction=FeatureReconstruction(self.model_dim).to(self.device)
        def before_model(module,args):self._reconstruction_mask=args[-1].reshape(args[4].shape)!=-1
        def before_disentangle(module,args):
            self._reconstruction_targets=[x[self._reconstruction_mask].detach() for x in args]
        def after_model(module,args,result):
            loss,errors,energy,_=module.reconstruction(list(result[1:4]),list(result[4:7]),self._reconstruction_targets)
            self.reconstruction_loss=loss
            if hasattr(self,'_reconstruction_totals'):
                self._reconstruction_totals['error']+=errors.detach().sum(0).cpu()
                self._reconstruction_totals['energy']+=energy.detach().sum(0).cpu()
                self._reconstruction_totals['n']+=len(errors)
            return result
        self.model.register_forward_pre_hook(before_model)
        self.model.multiattn.register_forward_pre_hook(before_disentangle)
        self.model.register_forward_hook(after_model)
    cls.get_model=get_model
    # Insert reconstruction objective in both warmup and combined stage. Never
    # hide it in AAO, which the existing trainer disables during warmup.
    import inspect
    method=textwrap.dedent(inspect.getsource(cls.train_or_eval_model_per_epoch))
    needle='            total_loss += loss.item()'
    assert needle.strip() in method
    method=method.replace('        total_loss += loss.item()',f'        loss = loss + {weight!r} * self.reconstruction_loss\n        total_loss += loss.item()',1)
    ns={};exec(compile(method,'decoder_epoch_objective','exec'),cls.train_or_eval_model_per_epoch.__globals__,ns)
    epoch_method=ns['train_or_eval_model_per_epoch']
    def epoch(self,dataloader,epoch,train):
        self._reconstruction_totals={'error':torch.zeros(3),'energy':torch.zeros(3),'n':0}
        result=epoch_method(self,dataloader,epoch,train)
        t=self._reconstruction_totals;n=t['n'];assert n>0
        row={'epoch':epoch,'split':train,'utterances':n,'weight':weight}
        for i,name in enumerate(['text','audio','visual']):
            row[name+'_mse']=float(t['error'][i]/n)
            row[name+'_target_energy']=float(t['energy'][i]/n)
            row[name+'_relative_mse']=float(t['error'][i]/t['energy'][i].clamp_min(1e-12))
        row['mean_mse']=sum(row[n+'_mse'] for n in ['text','audio','visual'])/3
        if output is not None:
            path=Path(output)/'reconstruction_metrics.csv';exists=path.exists()
            with path.open('a',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));
                if not exists:writer.writeheader()
                writer.writerow(row)
        self.last_reconstruction_metrics=row
        return result
    cls.train_or_eval_model_per_epoch=epoch

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['baseline'],required=True)
    opt=parser.parse_args();torch.set_num_threads(2)
    output=Path(r'D:\codex\shiyan\output\decoder_comparison_seed7_20261113')/opt.variant
    output.parent.mkdir(exist_ok=True)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='decoder_'+opt.variant)
    install(scope['AOFL'],opt.variant,output,.1)
    manifest={'variant':opt.variant,'decoder':'per modality Linear(512,256), ReLU, Linear(256,256)',
      'decoder_initial_seed':40007,'seed':7,'epochs':30,'reconstruction_weight':.1,
      'loss':'mean MSE across valid utterances, feature coordinates and three modalities; added every epoch',
      'input':'pre-OPR/pre-gate g,h concatenated','target':'detached pre-disentanglement x, 256 dims',
      'mask':'padded_labels != -1','manifold_base':'GAPD stage3 residual spectral gate' if opt.variant=='gapd' else None,
      'selection':'validation weighted F1, earliest tie','new_parameters':591360,
      'limitations':['reconstruction preserves recoverability, not necessarily emotion information','detached x target still changes across optimizer steps as upstream encoders change','single seed exploratory experiment'],
      'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['Model/FeatureReconstruction.py','Train/RunDecoderComparison.py']}}
    (output.parent/(opt.variant+'_plan.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    main=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(v,*a,**kw):
        if isinstance(v,nn.Module):v={'model_state_dict':v.state_dict(),'format':'decoder '+opt.variant+'; reconstruct with install()'}
        return save(v,*a,**kw)
    torch.save=weights
    try:exec(compile(textwrap.dedent(main),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
