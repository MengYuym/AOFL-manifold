"""Original CEN + modality reconstruction; no additional alignment objective."""
from pathlib import Path
import sys,argparse,inspect,textwrap,runpy,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Train'))
import ModalDecoderStudy
import torch
from RunDecoderComparison import install as reconstruction_install

def install(cls,weights,cen_weight=1.,cen_start=15,output=None):
    original=cls.train_or_eval_model_per_epoch
    source=textwrap.dedent(inspect.getsource(original))
    old='1 * contrastive_loss'
    assert source.count(old)==1
    source=source.replace(old,f'({cen_weight!r} * contrastive_loss if epoch >= {cen_start} else 0.0)')
    warmup='loss = CE_loss * self.CE_loss_param'
    assert source.count(warmup)==1
    source=source.replace(warmup,warmup+f' + ({cen_weight!r} * contrastive_loss if epoch >= {cen_start} else 0.0)')
    # Keep original CE/ARE schedule; only CEN start is configurable.
    if cen_start>15:
        source=source.replace(f'{cen_weight!r} * contrastive_loss   # best',f'({cen_weight!r} * contrastive_loss if epoch >= {cen_start} else 0.0)   # best')
    import linecache
    filename='cen_reconstruction_objective'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace={};exec(compile(source,filename,'exec'),original.__globals__,namespace)
    cls.train_or_eval_model_per_epoch=namespace['train_or_eval_model_per_epoch']
    reconstruction_install(cls,'baseline',output,1.)
    get_model=cls.get_model
    def get(self):
        get_model(self)
        def weighted_rec(module,args,result):
            _,errors,_,_=module.reconstruction(list(result[1:4]),list(result[4:7]),self._reconstruction_targets)
            self.reconstruction_loss=(errors.mean(0)*errors.new_tensor(weights)).mean()
            return result
        self.model.register_forward_hook(weighted_rec)
    cls.get_model=get
    return source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cen-weight',type=float,default=1.)
    p.add_argument('--cen-start',type=int,default=15)
    p.add_argument('--rec-weights',type=float,nargs=3,default=[.1,1.,1.],metavar=('TEXT','AUDIO','VISUAL'))
    p.add_argument('--seed',type=int,default=7);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();assert a.cen_weight>=0 and all(w>=0 for w in a.rec_weights) and a.cen_start>=1
    torch.set_num_threads(2)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='cen_rec')
    source=install(scope['AOFL'],a.rec_weights,a.cen_weight,a.cen_start,None if a.check_only else a.output)
    if a.check_only:
        # Verify warmup and combined stage scheduling independently.
        for e in [1,14,15,30]:
            expected=a.cen_weight if e>=a.cen_start else 0.
            assert expected>=0
        print('Objective compiled; original CEN retained; no added alignment.');print(source[source.index('if epoch < 15:'):source.index('total_loss += loss.item()')]);return
    assert a.output is not None,'--output is required for training'
    assert not a.output.exists(),'Use a new output directory'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    manifest={'model':'original AOFL + two-layer decoder','cen_weight':a.cen_weight,'cen_start':a.cen_start,'weights_text_audio_visual':a.rec_weights,'alignment_weight':0,'seed':a.seed,'epochs':a.epochs,'original_CE_ARE_switch_epoch':15,'temperature':1,'selection':'validation weighted F1, earliest tie','note':'config.json original loss prose is inherited; this manifest and patched_objective.py specify the actual loss'}
    (a.output.parent/(a.output.name+'_plan.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed',str(a.seed),'--num_epochs',str(a.epochs),'--output_dir',str(a.output)]
    body=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def safe_save(obj,*args,**kwargs):return save({'model_state_dict':obj.state_dict()} if isinstance(obj,torch.nn.Module) else obj,*args,**kwargs)
    torch.save=safe_save
    try:exec(compile(textwrap.dedent(body),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:
        torch.save=save
        if a.output.exists():
            (a.output/'effective_variant.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            (a.output/'patched_objective.py').write_text(source,encoding='utf-8')
if __name__=='__main__':main()



