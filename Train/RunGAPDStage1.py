"""Controlled first GAPD stage; does not claim full GAPD implementation."""
from pathlib import Path
import os,sys,runpy,json
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT/'Model'))
import torch
from torch import nn
from PrincipalAngleDisentangle import PrincipalAngleDisentangle

def install(trainer_class):
    original=trainer_class.get_model
    def get_model(self):
        original(self)
        with torch.random.fork_rng():
            torch.manual_seed(30007)
            self.model.gapd_stage1=PrincipalAngleDisentangle(self.model_dim,4,.1).to(self.device)
        def before_model(module,args):
            self._gapd_mask=args[-1].reshape(args[4].shape)!=-1
        def before_disentangle(module,args):
            mask=self._gapd_mask
            # x are [batch, utterances, d], mask excludes padding.
            x=torch.stack(args,0).mean(0)
            context=(x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            self._gapd_context=context[:,None,:].expand_as(x)[mask]
        def after_model(module,args,output):
            pa,rank=module.gapd_stage1(list(output[1:4]),list(output[4:7]),self._gapd_context)
            # Existing loop multiplies AAO by .09; divide here for explicit final weights.
            geometry=(.01*pa+.001*rank)/.09
            return (*output[:7],geometry,pa,rank)
        self.model.register_forward_pre_hook(before_model)
        self.model.multiattn.register_forward_pre_hook(before_disentangle)
        self.model.register_forward_hook(after_model)
    trainer_class.get_model=get_model

if __name__=='__main__':
    torch.set_num_threads(2)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='gapd_stage1')
    install(scope['AOFL'])
    output=Path(r'D:\codex\shiyan\output\gapd_stage1_seed7_20261111')
    # Reuse original main code in its own globals so the installed class is retained.
    text=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8')
    main=text.split("if __name__ == '__main__':",1)[1]
    import textwrap
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    # Local forward hooks cannot be pickled as whole-model objects. Always save
    # module weights explicitly; best_model.pth already uses a state-dict checkpoint.
    original_save=torch.save
    def save_weights(value,*args,**kwargs):
        if isinstance(value,nn.Module):
            value={'model_state_dict':value.state_dict(),'format':'GAPD stage1 weights; reconstruct with install()'}
        return original_save(value,*args,**kwargs)
    torch.save=save_weights
    try:
        exec(compile(textwrap.dedent(main),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:
        torch.save=original_save
    (output/'effective_variant.json').write_text(json.dumps({
        'stage':'GAPD stage 1, not full GAPD','seed':7,'rank':4,
        'changes':['per-utterance QR bases','full singular-value/principal-angle spectrum','detached context-conditioned target spectrum','all-other-modalities geodesic ranking'],
        'preserved':['original OPR','vector CEN','classifier','official split','original warmup and scheduler'],
        'final_geometry_weights':{'PA':.01,'rank':.001},
        'not_implemented':['spectral gate','Grassmann CEN','anti-collapse across utterances','R03 optimizer'],
        'reliability_proxy':'relative g norm, not calibrated confidence',
        'note':'classification CE still scaled by 10 after warmup; stage 1 auxiliary heads not directly in classifier; targets remain trainable despite detached conditions'
    },ensure_ascii=False,indent=2),encoding='utf-8')
