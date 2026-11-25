from pathlib import Path
import sys,json,csv,runpy,gc,html
ROOT=Path(r'D:/毕设/多模态情感识别/AOFL-main');sys.path.insert(0,str(ROOT/'Train'))
import torch,numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from ModalDecoderStudy import best
from RunCENReconstruction import install as cen_install
def install(cls,weights,alpha):return cen_install(cls,weights,1.,1)
BASE=Path(r'D:/codex/shiyan/output/cen_rec_20261124')
torch.set_num_threads(2)
plan=json.loads((BASE/'plan.json').read_text(encoding='utf-8-sig'));results=[]
for name,(weights,alpha) in plan['cases'].items():
 p=BASE/name;rows=list(csv.DictReader((p/'metrics.csv').open(encoding='utf-8-sig')));assert len(rows)==30
 manifest=json.loads((p/'effective_variant.json').read_text());assert manifest['weights_text_audio_visual']==weights and manifest['alignment_weight']==alpha
 scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='joint_analysis');cls=scope['AOFL'];install(cls,weights,alpha)
 t=cls('MELD',64,30,.0001,.0001,4,256,4,1024,.3,0,1,True,torch.device('cpu'))
 ck=torch.load(p/'best_model.pth',map_location='cpu',weights_only=False);r=best(p);assert ck['epoch']==int(r['epoch'])
 for fn in ['best_model.pth','final_model.pth']:
  c=torch.load(p/fn,map_location='cpu',weights_only=False);t.model.load_state_dict(c['model_state_dict'],strict=True);assert all(torch.isfinite(v).all() for v in c['model_state_dict'].values() if v.is_floating_point())
 t.model.load_state_dict(ck['model_state_dict'],strict=True);t.model.eval()
 loader=DataLoader(Subset(t.valid_dataloader.dataset,sorted(t.valid_indices)),batch_size=64,shuffle=False,collate_fn=t.valid_dataloader.collate_fn)
 cache=[]
 with torch.no_grad():
  for batch in loader:
   args=list(batch);args[-1]=args[-1].reshape(-1);o=t.model(*args);cache.append(torch.stack(list(o[1:4]),1).numpy())
 g=np.concatenate(cache);assert len(g)==1109;np.savez_compressed(BASE/(name+'_shared.npz'),g=g)
 def cos(a,b):return float(F.cosine_similarity(torch.from_numpy(a),torch.from_numpy(b),dim=-1).mean())
 pairs=[(0,1),(0,2),(1,2)];perm=np.random.default_rng(29).permutation(1109)
 row=dict(name=name,weights=weights,alignment=alpha,epoch=int(r['epoch']),valid_f1=float(r['valid_f1']),test_f1=float(r['test_f1']),shared_cosine=[cos(g[:,a],g[:,b]) for a,b in pairs],random_cosine=[cos(g[:,a],g[perm,b]) for a,b in pairs],centered_energy=[float(((g[:,m]-g[:,m].mean(0))**2).sum()/max((g[:,m]**2).sum(),1e-12)) for m in range(3)],checkpoint_checks=True)
 results.append(row);(BASE/'analysis.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(row),flush=True);t.writer.close();del t,ck,c;gc.collect()

