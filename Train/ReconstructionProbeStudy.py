from pathlib import Path
import sys,json,runpy,copy,csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
ROOT=Path(r'D:/毕设/多模态情感识别/AOFL-main');sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
from ModalDecoderStudy import install,best
B=Path(r'D:/codex/shiyan/output/reconstruction_probe_20261121');B.mkdir(exist_ok=True,parents=True)
P=Path(r'D:/codex/shiyan/output/modal_decoder_study_20261117/audio_visual')
torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plan={'source':str(P),'weights':[.1,1,1],'alignment':0,'freeze_backbone':True,'probe':'Linear512-256 ReLU Linear256-256; g-only uses [g,0], h-only uses [0,h]','seeds':[17,42,123],'epochs':40,'batch_size':512,'optimizer':'Adam lr=.001','selection':'fixed 40 epochs, no validation-based tuning','splits':'official train for fitting, official validation for evaluation; test unused','limitations':'equal architecture and nominal params; single-feature inputs have fewer active input weights; probes measure recoverability, not semantic disentanglement or classification gain'}
(B/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='probe');cls=scope['AOFL'];install(cls,[.1,1,1],0)
t=cls('MELD',64,30,.0001,.0001,4,256,4,1024,.3,0,1,True,device)
c=torch.load(P/'best_model.pth',map_location=device,weights_only=False);assert c['epoch']==int(best(P)['epoch']);t.model.load_state_dict(c['model_state_dict'],strict=True);t.model.eval()
for p in t.model.parameters():p.requires_grad_(False)
original={k:v.detach().clone() for k,v in t.model.state_dict().items()}
caches={}
for split in ['train','valid']:
 dl=getattr(t,split+'_dataloader');ids=sorted(getattr(t,split+'_indices'));loader=DataLoader(Subset(dl.dataset,ids),batch_size=64,shuffle=False,collate_fn=dl.collate_fn)
 cache={k:[] for k in ['x','g','h']}
 with torch.no_grad():
  for batch in loader:
   args=[a.to(device) for a in batch];args[-1]=args[-1].reshape(-1);o=t.model(*args)
   for k,values in [('x',t._reconstruction_targets),('g',o[1:4]),('h',o[4:7])]:cache[k].append(torch.stack(list(values),1).cpu())
 caches[split]={k:torch.cat(v) for k,v in cache.items()};print(split,len(caches[split]['x']),flush=True)
 assert len(caches[split]['x'])==(9989 if split=='train' else 1109)
 np.savez_compressed(B/(split+'_features.npz'),**{k:v.numpy() for k,v in caches[split].items()})
assert all(torch.equal(v,original[k]) for k,v in t.model.state_dict().items());t.writer.close()
results=[];logs=[]
for m,modality in enumerate(['text','audio','visual']):
 train={k:v[:,m].to(device) for k,v in caches['train'].items()};valid={k:v[:,m].to(device) for k,v in caches['valid'].items()}
 for seed in [17,42,123]:
  torch.manual_seed(seed);initial=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,256)).to(device).state_dict()
  for mode in ['g','h','gh']:
   def inp(d):return torch.cat([d['g'] if mode!='h' else torch.zeros_like(d['g']),d['h'] if mode!='g' else torch.zeros_like(d['h'])],-1)
   a=inp(train);av=inp(valid);net=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,256)).to(device);net.load_state_dict(initial);opt=torch.optim.Adam(net.parameters(),lr=.001)
   generator=torch.Generator().manual_seed(seed+9000)
   for epoch in range(1,41):
    net.train();order=torch.randperm(len(a),generator=generator).to(device);total=0
    for ids in order.split(512):
     opt.zero_grad();pred=net(a[ids]);loss=(pred-train['x'][ids]).square().mean();loss.backward();opt.step();total+=float(loss.detach())*len(ids)
    logs.append(dict(modality=modality,seed=seed,mode=mode,epoch=epoch,train_mse=total/len(a)))
   net.eval()
   with torch.no_grad():
    pred=net(av);x=valid['x'];mse=float((pred-x).square().mean());energy=float(x.square().mean());baseline=float((x-train['x'].mean(0)).square().mean());row=dict(modality=modality,seed=seed,mode=mode,mse=mse,relative_mse=mse/energy,r2_vs_train_mean=1-mse/baseline,cosine=float(torch.nn.functional.cosine_similarity(pred,x,dim=-1).mean()))
   results.append(row);torch.save(net.state_dict(),B/f'{modality}_{mode}_seed{seed}.pth');(B/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(row,flush=True)
with (B/'training.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(logs[0]));w.writeheader();w.writerows(logs)
summary=[]
for modality in ['text','audio','visual']:
 rr={mode:[r['relative_mse'] for r in results if r['modality']==modality and r['mode']==mode] for mode in ['g','h','gh']}
 summary.append(dict(modality=modality,**{mode:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))) for mode,v in rr.items()},paired_joint_gain_vs_best_single=[min(rr['g'][i],rr['h'][i])-rr['gh'][i] for i in range(3)]))
(B/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('DONE',summary,flush=True)
