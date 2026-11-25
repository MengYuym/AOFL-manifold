from pathlib import Path
import sys,json,runpy,csv
import numpy as np
ROOT=Path(r'D:/毕设/多模态情感识别/AOFL-main');sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
import torch
import ModalDecoderStudy
from torch.utils.data import DataLoader,Subset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score,accuracy_score,classification_report
B=Path(r'D:/codex/shiyan/output/baseline_probe_20261121');O=B/'emotion_probe';O.mkdir(exist_ok=True)
torch.set_num_threads(2)
scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='emotion_probe');cls=scope['AOFL']
from ModalDecoderStudy import install
install(cls,[.1,1,1],0)
t=cls('MELD',64,30,.0001,.0001,4,256,4,1024,.3,0,1,True,torch.device('cpu'))
labels={};caches={}
for split in ['train','valid']:
 dl=getattr(t,split+'_dataloader');loader=DataLoader(Subset(dl.dataset,sorted(getattr(t,split+'_indices'))),batch_size=64,shuffle=False,collate_fn=dl.collate_fn)
 yy=[]
 for batch in loader:
  y=batch[-1].reshape(-1);yy.append(y[y!=-1].numpy())
 labels[split]=np.concatenate(yy);caches[split]=np.load(B/(split+'_features.npz'));assert len(labels[split])==len(caches[split]['g'])
t.writer.close()
plan={'features':'frozen original AOFL without decoder, seed7 epoch13','classifier':'StandardScaler fitted on train then RidgeClassifier(alpha=1), no class weighting','selection':'fixed alpha1, no tuning or validation checkpoint selection','splits':'official training fit; validation evaluate; test unused','inputs':['g','h','gh'],'scope':'three separate modalities and concatenation of all modalities','limitations':['linear readout only; not original AOFL classifier','gh has twice input dimension and more classifier coefficients','one frozen backbone; no stable improvement claim','padding grouping same as cached feature extraction']}
(O/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');rows=[]
for m,name in [(0,'text'),(1,'audio'),(2,'visual'),(None,'all_modalities')]:
 for mode in ['g','h','gh']:
  arrays=[]
  for split in ['train','valid']:
   d=caches[split]
   def feature(k):return d[k][:,m] if m is not None else d[k].reshape(len(d[k]),-1)
   a=feature(mode) if mode!='gh' else np.concatenate([feature('g'),feature('h')],-1)
   arrays.append(a)
  clf=make_pipeline(StandardScaler(),RidgeClassifier(alpha=1.));clf.fit(arrays[0],labels['train']);pred=clf.predict(arrays[1]);train_pred=clf.predict(arrays[0])
  row=dict(modality=name,input=mode,dimensions=arrays[0].shape[1],train_weighted_f1=100*f1_score(labels['train'],train_pred,average='weighted'),valid_weighted_f1=100*f1_score(labels['valid'],pred,average='weighted'),valid_macro_f1=100*f1_score(labels['valid'],pred,average='macro'),valid_accuracy=100*accuracy_score(labels['valid'],pred))
  rows.append(row);np.savez_compressed(O/f'{name}_{mode}_predictions.npz',labels=labels['valid'],predictions=pred);(O/f'{name}_{mode}_classification.json').write_text(json.dumps(classification_report(labels['valid'],pred,output_dict=True,zero_division=0),indent=2),encoding='utf-8');print(row,flush=True)
(O/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
with (O/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

