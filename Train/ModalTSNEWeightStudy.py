from pathlib import Path
import sys,json,subprocess,csv,html
ROOT=Path(r'D:/毕设/多模态情感识别/AOFL-main');sys.path.insert(0,str(ROOT/'Train'))
BASE=Path(r'D:/codex/shiyan/output/modal_tsne_weights_20261117')
CASES={'stronger':([.1,1.,1.],.03),'lighter_rec':([.03,.3,.3],.03)}
def report():
 import ModalDecoderStudy as m
 rows=[]
 for n,(w,a) in CASES.items():
  p=BASE/n
  if not (p/'effective_variant.json').exists():continue
  rr=list(csv.DictReader((p/'metrics.csv').open(encoding='utf-8-sig')));assert len(rr)==30
  r=m.best(p);rows.append(dict(name=n,weights=w,alignment=a,epoch=int(r['epoch']),valid_f1=float(r['valid_f1']),test_f1=float(r['test_f1'])))
 (BASE/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:
  import torch,ModalDecoderStudy as m
  torch.set_num_threads(2);m.BASE=BASE;n=sys.argv[1];w,a=CASES[n];m.run(n,w,a)
 else:
  BASE.mkdir(exist_ok=True,parents=True)
  p=BASE/'plan.json';assert not p.exists(),'Existing study; do not overwrite'
  p.write_text(json.dumps({'model':'original AOFL + decoder, no GAPD','seed':7,'epochs':30,'cases':CASES,'budget':2,'selection':'validation weighted F1 only, earliest tie','references':'compare stronger against existing weights .1/1/1 alpha=0,.001,.003,.01,.1,1; lighter_rec is joint change, not isolated alignment ablation','split':'official MELD','alignment':'same utterance shared g pairwise mean 1-cos, all epochs; original CEN retained'},indent=2),encoding='utf-8')
  for n in CASES:
   code="from pathlib import Path; p=Path("+repr(str(Path(__file__)))+"); exec(compile(p.read_text(encoding='utf-8-sig'),str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})"
   with (BASE/(n+'.log')).open('w',encoding='utf-8') as log:r=subprocess.run([sys.executable,'-u','-c',code,n],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   assert r.returncode==0,(n,r.returncode)
   report()



