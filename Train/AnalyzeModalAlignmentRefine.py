from pathlib import Path
import sys,json,csv,runpy,gc,html
ROOT=Path(r'D:/毕设/多模态情感识别/AOFL-main');sys.path.insert(0,str(ROOT/'Train'))
import torch,numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from ModalDecoderStudy import install,best
BASE=Path(r'D:/codex/shiyan/output/modal_alignment_refine_20261117')
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
refs=[dict(name='原AOFL',weights='无decoder',alignment=0,epoch=13,valid_f1=66.6826,test_f1=65.7067),dict(name='0.1/1/1，无alignment',weights=[.1,1,1],alignment=0,epoch=13,valid_f1=67.0017,test_f1=65.0516)]
previous=json.loads(Path(r'D:/codex/shiyan/output/modal_alignment_joint_20261117/analysis.json').read_text())
refs += [r for r in previous if r['weights']==[.1,1,1]]
def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'
body='<h1>分模态权重与共享对齐细化实验</h1><p>原AOFL＋三个decoder，没有GAPD；官方MELD划分，种子7，每组30轮。重构位置保持不变，原CEN、ARE、OPR保留。重构损失为三个模态加权MSE的平均值；alignment为同一句话三个共享g两两1−余弦相似度的平均值。从第一轮开始加入。</p><p>预先固定四组候选，按验证F1选择轮次，测试只报告，不参与选择。权重顺序为文本／音频／视觉。alignment固定为0.003，围绕0.1／1／1分别减小文本、音频、视觉的重构权重。四组计划在训练前固定。</p>'
body+=table(['配置','重构权重','alignment','轮次','验证F1','对应测试F1','测试相比原AOFL（百分点）'],[[r['name'],r['weights'],r['alignment'],r['epoch'],f"{r['valid_f1']:.4f}",f"{r['test_f1']:.4f}",f"{r['test_f1']-65.7067:+.4f}"] for r in refs+results])
body+='<h2>共享特征检查</h2><p>余弦顺序为文本–音频、文本–视觉、音频–视觉；变化量顺序为文本、音频、视觉。变化量是去均值能量占总能量的比例。如果随机配对和正确配对都很接近，并且跨话语变化减少，不能只凭余弦增大认定对齐更好。这里没有设置阈值宣称坍缩。</p>'
body+=table(['配置','同话语余弦','随机话语余弦','跨话语变化量'],[[r['name'],[round(x,4) for x in r['shared_cosine']],[round(x,4) for x in r['random_cosine']],[round(x,4) for x in r['centered_energy']]] for r in results])
winner=max(results,key=lambda r:r['valid_f1']);body+=f"<h2>结论</h2><p>新增候选按验证选择：{winner['name']}，验证F1 {winner['valid_f1']:.4f}，对应测试F1 {winner['test_f1']:.4f}。与原AOFL测试差值 {winner['test_f1']-65.7067:+.4f} 个百分点。</p>"
center=next(r for r in results if r['name']=='center')
for r in results:
 body+=f"<p>{r['name']}：相比本轮中心配置（0.1／1／1、alignment 0.003），验证F1变化 {r['valid_f1']-center['valid_f1']:+.4f}，对应测试F1变化 {r['test_f1']-center['test_f1']:+.4f} 个百分点。</p>"
body+='<p>本轮并未为每种新权重补齐无alignment消融，因此不能把联合变化归因于alignment单独的作用。不同模态权重共同影响训练，四组不代表已找到每个模态的独立最优值。对齐系数增加不保证实际余弦单调上升，必须读取向量检查。文本、音频、视觉重构权重不是分类融合占比。</p>'
body+='<p>单种子探索不能证明稳定提升，也未找到各模态独立最优权重。共享诊断固定验证集排序和批次，原模型对padding处理的限制可能使该分组下预测与训练记录不完全一致；本报告分类F1全部使用训练期间的原始记录，诊断只用于比较向量。四组各30轮记录完整，最佳模型与验证选中轮次一致，最佳和最后模型严格加载且参数有限。</p>'
(BASE/'分模态权重与共享对齐细化报告.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1250px;margin:30px auto;padding:20px;font:16px/1.8 Microsoft YaHei;color:#25384a;background:#f8fafc}h1,h2{color:#255777}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}td,th{border:1px solid #ccd7e1;padding:9px}th{background:#e5eef6}</style>'+body,encoding='utf-8')

