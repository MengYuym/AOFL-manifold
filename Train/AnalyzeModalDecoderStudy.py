"""Validation-selected analysis of modality-specific decoder weights and alignment."""
from pathlib import Path
import sys,json,runpy,csv,gc,statistics,html
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ModalDecoderStudy import BASE,ROOT,install,best
import torch,numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,silhouette_score
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(2)
def diagnostics(a):
    a=a.astype(np.float64);center=a-a.mean(0);sv=np.linalg.svd(center,compute_uv=False);power=sv**2;power/=max(power.sum(),1e-30);nz=power[power>1e-15]
    return {'mean_coordinate_std':float(a.std(0).mean()),'mean_norm':float(np.linalg.norm(a,axis=1).mean()),'centered_energy_ratio':float(np.sum(center**2)/max(np.sum(a**2),1e-30)),'effective_rank':float(np.exp(-np.sum(nz*np.log(nz))))}
def cosine(a,b):return float(F.cosine_similarity(torch.from_numpy(a),torch.from_numpy(b),dim=-1).mean())
def emb(a):
    a=PCA(n_components=50,random_state=17).fit_transform(a)
    return TSNE(n_components=2,perplexity=30,random_state=17,init='pca',learning_rate='auto',n_jobs=2).fit_transform(a)
assets=BASE/'figures';assets.mkdir(exist_ok=True);results=[]
for name in ['audio_visual','audio','visual','align_0.1','align_1']:
    path=BASE/name;cfg=json.loads((path/'config.json').read_text(encoding='utf-8'));manifest=json.loads((path/'effective_variant.json').read_text())
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='modal_analysis');cls=scope['AOFL'];install(cls,manifest['weights_text_audio_visual'],manifest['alignment_weight'])
    t=cls('MELD',64,30,.0001,.0001,4,256,4,1024,.3,0,1,True,torch.device('cpu'))
    checkpoint=torch.load(path/'best_model.pth',map_location='cpu',weights_only=False);t.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    assert checkpoint['epoch']==int(best(path)['epoch'])
    assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values() if v.is_floating_point())
    final=torch.load(path/'final_model.pth',map_location='cpu',weights_only=False);t.model.load_state_dict(final['model_state_dict'],strict=True)
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values() if v.is_floating_point())
    t.model.load_state_dict(checkpoint['model_state_dict'],strict=True);t.model.eval()
    loader=DataLoader(Subset(t.valid_dataloader.dataset,sorted(t.valid_indices)),batch_size=64,shuffle=False,collate_fn=t.valid_dataloader.collate_fn)
    cache={k:[] for k in ['x','g','h','xh','labels','pred']}
    with torch.no_grad():
      for batch in loader:
        args=[a for a in batch];args[-1]=args[-1].reshape(-1);out=t.model(*args);g=list(out[1:4]);h=list(out[4:7]);x=t._reconstruction_targets
        _,_,_,xh=t.model.reconstruction(g,h,x)
        for key,a in [('x',x),('g',g),('h',h),('xh',xh)]:cache[key].append(torch.stack(a,1).numpy())
        cache['labels'].append(args[-1][args[-1]!=-1].numpy());cache['pred'].append(out[0].argmax(-1).numpy())
    cache={k:np.concatenate(v) for k,v in cache.items()};assert len(cache['labels'])==1109
    np.savez_compressed(BASE/(name+'_features.npz'),**cache)
    row=best(path);r={'name':name,'weights':manifest['weights_text_audio_visual'],'alignment':manifest['alignment_weight'],'epoch':int(row['epoch']),'valid_f1':float(row['valid_f1']),'test_f1':float(row['test_f1']),'fixed_group_valid_f1':float(f1_score(cache['labels'],cache['pred'],average='weighted')*100),'modalities':{}}
    ix=np.random.default_rng(17).choice(1109,200,replace=False);ix.sort()
    for m,modal in enumerate(['text','audio','visual']):
        x=cache['x'][:,m];xh=cache['xh'][:,m];mse=float(np.mean((x-xh)**2));center=float(np.mean((x-x.mean(0))**2))
        r['modalities'][modal]={'mse':mse,'relative_mse':mse/max(float(np.mean(x*x)),1e-12),'r2':1-mse/max(center,1e-12),'x_xh_cosine':cosine(x,xh),'g_variation':diagnostics(cache['g'][:,m]),'h_variation':diagnostics(cache['h'][:,m])}
    r['shared_cosine']={f'{a}_{b}':cosine(cache['g'][:,a],cache['g'][:,b]) for a,b in [(0,1),(0,2),(1,2)]}
    r['shared_random_utterance_cosine']={f'{a}_{b}':cosine(cache['g'][:,a],cache['g'][np.random.default_rng(29).permutation(1109),b]) for a,b in [(0,1),(0,2),(1,2)]}
    a=np.concatenate([cache[k][ix,m] for k in ['g','h'] for m in range(3)]);xy=emb(a);fig,axes=plt.subplots(1,2,figsize=(13,5))
    for j,label in enumerate(['g text','g audio','g visual','h text','h audio','h visual']):axes[0].scatter(*xy[j*200:(j+1)*200].T,s=10,alpha=.6,label=label)
    axes[0].legend(fontsize=8);axes[0].set_title('Joint embedding, six groups')
    sc=axes[1].scatter(*xy.T,c=np.tile(cache['labels'][ix],6),s=10,cmap='tab10',vmin=0,vmax=6,alpha=.6);fig.colorbar(sc,ax=axes[1]);axes[1].set_title('Identical coordinates, emotion label')
    fig.suptitle(f'{name}, weights={r["weights"]}, alignment={r["alignment"]}');fig.tight_layout();fig.savefig(assets/(name+'_six.png'),dpi=140);plt.close(fig)
    # Shared-only embedding and normalization sensitivity: equalize lengths, preserve directions.
    gs=np.concatenate([cache['g'][ix,m] for m in range(3)]);fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,a,title in [(axes[0],gs,'Shared only, raw vectors'),(axes[1],gs/np.maximum(np.linalg.norm(gs,axis=1,keepdims=True),1e-12),'Shared only, unit-length vectors')]:
        xy=emb(a)
        for m,label in enumerate(['text','audio','visual']):ax.scatter(*xy[m*200:(m+1)*200].T,s=12,alpha=.6,label=label)
        ax.legend();ax.set_title(title)
    fig.suptitle(name);fig.tight_layout();fig.savefig(assets/(name+'_shared.png'),dpi=140);plt.close(fig)
    results.append(r);t.writer.close();del t,checkpoint,final;gc.collect()
reference=json.loads(Path(r'D:/codex/shiyan/output/decoder_senior_study_20261116/analysis.json').read_text())
refs=[{'name':'uniform_'+str(r['weight']),'valid_f1':r['selected_valid_f1'],'shared_cosine':r['cross_modal_shared_cosine']} for r in reference if r['variant']=='baseline']
(BASE/'analysis.json').write_text(json.dumps({'runs':results,'uniform_references':refs,'validation_selected_candidate':max(results,key=lambda r:r['valid_f1'])['name'],'limits':'seed7 screening; test not selection; tSNE not information preservation proof; high cosine may reflect collapse'},indent=2),encoding='utf-8')
def table(headers,data):return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</table>'
body='<h1>分模态重构权重与共享特征对齐实验</h1><p>原AOFL，不加流形；种子7，每次30轮。先比较三组模态权重，按验证F1选候选，再在该候选上分别增加0.1和1.0的共享对齐。没有按测试分数选择方案。</p><h2>具体如何改</h2><p>新增重构损失为（三个模态各自权重乘各自MSE）的平均值。文本、音频、视觉权重顺序固定；相等权重0.1时与此前0.1×平均MSE完全等价。对齐损失为同一句话三个模态g两两之间的1−余弦相似度均值，仅作用于g，从第1轮开始加入。原有CEN、ARE及OPR保留。这是额外对齐约束，不等于修正了原论文目标。</p><p>模态加权仅决定辅助训练约束强度，不是融合时各模态的占比。模态共同训练，三个系数相互影响，本次少量候选不保证分别找到了全局最优值。候选优先增强此前恢复较弱的音频和视觉。</p><h2>验证选模结果</h2>'
body+=table(['配置','文本/音频/视觉权重','对齐权重','选中轮次','验证F1','对应测试F1'],[[r['name'],r['weights'],r['alignment'],r['epoch'],r['valid_f1'],r['test_f1']] for r in results])
body+='<p>测试结果仅在固定方案训练后读取，不参与候选与轮次选择。原AOFL种子7：验证66.6826，对应测试65.7067。统一权重的验证F1参照如下，来自此前同种子30轮实验。</p>'+table(['原AOFL+decoder统一权重','验证F1'],[[r['name'],r['valid_f1']] for r in refs])
for r in results:
    body+=f'<h2>{r["name"]}</h2><p>同话语共享特征余弦：{r["shared_cosine"]}；打乱话语对应后的余弦：{r["shared_random_utterance_cosine"]}。如果两者都很高且变化量很小，需警惕所有话语被压成相似方向。</p>'
    body+=table(['模态','相对重构误差','重构R²','g每坐标平均标准差','g去均值能量占比','g有效秩'],[[m,f"{z['relative_mse']:.4f}",f"{z['r2']:.4f}",f"{z['g_variation']['mean_coordinate_std']:.6f}",f"{z['g_variation']['centered_energy_ratio']:.4f}",f"{z['g_variation']['effective_rank']:.2f}"] for m,z in r['modalities'].items()])
    for suffix in ['six','shared']:body+=f'<img src="figures/{r["name"]}_{suffix}.png">'
body+='<h2>如何判断有没有达到目的</h2><p>共享g更重合只是一个现象。要结合原始256维余弦、同话语相对随机配对的差异、g的跨话语变化、有效秩与分类表现判断。单位长度图用于检查长度影响，和原始图不是同一坐标系，不能直接比较点位置。所有图固定抽取相同200条验证话语、PCA50、t-SNE种子17、perplexity30，没有选择最好看的图。六组联合图与共享单独图的坐标不可相互比较。</p><p>每坐标标准差和去均值能量显著下降、随机配对余弦接近同话语配对、分类下降，可能提示表示趋同，不能称为更好的解耦。有效秩描述跨话语变化的维度分布，单独高或低都不等于分类好坏。这里没有任意设定一个阈值自动宣布坍缩。</p><p>只用种子7做探索，不能宣称稳定提升；没有找到多个模态各自独立最优的权重，也没有证明所有信息保留。x随训练改变，各配置重构误差不能当成固定目标上的绝对公平排名。前一批剩余GAPD权重训练暂缓，已完成结果保留。</p>'
style='body{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.8 Microsoft YaHei,sans-serif;background:#f8fafc;color:#26364a}h1,h2{color:#214b70}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:9px;border:1px solid #d9e2eb;text-align:left}th{background:#e7eff6}img{width:100%;height:auto}'
import base64
for im in assets.glob('*.png'):body=body.replace('figures/'+im.name,'data:image/png;base64,'+base64.b64encode(im.read_bytes()).decode())
report=BASE/'分模态权重与共享对齐报告.html';report.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分模态权重与共享对齐</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
print(json.dumps([{k:r[k] for k in ['name','weights','alignment','epoch','valid_f1','test_f1','shared_cosine']} for r in results],indent=2));print(report)
