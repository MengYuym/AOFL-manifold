"""Fixed weight screening and validation-only feature diagnostics for senior review."""
from pathlib import Path
import os,sys,json,runpy,argparse,textwrap,csv,gc
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
import torch
from RunDecoderComparison import install
BASE=Path(r'D:/codex/shiyan/output/decoder_senior_study_20261116')
def path_for(variant,weight):
    if weight==.1:return Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113')/variant
    return BASE/f'{variant}_weight_{weight:g}'
def train(variant,weight):
    output=path_for(variant,weight)
    assert not output.exists(),output
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='weight_study')
    install(scope['AOFL'],variant,output,weight)
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed','7','--num_epochs','30','--output_dir',str(output)]
    body=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(v,*a,**kw):
        if isinstance(v,torch.nn.Module):v={'model_state_dict':v.state_dict()}
        return save(v,*a,**kw)
    torch.save=weights
    try:exec(compile(textwrap.dedent(body),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps({'variant':variant,'weight':weight,'seed':7,'epochs':30,'selection':'validation F1 only','target':'pre-OPR/pre-gate g,h to detached x'},indent=2),encoding='utf-8')
def analyze():
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import f1_score,silhouette_score
    from torch.utils.data import DataLoader,Subset
    from torch.nn import functional as F
    summary=[];features={};assets=BASE/'figures';assets.mkdir(exist_ok=True)
    def embed(a):
        q=PCA(n_components=min(50,a.shape[1],len(a)-1),random_state=17).fit_transform(a)
        return TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=17,n_jobs=2).fit_transform(q)
    for variant in ['baseline','gapd']:
      for weight in [.01,.1,1.]:
        path=path_for(variant,weight)
        if not (path/'summary.json').exists():continue
        checkpoint=torch.load(path/'best_model.pth',map_location='cpu',weights_only=False)
        scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='feature_analysis');cls=scope['AOFL'];install(cls,variant,weight=weight);scope['set_seed'](7)
        t=cls('MELD',64,30,.0001,.0001,4,256,4,1024,.3,0,1,True,torch.device('cpu'))
        t.model.load_state_dict(checkpoint['model_state_dict'],strict=True);t.model.eval()
        loader=DataLoader(Subset(t.valid_dataloader.dataset,sorted(t.valid_indices)),batch_size=64,shuffle=False,collate_fn=t.valid_dataloader.collate_fn)
        cache={k:[] for k in ['x','g','h','xh','labels','pred']}
        with torch.no_grad():
          for batch in loader:
            args=[b.to(t.device) for b in batch];args[-1]=args[-1].reshape(-1);out=t.model(*args)
            gs=list(out[1:4]);hs=list(out[4:7]);xs=t._reconstruction_targets
            _,_,_,xh=t.model.reconstruction(gs,hs,xs)
            for key,arr in [('x',xs),('g',gs),('h',hs),('xh',xh)]:cache[key].append(torch.stack(arr,1).cpu().numpy())
            cache['labels'].append(args[-1][args[-1]!=-1].cpu().numpy());cache['pred'].append(out[0].argmax(-1).cpu().numpy())
        cache={k:np.concatenate(v) for k,v in cache.items()};assert len(cache['labels'])==1109
        tag=f'{variant}_{weight:g}';np.savez_compressed(BASE/(tag+'_features.npz'),**cache)
        sample=np.random.default_rng(17).choice(1109,400,replace=False);sample.sort()
        item={'variant':variant,'weight':weight,'epoch':checkpoint['epoch'],'selected_valid_f1':checkpoint['metrics']['valid_f1'],'fixed_group_valid_f1':float(f1_score(cache['labels'],cache['pred'],average='weighted')*100),'modalities':{}}
        fig,axes=plt.subplots(1,3,figsize=(15,4.6))
        for mi,name in enumerate(['Text','Audio','Visual']):
            x=cache['x'][:,mi];xh=cache['xh'][:,mi];g=cache['g'][:,mi];h=cache['h'][:,mi]
            mse=float(np.mean((x-xh)**2));energy=float(np.mean(x*x));cos=float(F.cosine_similarity(torch.from_numpy(x),torch.from_numpy(xh),dim=-1).mean())
            shuffled=np.random.default_rng(19).permutation(len(x));shuffle_mse=float(np.mean((x-xh[shuffled])**2))
            centered=float(np.mean((x-x.mean(0))**2));r2=1-mse/max(centered,1e-12)
            item['modalities'][name]={'mse':mse,'relative_mse':mse/max(energy,1e-12),'cosine_x_xh':cos,'shuffled_pair_mse':shuffle_mse,'centered_mean_baseline_mse':centered,'reconstruction_r2':r2,'g_h_cosine':float(F.cosine_similarity(torch.from_numpy(g),torch.from_numpy(h),dim=-1).mean())}
            xy=embed(np.concatenate([x[sample],xh[sample]]));axes[mi].scatter(*xy[:400].T,s=9,alpha=.55,label='x');axes[mi].scatter(*xy[400:].T,s=9,alpha=.55,label='reconstructed x',marker='x');axes[mi].set_title(name);axes[mi].legend(fontsize=8)
        fig.suptitle(f'{variant}, reconstruction weight={weight:g}; joint t-SNE of x and reconstructed x');fig.tight_layout();fig.savefig(assets/(tag+'_reconstruction.png'),dpi=150);plt.close(fig)
        # All six groups share one embedding; class-coloured view uses identical coordinates.
        arrays=[cache[k][sample,mi] for k in ['g','h'] for mi in range(3)]
        a=np.concatenate(arrays);xy=embed(a);group=np.repeat(np.arange(6),400);labels=np.tile(cache['labels'][sample],6)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for j,name in enumerate(['g text','g audio','g visual','h text','h audio','h visual']):axes[0].scatter(*xy[group==j].T,s=8,alpha=.55,label=name)
        axes[0].legend(fontsize=8);axes[0].set_title('Same embedding: modality and feature type')
        sc=axes[1].scatter(*xy.T,c=labels,cmap='tab10',s=8,alpha=.55,vmin=0,vmax=6);fig.colorbar(sc,ax=axes[1],ticks=range(7));axes[1].set_title('Same embedding: emotion label (dataset ID)')
        fig.suptitle(f'{variant}, weight={weight:g}; pre-OPR/pre-gate g and h');fig.tight_layout();fig.savefig(assets/(tag+'_disentangle.png'),dpi=150);plt.close(fig)
        # Quantify paired cross-modal g alignment in original space.
        item['cross_modal_shared_cosine']={f'{a}_{b}':float(F.cosine_similarity(torch.from_numpy(cache['g'][:,a]),torch.from_numpy(cache['g'][:,b]),dim=-1).mean()) for a,b in [(0,1),(0,2),(1,2)]}
        item['private_modality_silhouette']=float(silhouette_score(np.concatenate([cache['h'][sample,mi] for mi in range(3)]),np.repeat(np.arange(3),400),sample_size=1000,random_state=17))
        summary.append(item);t.writer.close();del t,checkpoint;gc.collect();torch.cuda.empty_cache()
    (BASE/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report(summary)
def report(results):
    import html
    def table(headers,data):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</table>'
    body='<h1>Decoder 权重与特征可视化分析</h1><p>固定种子7，重构位置不变；权重0.01、0.1、1.0，每次30轮。0.1复用原实验，其余新增训练。所有图与原始空间分析均使用验证集，不用测试成绩选择权重。</p>'
    body+='<h2>权重筛选</h2><p>主要筛选依据为各运行记录中的最佳验证加权F1，不能从这一个种子的最好结果宣布稳定提升。固定分组F1仅为可复核特征提取时的指标；原代码注意力未完全屏蔽补齐位置，改变验证分组可能改变分数，不能把两种分组的F1混着选模型。</p>'
    body+=table(['模型','重构权重','选中轮次','训练记录验证F1','固定分组验证F1'],[[r['variant'],r['weight'],r['epoch'],f"{r['selected_valid_f1']:.4f}",f"{r['fixed_group_valid_f1']:.4f}"] for r in results])
    for variant in ['baseline','gapd']:
      rs=[r for r in results if r['variant']==variant];best=max(rs,key=lambda r:r['selected_valid_f1']);body+=f'<p>{variant} 在这三个权重中按验证F1选出的候选为 {best["weight"]:g}，仅用于后续讨论，不代表已完成多种子确认。</p>'
    body+='<h2>看图前先说明</h2><p>每个重构图中，同模态的x与x̂联合做一次PCA再做t-SNE；不是两张图独立降维后比较坐标。六类g、h也联合降维。不同模型、权重或模态面板的坐标不能直接比较。每图取相同400条验证话语，抽样和t-SNE随机种子17、perplexity30，没有试多张图挑最好看的。原始特征未经逐组标准化。</p><p>t-SNE只保留部分局部关系，二维重合不能证明信息没有丢失，二维分团也不保证解耦正确。h不必形成三个完全分离的团，g也不必不分情感类别地全部挤在一起。必须结合原始256维空间中的指标和分类结果判断。</p>'
    for r in results:
      tag=f'{r["variant"]}_{r["weight"]:g}';body+=f'<h2>{r["variant"]}，权重 {r["weight"]:g}</h2>'
      body+=table(['模态','相对重构误差','x与x̂余弦','重构R²','打乱配对MSE/真实配对MSE','g与h余弦'],[[name,f"{z['relative_mse']:.4f}",f"{z['cosine_x_xh']:.4f}",f"{z['reconstruction_r2']:.4f}",f"{z['shuffled_pair_mse']/max(z['mse'],1e-12):.3f}",f"{z['g_h_cosine']:.4f}"] for name,z in r['modalities'].items()])
      body+=f'<p>同话语跨模态g的平均余弦：{r["cross_modal_shared_cosine"]}；h按模态分组的原始空间轮廓系数：{r["private_modality_silhouette"]:.4f}。这只是分布描述，不是越分开越好。</p>'
      body+=f'<img src="figures/{tag}_reconstruction.png"><p>蓝色为x，橙色为恢复特征；每个模态面板独立联合降维，面板间坐标不可比较。</p><img src="figures/{tag}_disentangle.png"><p>左图区分六类g、h；右图使用完全相同坐标，按情感标签着色，帮助区分模态差异和情感类别结构。</p>'
    body+='<h2>指标如何理解</h2><p>相对误差=平方误差总和/目标平方值总和，越小表示总体尺度下重构更接近；余弦越接近1表示方向更一致。R²比较恢复误差与“把所有话语都预测为验证集均值”的误差，1理想、0相当于均值预测、负数更差。均值仅作为事后描述，不参与训练。打乱配对误差明显更大，说明恢复不只是匹配整体分布，也包含话语对应信息；仍不能证明全部信息保留。</p><p>x会随上游网络训练变化，不同权重的x不是固定目标，不能只比较MSE大小宣布某权重保留信息更好。重构只约束g、h合起来恢复x，不保证两者各自语义正确。当前是单种子权重筛选，后续先向学长展示完整结果，再决定是否值得做配对多种子确认。</p>'
    style='body{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.8 Microsoft YaHei,sans-serif;color:#26364a;background:#f8fafc}h1,h2{color:#214b70}h2{margin-top:32px}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d9e2eb;padding:9px;text-align:left}th{background:#e7eff6}img{width:100%;height:auto;margin-top:18px}'
    (BASE/'Decoder权重与特征可视化报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Decoder权重与特征分析</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','analyze']);p.add_argument('--variant',choices=['baseline','gapd']);p.add_argument('--weight',type=float);a=p.parse_args();torch.set_num_threads(2);BASE.mkdir(parents=True,exist_ok=True)
    if a.action=='train':train(a.variant,a.weight)
    else:analyze()
