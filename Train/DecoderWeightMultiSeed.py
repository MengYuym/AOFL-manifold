"""Predeclared multi-seed reconstruction-weight screening; no test-based selection."""
from pathlib import Path
import os,sys,runpy,textwrap,json,csv,hashlib,argparse,statistics,math,html
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'Train'),str(ROOT/'Model')]
import torch
from RunDecoderComparison import install
BASE=Path(r'D:/codex/shiyan/output/decoder_weights_multiseed_20261117')
SEEDS=[42,2025,123];WEIGHTS=[.01,.1,1.]
SOURCES=['Train/TrainMultiEMO.py','Train/RunDecoderComparison.py','Model/FeatureReconstruction.py','Model/AOFL_Model.py','Model/Multi_Disentangle.py','Train/RunGAPDStage3.py','Train/RunGAPDStage1.py']
def hashes():return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}
def location(v,s,w):
    if w==.1:return Path(r'D:/codex/shiyan/output')/('aofl_decoder_paired_20261113' if v=='baseline' else 'gapd_decoder_paired_20261113')/f'seed_{s}'/'with_decoder'
    return BASE/f'{v}_seed_{s}_weight_{w:g}'
def train(v,s,w):
    output=location(v,s,w);assert not output.exists(),output
    assert hashes()==json.loads((BASE/'plan.json').read_text())['source_sha256']
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='multiseed_weight');install(scope['AOFL'],v,output,w)
    sys.argv=['TrainMultiEMO.py','--dataset','MELD','--seed',str(s),'--num_epochs','30','--output_dir',str(output)]
    body=(ROOT/'Train/TrainMultiEMO.py').read_text(encoding='utf-8').split("if __name__ == '__main__':",1)[1]
    save=torch.save
    def weights(value,*a,**kw):
        if isinstance(value,torch.nn.Module):value={'model_state_dict':value.state_dict()}
        return save(value,*a,**kw)
    torch.save=weights
    try:exec(compile(textwrap.dedent(body),str(ROOT/'Train/TrainMultiEMO.py'),'exec'),scope)
    finally:torch.save=save
    (output/'effective_variant.json').write_text(json.dumps({'variant':v,'seed':s,'weight':w,'epochs':30,'decoder_seed':40007,'source_sha256':hashes()}),encoding='utf-8')
def prepare():
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'plan.json';assert not p.exists()
    p.write_text(json.dumps({'seeds':SEEDS,'weights':WEIGHTS,'new_runs':12,'epochs':30,'reuse':'.1 for all three seeds','selection':'per-run maximum validation F1 then weight by average validation F1 across seeds42,2025,123; no test selection','fixed':'same pre-gate reconstruction position, decoder seed40007, geometry seed30007','stop':'complete all runs, no interim stopping/tuning','seed7':'previous exploration shown separately, excluded from weight selection','source_sha256':hashes()},indent=2),encoding='utf-8')
def report():
    import gc
    from scipy.stats import t as student_t
    records=[];reference=None
    def readrows(p):
        with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    for v in ['baseline','gapd']:
      for s in SEEDS:
       for w in WEIGHTS:
        p=location(v,s,w);rr=readrows(p/'metrics.csv');assert len(rr)==30
        best=max(rr,key=lambda r:float(r['valid_f1']));cfg=json.loads((p/'config.json').read_text(encoding='utf-8'))
        assert cfg['torch_initial_seed']==s
        if reference is None:reference=cfg
        for k in ['batch_size','learning_rate','weight_decay','num_epochs','train_dialogue_ids','valid_dialogue_ids','test_dialogue_ids']:assert cfg[k]==reference[k]
        recon=readrows(p/'reconstruction_metrics.csv');assert len(recon)==90
        assert all(int(z['utterances'])=={'Train':9989,'Valid':1109,'Test':2610}[z['split']] for z in recon)
        assert all(math.isfinite(float(z[k])) for z in rr for k in ['valid_f1','test_f1','train_loss','valid_loss'])
        if w!=.1:
            scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='weight_audit');cls=scope['AOFL'];install(cls,v,weight=w)
            t=cls.__new__(cls)
            for k,val in cfg.items():setattr(t,k,val)
            t.device=torch.device('cpu');scope['set_seed'](s);t.get_model()
            for name in ['best_model.pth','final_model.pth']:
                ck=torch.load(p/name,map_location='cpu',weights_only=False);t.model.load_state_dict(ck['model_state_dict'],strict=True)
                assert all(torch.isfinite(a).all() for a in ck['model_state_dict'].values() if a.is_floating_point())
                if name=='best_model.pth':assert ck['epoch']==int(best['epoch'])
                del ck
            del t;gc.collect()
        records.append({'variant':v,'seed':s,'weight':w,'epoch':int(best['epoch']),'valid_f1':float(best['valid_f1']),'test_f1':float(best['test_f1']),'path':str(p)})
    summaries=[];selected={}
    for v in ['baseline','gapd']:
      for w in WEIGHTS:
        rs=[r for r in records if r['variant']==v and r['weight']==w]
        summaries.append({'variant':v,'weight':w,**{k:{'mean':statistics.mean(r[k] for r in rs),'sd':statistics.stdev(r[k] for r in rs)} for k in ['valid_f1','test_f1']}})
      selected[v]=max([r for r in summaries if r['variant']==v],key=lambda r:r['valid_f1']['mean'])['weight']
    comparisons=[]
    for v,w in selected.items():
      ds=[]
      for s in SEEDS:
        p=Path(r'D:/codex/shiyan/output/aofl_decoder_paired_20261113')/f'seed_{s}'/'without_decoder'
        b=max(readrows(p/'metrics.csv'),key=lambda r:float(r['valid_f1']))
        chosen=next(r for r in records if r['variant']==v and r['weight']==w and r['seed']==s)
        ds.append(chosen['test_f1']-float(b['test_f1']))
      mean=statistics.mean(ds);sd=statistics.stdev(ds);half=float(student_t.ppf(.975,2))*sd/math.sqrt(3)
      comparisons.append({'variant':v,'selected_weight':w,'test_delta_vs_original_AOFL':ds,'mean':mean,'ci95':[mean-half,mean+half]})
    result={'records':records,'summaries':summaries,'selected_by_mean_validation':selected,'comparisons_to_original_AOFL':comparisons,'audits':'12 new runs strict best/final load finite, 30 epochs, 90 recon rows, matching split/settings','limits':'3 seeds used for tuning, not independent confirmation; fixed MELD; no significance claim'}
    (BASE/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    def table(h,data):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</table>'
    body='<h1>Decoder 多种子权重对照</h1><p>种子42、2025、123；权重0.01、0.1、1.0；原AOFL与GAPD各三种权重。新增十二次、每次三十轮，权重0.1复用已完成结果。固定重构位置与模块初始化。种子7已用于前期探索，本次不参与权重选择。</p><h2>平均结果</h2><p>先在每次运行中按验证F1选轮次，再根据三个种子的平均验证F1选权重。测试结果只在选择后报告。</p>'
    body+=table(['版本','权重','验证F1 均值 ± 标准差','对应测试F1 均值 ± 标准差'],[[r['variant'],r['weight'],*[f"{r[k]['mean']:.4f} ± {r[k]['sd']:.4f}" for k in ['valid_f1','test_f1']]] for r in summaries])
    body+='<h2>按验证集选出的候选与原AOFL比较</h2>'
    for c in comparisons:body+=f'<p>{c["variant"]}候选权重为{c["selected_weight"]}，同种子下相对原AOFL的测试差值为{[round(d,4) for d in c["test_delta_vs_original_AOFL"]]}，平均{c["mean"]:+.4f}个百分点；近似95%配对t区间[{c["ci95"][0]:+.4f}, {c["ci95"][1]:+.4f}]。</p>'
    body+='<p>区间只是三个种子的近似描述；这些种子同时用于权重筛选，不能当作独立确认。区间包含0表示不能排除没有平均提升。不得只挑测试分数最高的权重宣布超过原AOFL。</p><h2>逐次结果</h2>'
    body+=table(['版本','种子','权重','验证选中轮次','验证F1','对应测试F1'],[[r['variant'],r['seed'],r['weight'],r['epoch'],r['valid_f1'],r['test_f1']] for r in records])
    body+='<h2>检查与局限</h2><p>十二次新增训练全部完成；最佳、最终模型严格加载且权重有限。每次九十行重构记录，排除补齐位置。划分及关键设置一致。原代码注意力的补齐处理未改，批次分组影响仍存在。decoder随机种子固定40007，几何模块固定30007，只改变原模型及训练种子。尚不包含跨数据集验证；权重调优不能代替独立多种子确认，也不能证明重构完全没有信息损失。完整记录与模型按版本、种子、权重保存。</p>'
    style='body{max-width:1150px;margin:32px auto;padding:0 24px;font:16px/1.8 Microsoft YaHei,sans-serif;color:#26364a;background:#f8fafc}h1,h2{color:#214b70}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:10px;border:1px solid #d9e2eb;text-align:left}th{background:#e7eff6}'
    (BASE/'Decoder多种子权重对照.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Decoder多种子权重对照</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
    print(json.dumps({'summaries':summaries,'selected':selected,'comparisons':comparisons},indent=2))
if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','report']);p.add_argument('--variant');p.add_argument('--seed',type=int);p.add_argument('--weight',type=float);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='train':train(a.variant,a.seed,a.weight)
    else:report()
