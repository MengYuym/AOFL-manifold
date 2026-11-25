"""Audit original AOFL paired decoder runs, then compare four matched configurations."""
from pathlib import Path
import sys,json,csv,math,statistics,runpy,gc,html
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunAOFLDecoderPaired import BASE,ROOT,NEW,install
import torch
from scipy.stats import t as student_t
torch.set_num_threads(2)
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(headers,data):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table></div>'
def describe(values):
    n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values);half=float(student_t.ppf(.975,n-1))*sd/math.sqrt(n)
    return {'n':n,'mean':mean,'sample_sd':sd,'ci95_low':mean-half,'ci95_high':mean+half,'wins':sum(v>0 for v in values),'losses':sum(v<0 for v in values)}
plan=js(BASE/'experiment_plan.json');records=[];audits=[];ref=None
for seed in [7]+NEW:
    for decoder in [False,True]:
        if seed==7:path=Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113/baseline') if decoder else ROOT/'save_model_MELD/multiseed_20261106_144852_867353/seed_7'
        else:path=BASE/f'seed_{seed}'/('with_decoder' if decoder else 'without_decoder')
        cfg=js(path/'config.json');rr=rows(path/'metrics.csv')
        assert len(rr)>=30
        rr=rr[:30]
        assert [int(r['epoch']) for r in rr]==list(range(1,31))
        assert all(math.isfinite(float(r[k])) for r in rr for k in ['train_loss','valid_loss','test_loss','valid_f1','test_f1'])
        assert cfg['torch_initial_seed']==seed
        if ref is None:ref=cfg
        for k in ['dataset','batch_size','learning_rate','weight_decay','num_layers','model_dim','num_heads','hidden_dim','dropout_rate','dropout_rec','CE_loss_param','multi_attn_flag','train_dialogue_ids','valid_dialogue_ids','test_dialogue_ids','selection_metric','tie_policy']:
            assert cfg[k]==ref[k],(seed,decoder,k)
        best=max(rr,key=lambda r:float(r['valid_f1']))
        r={'seed':seed,'decoder':decoder,'best_epoch':int(best['epoch']),'valid_f1':float(best['valid_f1']),'test_f1':float(best['test_f1']),'test_acc':float(best['test_acc']),'path':str(path)}
        if decoder:
            recon=rows(path/'reconstruction_metrics.csv');assert len(recon)==90
            assert all(int(z['utterances'])=={'Train':9989,'Valid':1109,'Test':2610}[z['split']] for z in recon)
            assert all(math.isfinite(float(z[m+'_relative_mse'])) for z in recon for m in ['text','audio','visual'])
            r['selected_valid_reconstruction']=next(z for z in recon if z['split']=='Valid' and int(z['epoch'])==r['best_epoch'])
        if seed in NEW:
            assert cfg['num_epochs']==30 and len(rows(path/'metrics.csv'))==30
            manifest=js(path/'effective_variant.json');assert manifest['manifold'] is False
            assert manifest['source_sha256']==plan['source_sha256']
            summary=js(path/'summary.json');assert summary['best_epoch']==r['best_epoch'] and summary['test_f1_at_best_valid']==r['test_f1']
            r['minutes']=summary['total_training_minutes']
            scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='original_decoder_audit');cls=scope['AOFL']
            if decoder:install(cls,'baseline')
            t=cls.__new__(cls)
            for k,v in cfg.items():setattr(t,k,v)
            t.device=torch.device('cpu');scope['set_seed'](seed);t.get_model()
            assert not hasattr(t.model,'gapd_stage1') and not hasattr(t.model.multiattn,'subspace_refiners')
            assert sum(p.numel() for p in t.model.parameters())==(21769747 if decoder else 21178387)
            for filename in ['best_model.pth','final_model.pth']:
                checkpoint=torch.load(path/filename,map_location='cpu',weights_only=False)
                t.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
                assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values() if v.is_floating_point())
                if filename=='best_model.pth':assert checkpoint['epoch']==r['best_epoch']
                del checkpoint
            del t;gc.collect()
            audits.append({'seed':seed,'decoder':decoder,'strict_finite_best_final':True,'original_aofl_no_manifold':True,'matching_settings_splits_source':True,'complete_30_epochs':True})
        records.append(r)
pairs=[]
for seed in [7]+NEW:
    a=next(r for r in records if r['seed']==seed and not r['decoder']);b=next(r for r in records if r['seed']==seed and r['decoder'])
    pairs.append({'seed':seed,'valid_delta':b['valid_f1']-a['valid_f1'],'test_delta':b['test_f1']-a['test_f1']})
groups={'all_six':describe([p['test_delta'] for p in pairs]),'new_five':describe([p['test_delta'] for p in pairs if p['seed']!=7])}
gapd=js(Path(r'D:/codex/shiyan/output/gapd_decoder_paired_20261113/six_seed_results.json'))
configurations={'AOFL':[r for r in records if not r['decoder']],'AOFL + decoder':[r for r in records if r['decoder']],'GAPD':[r for r in gapd['runs'] if not r['decoder']],'GAPD + decoder':[r for r in gapd['runs'] if r['decoder']]}
stats={name:{k:{'mean':statistics.mean(r[k] for r in rs),'sample_sd':statistics.stdev(r[k] for r in rs)} for k in ['valid_f1','test_f1']} for name,rs in configurations.items()}
comparisons={}
for name,left,right in [('AOFL decoder effect','AOFL','AOFL + decoder'),('GAPD decoder effect','GAPD','GAPD + decoder'),('GAPD decoder vs AOFL','AOFL','GAPD + decoder')]:
    values=[]
    for seed in [7]+NEW:
        a=next(r for r in configurations[left] if r['seed']==seed);b=next(r for r in configurations[right] if r['seed']==seed);values.append(b['test_f1']-a['test_f1'])
    comparisons[name]={'description':right+' minus '+left,**describe(values),'paired_differences':values}
result={'runs':records,'pairs':pairs,'groups':groups,'four_configuration_statistics':stats,'comparisons':comparisons,'new_run_audits':audits,'limitations':['fixed MELD split','decoder initialization fixed40007','six seeds, t interval approximate','seed7 result already seen','GAPD comparison descriptive, not all-new independent confirmation','no test-example uncertainty estimate']}
(BASE/'six_seed_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
g=groups['all_six'];fresh=groups['new_five']
if g['ci95_low']>0 and fresh['ci95_low']>0:verdict='当前六个种子及新增五组结果支持原 AOFL 上加入 decoder 存在平均提升，但不保证每次都提高，也不代表跨数据集有效。'
elif g['mean']>0:verdict='平均测试 F1 有增加，但尚不足以确认稳定提升，应结合新增五组结果和区间判断。'
else:verdict='扩大到六个种子后，平均测试 F1 没有提高，目前不支持在原 AOFL 上加 decoder 带来稳定提升。'
body=f'<h1>原 AOFL＋decoder：六种子配对验证</h1><p class="muted">原 AOFL 保留 OPR · 不加流形模块 · 固定重构位置与权重 0.1</p><div class="callout"><b>{verdict}</b><br>全部六组平均测试变化 {g["mean"]:+.4f} 个百分点，近似 95% 置信区间 [{g["ci95_low"]:+.4f}, {g["ci95_high"]:+.4f}]，提高 {g["wins"]} 组，下降 {g["losses"]} 组。</div>'
body+='<h2>这次具体做了什么</h2><p>此前原 AOFL 加 decoder 只有种子 7。本次固定补充种子 42、2025、123、1024、10007，每个种子分别运行原 AOFL 与原 AOFL＋decoder，共新增十次，每次三十轮。所有运行均完成，不按中途测试结果修改方案。种子 42 的历史 100 轮实验保留，另行重跑本次 30 轮版本；种子 7 的原 AOFL 使用历史实验前 30 轮内验证 F1 最佳结果。</p><p>decoder 输入同模态、OPR 处理前的 g、h，拼接后通过两层全连接网络重构解耦前的 x。三个模态各用一个 decoder；目标 x 分支断开梯度，补齐位置不计入，重构损失权重固定 0.1，从第一轮起加入。原 OPR、CEN、ARE、分类器和损失阶段设置保留。</p>'
body+='<h2>原 AOFL 的逐种子配对结果</h2><p>每版分别用验证 F1 选模型，再读取对应测试 F1。变化 = 加 decoder − 原 AOFL，单位为百分点。</p>'
data=[]
for p in pairs:
    a=next(r for r in records if r['seed']==p['seed'] and not r['decoder']);b=next(r for r in records if r['seed']==p['seed'] and r['decoder'])
    data.append([p['seed'],a['best_epoch'],b['best_epoch'],f"{a['valid_f1']:.4f}",f"{b['valid_f1']:.4f}",f"{a['test_f1']:.4f}",f"{b['test_f1']:.4f}",f"{p['test_delta']:+.4f}"])
body+=table(['种子','原版轮次','decoder轮次','原版验证F1','decoder验证F1','原版测试F1','decoder测试F1','测试变化'],data)
body+='<h2>平均改善是否可靠</h2>'+table(['范围','平均测试变化','差值样本标准差','近似95%置信区间','提高／总数'],[['全部六组' if name=='all_six' else '本次新增五组',f"{z['mean']:+.4f}",f"{z['sample_sd']:.4f}",f"[{z['ci95_low']:+.4f}, {z['ci95_high']:+.4f}]",f"{z['wins']} / {z['n']}"] for name,z in groups.items()])
body+='<p>置信区间按同一种子两版的测试 F1 差值，用双侧 Student-t 方法估计。区间包含 0，说明还不能排除平均没有改善；不是“95% 的运行都会提高”。种子少，且只针对固定数据划分与初始化方式，区间属于近似描述。种子 7 已经看过，新增五组单独报告，不把所有数据都称为全新的确认实验。</p>'
body+='<h2>学长要求的两条线放在一起看</h2><p>下面四版均使用相同六个种子、MELD 官方划分、每次前三十轮的验证选模规则。GAPD 两版沿用已完成的结果，只作完整对照，不重新调参。</p>'
body+=table(['模型','验证F1：均值 ± 标准差','测试F1：均值 ± 标准差'],[[name,*[f"{s[k]['mean']:.4f} ± {s[k]['sample_sd']:.4f}" for k in ['valid_f1','test_f1']]] for name,s in stats.items()])
body+=table(['比较（后者减前者）','平均测试变化','近似95%置信区间','提高／总数'],[[z['description'],f"{z['mean']:+.4f}",f"[{z['ci95_low']:+.4f}, {z['ci95_high']:+.4f}]",f"{z['wins']} / {z['n']}"] for z in comparisons.values()])
body+='<p>不能只凭四个平均值的大小宣布显著优劣。多个对照是描述性分析，未进行多重比较校正；GAPD 实验结果此前已经观察过。训练种子差异也不等于测试样本抽样不确定性。</p>'
body+='<h2>检查结果与限制</h2><ul><li>新增十次均有完整三十轮指标，最佳轮次与保存指标对应；最佳和最终模型均可严格加载，权重没有 NaN 或无穷值。</li><li>确认两版均为原 AOFL，没有 GAPD 或子空间处理模块；参数量分别为 21,178,387 与 21,769,747。</li><li>相同种子的原模型初始权重与 CPU 随机数状态一致；五个新种子的原模型初始化确实不同。</li><li>数据划分、关键训练设置与选模规则一致，源码哈希与训练前计划一致，未中途修改实现。</li><li>加 decoder 的每次有九十行重构指标，训练／验证／测试有效话语数分别为 9989／1109／2610。</li><li>decoder 初始化固定种子40007，沿用既有设置；不覆盖全部新增模块初始化的不确定性。只验证固定 MELD 划分，不能直接推广到其他数据集。</li><li>原代码注意力对补齐位置的处理仍保持旧设置，批次分组影响尚未消除；重构误差降低也不等于保留了更多情感信息。</li></ul>'
body+='<h2>如何汇报</h2><p>把两条线分别描述为“原 AOFL 加重构约束”和“GAPD 加同样重构约束”，展示各自配对差值及波动。当前重构位置已确认，无需再询问位置。先按这批固定结果判断是否保留方向，不继续为追求正结果反复更换种子或权重。</p><p>本报告对应数据：aofl_decoder_paired_20261113/six_seed_results.json。每种子目录包含逐轮指标、配置、最佳和最终模型；原有种子7与GAPD文件均保留。</p>'
style='body{max-width:1250px;margin:36px auto;padding:0 24px;font:16px/1.85 "Microsoft YaHei",sans-serif;background:#f8fafc;color:#26364a}h1{font-size:28px;color:#214b70}h2{font-size:21px;margin-top:32px;border-bottom:1px solid #cbd8e5;color:#285f85}.muted{color:#64748b}.callout{background:#eaf2f8;border-left:4px solid #387cac;padding:16px}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:10px 12px;border:1px solid #d9e2eb;text-align:left}th{background:#e7eff6}tbody tr:nth-child(even){background:#f3f7fa}.scroll{overflow-x:auto}b{color:#1f5d84}'
report=BASE.parent/'原AOFL_Decoder_六种子配对与四版对照.html'
report.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原AOFL Decoder 六种子验证</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
print(json.dumps({'pairs':pairs,'groups':groups,'four_configuration_statistics':stats,'comparisons':comparisons},indent=2));print(report)
