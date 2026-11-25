"""Audit six new runs and report paired uncertainty without overwriting prior report."""
from pathlib import Path
import sys,json,csv,statistics,html,runpy,gc,math
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunGAPDDecoderPaired import BASE,ROOT,install_decoder,install_gapd
import torch
from scipy.stats import t as student_t
torch.set_num_threads(2)
NEW=[123,1024,10007]
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def table(headers,data):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table></div>'
prior=js(BASE/'paired_results.json');runs=prior['runs'].copy();audits=[]
ref=js(Path(runs[0]['path'])/'config.json')
plan=js(BASE/'extension_plan.json')
assert plan['new_seeds']==NEW
for seed in NEW:
    for decoder in [False,True]:
        path=BASE/f'seed_{seed}'/('with_decoder' if decoder else 'without_decoder')
        cfg=js(path/'config.json');summary=js(path/'summary.json');rr=rows(path/'metrics.csv');manifest=js(path/'effective_variant.json')
        assert len(rr)==30 and [int(r['epoch']) for r in rr]==list(range(1,31))
        assert all(math.isfinite(float(r[k])) for r in rr for k in ['train_loss','valid_loss','test_loss','valid_f1','test_f1'])
        assert cfg['torch_initial_seed']==seed and manifest['decoder_weight']==(.1 if decoder else 0)
        assert manifest['source_sha256']==plan['source_sha256']
        for k in ['dataset','batch_size','num_epochs','learning_rate','weight_decay','num_layers','model_dim','num_heads','hidden_dim','dropout_rate','dropout_rec','CE_loss_param','multi_attn_flag','train_dialogue_ids','valid_dialogue_ids','test_dialogue_ids','selection_metric','tie_policy']:
            assert cfg[k]==ref[k],(seed,decoder,k)
        best=max(rr,key=lambda r:float(r['valid_f1']))
        assert int(best['epoch'])==summary['best_epoch']
        assert float(best['test_f1'])==summary['test_f1_at_best_valid']
        item={'seed':seed,'decoder':decoder,'best_epoch':int(best['epoch']),'valid_f1':float(best['valid_f1']),'test_f1':float(best['test_f1']),'test_acc':float(best['test_acc']),'minutes':summary['total_training_minutes'],'path':str(path)}
        if decoder:
            recon=rows(path/'reconstruction_metrics.csv');assert len(recon)==90
            assert all(int(r['utterances'])=={'Train':9989,'Valid':1109,'Test':2610}[r['split']] for r in recon)
            assert all(math.isfinite(float(r[m+'_relative_mse'])) for r in recon for m in ['text','audio','visual'])
            item['selected_reconstruction_valid']=next(r for r in recon if r['split']=='Valid' and int(r['epoch'])==item['best_epoch'])
        scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='extension_audit');cls=scope['AOFL']
        if decoder:install_decoder(cls,'gapd')
        else:install_gapd(cls)
        t=cls.__new__(cls)
        for k,v in cfg.items():setattr(t,k,v)
        t.device=torch.device('cpu');scope['set_seed'](seed);t.get_model()
        assert sum(p.numel() for p in t.model.parameters())==(22872927 if decoder else 22281567)
        for filename in ['best_model.pth','final_model.pth']:
            checkpoint=torch.load(path/filename,map_location='cpu',weights_only=False)
            t.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values() if v.is_floating_point())
            if filename=='best_model.pth':assert checkpoint['epoch']==item['best_epoch']
            del checkpoint
        del t;gc.collect()
        audits.append({'seed':seed,'decoder':decoder,'complete_30_epochs':True,'settings_and_splits_match':True,'source_hash_matches_predeclared_plan':True,'strict_finite_best_final_checkpoints':True})
        runs.append(item)
pairs=[]
for seed in [7,42,2025]+NEW:
    a=next(r for r in runs if r['seed']==seed and not r['decoder']);b=next(r for r in runs if r['seed']==seed and r['decoder'])
    pairs.append({'seed':seed,'valid_delta':b['valid_f1']-a['valid_f1'],'test_delta':b['test_f1']-a['test_f1']})
def describe(ps):
    ds=[r['test_delta'] for r in ps];n=len(ds);mean=statistics.mean(ds);sd=statistics.stdev(ds);half=float(student_t.ppf(.975,n-1))*sd/math.sqrt(n)
    return {'n':n,'mean_test_delta':mean,'paired_sample_sd':sd,'ci95_low':mean-half,'ci95_high':mean+half,'wins':sum(d>0 for d in ds),'losses':sum(d<0 for d in ds),'ties':sum(d==0 for d in ds),'method':'two-sided Student-t interval on paired per-seed differences; approximate small-sample description'}
groups={'all_six':describe(pairs),'followup_five':describe([p for p in pairs if p['seed']!=7]),'new_three':describe([p for p in pairs if p['seed'] in NEW])}
stats={}
for decoder in [False,True]:
    rs=[r for r in runs if r['decoder']==decoder]
    stats[str(decoder)]={k:{'mean':statistics.mean(r[k] for r in rs),'sample_sd':statistics.stdev(r[k] for r in rs)} for k in ['valid_f1','test_f1']}
result={'runs':runs,'pairs':pairs,'groups':groups,'statistics':stats,'new_run_audits':audits,'limitations':['fixed MELD split','fixed additional-module initialization seeds','seed7 exploratory and earlier results seen before extension','small sample t approximation','no test-example sampling uncertainty assessed','not a comparison against original AOFL']}
(BASE/'six_seed_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
allg=groups['all_six'];newg=groups['new_three'];follow=groups['followup_five']
if allg['ci95_low']>0 and newg['mean_test_delta']>0:
    verdict='结果支持在当前设置下存在平均提升，但证据仍限于这些种子、固定数据划分与初始化方式，不能保证每次运行都提高。'
elif allg['mean_test_delta']>0:
    verdict='平均分有提高，但估计范围仍包含 0，目前不能确认存在稳定的平均提升。'
else:verdict='扩大种子后，平均分没有提高，目前不支持 decoder 带来稳定提升。'
body=f'<h1>GAPD＋decoder：六种子配对验证</h1><p class="muted">固定重构位置 · 权重 0.1 · MELD 官方划分 · 每次 30 轮</p><div class="callout"><b>{verdict}</b><br>六组平均测试 F1 变化：{allg["mean_test_delta"]:+.4f} 个百分点；近似 95% 置信区间：[{allg["ci95_low"]:+.4f}, {allg["ci95_high"]:+.4f}]；提高 {allg["wins"]} 组、下降 {allg["losses"]} 组。</div>'
body+='<h2>这次做了什么</h2><p>沿用前三个种子 7、42、2025，事先确定再补充种子 123、1024、10007。每个新种子分别运行不加 decoder 与加 decoder，共新增 6 次、180 轮。全部完成，没有根据中途测试成绩调参、换种子或停止实验。</p><p>位置保持为：同模态、OPR／流形门控前的 g、h 拼接 → decoder → 重构解耦前的 x。三个模态各有独立 decoder，重构目标分支断开梯度，排除补齐位置，损失权重固定 0.1。用户已明确确认采用这一位置。</p>'
body+='<h2>逐种子结果</h2><p>两版分别按验证 F1 选最佳轮次，再比较对应测试 F1。测试分数不参与选模型。变化 = 加 decoder − 不加 decoder，正数为改善，单位为百分点。</p>'
data=[]
for p in pairs:
    a=next(r for r in runs if r['seed']==p['seed'] and not r['decoder']);b=next(r for r in runs if r['seed']==p['seed'] and r['decoder'])
    data.append([p['seed'],'本次新增' if p['seed'] in NEW else '已有',a['best_epoch'],b['best_epoch'],f"{a['test_f1']:.4f}",f"{b['test_f1']:.4f}",f"{p['test_delta']:+.4f}",f"{p['valid_delta']:+.4f}"])
body+=table(['种子','来源','不加：轮次','加：轮次','不加：测试 F1','加：测试 F1','测试变化','验证变化'],data)
body+='<h2>平均值与随机波动</h2>'+table(['版本','验证 F1 平均 ± 样本标准差','测试 F1 平均 ± 样本标准差'],[['加 decoder' if d else '不加 decoder',*[f"{stats[str(d)][k]['mean']:.4f} ± {stats[str(d)][k]['sample_sd']:.4f}" for k in ['valid_f1','test_f1']]] for d in [False,True]])
body+='<p>分数波动小和平均提升是两个问题。判断平均提升主要看同一种子两版的差值，而不是只比较两版各自的标准差。增加种子不能代替补充新的数据集。</p>'
body+='<h2>提升范围：是否仍可能没有改善</h2>'
labels={'all_six':'全部 6 组','followup_five':'后续 5 组（不含探索种子 7）','new_three':'本次事先确定的 3 组'}
body+=table(['统计范围','平均测试变化','差值的样本标准差','近似 95% 置信区间','提高／组数'],[[labels[k],f"{g['mean_test_delta']:+.4f}",f"{g['paired_sample_sd']:.4f}",f"[{g['ci95_low']:+.4f}, {g['ci95_high']:+.4f}]",f"{g['wins']} / {g['n']}"] for k,g in groups.items()])
body+='<p>这里用每个种子的“加 − 不加”差值计算均值，再用 Student-t 方法估计平均差值的范围。若范围包含 0，说明这些实验还不能排除平均没有改善的情况；若整体高于 0，会更支持平均提升。它不是“95% 的运行都会提高”，也不是下一次实验分数的预测范围。</p><p>这些区间只是小样本近似：依赖差值独立、分布适合 t 方法等假设。六个种子仍少，且种子 7 是最初发现改善的探索结果、前两组也已被看过，因此同时列出后续五组和本次新增三组，避免把所有观察都当成独立的确认。没有因区间结果继续追加种子直到过线。</p>'
body+='<h2>这次检查了什么</h2><ul><li>本次六次运行均完整记录 30 轮，指标为有限数值；每次最佳轮次与保存的模型一致。</li><li>官方训练、验证、测试话语划分与学习率、批量大小、网络结构、训练预算、选模规则逐项一致。</li><li>训练涉及的源码哈希与运行前预定计划一致，说明没有中途修改模型实现。</li><li>同一种子的 GAPD 原有权重及 CPU 随机数状态保持配对；不同种子的原模型初始化不同。</li><li>新增六次的最佳和最终模型均严格加载成功，权重未出现 NaN 或无穷值。</li><li>加 decoder 的每次均有 90 行重构记录，训练／验证／测试有效话语分别为 9989／1109／2610。</li></ul>'
body+='<h2>结论适用于什么范围</h2><p>本次只比较 GAPD 第三阶段与它加 decoder 的版本，不证明优于原 AOFL。几何模块初始化种子仍固定 30007，decoder 固定 40007；变化的是原模型和训练随机性，因此不是所有新增模块初始化的全面验证。仍使用固定 MELD 数据划分，没有衡量更换测试样本或数据集带来的不确定性。原代码注意力对补齐位置的处理保持不变，批次分组影响仍存在。</p><p>decoder 约束 g、h 合在一起能恢复 x，不保证共享与特定信息被理想分开，也不直接保证流形处理后的信息保留。重构误差下降不能单独证明保留了更多情感信息。</p>'
body+='<h2>怎么向学长汇报</h2><p>可以直接使用前面的逐种子表和平均差值区间，说明结果是否支持平均改善，不把“平均正数”说成“每次都有效”。接下来应根据这批固定实验决定是否保留该方向；若目标是论文级对照，再补齐原 AOFL 的同种子结果及必要的机制分析，而不是继续按测试表现更换权重。</p><p>详细数据见同目录 gapd_decoder_paired_20261113/six_seed_results.json；逐轮 CSV、最佳和最终模型按种子分别保存。原三种子报告与所有已有结果均保留。</p>'
style='body{max-width:1200px;margin:36px auto;padding:0 24px;font:16px/1.85 "Microsoft YaHei",sans-serif;background:#f8fafc;color:#26364a}h1{font-size:28px;color:#214b70}h2{font-size:21px;margin-top:32px;border-bottom:1px solid #cbd8e5;color:#285f85}.muted{color:#64748b}.callout{background:#eaf2f8;border-left:4px solid #387cac;padding:16px}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:10px 12px;border:1px solid #d9e2eb;text-align:left}th{background:#e7eff6}tbody tr:nth-child(even){background:#f3f7fa}.scroll{overflow-x:auto}b{color:#1f5d84}'
report=BASE.parent/'GAPD_Decoder_六种子稳定性验证.html'
report.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GAPD decoder 六种子验证</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
print(json.dumps({'pairs':pairs,'groups':groups,'statistics':stats},indent=2));print(report)
