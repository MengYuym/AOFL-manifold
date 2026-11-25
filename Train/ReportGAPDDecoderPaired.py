"""Audit completed paired runs and generate an offline Chinese report."""
from pathlib import Path
import sys,json,csv,statistics,html,runpy,gc,math
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunGAPDDecoderPaired import BASE,ROOT,install_decoder,install_gapd
import torch
torch.set_num_threads(2)
def readrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def loadjson(path):return json.loads(path.read_text(encoding='utf-8'))
def table(headers,data):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</table></div>'
records=[];audits=[];reference=None
for seed in [7,42,2025]:
    for decoder in [False,True]:
        if seed==7:
            path=Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113/gapd') if decoder else Path(r'D:/codex/shiyan/output/gapd_stage3_seed7_20261112')
        else:path=BASE/f'seed_{seed}'/('with_decoder' if decoder else 'without_decoder')
        config=loadjson(path/'config.json');rows=readrows(path/'metrics.csv');summary=loadjson(path/'summary.json')
        assert len(rows)==30 and [int(r['epoch']) for r in rows]==list(range(1,31))
        assert all(math.isfinite(float(r[k])) for r in rows for k in ['train_loss','valid_loss','test_loss','valid_f1','test_f1'])
        assert config['torch_initial_seed']==seed
        if reference is None:reference=config
        for k in ['dataset','batch_size','num_epochs','learning_rate','weight_decay','num_layers','model_dim','num_heads','hidden_dim','dropout_rate','dropout_rec','CE_loss_param','multi_attn_flag','train_dialogue_ids','valid_dialogue_ids','test_dialogue_ids','selection_metric','tie_policy']:
            assert config[k]==reference[k],(seed,decoder,k)
        best=max(rows,key=lambda r:float(r['valid_f1']))
        assert int(best['epoch'])==summary['best_epoch']
        item={'seed':seed,'decoder':decoder,'best_epoch':int(best['epoch']),'valid_f1':float(best['valid_f1']),'test_f1':float(best['test_f1']),'test_acc':float(best['test_acc']),'minutes':summary['total_training_minutes'],'path':str(path)}
        if decoder:
            recon=readrows(path/'reconstruction_metrics.csv');assert len(recon)==90
            assert all(int(r['utterances'])=={'Train':9989,'Valid':1109,'Test':2610}[r['split']] for r in recon)
            item['selected_reconstruction_valid']=next(r for r in recon if r['split']=='Valid' and int(r['epoch'])==item['best_epoch'])
        if seed!=7:
            scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='pair_checkpoint_audit');cls=scope['AOFL']
            if decoder:install_decoder(cls,'gapd')
            else:install_gapd(cls)
            t=cls.__new__(cls)
            for k,v in config.items():setattr(t,k,v)
            t.device=torch.device('cpu');scope['set_seed'](seed);t.get_model()
            for filename in ['best_model.pth','final_model.pth']:
                checkpoint=torch.load(path/filename,map_location='cpu',weights_only=False)
                t.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
                assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values() if v.is_floating_point())
                if filename=='best_model.pth':assert checkpoint['epoch']==item['best_epoch']
                del checkpoint
            del t;gc.collect()
        records.append(item);audits.append({'seed':seed,'decoder':decoder,'epochs_complete':True,'split_and_settings_match':True,'checkpoint_audit':'strict finite best+final' if seed!=7 else 'reuse previous verified audit'})
pairs=[]
for seed in [7,42,2025]:
    a=next(r for r in records if r['seed']==seed and not r['decoder']);b=next(r for r in records if r['seed']==seed and r['decoder'])
    pairs.append({'seed':seed,'valid_delta':b['valid_f1']-a['valid_f1'],'test_delta':b['test_f1']-a['test_f1']})
stats={}
for decoder in [False,True]:
    rs=[r for r in records if r['decoder']==decoder]
    stats[str(decoder)]={k:{'mean':statistics.mean(r[k] for r in rs),'sample_sd':statistics.stdev(r[k] for r in rs)} for k in ['valid_f1','test_f1']}
delta_mean=statistics.mean(r['test_delta'] for r in pairs);new_mean=statistics.mean(r['test_delta'] for r in pairs if r['seed']!=7)
wins=sum(r['test_delta']>0 for r in pairs)
result={'runs':records,'pairs':pairs,'statistics':stats,'test_delta_mean':delta_mean,'test_delta_sample_sd':statistics.stdev(r['test_delta'] for r in pairs),'new_seeds_test_delta_mean':new_mean,'audits':audits}
(BASE/'paired_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
body='<h1>GAPD＋decoder：多种子配对验证</h1><p class="muted">MELD 官方划分 · 每次 30 轮 · 种子 7、42、2025</p>'
body+=f'<div class="callout">三个种子中，加入 decoder 后对应测试 F1 提高了 {wins} 次；平均变化为 <b>{delta_mean:+.4f}</b> 个百分点。新增种子 42、2025 的平均变化为 <b>{new_mean:+.4f}</b> 个百分点。</div>'
body+='<h2>本次在验证什么</h2><p>此前种子 7 中，GAPD 加 decoder 后测试 F1 提高。此次不修改模型、重构权重或训练设置，补充种子 42 和 2025，每个种子各运行加 decoder 与不加 decoder 两次。已有种子 7 的结果复用，总计比较三组配对结果。这不是三次都重新运行原 AOFL，而是在同一个 GAPD 第三阶段上检查 decoder 的作用。</p>'
body+='<h2>逐种子结果</h2><p>每次独立用验证 F1 选出最佳轮次，再读取该轮测试 F1。加与不加 decoder 可以选中不同轮次。变化值 = 加 decoder − 不加 decoder，单位为百分点。</p>'
body+=table(['种子','版本','最佳轮次','验证 F1（%）','对应测试 F1（%）'],[[r['seed'],'加 decoder' if r['decoder'] else '不加 decoder',r['best_epoch'],f"{r['valid_f1']:.4f}",f"{r['test_f1']:.4f}"] for r in records])
body+=table(['种子','验证 F1 变化','对应测试 F1 变化'],[[r['seed'],f"{r['valid_delta']:+.4f}",f"{r['test_delta']:+.4f}"] for r in pairs])
body+='<h2>平均表现和波动</h2><p>平均值表示三个种子的整体水平；样本标准差表示不同种子结果的波动，不是误差范围或显著性证明。</p>'
body+=table(['版本','验证 F1：平均 ± 标准差','测试 F1：平均 ± 标准差'],[['加 decoder' if d else '不加 decoder',*[f"{stats[str(d)][k]['mean']:.4f} ± {stats[str(d)][k]['sample_sd']:.4f}" for k in ['valid_f1','test_f1']]] for d in [False,True]])
body+='<h2>怎么理解结论</h2>'
if wins==3:body+='<p>本次三个种子的变化方向一致，是 decoder 对这一 GAPD 配置有帮助的初步证据。但种子 7 是提出继续验证的依据，新增两组才是这次的后续验证，不能把三组都描述成事先独立的确认实验；样本仍少，暂不宣称稳定或显著提升。</p>'
elif new_mean<=0:body+='<p>新增种子平均没有延续种子 7 的正向改善，因此目前不能认为 decoder 带来了可重复的分类提升。应如实汇报结果，不根据测试表现继续挑选权重或种子。</p>'
else:body+='<p>新增种子平均有正向变化，但不同种子的变化方向并非全部一致。目前可视为有一定积极信号，尚不足以证明稳定提升；应同时展示每组结果和波动。</p>'
body+='<p>本次没有补齐“原 AOFL”的全部相同种子实验，因此无论本表是否改善，都不能据此宣布优于原 AOFL。验证 F1 可能提高而测试 F1 不提高，这是两个不同数据集上的结果，不应只挑其中较好的一项汇报。</p>'
body+='<h2>保持不变的条件与已做检查</h2><ul><li>官方训练、验证、测试划分和训练设置一致；每次均完成 30 轮，不根据测试分数中途停止或调参。</li><li>重构仍为每模态拼接门控前 g、h，重构断开目标梯度的 x，权重固定 0.1，补齐位置不计入。</li><li>同一种子下，新增 decoder 不改变原 GAPD 初始权重和 CPU 随机数状态；不同种子的原模型初始化确实不同。</li><li>GAPD 新增几何模块初始化种子仍固定 30007，decoder 固定 40007，沿用先前实验。这里验证的是原模型和训练随机性的变化，不覆盖所有新增模块初始化的变化。</li><li>逐轮指标完整且有限，新增四次的最佳和最终模型均严格加载成功，权重没有 NaN 或无穷值。</li></ul>'
recondata=[]
for r in records:
    if not r['decoder']:continue
    z=r['selected_reconstruction_valid']
    recondata.append([r['seed'],r['best_epoch'],*[f"{100*float(z[m+'_relative_mse']):.2f}%" for m in ['text','audio','visual']]])
body+='<h2>辅助查看：选中模型的重构误差</h2><p>相对重构误差 = 总重构平方误差 ÷ 目标特征总平方值。它不是分类错误率；目标 x 随训练变化，因此不能只凭误差下降就认定保留了更多情感信息。</p>'+table(['种子','选中轮次','文本相对误差','音频相对误差','视觉相对误差'],recondata)
body+='<h2>下一步建议</h2><p>先把本表交给学长，确认门控前重构的位置是否符合他的目的。如果要证明改造优于原 AOFL，需要在相同种子、相同预算下补齐原 AOFL 对照；如果要解释信息保留机制，则需要另行设计门控前后特征的分析，不能仅用这次分类分数作证明。原代码注意力对补齐位置的处理仍沿用旧设置，批次分组影响尚未消除。</p><p>完整配置、配对条件检查、逐轮 CSV、模型与本报告的数据位于 gapd_decoder_paired_20261113 文件夹。种子 7 保留在原实验文件夹，未覆盖。</p>'
style='body{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.85 "Microsoft YaHei",sans-serif;background:#f8fafc;color:#26364a}h1{font-size:28px;color:#214b70}h2{font-size:21px;margin-top:32px;border-bottom:1px solid #cbd8e5;color:#285f85}.muted{color:#64748b}.callout{background:#eaf2f8;border-left:4px solid #387cac;padding:16px}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:10px 12px;border:1px solid #d9e2eb;text-align:left}th{background:#e7eff6}tr:nth-child(even){background:#f3f7fa}.scroll{overflow-x:auto}b{color:#1f5d84}'
report=BASE.parent/'GAPD_Decoder_多种子配对验证.html'
report.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GAPD Decoder 配对验证</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
print(json.dumps({'pairs':pairs,'stats':stats,'new_mean':new_mean},indent=2));print(report)
