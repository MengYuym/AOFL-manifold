"""Validate finished checkpoints and report the predeclared paired experiment."""
from pathlib import Path
import sys,json,csv,runpy,html
sys.path.insert(0,str(Path(__file__).resolve().parent))
from RunDecoderComparison import install,ROOT
import torch
BASE=Path(r'D:/codex/shiyan/output/decoder_comparison_seed7_20261113')
torch.set_num_threads(2)
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def best(p):return max(rows(p/'metrics.csv')[:30],key=lambda r:float(r['valid_f1']))
records={
 'AOFL':best(ROOT/'save_model_MELD/multiseed_20261106_144852_867353/seed_7'),
 'GAPD第三阶段':best(Path(r'D:/codex/shiyan/output/gapd_stage3_seed7_20261112'))}
diagnostics={}
for variant,label in [('baseline','AOFL＋decoder'),('gapd','GAPD第三阶段＋decoder')]:
    path=BASE/variant;metrics=rows(path/'metrics.csv');recon=rows(path/'reconstruction_metrics.csv')
    assert len(metrics)==30 and len(recon)==90
    assert all(int(r['utterances'])=={'Train':9989,'Valid':1109,'Test':2610}[r['split']] for r in recon)
    records[label]=best(path)
    scope=runpy.run_path(str(ROOT/'Train/TrainMultiEMO.py'),run_name='inspect_'+variant)
    cls=scope['AOFL'];install(cls,variant);scope['set_seed'](7)
    t=cls('MELD',64,30,.0001,.0001,4,256,4,1024,.3,0,1,True,torch.device('cuda'))
    checkpoint=torch.load(path/'best_model.pth',map_location='cpu',weights_only=False)
    t.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    assert checkpoint['epoch']==int(records[label]['epoch'])
    assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values() if v.is_floating_point())
    final=torch.load(path/'final_model.pth',map_location='cpu',weights_only=False)
    t.model.load_state_dict(final['model_state_dict'],strict=True)
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values() if v.is_floating_point())
    t.model.load_state_dict(checkpoint['model_state_dict'],strict=True);t.model.eval()
    batch=next(iter(t.valid_dataloader));args=[v.to(t.device) for v in batch];args[-1]=args[-1].reshape(-1)
    with torch.no_grad():
        output=t.model(*args)[0].clone();mse=float(t.reconstruction_loss)
        for p in t.model.reconstruction.parameters():p.zero_()
        output_without_decoder=t.model(*args)[0]
    assert torch.equal(output,output_without_decoder)
    diagnostics[variant]={'epochs':len(metrics),'reconstruction_rows':len(recon),'strict_best_and_final_load':True,'finite_weights':True,'selected_epoch':checkpoint['epoch'],'decoder_zeroing_leaves_logits_identical':True,'one_validation_batch_mse':mse,'training_minutes':json.loads((path/'summary.json').read_text(encoding='utf-8'))['total_training_minutes']}
    t.writer.close();del t,checkpoint,final;torch.cuda.empty_cache()
(BASE/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')

def table(headers,data):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table></div>'
comparison=[]
for label,r in records.items():comparison.append([label,r['epoch'],r['valid_f1'],r['test_f1'],r['test_acc']])
sections=[]
for variant,label,parent in [('baseline','AOFL＋decoder','AOFL'),('gapd','GAPD第三阶段＋decoder','GAPD第三阶段')]:
    r=records[label];old=records[parent]
    dv=float(r['valid_f1'])-float(old['valid_f1']);dt=float(r['test_f1'])-float(old['test_f1'])
    recon=rows(BASE/variant/'reconstruction_metrics.csv')
    data=[]
    for epoch in [1,int(r['epoch']),30]:
        z=next(z for z in recon if z['split']=='Valid' and int(z['epoch'])==epoch)
        for m,name in [('text','文本'),('audio','音频'),('visual','视觉')]:
            data.append([epoch,name,f"{float(z[m+'_mse']):.6f}",f"{float(z[m+'_target_energy']):.6f}",f"{100*float(z[m+'_relative_mse']):.2f}%"])
    sections.append(f'<h2>{label}</h2><p>相对对应的无 decoder 版本：验证 F1 变化 <b>{dv:+.4f}</b> 个百分点，对应测试 F1 变化 <b>{dt:+.4f}</b> 个百分点。用时 {diagnostics[variant]["training_minutes"]:.2f} 分钟。</p>'+table(['轮次','模态','验证集重构 MSE','目标特征平均平方值','相对误差'],data))
body='''<h1>AOFL 与 GAPD：加入重构 decoder 的两版实验</h1>
<p class="muted">2026-11-13 · MELD 官方划分 · 随机种子 7 · 每版 30 轮 · 探索性结果</p>
<h2>这次具体做了什么</h2><p>分别在原 AOFL 和 GAPD 第三阶段上加入完全相同的重构分支。每个模态把解耦后、OPR 或流形门控处理前的 g 与 h 拼接，用 decoder 重构同模态解耦前的 256 维 x。目标是检查：要求 g、h 一起保留更多可重构的信息，是否有利于情感分类。</p>
<div class="formula">同模态 [g, h]（512 维） → 全连接层（256 维） → ReLU → 全连接层（256 维） → x̂</div>
<p>三个模态分别有一个 decoder，互不共用参数，共增加 591,360 个参数。每个模态先对 256 个坐标求平方误差均值，再对有效话语与三个模态取平均。总损失 = 原有损失 + 0.1 × 重构损失。从第 1 轮起加入重构损失；原有第 15 轮损失切换规则保留。</p>
<p>选择 GAPD 第三阶段是因为它保留了特定子空间以外的 h 分量，适合作为结构较简单、信息保留较完整的流形对照，并不是因为它已经证明性能最好。该版使用 QR 正交化和 Adam，不是 R03 的黎曼优化器，也不是完整实现导师所有设想。</p>
<h2>如何读结果</h2><p>验证 F1 用于挑选 30 轮中最好的模型；对应测试 F1 是选定那一轮后，在测试集上的结果。数值越高越好，单位为百分数。下面原有两行采用历史实验前 30 轮内的验证集最佳模型，后两行是本次新增实验；未按测试集挑轮次。</p>'''
body+=table(['模型','验证选中的轮次','验证 F1（%）','对应测试 F1（%）','对应测试准确率（%）'],comparison)
body+=''.join(sections)
body+='''<h2>重构误差能说明什么</h2><p>MSE 衡量重构的 x̂ 与 x 的差距，越小表示越容易从 g、h 恢复 x。相对误差 = 总重构平方误差 ÷ 目标 x 的总平方值；表中按百分数展示，用于辅助判断不同模态、不同轮次的误差尺度，不能当成准确率。“目标特征平均平方值”记录 x 的数值尺度。</p>
<p>x 在每一步计算重构损失时被断开目标分支的梯度，避免通过直接移动目标降低误差；但上游网络仍会在训练过程中更新，因此各轮的 x 不是完全固定的数据。MSE 降低并不自动证明原始信息全部保留，也不保证保留的是对情感分类有用的信息。</p>
<h2>已经完成的检查</h2><ul><li>两版 decoder 初始化完全相同，参数均已加入优化器；真实小批次训练能更新 decoder。</li><li>仅使用重构损失反向传播时，共享编码器与三个特定编码器均有非零梯度，目标分支没有梯度。</li><li>每轮重构记录均为训练 9989、验证 1109、测试 2610 条话语，补齐位置没有计入。</li><li>两版均完成 30 轮；最佳和最终模型均能严格加载，权重未出现 NaN 或无穷值。</li><li>将 decoder 参数置零后，同一批次的分类得分完全不变，确认 decoder 是辅助训练分支，不直接参与最终分类输出。</li></ul>
<h2>结论的边界与下一步</h2><p>当前只有一个随机种子，不能据此宣布稳定改进。历史对照和本次均沿用同一官方划分、原有分类流程和验证选择规则；原代码注意力没有完整屏蔽补齐位置，因此结果仍可能受批次分组影响。未在本次修改这一点，以保留对照条件。</p>
<p>重构约束的是 g 与 h 合在一起的可恢复性，不保证两者各自包含理想的共享或特定信息；因为使用门控前的 g、h，也没有直接约束流形处理后的信息保留。如果要继续，先与学长确认这一重构位置符合预期，再决定是否对有希望的配置补充相同种子的配对实验，不按测试集反复尝试设置。</p>
<h2>文件位置</h2><p>本报告旁的 decoder_comparison_seed7_20261113 文件夹包含两版的逐轮分类指标、逐模态重构指标、最佳模型、最终模型、配置和检查结果。CSV 中 CE_only / combined 仍是原有损失阶段的名称，两阶段实际上都额外加了重构损失；新增设置见对应的 plan.json 与 effective_variant.json。</p>'''
style='body{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.85 "Microsoft YaHei",sans-serif;color:#243247;background:#f7f9fc}h1{font-size:28px;color:#163c65}h2{margin-top:32px;font-size:21px;color:#235a80;border-bottom:1px solid #cedbe7}p,li{max-width:1000px}.muted{color:#64748b}.formula{padding:18px;background:#eaf2fa;border-left:4px solid #3577aa}table{width:100%;border-collapse:collapse;background:white;margin:18px 0}th,td{padding:10px 13px;border:1px solid #d8e1ea;text-align:left}th{background:#e7eff7}tr:nth-child(even){background:#f4f7fa}.scroll{overflow-x:auto}b{color:#24618b}'
report=BASE.parent/'AOFL与GAPD_重构Decoder两版实验.html'
report.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>重构Decoder两版实验</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
(BASE/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2));print(report)
