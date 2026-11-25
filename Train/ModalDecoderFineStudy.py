from pathlib import Path
import sys,json,subprocess,csv,html,time
ROOT=Path(r'D:/毕设/多模态情感识别/AOFL-main')
sys.path.insert(0,str(ROOT/'Train'))
BASE=Path(r'D:/codex/shiyan/output/modal_decoder_fine_20261117')
CASES={'joint_03':(.1,.3,.3),'joint_05':(.1,.5,.5),'audio_2':(.1,2.,.1),'visual_2':(.1,.1,2.)}
def report():
    import ModalDecoderStudy as m
    rows=[]
    for n,w in CASES.items():
        p=BASE/n
        if not (p/'summary.json').exists():continue
        allrows=list(csv.DictReader((p/'metrics.csv').open(encoding='utf-8-sig')))
        if len(allrows)!=30:continue
        r=m.best(p);rows.append({'name':n,'weights':w,'epoch':int(r['epoch']),'valid_f1':float(r['valid_f1']),'test_f1':float(r['test_f1'])})
    (BASE/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    old=json.loads(Path(r'D:/codex/shiyan/output/modal_decoder_study_20261117/analysis.json').read_text())['runs']
    refs=[{'name':'原AOFL（无decoder）','weights':'—','epoch':13,'valid_f1':66.6826,'test_f1':65.7067},{'name':'统一权重0.01','weights':[.01]*3,'epoch':13,'valid_f1':67.1792,'test_f1':65.4970},{'name':'统一权重0.1','weights':[.1]*3,'epoch':13,'valid_f1':67.0064,'test_f1':65.6009}]+[{k:r[k] for k in ['name','weights','epoch','valid_f1','test_f1']} for r in old if r['alignment']==0]
    best=max(refs[1:]+rows,key=lambda r:r['valid_f1'])
    body='<h1>原AOFL：分模态decoder权重细化实验</h1><p>种子7，官方MELD划分，每组30轮。原CEN、ARE、OPR保留，没有GAPD，没有额外alignment。重构位置保持不变：[g,h]→decoder→重构解耦前x，目标x停止梯度。新增损失为三个模态加权重构误差的平均值。</p>'
    body+='<p>候选预先固定，按验证F1选择轮次；测试F1不参与选择。权重顺序是文本、音频、视觉。这是单种子探索，不足以证明稳定提升。</p><table><tr><th>配置</th><th>重构权重</th><th>选中轮次</th><th>验证F1</th><th>对应测试F1</th><th>测试相比原AOFL（百分点）</th></tr>'
    for r in refs+rows:body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['name'],r['weights'],r['epoch'],f"{r['valid_f1']:.4f}",f"{r['test_f1']:.4f}",f"{r['test_f1']-65.7067:+.4f}"])+'</tr>'
    body+='</table><p>已完成新增配置：'+str(len(rows))+'/4。上述decoder候选中验证F1最高的是 '+html.escape(best['name'])+'。验证F1更高不等于测试表现更好，也不能按本表测试分数反向选择权重。</p><p>每个配置保留逐轮日志、最佳及最后模型。后续若验证结果有希望，应固定配置后补充相同种子的配对实验，比较多个种子的平均差异。</p>'
    (BASE/'分模态权重细化实验报告.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:35px auto;padding:20px;font:16px/1.8 Microsoft YaHei;color:#26364a;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cdd8e2;padding:10px}th{background:#e6eef5}h1{color:#245477}</style>'+body,encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False),flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:
        import torch,ModalDecoderStudy as m
        torch.set_num_threads(2);m.BASE=BASE;m.run(sys.argv[1],CASES[sys.argv[1]],0)
    else:
        BASE.mkdir(exist_ok=True,parents=True)
        plan=BASE/'plan.json'
        if not plan.exists():plan.write_text(json.dumps({'model':'original AOFL + decoder','seed':7,'epochs':30,'cases':CASES,'alignment':0,'selection':'validation weighted F1; earliest tie','split':'official MELD','budget':'4 runs, no automatic extension'},indent=2),encoding='utf-8')
        for name in CASES:
            if (BASE/name/'summary.json').exists():continue
            assert not (BASE/name).exists(), 'Incomplete run exists; do not silently overwrite'
            source="from pathlib import Path; p=Path("+repr(str(Path(__file__)))+"); exec(compile(p.read_text(encoding='utf-8'),str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})"
            with (BASE/(name+'.log')).open('w',encoding='utf-8') as log:
                p=subprocess.run([sys.executable,'-u','-c',source,name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            assert p.returncode==0,(name,p.returncode)
            report()
        report()
