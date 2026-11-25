"""Run from PyCharm or Python. Uses official MELD splits; does not alter model/losses."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 42, 3407, 12345, 98765]  # Prespecified additional seeds; not claimed to be the paper's seeds.


def summarize(rows):
    result = {}
    for key in ('test_f1', 'test_acc'):
        values = [r[key] for r in rows]
        result[key] = dict(mean=statistics.mean(values),
                           sample_std=statistics.stdev(values) if len(values) > 1 else None,
                           count=len(values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--dry-run', action='store_true', help='show commands without training')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    training = ROOT / 'Train/TrainMultiEMO.py'
    manifest = ROOT / 'Data/MELD/official_split_ids.json'
    split = json.loads(manifest.read_text(encoding='utf-8'))
    assert len(split['validation_ids']) == 114 and len(split['test_ids']) == 280
    group = ROOT / 'save_model_MELD' / ('multiseed_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    commands = [(seed, group / f'seed_{seed}',
                 [sys.executable, '-u', str(training), '--dataset', 'MELD',
                  '--seed', str(seed), '--output_dir', str(group / f'seed_{seed}')])
                for seed in args.seeds]
    if args.dry_run:
        for seed, _, cmd in commands:
            print(f'Seed {seed}: {subprocess.list2cmdline(cmd)}')
        print('Dry run only. No training started.')
        return
    group.mkdir(parents=True, exist_ok=False)
    sources = [training, manifest, ROOT/'Model/AOFL_Model.py',
               ROOT/'Model/Multi_Disentangle.py', ROOT/'Dataset/MELDDataset.py']
    (group/'experiment.json').write_text(json.dumps(dict(
        seeds=args.seeds, seed_note='Preliminary seed list; paper seed list not confirmed',
        selection='validation weighted F1; corresponding test metrics',
        std_definition='sample standard deviation, denominator n-1',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    ), indent=2), encoding='utf-8')
    rows = []
    for seed, directory, cmd in commands:
        print(f'\nStarting seed {seed}. Results: {directory}', flush=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        with (group/f'seed_{seed}_console.txt').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
        if code != 0:
            raise RuntimeError(f'Seed {seed} failed (exit {code}); check its console log. No complete average reported.')
        config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        assert config['torch_initial_seed'] == seed
        assert config['split_strategy'].startswith('MELD official dev')
        assert len(config['valid_dialogue_ids']) == 114
        best = json.loads((directory/'best_metrics.json').read_text(encoding='utf-8'))
        rows.append(dict(seed=seed, best_epoch=best['epoch'], valid_f1=best['valid_f1'],
                         test_f1=best['test_f1'], test_acc=best['test_acc'], directory=str(directory)))
        with (group/'seed_results.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        stats = summarize(rows)
        (group/'aggregate.json').write_text(json.dumps(dict(
            complete=len(rows)==len(args.seeds), planned_seeds=args.seeds,
            completed_seeds=[r['seed'] for r in rows], statistics=stats), indent=2), encoding='utf-8')
    lines = ['# MELD 官方划分：多种子结果', '',
             '种子为预先指定的实验列表，尚未确认与论文一致。每个种子均使用验证集选模。', '',
             '| 种子 | 最佳轮 | 验证 F1 | 测试 F1 | 测试准确率 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['best_epoch']} | {r['valid_f1']:.4f}% | {r['test_f1']:.4f}% | {r['test_acc']:.4f}% |")
    lines += ['', '标准差采用样本标准差（n−1），单位为百分点。']
    for key, label in [('test_f1','测试加权 F1'), ('test_acc','测试准确率')]:
        item = stats[key]
        deviation = f"{item['sample_std']:.4f}" if item['sample_std'] is not None else '不可计算（仅一次）'
        lines += ['', f"{label}：平均 {item['mean']:.4f}%；标准差 {deviation}。"]
    (group/'多种子结果汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'\nAll seeds completed. Summary: {group}', flush=True)


if __name__ == '__main__':
    main()
