"""Fixed-design exploration: seed 42, QR and free, official MELD split."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    group = root/'save_model_MELD'/('seed42_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    commands = [(mode, [sys.executable, '-u', str(root/'Train/TrainMultiEMO.py'),
                        '--dataset', 'MELD', '--seed', '42', '--refinement', mode,
                        '--subspace_rank', '4', '--num_epochs', '100',
                        '--output_dir', str(group/mode)]) for mode in ('qr', 'free')]
    if args.dry_run:
        for mode, cmd in commands:
            print(mode+': '+subprocess.list2cmdline(cmd))
        print('No training started.')
        return
    group.mkdir(parents=True, exist_ok=False)
    metadata = dict(seed=42, rank=4, epochs=100, variants=['qr', 'free'],
                    selection='validation weighted F1', status='running')
    def save_status():
        (group/'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    save_status()
    rows = []
    for mode, cmd in commands:
        print(f'Starting {mode}, seed 42. Output: {group/mode}', flush=True)
        with (group/f'{mode}_console.txt').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, cwd=root, env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace')
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                metadata['status'] = 'interrupted'
                save_status()
                raise
        if code:
            metadata.update(status='failed', failed_variant=mode)
            save_status()
            raise RuntimeError(f'{mode} failed; inspect {mode}_console.txt')
        config = json.loads((group/mode/'config.json').read_text(encoding='utf-8'))
        assert config['refinement'] == mode and config['torch_initial_seed'] == 42
        assert config['split_strategy'].startswith('MELD official dev')
        best = json.loads((group/mode/'best_metrics.json').read_text(encoding='utf-8'))
        rows.append(dict(variant=mode, epoch=best['epoch'], valid_f1=best['valid_f1'],
                         test_f1=best['test_f1'], test_acc=best['test_acc']))
        with (group/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata['status'] = 'complete'
    save_status()
    print(f'Both variants completed. Comparison: {group/"comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
