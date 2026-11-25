"""Second prototype: original OPR/AAC, per-utterance subspace CSR. Seed 7."""
from pathlib import Path
import argparse
import os
import runpy
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.argv = [str(root/'Train/TrainMultiEMO.py'), '--dataset', 'MELD',
                '--seed', '7', '--csr_mode', 'subspace', '--subspace_rank', '4',
                '--subspace_margin', '0.1', '--num_epochs', str(args.epochs)]
    runpy.run_path(sys.argv[0], run_name='__main__')
