"""Initial QR subspace experiment: one seed, official MELD split, 100 epochs."""
from pathlib import Path
import os
import runpy
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.argv = [str(root/'Train/TrainMultiEMO.py'), '--dataset', 'MELD',
                '--seed', '7', '--refinement', 'qr', '--subspace_rank', '4']
    runpy.run_path(sys.argv[0], run_name='__main__')
