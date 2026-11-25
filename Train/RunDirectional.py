"""Exploratory directional OPR, original vector CSR; seed 7, 30 epochs."""
from pathlib import Path
import os
import runpy
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.argv = [str(root/'Train/TrainMultiEMO.py'), '--dataset', 'MELD',
                '--seed', '7', '--refinement', 'directional', '--csr_mode', 'vector',
                '--num_epochs', '30']
    runpy.run_path(sys.argv[0], run_name='__main__')
