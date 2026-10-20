"""Simulate the experiments in parallel -> results/<name>.npz.

    python simulate.py                    # all experiments
    python simulate.py system1 system3    # some of them
"""
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')

from core.experiments import EXPERIMENTS, run  # noqa: E402


def simulate(name):
    start = time.time()
    run(name)
    return f'{name}: {time.time() - start:.0f} s'


if __name__ == '__main__':
    names = sys.argv[1:] or list(EXPERIMENTS)
    unknown = sorted(set(names) - set(EXPERIMENTS))
    if unknown:
        sys.exit(f'unknown experiments: {", ".join(unknown)}')
    with Pool(len(names)) as pool:
        for message in pool.imap_unordered(simulate, names):
            print(message, flush=True)
