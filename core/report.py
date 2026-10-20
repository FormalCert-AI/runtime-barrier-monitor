"""Metrics (in %) and plain-text tables."""
import numpy as np

from .config import ALPHA, DEFAULT_INDEX, KMAX_LN_A
from .experiments import run
from .simulate import MIXTURES, MONITORS


def rho(p):
    return 100 * np.mean(p > ALPHA)


def p_max(p):
    return 100 * p.max()


def rates(results, monitor, kmax_ln_a=KMAX_LN_A[DEFAULT_INDEX]):
    if monitor in MIXTURES:
        return results['mixture_rate'][list(MIXTURES).index(monitor)]
    return results['alarm_rate'][list(MONITORS).index(monitor), list(KMAX_LN_A).index(kmax_ln_a)]


def stay_rate(results):
    return 100 * (1 - results['n_exit'].sum() / (results['N'] * results['R']))


def pre_exit_alarm_rate(results, monitor, kmax_ln_a=KMAX_LN_A[DEFAULT_INDEX]):
    n_exit = results['n_exit'].sum()
    if n_exit == 0:
        return np.nan
    m, j = list(MONITORS).index(monitor), list(KMAX_LN_A).index(kmax_ln_a)
    return 100 * results['caught'][m, j] / n_exit


def table1(systems):
    rows = []
    for name, label in systems:
        results = run(name)
        for c in (0.0, KMAX_LN_A[DEFAULT_INDEX]):
            p = [rates(results, m, c) for m in MONITORS]
            rows.append([label, f'{c:g}', rho(p[0]), p_max(p[0]), rho(p[1]), p_max(p[1])])
    return table(['system', 'K_max ln a', 'Direct rho', 'Direct p_max', 'DAB rho', 'DAB p_max'], rows)


def table(header, rows, fmt='{:.2f}'):
    cells = [header] + [[v if isinstance(v, str) else fmt.format(v) for v in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return '\n'.join('  '.join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths)))
                     for row in cells)
