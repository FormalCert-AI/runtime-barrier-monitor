"""Figure 7: sensitivity to K_max, DAB memory gamma and cap U, varied one at a time."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core.config import ALPHA, CAP, GAMMA, HORIZONS, K_MAX, SENS_CAP, SENS_GAMMA
from core.experiments import run
from core.report import rates
from core.simulate import MONITORS
from plotting import violin
from plotting.style import DAB, DAB_FILL, DIRECT, DIRECT_FILL, WIDTH, legend_below, plt, save

plt.rcParams['axes.titlepad'] = 4
SYSTEMS = (('system1', 'System1 (valid)', 10), ('system2', 'System2 (valid)', 10),
           ('system2_input', 'System2 + input (invalid)', 100), ('system3', 'System3 (invalid)', 100))
COLUMNS = (('horizon', HORIZONS, r'horizon $K_{\max}$', K_MAX),
           ('gamma', SENS_GAMMA, r'DAB memory $\gamma$', GAMMA),
           ('cap', SENS_CAP, r'DAB cap $U$', CAP))
STYLE = {'direct': (DIRECT_FILL, DIRECT, 'left', '^'), 'dab': (DAB_FILL, DAB, 'right', 'o')}
HALF_WIDTH, OFFSET = 0.42, 0.16


def column_rates(results, monitor, column):
    """p_hat_i, one row per value on the x-axis; Direct has no gamma or U."""
    if column == 'horizon':
        return results['horizon_rate'][list(MONITORS).index(monitor)]
    if monitor == 'direct':
        return np.repeat(rates(results, 'direct')[None], len(results[f'sens_{column}']), 0)
    return results[f'{column}_rate']


fig, axes = plt.subplots(len(SYSTEMS), len(COLUMNS), figsize=(WIDTH, 4.7),
                         gridspec_kw=dict(height_ratios=[0.65, 0.65, 1, 1], wspace=0.2, hspace=0.4, left=0.07,
                                          right=0.99, top=0.96, bottom=0.11))
letters = iter('abcdefghijklmnop')
for row, (name, title, top) in zip(axes, SYSTEMS):
    results = run(name)
    for ax, (column, values, xlabel, default) in zip(row, COLUMNS):
        x = np.arange(len(values))
        ax.axvline(list(values).index(default), color='#c8c8c8', lw=0.8, zorder=1)
        for monitor, (fill, line, side, marker) in STYLE.items():
            p = 100 * column_rates(results, monitor, column)
            for xi, pi in zip(x, p):
                violin.alarm_rates(ax, xi, pi, fill, HALF_WIDTH, side, top)
            dx = OFFSET if side == 'right' else -OFFSET
            ax.plot(x + dx, [100 * np.mean(pi > 100 * ALPHA) for pi in p], color=line, lw=1.0, ls='-', marker=marker,
                    ms=2.8, zorder=4)
        violin.alarm_axis(ax, label_alpha=column == 'horizon', top=top)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:g}' for v in values], fontsize=6.5)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_title(f'({next(letters)}) {title}', fontsize=8)
        if column != 'horizon':
            ax.tick_params(labelleft=False)
        if ax in axes[-1]:
            ax.set_xlabel(xlabel)
for ax in axes.flat:
    ax.minorticks_off()
fig.supylabel(r'alarm rate $\hat p_i$ and $\rho_\alpha$ (%)', x=0.008, fontsize=8)

legend_below(fig, [Patch(color=DIRECT_FILL, label=r'Direct: distribution of $\hat p_i$ (top: $\hat p^{\max}$)'),
                   Line2D([], [], color=DIRECT, lw=1.0, marker='^', ms=2.8, label=r'Direct: $\rho_\alpha$'),
                   Patch(color=DAB_FILL, label=r'DAB: distribution of $\hat p_i$'),
                   Line2D([], [], color=DAB, lw=1.0, marker='o', ms=2.8, label=r'DAB: $\rho_\alpha$'),
                   Line2D([], [], color='#333333', lw=0.8, ls='--', label=r'$\alpha$'),
                   Line2D([], [], color='#c8c8c8', lw=0.8, label='default')], columnspacing=1.2)
save(fig, 'fig_sensitivity')
