"""Figure 3: alarm-rate distributions, Direct (left) and DAB (right), K_max ln a = 0 and 4."""
import numpy as np
from matplotlib.patches import Patch

from core.config import ALPHA, DEFAULT_KMAX_LN_A, KMAX_LN_A
from core.experiments import run
from plotting.style import ALARM_BG, DAB, DIRECT, WIDTH, legend_below, plt, save

ROWS = (0.0, DEFAULT_KMAX_LN_A)
#           experiment, title, top of the y-axis (%), bin width (%)
SYSTEMS = (('system1', 'System1 (valid)', 10, 0.5),
           ('system2', 'System2 (valid)', 10, 0.5),
           ('system2_input', 'System2 + input (invalid)', 100, 4),
           ('system3', 'System3 (invalid)', 100, 4))


def panel(ax, alarm_rate, title, top, width):
    """Mirrored histograms (% of the initial states) of the rows of alarm_rate: Direct, DAB."""
    edges = np.arange(0, top + width, width)
    gap, zero_height = 0.25 * width, 0.9 * width
    ax.axhspan(100 * ALPHA, top, color=ALARM_BG, zorder=0)
    ax.axhspan(-gap - zero_height - 0.1 * width, -0.5 * gap, color='#f6f6f6', zorder=0)
    for p, color, side in ((alarm_rate[0], DIRECT, -1), (alarm_rate[1], DAB, 1)):
        p = 100 * p
        counts, _ = np.histogram(np.minimum(p[p > 0], top - 1e-9), bins=edges)
        ax.barh(edges[:-1], side * 100 * counts / len(p), height=width, align='edge', color=color,
                edgecolor='white', linewidth=0.3, zorder=3)
        zero = np.mean(p == 0)
        if zero > 0:
            ax.barh(-gap - zero_height, side * 100 * zero, height=zero_height, align='edge', color=color,
                    edgecolor='white', linewidth=0.3, zorder=3)
    ax.axhline(100 * ALPHA, color='#333333', lw=0.8, ls='--', zorder=4)
    ax.axvline(0, color='#888888', lw=0.5, zorder=4)
    ax.text(0.03, 100 * ALPHA, r'$\alpha$', transform=ax.get_yaxis_transform(), fontsize=8, va='bottom', ha='left', zorder=5)
    ax.axhline(-0.5 * gap, color='#bbbbbb', lw=0.5, zorder=4)
    ax.set_xlim(-105, 105)
    ax.set_xticks([-100, 0, 100])
    ax.set_xticklabels(['100\nDirect', '0', '100\nDAB'])
    ax.set_ylim(-gap - zero_height - 0.1 * width, top)
    ticks = list(np.linspace(0, top, 6)[1:])
    ax.set_yticks([-gap - zero_height / 2] + ticks)
    ax.set_yticklabels(['0'] + [f'{t:g}' for t in ticks])
    ax.set_title(title, fontsize=8)


fig, axes = plt.subplots(len(ROWS), len(SYSTEMS), figsize=(WIDTH, 3.8),
                         gridspec_kw=dict(wspace=0.32, hspace=0.5, left=0.07, right=0.97, top=0.93, bottom=0.13))
letters = iter('abcdefgh')
for row, c in zip(axes, ROWS):
    j = list(KMAX_LN_A).index(c)
    for ax, (experiment, title, top, width) in zip(row, SYSTEMS):
        panel(ax, run(experiment)['alarm_rate'][:, j], f'({next(letters)}) {title}', top, width)
    row[0].set_ylabel(r'alarm rate $\hat p_i$ (%)')
    row[-1].text(1.04, 0.5, f'$K_{{\\max}}\\ln a={c:g}$' + (' (default)' if c == DEFAULT_KMAX_LN_A else ' ($a=1$)'),
                 transform=row[-1].transAxes, rotation=270, va='center', ha='left', fontsize=8)
legend_below(fig, [Patch(color=DIRECT, label=r'Direct ($B_k/B_0$)'), Patch(color=DAB, label='DAB'),
                   Patch(color=ALARM_BG, label=r'$\hat p_i>\alpha$')], columnspacing=2.0)
save(fig, 'fig_phat_density')
