"""Figure 4: effect of a on DAB alarm rates and evidence for the invalid certificates."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core.config import ALPHA, DEFAULT_KMAX_LN_A, K_MAX, KMAX_LN_A, LOGM_BINS, LOGM_RANGE, LOGM_STEPS
from core.experiments import run
from core.report import p_max, rho, table
from core.simulate import MONITORS
from plotting import violin
from plotting.style import DAB, DAB_FILL, WIDTH, legend_below, plt, save

SYSTEMS = (('system2_input', 'System2 + input (invalid)'), ('system3', 'System3 (invalid)'))
DAB_ROW = list(MONITORS).index('dab')
EDGES = np.log(10) * np.linspace(*LOGM_RANGE, LOGM_BINS + 1)        # stored as log10 M_k
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
SHADES = plt.get_cmap('Greys')(np.linspace(0.95, 0.4, len(KMAX_LN_A)))
RHO_STYLE = dict(color=DAB, lw=1.1, marker='o', ms=3.4, mfc='white', mec=DAB, mew=0.8, zorder=4)

rows = []
for name, title in SYSTEMS:
    for c, p in zip(KMAX_LN_A, run(name)['alarm_rate'][DAB_ROW]):
        rows.append([title.split(' (')[0], f'{c:g}', rho(p), p_max(p)])
print(table(['system', 'K_max ln a', 'DAB rho', 'DAB p_max'], rows))

fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.25), gridspec_kw=dict(wspace=0.36, left=0.07, right=0.99, top=0.88,
                                                                       bottom=0.3))
positions = np.arange(len(KMAX_LN_A))
for ax, (name, title), letter in zip(axes[:2], SYSTEMS, 'ab'):
    p_all = 100 * run(name)['alarm_rate'][DAB_ROW]
    ax.axvline(positions[list(KMAX_LN_A).index(DEFAULT_KMAX_LN_A)], color='#c8c8c8', lw=0.8, zorder=1)
    for x, p in zip(positions, p_all):
        violin.alarm_rates(ax, x, p, DAB_FILL, 0.42)
    ax.plot(positions, [100 * np.mean(p > 100 * ALPHA) for p in p_all], **RHO_STYLE)
    violin.alarm_axis(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{c:g}' for c in KMAX_LN_A])
    ax.set_xlim(-0.6, len(KMAX_LN_A) - 0.4)
    ax.set_xlabel(r'$K_{\max}\ln a$')
    ax.set_title(f'({letter}) {title}', fontsize=8)
axes[0].set_ylabel(r'DAB alarm rate $\hat p_i$ and $\rho_\alpha$ (%)')

steps = np.arange(K_MAX + 1)
for ax, (name, title), letter in zip(axes[2:], SYSTEMS, 'cd'):
    for k, counts in zip(LOGM_STEPS, run(name)['logm_hist'][DAB_ROW]):
        violin.draw(ax, k, CENTERS, violin.smooth(counts.astype(float)), DAB_FILL, 21)
    for c, color in zip(KMAX_LN_A, SHADES):
        ax.plot(steps, c * (1 - steps / K_MAX) + np.log(1 / ALPHA), color=color, lw=1.2,
                ls='--' if c == 0 else '-', zorder=3)
    ax.text(0.02, np.log(1 / ALPHA), r'$\ln(1/\alpha)$', transform=ax.get_yaxis_transform(), fontsize=8,
            va='bottom', ha='left')
    ax.set_xlim(0, K_MAX + 25)
    ax.set_xticks(range(0, K_MAX + 1, 100))
    ax.set_ylim(-1.75, 8.5)
    ax.set_xlabel('time step $k$')
    ax.set_title(f'({letter}) {title}', fontsize=8)
axes[2].set_ylabel(r'DAB evidence $\ln M_k$')
for ax in axes:
    ax.minorticks_off()

handles = ([Line2D([], [], ls='', label=r'threshold, $K_{\max}\ln a=$')]
           + [Line2D([], [], color=color, lw=1.2, ls='--' if c == 0 else '-', label=f'{c:g}')
              for c, color in zip(KMAX_LN_A, SHADES)]
           + [Patch(color=DAB_FILL, label='DAB: distribution'),
              Line2D([], [], **{**RHO_STYLE, 'zorder': 1}, label=r'$\rho_\alpha$'),
              Line2D([], [], color='#c8c8c8', lw=0.8, label='default')])
legend_below(fig, handles, handlelength=1.4, columnspacing=1.0)
save(fig, 'fig_factor_a')
