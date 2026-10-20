"""Table 3 and Figure 6: Direct, fixed-bet mixtures and DAB on the same runs (K_max ln a = 4)."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core.config import ALPHA
from core.experiments import run
from core.report import p_max, rates, rho, table
from plotting import violin
from plotting.style import DAB, DIRECT, LEGEND_GAP, WIDTH, legend_below, plt, save

#           monitor, label, colour
MONITORS = (('direct', 'Direct', DIRECT), ('fraction', 'Fraction mixture', '#8c8c8c'),
            ('absolute1', r'Absolute mixture $\mathcal{A}_1$', '#a9a9a9'),
            ('absolute2', r'Absolute mixture $\mathcal{A}_2$', '#c4c4c4'),
            ('dab', 'DAB', DAB))
#           experiment, title, top of the y-axis (%)
SYSTEMS = (('system1', '(a) System1 (valid)', 10), ('system2', '(b) System2 (valid)', 10),
           ('system2_input', '(c) System2 + input (invalid)', 100), ('system3', '(d) System3 (invalid)', 100))
RHO = dict(marker='o', ms=3.4, mfc='white', mec='#111111', mew=0.8, ls='', zorder=5)

rows = []
for name, title, _ in SYSTEMS:
    results = run(name)
    rows.append([title[4:]] + [v for m, _, _ in MONITORS for v in (rho(rates(results, m)), p_max(rates(results, m)))])
print(table(['system'] + [f'{m} {s}' for m, _, _ in MONITORS for s in ('rho', 'p_max')], rows))

fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(WIDTH, 1.95), gridspec_kw=dict(wspace=0.3, left=0.07, right=0.99,
                                                                                  top=0.86, bottom=0.16))
x = np.arange(len(MONITORS))
for ax, (name, title, top) in zip(axes, SYSTEMS):
    results = run(name)
    for xi, (monitor, _, color) in zip(x, MONITORS):
        p = 100 * rates(results, monitor)
        violin.alarm_rates(ax, xi, p, color, 0.42, top=top)
        ax.plot(xi, 100 * np.mean(p > 100 * ALPHA), **RHO)
    violin.alarm_axis(ax, top=top)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(MONITORS) - 0.4)
    ax.minorticks_off()
    ax.set_title(title, fontsize=8)
axes[0].set_ylabel(r'alarm rate $\hat p_i$ and $\rho_\alpha$ (%)')

handles = ([Patch(color=color, label=label) for _, label, color in MONITORS]
           + [Line2D([], [], **{**RHO, 'zorder': 1}, label=r'$\rho_\alpha$')])
legend_below(fig, handles, handlelength=1.4, gap=LEGEND_GAP + 0.05)
save(fig, 'fig_baselines')
